from data_ho_2 import QmdlDataset
from pathlib import Path
from tqdm import tqdm
from torchsampler import ImbalancedDatasetSampler
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, random_split
from utils import CHUNK_SIZE, LOG_LENGTH
import math
import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, d_model=80, n_signals=2, dropout=0.1):
        super().__init__()
        # Project the dimension of features from that of input into d_model.
        self.prenet = nn.Linear(LOG_LENGTH, d_model)
        # TODO:
        #   Change Transformer to Conformer.
        #   https://arxiv.org/abs/2005.08100
        self.encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, dim_feedforward=256, nhead=2
        )
        # self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

        # Project the the dimension of features from d_model into speaker nums.
        self.pred_layer = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.Sigmoid(),
                nn.Linear(d_model, n_signals),
        )

    def forward(self, logs):
        """
        args:
    logs: (batch_size, CHUNK_SIZE, LOG_LENGTH)
        return:
                out: (batch size, n_signals)
        """
        # out: (batch size, length, d_model)
        out = self.prenet(logs)
        # out: (length, batch size, d_model)
        out = out.permute(1, 0, 2)
        # The encoder layer expect features in the shape of (length, batch size, d_model).
        out = self.encoder_layer(out)
        # out: (batch size, length, d_model)
        out = out.transpose(0, 1)
        # mean pooling
        stats = out.mean(dim=1)

        # out: (batch, n_spks)
        out = self.pred_layer(stats)
        return out

def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
            optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
            num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
            num_training_steps (:obj:`int`):
            The total number of training steps.
            num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
            last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
            :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    def lr_lambda(current_step):
        # Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # decadence
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def model_fn(batch, model, criterion, device):
    """Forward a batch through the model."""

    logs, labels = batch
    logs = logs.to(device)
    labels = labels.to(device)

    outs = model(logs)

    # print('\nhaha', outs.shape, outs.dtype, outs, labels.shape, labels.dtype, labels)
    loss = criterion(outs, labels)

    # Get the speaker id with highest probability.
    preds = outs.argmax(1)
    # Compute accuracy.
    accuracy = torch.mean((preds == labels).float())

    return loss, accuracy

def valid(dataloader, model, criterion, device): 
    """Validate on validation set."""

    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    pbar = tqdm(total=len(dataloader.dataset), ncols=0, desc="Valid", unit=" uttr")

    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            loss, accuracy = model_fn(batch, model, criterion, device)
            running_loss += loss.item()
            running_accuracy += accuracy.item()

        pbar.update(dataloader.batch_size)
        pbar.set_postfix(
            loss=f"{running_loss / (i+1):.2f}",
            accuracy=f"{running_accuracy / (i+1):.2f}",
        )

    pbar.close()
    model.train()

    return running_accuracy / len(dataloader)

def parse_args():
    """arguments"""
    config = {
        "data_dir": "./Dataset",
        "save_path": "model.ckpt",
        "batch_size": 32,
        "n_workers": 8,
        "valid_steps": 2000,
        "warmup_steps": 1000,
        "save_steps": 10000,
        "total_steps": 70000,
    }
    return config

def main(
    data_dir,
    save_path,
    batch_size,
    n_workers,
    valid_steps,
    warmup_steps,
    total_steps,
    save_steps,
):
    """Main function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info]: Use {device} now!")

    # data train
    path_project = Path('../')
    path_data_1 = Path('data.hangover/trip_1.forward/')
    path_rawlogs_list_1 = [path_project / path_data_1 / Path(f'qmdl_{i}.qmdl') for i in range(1, 20+1)]
    path_data_2 = Path('data.hangover/trip_1.backward/')
    path_rawlogs_list_2 = [path_project / path_data_2 / Path(f'qmdl_{i}.qmdl') for i in range(1, 23+1)]
    path_rawlogs_list = path_rawlogs_list_1 + path_rawlogs_list_2
    assert len(path_rawlogs_list) == 43
    ds_train = QmdlDataset(path_rawlogs_list, chunk_size=CHUNK_SIZE, negative_ratio=100)
    dl_train = DataLoader(ds_train, sampler=ImbalancedDatasetSampler(ds_train), batch_size=16)

    # data test
    path_test_data = Path('data.hangover/trip_2.samples/')
    path_test_rawlogs_list = sorted((path_project / path_test_data).glob('*.qmdl'))
    assert len(path_test_rawlogs_list) == 3
    ds_test = QmdlDataset(path_test_rawlogs_list, chunk_size=CHUNK_SIZE, negative_ratio=0)
    dl_test = DataLoader(ds_test, batch_size=8)

    train_iterator = iter(dl_train)
    print(f"[Info]: Finish loading data!",flush = True)

    model = Classifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"[Info]: Finish creating model!",flush = True)

    best_accuracy = -1.0
    best_state_dict = None

    pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

    for step in range(total_steps):
        # Get data
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(dl_train)
            batch = next(train_iterator)

        loss, accuracy = model_fn(batch, model, criterion, device)
        batch_loss = loss.item()
        batch_accuracy = accuracy.item()

        # Updata model
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Log
        pbar.update()
        pbar.set_postfix(
            loss=f"{batch_loss:.2f}",
            accuracy=f"{batch_accuracy:.2f}",
            step=step + 1,
        )

        # Do validation
        if (step + 1) % valid_steps == 0:
            pbar.close()

            valid_accuracy = valid(dl_test, model, criterion, device)

            # keep the best model
            if valid_accuracy > best_accuracy:
                best_accuracy = valid_accuracy
                best_state_dict = model.state_dict()

            pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

        # Save the best model so far.
        if (step + 1) % save_steps == 0 and best_state_dict is not None:
            torch.save(best_state_dict, save_path)
            pbar.write(f"Step {step + 1}, best model saved. (accuracy={best_accuracy:.4f})")

    pbar.close()

if __name__ == "__main__":
    main(**parse_args())
