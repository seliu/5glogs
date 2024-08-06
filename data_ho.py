from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from utils import QmdlLogsHelper
import numpy as np

class QmdlDataset(Dataset):
    def __init__(self, filepath: Path, chunk_size: int = 100, split: str = None):
        logs_helper = QmdlLogsHelper(filepath)
        data = logs_helper.get_dataset_ho(chunk_size=chunk_size) # 3 entries per data: start_id, end_id, label
        data_label0 = data[np.where(data[:,2] == 0)[0]]
        data_label1 = data[np.where(data[:,2] == 1)[0]]
        assert len(data) == len(data_label0) + len(data_label1)
        data_balanced = np.vstack([data_label0, data_label1])
        # assert len(data_balanced) == len(data_label0_0) + len(data_label1_1)
        # # print(f"{len(data_balanced)} == {len(data_label0_0) + len(data_label1_1)} ( = {len(data_label0_0)} + {len(data_label1_1)} )")
        x = data_balanced[:, 0:2]
        y = data_balanced[:, 2].astype(np.int64)

        # # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42, stratify=y)

        idx_split_train_0 = int(len(data_label0) * 0.8)
        # # print(f'idx_split_train_0: {idx_split_train_0} / {len(data_label0_0)}')
        data_label0_train = data_label0[:idx_split_train_0]
        data_label0_test =  data_label0[idx_split_train_0:]
        idx_split_train_1 = int(len(data_label1) * 0.8)
        # # print(f'idx_split_train_1: {idx_split_train_1} / {len(data_label1_1)}')
        data_label1_train = data_label1[:idx_split_train_1]
        data_label1_test =  data_label1[idx_split_train_1:]
        data_balanced_train = np.vstack([data_label0_train, data_label1_train])
        data_balanced_test = np.vstack([data_label0_test, data_label1_test])
        x_train = data_balanced_train[:, 0:2]
        y_train = data_balanced_train[:, 2].astype(np.int64)
        x_test = data_balanced_test[:, 0:2]
        y_test = data_balanced_test[:, 2].astype(np.int64)

        self.data_array = logs_helper.get_logs_array()

        # split: train, test, all
        match split:
            case 'train':
                self.data_x, self.data_y = x_train, y_train
            case 'test':
                self.data_x, self.data_y = x_test, y_test
            case _:
                self.data_x, self.data_y = x, y

    def __len__(self):
        return len(self.data_y)

    def __getitem__(self, idx):
        data = self.data_x[idx] # 3 entries per data: start_id, end_id, label
        data_item = self.data_array[data[0]:data[1], :]
        label = self.data_y[idx]
        return data_item, label

if __name__ == '__main__':
    path_project = "/disk/sean/5glogs"
    path_file = "sa_ho_event.qmdl"
    qmdl_logs_path = Path(path_project) / Path(path_file)
    print(qmdl_logs_path)
    dataset = QmdlDataset(qmdl_logs_path, split='test')
    data_item, label = dataset.__getitem__(0)
    print(dataset.__len__(), data_item.shape, label, type(label))

    from torch.utils.data import DataLoader
    dl_test = DataLoader(dataset, batch_size=32, shuffle=False)
    # test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
    test_features, test_labels = next(iter(dl_test))
    print(test_features.shape, test_features.dtype, test_labels.shape, test_labels.dtype)
