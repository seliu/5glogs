from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Sequence
from torch.utils.data import DataLoader, Dataset
from torchsampler import ImbalancedDatasetSampler
from utils_s4 import LOG_LENGTH, QmdlLogsHelper
import numpy as np

class QmdlDataset(Dataset):
    def __init__(self, filepaths: Sequence[Path], chunk_size: int = 100, negative_ratio: int = 1, split: str = None):
        logs_helper = QmdlLogsHelper(filepaths)
        self.df_logs = logs_helper.df_logs
        data = logs_helper.get_dataset_ho_2(chunk_size, negative_ratio) # 3 entries per data: start_id, end_id, label

        self.data_x = data[:, 0:2]
        self.data_y = data[:, 2]
        
        # data_label0 = data[np.where(data[:,2] == 0)[0]]
        # data_label1 = data[np.where(data[:,2] == 1)[0]]
        # assert len(data) == len(data_label0) + len(data_label1)
        # data_balanced = np.vstack([data_label0, data_label1])

        # x = data_balanced[:, 0:2]
        # y = data_balanced[:, 2].astype(np.int64)

        # # # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42, stratify=y)

        # idx_split_train_0 = int(len(data_label0) * 0.8)
        # # # print(f'idx_split_train_0: {idx_split_train_0} / {len(data_label0_0)}')
        # data_label0_train = data_label0[:idx_split_train_0]
        # data_label0_test =  data_label0[idx_split_train_0:]
        # idx_split_train_1 = int(len(data_label1) * 0.8)
        # # # print(f'idx_split_train_1: {idx_split_train_1} / {len(data_label1_1)}')
        # data_label1_train = data_label1[:idx_split_train_1]
        # data_label1_test =  data_label1[idx_split_train_1:]
        # data_balanced_train = np.vstack([data_label0_train, data_label1_train])
        # data_balanced_test = np.vstack([data_label0_test, data_label1_test])
        # x_train = data_balanced_train[:, 0:2]
        # y_train = data_balanced_train[:, 2].astype(np.int64)
        # x_test = data_balanced_test[:, 0:2]
        # y_test = data_balanced_test[:, 2].astype(np.int64)

        # self.data_array = logs_helper.get_logs_array()

        # # split: train, test, all
        # match split:
        #     case 'train':
        #         self.data_x, self.data_y = x_train, y_train
        #     case 'test':
        #         self.data_x, self.data_y = x_test, y_test
        #     case _:
        #         self.data_x, self.data_y = x, y

    def __len__(self):
        return len(self.data_y)

    def __getitem__(self, idx):
        data = self.data_x[idx] # 3 entries per data: start_id, end_id, label
        logs = self.df_logs[data[0]:data[1]].log

        # bytes -> uint8
        logs_uint8_list = [np.frombuffer(log, dtype=np.uint8) for log in logs]

        # uint8 -> float
        logs_float_array = np.zeros([len(logs), LOG_LENGTH], dtype=np.float32)
        for i, log in enumerate(logs_uint8_list):
            if len(log) > LOG_LENGTH:
                logs_float_array[i][:LOG_LENGTH] = log[:LOG_LENGTH]
            else:
                logs_float_array[i][:len(log)] = log
        logs_float_array /= 255.0
        
        label = self.data_y[idx]
        return logs_float_array, label

    #
    # required by torchsampler.ImbalancedDatasetSampler
    #
    def get_labels(self):
        return self.data_y


if __name__ == '__main__':
    path_project = Path('/disk/sean/5glogs/')
    
    # path_data_1 = Path('data.hangover/trip_1.forward/')
    # path_rawlogs_list_1 = [path_project / path_data_1 / Path(f'qmdl_{i}.qmdl') for i in range(1, 20+1)]
    # path_data_2 = Path('data.hangover/trip_1.backward/')
    # path_rawlogs_list_2 = [path_project / path_data_2 / Path(f'qmdl_{i}.qmdl') for i in range(1, 23+1)]
    # path_rawlogs_list = path_rawlogs_list_1 + path_rawlogs_list_2
    # assert len(path_rawlogs_list) == 43
    # ds_train = QmdlDataset(path_rawlogs_list, chunk_size=100, negative_ratio=100)
    # x_train, y_train = ds_train.__getitem__(0)
    # print(x_train, y_train, ds_train.__len__())

    path_data_3 = Path('data.hangover/trip_2.samples/')
    path_rawlogs_list_3 = sorted((path_project / path_data_3).glob('*.qmdl'))
    assert len(path_rawlogs_list_3) == 3
    ds_test = QmdlDataset(path_rawlogs_list_3, chunk_size=100, negative_ratio=100)
    x_test, y_test = ds_test.__getitem__(0)
    print(x_test, y_test, ds_test.__len__())

    dl_test = DataLoader(ds_test, sampler=ImbalancedDatasetSampler(ds_test), batch_size=16)
    x_test, y_test = next(iter(dl_test))
    print(x_test.shape, y_test)

    # data_item, label = dataset.__getitem__(0)
    # print(dataset.__len__(), data_item.shape, label, type(label))

    # from torch.utils.data import DataLoader
    # dl_test = DataLoader(dataset, batch_size=32, shuffle=False)
    # # test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
    # test_features, test_labels = next(iter(dl_test))
    # print(test_features.shape, test_features.dtype, test_labels.shape, test_labels.dtype)
