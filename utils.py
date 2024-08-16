from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd

# LOG_LENGTH = 5480 # len(max(logs_list, key=len)) == 5480
LOG_LENGTH = 2560 # for hangover dataset

class QmdlLogsHelper:
    def __init__(self, qmdl_paths: Sequence[Path]):
        self.logs_list = []
        self.logs_len_list = []
        for qmdl_path in qmdl_paths:
            with qmdl_path.open("rb") as f:
                logs_bytes = f.read()
                logs_list = self.clean_and_split(logs_bytes)
                self.logs_list.extend(logs_list)
                logs_len_list = [len(l) for l in logs_list]
                self.logs_len_list.extend(logs_len_list)

        # self.logs_float = self.to_float_array(self.logs_list)
        # self.labels_array = self.get_detach_request_labels_array()
        self.df_logs = pd.DataFrame({
            'log': self.logs_list,
            'log_len': self.logs_len_list,
            'hangover_start': [0]*len(self.logs_list),
        })
        self.df_logs = self.label_hangover_start(self.df_logs)

    def clean_and_split(self, logs: bytes) -> list:
        # raw logs -> logs list
        logs = logs.replace(b'\x7d\x5d', b'\x7d') # 7D5D -> 7D
        logs = logs.replace(b'\x7d\x5e', b'\x7e') # 7D5E -> 7E
        logs_list = logs.split(b'\x7e')
        return logs_list

    def label_hangover_start(self, df_logs: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
        for i, log in enumerate(df_logs['log']):
            if log[0:1] == b'\x60' and (a := b'\x74') in log:
                id1 = log.index(a)
                if (id2 := id1 + 1) < len(log) and (nxt := log[id2] & b'\x0f'[0]) == b'\x0c'[0]:
                    df_logs.at[i, 'hangover_start'] = 1
        return df_logs

    def to_float_array(self, logs_list: list) -> np.ndarray:
        # bytes -> uint8
        logs_uint8_list = [np.frombuffer(log, dtype=np.int8) for log in logs_list]

        # uint8 -> float
        logs_float_array = np.zeros([len(logs_list), LOG_LENGTH], dtype=np.float32)
        for i, log in enumerate(logs_uint8_list):
            if len(log) > LOG_LENGTH:
                logs_float_array[i][:LOG_LENGTH] = log[:LOG_LENGTH]
            else:
                logs_float_array[i][:len(log)] = log
        logs_float_array /= 255.0
        
        return logs_float_array

    # def get_detach_request_logs_index(self) -> list:
    #     logs_index = [i for i, log in enumerate(self.logs_list) if b'\x07\x45' in log]
    #     return logs_index

    def get_detach_request_labels_array(self):
        labels_array = np.zeros([len(self.logs_list)], dtype=np.uint8)
        for i, log in enumerate(self.logs_list):
            if b'\x07\x45' in log: # detach request
                labels_array[i] = 1
        return labels_array

    def get_logs_array(self):
        return self.logs_float

    def get_dataset(self, chunk_size: int = 100): # detach request
        # 3 entries per data: start_id, end_id, label
        data = [np.array([i, i+chunk_size, self.labels_array[i:i+chunk_size].sum(dtype=np.uint32)], dtype=np.uint32) for i in range(len(self.logs_float)-chunk_size)]
        data = np.vstack(data)
        return data

    def get_dataset_ho(self, chunk_size: int = 100):
        modem_event_logs = {i:log for i, log in enumerate(self.logs_list) if log[0:1] == b'\x60'}

        ho_start_logs = {} # hangover start logs
        for k in modem_event_logs.keys():
            if (a := b'\x74') in (log := self.logs_list[k]):
                id1 = log.index(a)
                if (id2 := id1 + 1) < len(log):
                    nxt = log[id2] & b'\x0f'[0]
                    if nxt == b'\x0c'[0]:
                        ho_start_logs[k] = log

        positive_ids = []
        for k in ho_start_logs.keys():
            positive_ids.append([i for i in range(k-chunk_size, k)])
            
        neg_start, neg_delta = 40000, 40000
        negative_logs_keys = [k for k in range(neg_start, neg_start + 5 * neg_delta, neg_delta)]
        negative_ids = []
        for k in negative_logs_keys:
            negative_ids.append([i for i in range(k-chunk_size, k)])

        # 3 entries per data: start_id, end_id, label
        positive_data = [np.array([k-chunk_size, k, 1]) for k in ho_start_logs.keys()]
        negative_data = [np.array([k-chunk_size, k, 0]) for k in negative_logs_keys]
        data = np.vstack([positive_data, negative_data])
        return data
