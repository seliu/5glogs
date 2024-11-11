from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd

CHUNK_SIZE = 200

# LOG_LENGTH = 5480 # len(max(logs_list, key=len)) == 5480
# LOG_LENGTH = 2560 # for hangover dataset
LOG_LENGTH = 1000 # for hangover dataset 2

class QmdlLogsHelper:
    def __init__(self, qmdl_paths: Sequence[Path]):
        df_logs_list = []
        for qmdl_path in qmdl_paths:
            df_logs_list.append(self.get_dataframe_with_handover_start_label(qmdl_path))
        self.df_logs = pd.concat(df_logs_list, ignore_index=True, axis=0)

    def split_and_replace(self, logs: bytes) -> list:
        # raw logs -> logs list
        logs_list = logs.split(b'\x7e')
        for i, log in enumerate(logs_list):
            log = log.replace(b'\x7d\x5d', b'\x7d') # 7D5D -> 7D
            log = log.replace(b'\x7d\x5e', b'\x7e') # 7D5E -> 7E
            logs_list[i] = log
        return logs_list

    def get_logs_list(self, qmdl_path: Path) -> list[bytes]:
        with qmdl_path.open("rb") as f:
            logs_bytes = f.read()
            logs_list = self.split_and_replace(logs_bytes)
        return logs_list

    def get_handover_start_bytes_list(self, qmdl_path: Path) -> list:
        from subprocess import run
        result = run(['/disk/sean/5glogs/pattern_recognizer/x86-pattern-recognizer', 'debug',  qmdl_path], capture_output=True, text=True)
        ho_start_hex_str_list = result.stdout.split('\n')[1:-1]
        ho_start_hex_bytes_list = [bytes.fromhex(hex_str) for hex_str in ho_start_hex_str_list]
        return ho_start_hex_bytes_list

    def get_handover_start_logs_index_list(self, logs_list: list, qmdl_path: Path) -> list[int]:
        ho_start_hex_bytes_list = self.get_handover_start_bytes_list(qmdl_path)
        match_log_index_list = []
        for hex_bytes in ho_start_hex_bytes_list:
            match_index = []
            for i, log in enumerate(logs_list):
                if hex_bytes in log:
                    match_index.append(i)
            if len(match_index) == 1:
                match_log_index_list.append(match_index[0])
            elif len(match_index) > 1:
                print(f'{qmdl_path}: ho_start {hex_bytes.hex()} got {len(match_index)} matched logs.')
        print(f'{qmdl_path}: ho_start = {len(ho_start_hex_bytes_list):02d}, matched_logs = {len(match_log_index_list):02d}')
        return match_log_index_list

    def get_dataframe_with_handover_start_label(self, qmdl_path: Path) -> pd.core.frame.DataFrame:
        # logs list
        logs_list = self.get_logs_list(qmdl_path)
        logs_len_list = [len(l) for l in logs_list]
        
        # handover start logs index list
        ho_start_logs_id_list = self.get_handover_start_logs_index_list(logs_list, qmdl_path)
        
        # build dataframe
        df_logs = pd.DataFrame({
            'log': logs_list,
            'log_len': logs_len_list,
            'handover_start': [0]*len(logs_list),
        })
        df_logs.loc[ho_start_logs_id_list, 'handover_start'] = 1
        return df_logs

    ##############################################################################################

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

    # def get_dataset_ho(self, chunk_size: int = 100):
    #     modem_event_logs = {i:log for i, log in enumerate(self.logs_list) if log[0:1] == b'\x60'}

    #     ho_start_logs = {} # hangover start logs
    #     for k in modem_event_logs.keys():
    #         if (a := b'\x74') in (log := self.logs_list[k]):
    #             id1 = log.index(a)
    #             if (id2 := id1 + 1) < len(log):
    #                 nxt = log[id2] & b'\x0f'[0]
    #                 if nxt == b'\x0c'[0]:
    #                     ho_start_logs[k] = log

    #     positive_ids = []
    #     for k in ho_start_logs.keys():
    #         positive_ids.append([i for i in range(k-chunk_size, k)])
            
    #     neg_start, neg_delta = 40000, 40000
    #     negative_logs_keys = [k for k in range(neg_start, neg_start + 5 * neg_delta, neg_delta)]
    #     negative_ids = []
    #     for k in negative_logs_keys:
    #         negative_ids.append([i for i in range(k-chunk_size, k)])

    #     # 3 entries per data: start_id, end_id, label
    #     positive_data = [np.array([k-chunk_size, k, 1]) for k in ho_start_logs.keys()]
    #     negative_data = [np.array([k-chunk_size, k, 0]) for k in negative_logs_keys]
    #     data = np.vstack([positive_data, negative_data])
    #     return data

    def sample_negative_ho_start_id(self, hostart_id_list, width: int = 100):
        id_fine = False
        while id_fine is False:
            idx = np.random.randint(0, len(self.df_logs))
            for i in hostart_id_list:
                w = width * 1.5
                if i - w <= idx <= i + w:
                    id_fine = False
                    break
                else:
                    id_fine = True
            # id_fine = True
        return idx

    def get_dataset_ho(self, chunk_size: int = 100, negative_ratio: int = 1):
        df_hostart = self.df_logs[self.df_logs.handover_start == 1] # hangover start

        # positive_ids = []
        # for k in df_hostart.index:
        #     positive_ids.append([i for i in range(k-chunk_size, k)])
        negative_logs_keys = []
        # negative_ids = []
        for _ in range(len(df_hostart) * negative_ratio):
            k = self.sample_negative_ho_start_id(df_hostart.index, width=chunk_size)
            negative_logs_keys.append(k)
            # negative_ids.append([i for i in range(k-chunk_size, k)])
        assert len(negative_logs_keys) == len(df_hostart) * negative_ratio
        
        # 3 entries per data: start_id, end_id, label
        positive_data = [np.array([k-chunk_size, k, 1]) for k in df_hostart.index]
        negative_data = [np.array([k-chunk_size, k, 0]) for k in negative_logs_keys]
        data = np.vstack([positive_data, negative_data]) if negative_data else np.stack(positive_data)
        return data
