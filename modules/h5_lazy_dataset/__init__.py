import h5py
import ast
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class H5LazyDataset(Dataset):
    def __init__(self, path: str, data_key: str, label_key: str) -> None:
        self.path: str = path
        self.data_key: str = data_key
        self.label_key: str = label_key

        with h5py.File(self.path, 'r') as f:
            self.length = len(f[self.label_key])

    def __len__(self):
        return self.length

    def __getitem__(self, index) -> Any:
        with h5py.File(self.path, 'r') as f:
            data: torch.Tensor = torch.tensor(f[self.data_key][index])

            raw_label: np.bytes_ = f[self.label_key][index]
            str_label: str = raw_label.decode('utf-8')
            label: torch.Tensor = torch.tensor(ast.literal_eval(str_label))

        return data, label

