import h5py
import ast
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class H5LazyDataset(Dataset):

    def _get_file(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.path, 'r')

        return self._file

    def __init__(self, path: str, data_key: str, label_key: str) -> None:
        self._file: h5py.File | None = None
        self.path: str = path
        self.data_key: str = data_key
        self.label_key: str = label_key

        with h5py.File(self.path, 'r') as f:
            self.length = len(f[self.label_key])

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index) -> Any:
        f = self._get_file()
        data: torch.Tensor = torch.tensor(f[self.data_key][index])

        raw_label: np.bytes_ = f[self.label_key][index]
        str_label: str = raw_label.decode('utf-8')
        label: torch.Tensor = torch.tensor(ast.literal_eval(str_label))

        return data, label

    def __del__(self) -> None:
        if self._file is not None:
            self._file.close()
