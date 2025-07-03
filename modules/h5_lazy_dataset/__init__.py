import ast
from typing import Any, Callable, Optional, Tuple
from functools import reduce

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class H5LazyDataset(Dataset):

    def __init__(
        self, path: str, data_key: str, label_key: str,
        data_transforms: Tuple[Callable[[Any], Any], ...] = (),
        label_transforms: Tuple[Callable[[Any], Any], ...] = (),
    ) -> None:
        self._file: h5py.File | None = None
        self.path: str = path
        self.data_key: str = data_key
        self.label_key: str = label_key
        self.data_transforms: Tuple[Callable[[Any], Any], ...] = data_transforms
        self.label_transforms: Tuple[Callable[[Any], Any], ...] = label_transforms

        self.length = 0

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index) -> Tuple[Any, Any]:
        f = self.get_file()
        data: Any = f[self.data_key][index]

        label: Any = f[self.label_key][index]

        data_transformed: torch.Tensor = reduce(lambda _data, transformer: transformer(_data), self.data_transforms, data)
        label_transformed: torch.Tensor = reduce(lambda _label, transformer: transformer(_label), self.label_transforms, label)

        return data_transformed, label_transformed

    def __del__(self) -> None:
        if self._file is not None:
            self._file.close()

    def get_file(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.path, 'r')

        return self._file