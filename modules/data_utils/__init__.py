from typing import Any, Callable, cast, Tuple
from functools import reduce

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class H5LazyDataset(Dataset):

    def __init__(
        self, path: str, data_key: str | None = None, label_key: str | None = None,
        data_transforms: Tuple[Callable[[Any], Any], ...] = (),
        label_transforms: Tuple[Callable[[Any], Any], ...] = (),
    ) -> None:
        self._file: h5py.File | None = None
        self.path: str = path
        self.data_key: str | None = data_key
        self.label_key: str | None = label_key
        self.data_transforms: Tuple[Callable[[Any], Any], ...] = data_transforms
        self.label_transforms: Tuple[Callable[[Any], Any], ...] = label_transforms

        self.length = len(self.get_file())

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        file: h5py.File = self.get_file()

        data_transformed: Any = None
        label_transformed: Any = None

        if self.data_key is not None:
            data = cast(h5py.Dataset, file[self.data_key])[index]
            data_transformed = reduce(lambda _data, transformer: transformer(_data), self.data_transforms, data)

        if self.label_key is not None:
            label: Any = cast(h5py.Dataset, file[self.label_key])[index]
            label_transformed = reduce(lambda _label, transformer: transformer(_label), self.label_transforms, label)

        return data_transformed, label_transformed

    def __del__(self) -> None:
        if self._file is not None:
            self._file.close()

    def get_file(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.path, 'r')

        return self._file