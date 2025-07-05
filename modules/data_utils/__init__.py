from typing import Any, Callable, cast, Tuple, TypeVar, Generic, Callable
from functools import reduce
from abc import ABC, abstractmethod

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from constants import FREQ_RES_MHZ, FREQ_START_MHZ, FREQS_MHZ

T = TypeVar('T', bound=torch.Tensor)

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

class DataProcessor(ABC):

    @abstractmethod
    def __sentinel__(self) -> None:
        """A sentinel method to ensure this class is abstract."""

    @staticmethod
    def min_max_normalize(data: torch.Tensor) -> torch.Tensor:
        _min, _max = data.aminmax()
        if _min == _max:
            return torch.zeros_like(data)

        return (data - _min) / (_max - _min)

    @staticmethod
    def sparse_centers(data: torch.Tensor) -> torch.Tensor:
        data_out: torch.Tensor = torch.zeros_like(FREQS_MHZ, dtype=torch.float64)
        midpoints: torch.Tensor = data.mean(dim=1)
        indexes: torch.Tensor = ((midpoints - FREQ_START_MHZ) / FREQ_RES_MHZ).round().to(torch.int64)
        indexes = indexes.clamp(0, data_out.numel() - 1)

        data_out[indexes] = 1

        return data_out

    @staticmethod
    def identity(data: torch.Tensor) -> torch.Tensor:
        return data

class MultiPipeline(Tuple[torch.Tensor, ...]):

    @classmethod
    def multiply(cls, n: int) -> Callable[[torch.Tensor], 'MultiPipeline']:
        def mark(data: torch.Tensor) -> torch.Tensor:
            setattr(data, '__multiplied__', True)
            return data

        def _multiply(data: torch.Tensor) -> MultiPipeline:
            if getattr(data, '__multiplied__', False):
                raise RuntimeError(
                    'multiply() can only be called once within a transform pipeline.'
                )

            return cls(mark(data.clone()) for _ in range(n))

        return _multiply

    @classmethod
    def apply(cls, *args: Callable[[torch.Tensor], torch.Tensor]) -> Callable[['MultiPipeline'], 'MultiPipeline']:
        def _apply(multi_pipeline: MultiPipeline) -> MultiPipeline:
            if not isinstance(multi_pipeline, cls):
                raise TypeError(f'Expected MultiPipeline, got {type(multi_pipeline).__name__}')

            return cls(
                arg(data) for
                arg, data in zip(args, multi_pipeline, strict=True)
            )

        return _apply