from typing import Any, Callable, cast, Tuple, Callable, Protocol
from functools import reduce
from abc import ABC, abstractmethod

import h5py
import matplotlib.axes
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

from constants import FREQ_RES_MHZ, FREQ_START_MHZ, FREQS_MHZ

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


class DataUtils(ABC):

    @abstractmethod
    def __sentinel__(self) -> None:
        """A sentinel method to ensure this class is abstract."""

    @staticmethod
    def plot_spectrum(
        data: torch.Tensor, freqs: torch.Tensor = FREQS_MHZ, axes: matplotlib.axes.Axes | None = None, title: str | None = None
    ) -> None:

        if axes is None:
            _, axes = plt.subplots()

        axes.plot(freqs.numpy(force=True), data.numpy(force=True))
        axes.set_xlabel('Frequency')
        axes.set_ylabel('Magnitude')
        if title is not None:
            axes.set_title(title)

        axes.grid(True)

    @staticmethod
    def identity(data: torch.Tensor) -> torch.Tensor:
        return data

    @staticmethod
    def min_max_normalize(data: torch.Tensor) -> torch.Tensor:
        _min, _max = data.aminmax()
        if _min == _max:
            return torch.zeros_like(data)

        return (data - _min) / (_max - _min)

    @staticmethod
    def _calculate_indexes(freqs: torch.Tensor) -> torch.Tensor:
        indexes: torch.Tensor = ((freqs - FREQ_START_MHZ) / FREQ_RES_MHZ).round().to(torch.int64)

        return indexes.clamp(0, FREQS_MHZ.numel() - 1)

    @classmethod
    def _sparse(cls, data: torch.Tensor) -> torch.Tensor:
        indexes: torch.Tensor = cls._calculate_indexes(data)
        data_out: torch.Tensor = torch.zeros_like(FREQS_MHZ, dtype=torch.float64)
        data_out[indexes] = 1

        return data_out

    @classmethod
    def sparse_band_centers(cls, data: torch.Tensor) -> torch.Tensor:
        band_centers: torch.Tensor = data.mean(dim=1)

        return cls._sparse(band_centers)

    @classmethod
    def sparse_band_starts(cls, data: torch.Tensor) -> torch.Tensor:
        band_starts: torch.Tensor = data[:, 0]

        return cls._sparse(band_starts)


    @classmethod
    def sparse_band_stops(cls, data: torch.Tensor) -> torch.Tensor:
        band_stops: torch.Tensor = data[:, 1]

        return cls._sparse(band_stops)


class MultiPipeline(tuple['torch.Tensor | MultiPipeline | Any', ...]):

    def __str__(self, n: int = 0) -> str:
        body: str = ',\n'.join(
            (n + 1) * '\t' + data_str for data_str in (
                data.__str__(n + 1) if isinstance(data, self.__class__) else data.__str__()
                for data in self
            )
        )
        return f"""MultiPipeline(\n{body}\n{n * '\t'})"""

    @classmethod
    def split(cls, n: int) -> Callable[['torch.Tensor | MultiPipeline'], 'MultiPipeline']:
        def _split(data: torch.Tensor | MultiPipeline) -> MultiPipeline:
            return cls(data.clone() for _ in range(n))

        return _split

    @classmethod
    def apply(cls, *args: Callable[[Any], Any]) -> Callable[['MultiPipeline'], 'MultiPipeline']:
        def _apply(multi_pipeline: MultiPipeline) -> MultiPipeline:
            if not isinstance(multi_pipeline, cls):
                raise TypeError(f'Expected MultiPipeline, got {type(multi_pipeline).__name__}')

            return cls(
                arg(data) for
                arg, data in zip(args, multi_pipeline, strict=True)
            )

        return _apply

    @classmethod
    def _flatten(cls, data: Any) -> Tuple:
        if isinstance(data, cls):
            return cls.flatten(data)

        return (data,)

    @classmethod
    def flatten(cls, multi_pipeline: 'MultiPipeline') -> 'MultiPipeline':
        if not isinstance(multi_pipeline, cls):
            raise TypeError(f'Expected MultiPipeline, got {type(multi_pipeline).__name__}')

        return cls(
            reduce(lambda accumulator, data: accumulator + cls._flatten(data), multi_pipeline, ())
        )

    def clone(self) -> 'MultiPipeline':
        return self.__class__(data.clone() for data in self)