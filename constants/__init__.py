import torch

DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'
"""Device."""

NUM_SAMPLE: int = 100_000
"""Number of samples."""

FREQ_SAMPLE_HZ: float = 1e8
"""Sampling frequency (Hz)."""

FREQ_SAMPLE_MHZ: float = FREQ_SAMPLE_HZ / 1e6
"""Sampling frequency (MHz)."""

FREQ_RES_HZ: float = 1e3
"""Frequency resolution (Hz)."""

FREQ_RES_MHZ: float = FREQ_RES_HZ / 1e6
"""Frequency resolution (MHz)."""

FREQ_START_HZ: float = 2400e6
"""Start frequency for the FFT (Hz)."""

FREQ_START_MHZ: float = FREQ_START_HZ / 1e6
"""Start frequency for the FFT (MHz)."""

FREQ_END_HZ: float = 2500e6
"""End frequency for the FFT (Hz)."""

FREQ_END_MHZ: float = FREQ_END_HZ / 1e6
"""End frequency for the FFT (MHz)."""

FREQ_CENTER_HZ: float = 2450e6
"""Center frequency for the FFT (Hz)"""

FREQ_CENTER_MHZ: float = FREQ_CENTER_HZ / 1e6
"""Center frequency for the FFT (MHz)."""

FREQS_HZ: torch.Tensor = (
    torch.fft.fftshift(
        torch.fft.fftfreq(NUM_SAMPLE, d=1 / FREQ_SAMPLE_HZ)
    ).to(dtype=torch.float64) + FREQ_CENTER_HZ
)

FREQS_MHZ: torch.Tensor = FREQS_HZ / 1e6
"""Frequencies for the FFT (MHz), shifted to center around FREQ_CENTER."""
