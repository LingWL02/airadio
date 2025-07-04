import torch

DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'
"""Device."""

NUM_SAMPLE: int = 100_000
"""Number of samples."""

FREQ_SAMPLE: float = 1e8
"""Sampling frequency (Hz)."""

FREQ_CENTER: float = 2450e6
"""Center frequency for the FFT (Hz), used for shifting the spectrum."""

FREQS_MHZ: torch.Tensor = (
    torch.fft.fftshift(
        torch.fft.fftfreq(NUM_SAMPLE, d=1 / FREQ_SAMPLE)
    ).to(dtype=torch.float64) + FREQ_CENTER
) / 1e6
"""Frequencies for the FFT (MHz), shifted to center around FREQ_CENTER."""
