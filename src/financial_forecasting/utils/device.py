"""Hardware device selection utilities."""

from __future__ import annotations

import torch


def get_device(device_setting: str = "auto") -> torch.device:
    """Select the best computational device based on hardware and configuration.

    Parameters
    ----------
    device_setting:
        Device name like "cpu", "cuda", "mps", or "auto".

    Returns
    -------
    torch.device:
        The selected PyTorch device.
    """
    if device_setting == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    return torch.device(device_setting)
