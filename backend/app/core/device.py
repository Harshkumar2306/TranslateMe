"""
Hardware Device Detection and Memory Utilities
Supports Apple Silicon MPS, NVIDIA CUDA, and CPU fallback.
"""

import torch
import logging

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """
    Automatically selects the best available device:
    1. CUDA if available (NVIDIA GPU / Google Colab)
    2. MPS if available (Apple Silicon Mac)
    3. CPU as fallback
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using NVIDIA CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon Metal Performance Shaders (MPS)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    return device


def get_device_info() -> dict:
    device = get_device()
    info = {
        "device_type": device.type,
        "device_name": "CPU",
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda":
        info["device_name"] = torch.cuda.get_device_name(0)
        info["memory_allocated_mb"] = round(torch.cuda.memory_allocated(0) / (1024 * 1024), 2)
        info["memory_reserved_mb"] = round(torch.cuda.memory_reserved(0) / (1024 * 1024), 2)
    elif device.type == "mps":
        info["device_name"] = "Apple Silicon GPU (MPS)"
    return info
