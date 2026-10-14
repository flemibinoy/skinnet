import importlib.metadata
import platform
import random

import numpy as np
import torch


def seed_all(seed, threads=2):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def device_for(request="auto"):
    if request == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; use cpu or auto")
    return (
        "cuda" if request == "cuda" or (request == "auto" and torch.cuda.is_available()) else "cpu"
    )


def environment():
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else platform.processor(),
        "packages": {
            p: importlib.metadata.version(p)
            for p in (
                "torch",
                "torchvision",
                "numpy",
                "pandas",
                "scikit-learn",
                "Pillow",
                "streamlit",
            )
        },
    }
