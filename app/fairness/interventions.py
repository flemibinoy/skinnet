import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def disease_strategy(labels, strategy, seed):
    counts = np.bincount(labels, minlength=7)
    if (counts == 0).any():
        raise ValueError("Training requires all seven disease classes")
    weights = 1.0 / counts
    if strategy == "class_weighted":
        return torch.tensor(weights / weights.mean(), dtype=torch.float32), None
    if strategy == "balanced_sampling":
        sampler = WeightedRandomSampler(
            weights[np.asarray(labels)],
            len(labels),
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
        return None, sampler
    if strategy != "baseline":
        raise ValueError("Unsupported intervention; skin-tone balancing is unavailable")
    return None, None
