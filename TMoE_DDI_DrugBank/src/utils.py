import random

import numpy as np
import scipy.sparse as sp
import torch


def set_random_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic


def normalize_rows(matrix):
    rowsum = np.array(matrix.sum(1))
    inv = np.power(rowsum, -1).flatten()
    inv[np.isinf(inv)] = 0.0
    return sp.diags(inv).dot(matrix)
