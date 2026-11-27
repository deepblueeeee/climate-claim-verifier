import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

DEFAULT_SEED = 42


def set_seed(seed=DEFAULT_SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SampleDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def save_model(model, path):
    os.makedirs(path, exist_ok=True)
    model.save(path)


def make_torch_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def add_negatives(output, used, gold_set, candidates, count):
    added = 0
    for pos in candidates:
        if pos not in gold_set and pos not in used:
            output.append(pos)
            used.add(pos)
            added += 1
        if added >= count:
            break
