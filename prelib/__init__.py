"""Shared helpers for the pre-training pathway notebooks."""
import os, random
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(ROOT, "checkpoints")
RESULTS_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(ROOT, "figures")


def set_seed(seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
