# Standard libraries
import sys
import os
import json
import datetime
import gc
import pickle
import copy
from pathlib import Path
from hashlib import sha256
from multiprocessing import Pool

# Third-party libraries
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

# Project-specific imports
from .build import DATASETS


@DATASETS.register_module()
class dummy_tracklet(Dataset):
    def __init__(self, config):
        self.config = config
    
    def __len__(self):
        return 10000

    def __getitem__(self, index):
        i = np.random.randint(0, 9)
        padding = max(0, 8 - i)
        # padding = 0
        padding = torch.tensor(padding)
        # generating dummy tracklet and center with random values
        # tracklet = torch.concat([torch.zeros((i, 32 , 3)),torch.zeros((padding, 32, 3))]).type(torch.float32)
        # center = torch.concat([torch.zeros((i, 3)),torch.zeros((padding, 3))]).type(torch.float32)
        tracklet = torch.rand((8,32,3)).type(torch.float32)
        center = torch.rand((8,3)).type(torch.float32)
        return 0, 0, (tracklet, center, padding)