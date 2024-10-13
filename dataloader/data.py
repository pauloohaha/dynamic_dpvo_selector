import torch
import torch.nn as nn
from tqdm import tqdm
import pickle
import sys
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import math
import numpy as np

class ConfidenceDataset(Dataset):
    def __init__(self, confidence_data, difficulty_labels):
        self.confidence_data = confidence_data  # List of confidence sequences
        self.difficulty_labels = difficulty_labels  # List of corresponding difficulty scores

    def __len__(self):
        return len(self.confidence_data)

    def __getitem__(self, idx):
        return self.confidence_data[idx], self.difficulty_labels[idx]