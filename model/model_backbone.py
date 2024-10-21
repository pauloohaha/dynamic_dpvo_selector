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



class ConfidenceToDifficultyModel(nn.Module):
    def __init__(self,seq_len = 100, input_dim=3, d_model=64, nhead=8, num_layers=15, dim_feedforward=128):
        super(ConfidenceToDifficultyModel, self).__init__()

        # Transformer encoder layers
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        # Linear layers for final difficulty estimation
        self.fc1 = nn.Linear(d_model, 64)
        self.fc2 = nn.Linear(64, 1)  # Output single difficulty score

        # Positional encoding (to help with sequence order)
        self.positional_encoding = nn.Parameter(torch.zeros(1248, d_model))  # Maximum length = 1248

        # Input embedding layer to map 1D confidence scores to d_model dimension
        self.embedding = nn.Linear(input_dim, d_model)

        self.pooling_layer = nn.Linear(seq_len, 1)
        

    def forward(self, x):
        # x: input confidence scores (batch_size, sequence_length, input_dim)
        
        # Get the batch size and sequence length
        batch_size, seq_len, _ = x.size()
        
        # Apply embedding to map confidence scores to d_model
        x = self.embedding(x)  # (batch_size, sequence_length, d_model)
        
        # Add positional encoding
        x = x + self.positional_encoding[:seq_len, :].unsqueeze(0)

        # Transformer encoder expects (sequence_length, batch_size, d_model)
        x = x.permute(1, 0, 2)

        # Pass through the transformer encoder
        x = self.transformer_encoder(x)  # (sequence_length, batch_size, d_model)

        # linear layer
        x = x.permute(1, 2, 0)  # (batch_size, d_model, sequence_length)
        x = self.pooling_layer(x).squeeze(-1)  # (batch_size, d_model)

        # Pass through the fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # (batch_size, 1)

        return x
