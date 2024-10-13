import torch
import torch.nn as nn
from tqdm import tqdm
import pickle
import wandb
import sys
print("Python version:", sys.version)
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
import numpy as np
from model.model_backbone import ConfidenceToDifficultyModel
from dataloader.data import ConfidenceDataset
from utils.preprocess import aglin_and_preprocess_log_data
from test_model import test_model

load_input_label = 0

# Define the path where you want to save the model
model_path = "confidence_to_difficulty_model.pth"
new_model_path = 'confidence_to_difficulty_model_new.pth'

euroc_scenes = [
    "MH_01_easy",
    "MH_02_easy",
    "MH_03_medium",
    "MH_04_difficult",
    "MH_05_difficult",
    "V1_01_easy",
    "V1_02_medium",
    "V1_03_difficult",
    "V2_01_easy",
    "V2_02_medium",
    "V2_03_difficult",
]

dynamic_log_path = '/pool0/piaodeng/distributed_dpvo/dynamic_slam_log/logs/'


def train():
    wandb.init(project="confidence-to-difficulty")
    if load_input_label == 0:
      confidence_data = torch.tensor([])
      difficulty_labels = torch.tensor([])
      for scene_name in euroc_scenes:
        print(scene_name)
        for num_patches in range(96, 0, -16):
          for num_frame in range(22, 4, -6):
              print(str(num_patches) + str(num_frame))
              for trails_idx in range(0, 1):
                current_input_log = dynamic_log_path + 'dynamic_slam_log_confidence_array_'+scene_name+'_'+str(num_patches)+'_patches_'+str(num_frame)+'_frames_trials_'+str(trails_idx)+'.pickle'
                dynamic_log_picke_file = open(current_input_log, 'rb')
                logged_data = pickle.load(dynamic_log_picke_file)

                confidence_data_1_test,  difficulty_labels_1_test= aglin_and_preprocess_log_data(logged_data, num_patches, num_frame)

                confidence_data = torch.cat((confidence_data, torch.tensor(confidence_data_1_test, dtype=torch.float32)), dim=0)
                difficulty_labels = torch.cat((difficulty_labels, torch.tensor(difficulty_labels_1_test, dtype=torch.float32)), dim=0)

      torch.save(confidence_data, '../confidence_data_train.pth')
      torch.save(difficulty_labels, '../difficulty_labels_train.pth')
    else:
      #bypass the data preparation
      confidence_data = torch.load("../confidence_data_train.pth")
      difficulty_labels = torch.load("../difficulty_labels_train.pth")

    confidence_data = confidence_data
    dataset = ConfidenceDataset(confidence_data, difficulty_labels)
    data_loader = DataLoader(dataset, 30, shuffle=True)

    criterion = nn.MSELoss()

    model = ConfidenceToDifficultyModel(num_layers=15).to('cuda')# Initialize the model
    model.load_state_dict(torch.load(model_path))
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    # Training loop
    num_epochs = 999999
    model.train()  # Set the model to training mode

    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}")
        for i, (inputs, targets) in enumerate(progress_bar):
            # Move inputs and targets to device (e.g., GPU if available)
            inputs, targets = inputs.to('cuda'), targets.to('cuda').unsqueeze(1)

            # Forward pass: Compute predicted difficulty
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, targets)
            #print(outputs[0].item(), targets[0].item())

            # Backward pass: Compute gradients
            optimizer.zero_grad()
            loss.backward()
            wandb.log({"loss": loss.item()})
            # Update weights
            optimizer.step()

            progress_bar.set_postfix(loss=running_loss / (i + 1))
            
            # Track loss for logging
            running_loss += loss.item()

        # Print average loss per epoch
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(data_loader)}')

        if epoch % 10 == 9:
            # Save the model's state dictionary
            torch.save(model.state_dict(), new_model_path)
            test_model()
    
    wandb.finish()

if __name__ == '__main__':
  train()