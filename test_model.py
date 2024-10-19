import torch
import torch.nn as nn
from tqdm import tqdm
import pickle
import wandb
import sys
print("Python version:", sys.version)
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import math
import numpy as np
from model.model_backbone import ConfidenceToDifficultyModel
from dataloader.data import ConfidenceDataset
from utils.preprocess import aglin_and_preprocess_log_data



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
model_path = "confidence_to_difficulty_model_new.pth0"

def test_model(newest_model = None):
      
      if newest_model != None:
         model_path = newest_model

      for scene_name in euroc_scenes:
        print(scene_name)
        for num_patches in range(96, 90, -16):
          for num_frame in range(22, 20, -6):
              print(str(num_patches) + str(num_frame))
              for trails_idx in range(1, 2):
                current_input_log = dynamic_log_path + 'dynamic_slam_log_confidence_array_'+scene_name+'_'+str(num_patches)+'_patches_'+str(num_frame)+'_frames_trials_'+str(trails_idx)+'.pickle'

                current_saved_label_data = 'test_processed_data/dynamic_slam_log_confidence_array_'+scene_name+'_'+str(num_patches)+'_patches_'+str(num_frame)+'_frames_trials_'+str(trails_idx)+'_label.pth'
                current_saved_confidence_data = 'test_processed_data/dynamic_slam_log_confidence_array_'+scene_name+'_'+str(num_patches)+'_patches_'+str(num_frame)+'_frames_trials_'+str(trails_idx)+'_confidence.pth'

                if os.path.exists(current_saved_confidence_data):
                    #load processed data
                    confidence_data = torch.load(current_saved_confidence_data)
                    difficulty_labels = torch.load(current_saved_label_data)
                else:
                    #load from otiginal data
                    dynamic_log_picke_file = open(current_input_log, 'rb')
                    logged_data = pickle.load(dynamic_log_picke_file)

                    confidence_data_1_test,  difficulty_labels_1_test= aglin_and_preprocess_log_data(logged_data)

                    confidence_data = torch.tensor(confidence_data_1_test, dtype=torch.float32)
                    difficulty_labels = torch.tensor(difficulty_labels_1_test, dtype=torch.float32)

                    torch.save(confidence_data, current_saved_confidence_data)
                    torch.save(difficulty_labels, current_saved_label_data)

                dataset = ConfidenceDataset(confidence_data, difficulty_labels)
                data_loader = DataLoader(dataset, 30, shuffle=False)

                model = ConfidenceToDifficultyModel().to('cuda')# Initialize the model
                model.load_state_dict(torch.load(model_path))
                model.eval()

                all_targets = []
                all_predictions = []
                progress_bar = tqdm(data_loader, desc=f"test")
                with torch.no_grad():
                  for i, (inputs, targets) in enumerate(progress_bar):
                      # Move inputs and targets to device (e.g., GPU if available)
                      inputs, targets = inputs.to('cuda'), targets.to('cuda').unsqueeze(1)

                      # Forward pass: Compute predicted difficulty
                      outputs = model(inputs)
                      
                      all_predictions.extend(outputs.cpu().numpy())
                      all_targets.extend(targets.cpu().numpy())

                tstamp = []

                for i in range(0, len(all_predictions)):
                  tstamp.append(i)

                fig, (ax1) = plt.subplots(figsize=(40, 28))

                ax1.plot(tstamp, all_predictions, 'c-', label='prediction')
                ax1.set_ylabel('prediction', color='c')
                ax1.set_xlabel('time stamp')
                ax1.set_ylim(0, 0.08)
                ax1.tick_params(axis='y', labelcolor='c')

                ax3 = ax1.twinx()
                ax3.plot(tstamp, all_targets, 'b-', label='ground truth')
                ax3.set_ylabel('ground truth', color='b')
                ax3.set_ylim(0, 0.08)
                ax3.tick_params(axis='y', labelcolor='b')

                plt.title(scene_name+' '+str(num_patches)+"patches " + str(num_frame) + 'frames ' + str(trails_idx) + 'trail' )

                output_folder = "plot/dynamic_slam_prediction_"+scene_name+"_"+str(num_patches)+"_patches_"+str(num_frame)+"_frames_trail_"+str(trails_idx)+"/"
                os.makedirs(output_folder, exist_ok=True)

                if os.path.exists(output_folder+"100_input.png"):
                  os.remove(output_folder+"100_input.png")

                plt.savefig(output_folder+"100_input.png", dpi=100, bbox_inches='tight')
                plt.close('all')

                      
                
    # Save the model's state dictionary

if __name__ == '__main__':
  test_model(model_path)