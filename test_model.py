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
import pandas as pd


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

tartan_scenes = [
  #"debug/P000",
  "abandonedfactory/abandonedfactory/Easy/P000",
  # "abandonedfactory/abandonedfactory/Easy/P001",
  # "abandonedfactory/abandonedfactory/Easy/P002",
  # "abandonedfactory/abandonedfactory/Easy/P004",
  # "abandonedfactory/abandonedfactory/Easy/P005",
  # "abandonedfactory/abandonedfactory/Easy/P006",
  # "abandonedfactory/abandonedfactory/Easy/P008",
  # "abandonedfactory/abandonedfactory/Easy/P009",
  # "abandonedfactory/abandonedfactory/Easy/P010",
  # "abandonedfactory/abandonedfactory/Easy/P011",
  "abandonedfactory/abandonedfactory/Hard/P000",
  # "abandonedfactory/abandonedfactory/Hard/P001",
  # "abandonedfactory/abandonedfactory/Hard/P002",
  # "abandonedfactory/abandonedfactory/Hard/P003",
  # "abandonedfactory/abandonedfactory/Hard/P004",
  # "abandonedfactory/abandonedfactory/Hard/P005",
  # "abandonedfactory/abandonedfactory/Hard/P006",
  # "abandonedfactory/abandonedfactory/Hard/P007",
  # "abandonedfactory/abandonedfactory/Hard/P008",
  # "abandonedfactory/abandonedfactory/Hard/P009",
  # "abandonedfactory/abandonedfactory/Hard/P010",
  # "abandonedfactory/abandonedfactory/Hard/P011",
  "abandonedfactory_night/abandonedfactory_night/Easy/P001",
  # "abandonedfactory_night/abandonedfactory_night/Easy/P002",
  # "abandonedfactory_night/abandonedfactory_night/Easy/P003",
  # "abandonedfactory_night/abandonedfactory_night/Easy/P004",
  # "abandonedfactory_night/abandonedfactory_night/Easy/P005",
  # "abandonedfactory_night/abandonedfactory_night/Easy/P006",
  # "abandonedfactory_night/abandonedfactory_night/Easy/P007",
  # "abandonedfactory_night/abandonedfactory_night/Easy/P008",
  # "abandonedfactory_night/abandonedfactory_night/Easy/P009",
  # "abandonedfactory_night/abandonedfactory_night/Easy/P010",
  # "abandonedfactory_night/abandonedfactory_night/Easy/P011",
  # "abandonedfactory_night/abandonedfactory_night/Easy/P012",
  # "abandonedfactory_night/abandonedfactory_night/Easy/P013",
  # "abandonedfactory_night/abandonedfactory_night/Hard/P000",
  # "abandonedfactory_night/abandonedfactory_night/Hard/P001",
  "abandonedfactory_night/abandonedfactory_night/Hard/P002",
  # "abandonedfactory_night/abandonedfactory_night/Hard/P003",
  # "abandonedfactory_night/abandonedfactory_night/Hard/P005",
  # "abandonedfactory_night/abandonedfactory_night/Hard/P006",
  # "abandonedfactory_night/abandonedfactory_night/Hard/P007",
  # "abandonedfactory_night/abandonedfactory_night/Hard/P008",
  # "abandonedfactory_night/abandonedfactory_night/Hard/P009",
  # "abandonedfactory_night/abandonedfactory_night/Hard/P010",
  # "abandonedfactory_night/abandonedfactory_night/Hard/P011",
  # "abandonedfactory_night/abandonedfactory_night/Hard/P012",
  # "abandonedfactory_night/abandonedfactory_night/Hard/P013",
  # "abandonedfactory_night/abandonedfactory_night/Hard/P014",
  # "amusement/amusement/Easy/P001",
  # "amusement/amusement/Easy/P002",
  # "amusement/amusement/Easy/P003",
  # "amusement/amusement/Easy/P004",
  # "amusement/amusement/Easy/P006",
  # "amusement/amusement/Easy/P007",
  # "amusement/amusement/Easy/P008",
  # "amusement/amusement/Hard/P000",
  # "amusement/amusement/Hard/P001",
  "amusement/amusement/Hard/P002",
  # "amusement/amusement/Hard/P003",
  # "amusement/amusement/Hard/P004",
  # "amusement/amusement/Hard/P005",
  # "amusement/amusement/Hard/P006",
  # "amusement/amusement/Hard/P007",
  # "carwelding/carwelding/Easy/P001",
  # "carwelding/carwelding/Easy/P002",
  # "carwelding/carwelding/Easy/P004",
  # "carwelding/carwelding/Easy/P005",
  # "carwelding/carwelding/Easy/P006",
  "carwelding/carwelding/Easy/P007",
  # "carwelding/carwelding/Hard/P000",
  # "carwelding/carwelding/Hard/P001",
  # "carwelding/carwelding/Hard/P002",
  # "carwelding/carwelding/Hard/P003",
  # "endofworld/endofworld/Easy/P000",
  # "endofworld/endofworld/Easy/P001",
  # "endofworld/endofworld/Easy/P002",
  # "endofworld/endofworld/Easy/P003",
  # "endofworld/endofworld/Easy/P004",
  # "endofworld/endofworld/Easy/P005",
  # "endofworld/endofworld/Easy/P006",
  # "endofworld/endofworld/Easy/P007",
  # "endofworld/endofworld/Easy/P008",
  # "endofworld/endofworld/Easy/P009",
  "endofworld/endofworld/Hard/P000",
  # "endofworld/endofworld/Hard/P001",
  # "endofworld/endofworld/Hard/P002",
  # "endofworld/endofworld/Hard/P005",
  # "endofworld/endofworld/Hard/P006",
  "gascola/gascola/Easy/P001",
  # "gascola/gascola/Easy/P003",
  # "gascola/gascola/Easy/P004",
  # "gascola/gascola/Easy/P005",
  # "gascola/gascola/Easy/P006",
  # "gascola/gascola/Easy/P007",
  # "gascola/gascola/Easy/P008",
  "gascola/gascola/Hard/P000",
  # "gascola/gascola/Hard/P001",
  # "gascola/gascola/Hard/P002",
  # "gascola/gascola/Hard/P003",
  # "gascola/gascola/Hard/P004",
  # "gascola/gascola/Hard/P005",
  # "gascola/gascola/Hard/P006",
  # "gascola/gascola/Hard/P007",
  # "gascola/gascola/Hard/P008",
  # "gascola/gascola/Hard/P009",
  "hospital/hospital/Easy/P000",
  # "hospital/hospital/Easy/P001",
  # "hospital/hospital/Easy/P002",
  # "hospital/hospital/Easy/P003",
  # "hospital/hospital/Easy/P004",
  # "hospital/hospital/Easy/P005",
  # "hospital/hospital/Easy/P006",
  # "hospital/hospital/Easy/P007",
  # "hospital/hospital/Easy/P008",
  # "hospital/hospital/Easy/P009",
  # "hospital/hospital/Easy/P010",
  # "hospital/hospital/Easy/P011",
  # "hospital/hospital/Easy/P012",
  # "hospital/hospital/Easy/P013",
  # "hospital/hospital/Easy/P014",
  # "hospital/hospital/Easy/P015",
  # "hospital/hospital/Easy/P016",
  # "hospital/hospital/Easy/P017",
  # "hospital/hospital/Easy/P018",
  # "hospital/hospital/Easy/P019",
  # "hospital/hospital/Easy/P020",
  # "hospital/hospital/Easy/P021",
  # "hospital/hospital/Easy/P022",
  # "hospital/hospital/Easy/P023",
  # "hospital/hospital/Easy/P024",
  # "hospital/hospital/Easy/P025",
  # "hospital/hospital/Easy/P026",
  # "hospital/hospital/Easy/P027",
  # "hospital/hospital/Easy/P028",
  # "hospital/hospital/Easy/P029",
  # "hospital/hospital/Easy/P030",
  # "hospital/hospital/Easy/P031",
  # "hospital/hospital/Easy/P032",
  # "hospital/hospital/Easy/P033",
  # "hospital/hospital/Easy/P034",
  # "hospital/hospital/Easy/P035",
  # "hospital/hospital/Easy/P036",
  "hospital/hospital/Hard/P037",
  # "hospital/hospital/Hard/P038",
  # "hospital/hospital/Hard/P039",
  # "hospital/hospital/Hard/P040",
  # "hospital/hospital/Hard/P041",
  # "hospital/hospital/Hard/P042",
  # "hospital/hospital/Hard/P043",
  # "hospital/hospital/Hard/P044",
  # "hospital/hospital/Hard/P045",
  # "hospital/hospital/Hard/P046",
  # "hospital/hospital/Hard/P047",
  # "hospital/hospital/Hard/P048",
  # "hospital/hospital/Hard/P049",
  "japanesealley/japanesealley/Easy/P001",
  # "japanesealley/japanesealley/Easy/P002",
  # "japanesealley/japanesealley/Easy/P003",
  # "japanesealley/japanesealley/Easy/P004",
  # "japanesealley/japanesealley/Easy/P005",
  # "japanesealley/japanesealley/Easy/P007",
  "japanesealley/japanesealley/Hard/P000",
  # "japanesealley/japanesealley/Hard/P001",
  # "japanesealley/japanesealley/Hard/P002",
  # "japanesealley/japanesealley/Hard/P003",
  # "japanesealley/japanesealley/Hard/P004",
  # "japanesealley/japanesealley/Hard/P005",
  "neighborhood/neighborhood/Easy/P000",
  # "neighborhood/neighborhood/Easy/P001",
  # "neighborhood/neighborhood/Easy/P002",
  # "neighborhood/neighborhood/Easy/P003",
  # "neighborhood/neighborhood/Easy/P004",
  # "neighborhood/neighborhood/Easy/P005",
  # "neighborhood/neighborhood/Easy/P007",
  # "neighborhood/neighborhood/Easy/P008",
  # "neighborhood/neighborhood/Easy/P009",
  # "neighborhood/neighborhood/Easy/P010",
  # "neighborhood/neighborhood/Easy/P012",
  # "neighborhood/neighborhood/Easy/P013",
  # "neighborhood/neighborhood/Easy/P014",
  # "neighborhood/neighborhood/Easy/P015",
  # "neighborhood/neighborhood/Easy/P016",
  # "neighborhood/neighborhood/Easy/P017",
  # "neighborhood/neighborhood/Easy/P018",
  # "neighborhood/neighborhood/Easy/P019",
  # "neighborhood/neighborhood/Easy/P020",
  # "neighborhood/neighborhood/Easy/P021",
  "neighborhood/neighborhood/Hard/P000",
  # "neighborhood/neighborhood/Hard/P001",
  # "neighborhood/neighborhood/Hard/P002",
  # "neighborhood/neighborhood/Hard/P003",
  # "neighborhood/neighborhood/Hard/P004",
  # "neighborhood/neighborhood/Hard/P005",
  # "neighborhood/neighborhood/Hard/P006",
  # "neighborhood/neighborhood/Hard/P007",
  # "neighborhood/neighborhood/Hard/P008",
  # "neighborhood/neighborhood/Hard/P009",
  # "neighborhood/neighborhood/Hard/P010",
  # "neighborhood/neighborhood/Hard/P011",
  # "neighborhood/neighborhood/Hard/P012",
  # "neighborhood/neighborhood/Hard/P013",
  # "neighborhood/neighborhood/Hard/P014",
  # "neighborhood/neighborhood/Hard/P015",
  # "neighborhood/neighborhood/Hard/P016",
  # "neighborhood/neighborhood/Hard/P017",
  "ocean/ocean/Easy/P000",
  # "ocean/ocean/Easy/P001",
  # "ocean/ocean/Easy/P002",
  # "ocean/ocean/Easy/P004",
  # "ocean/ocean/Easy/P005",
  # "ocean/ocean/Easy/P006",
  # "ocean/ocean/Easy/P008",
  # "ocean/ocean/Easy/P009",
  # "ocean/ocean/Easy/P010",
  # "ocean/ocean/Easy/P011",
  # "ocean/ocean/Easy/P012",
  # "ocean/ocean/Easy/P013",
  "ocean/ocean/Hard/P000",
  # "ocean/ocean/Hard/P001",
  # "ocean/ocean/Hard/P002",
  # "ocean/ocean/Hard/P003",
  # "ocean/ocean/Hard/P004",
  # "ocean/ocean/Hard/P005",
  # "ocean/ocean/Hard/P006",
  # "ocean/ocean/Hard/P007",
  # "ocean/ocean/Hard/P008",
  # "ocean/ocean/Hard/P009",
  "office/office/Easy/P000",
  # "office/office/Easy/P001",
  # "office/office/Easy/P002",
  # "office/office/Easy/P003",
  # "office/office/Easy/P004",
  # "office/office/Easy/P005",
  # "office/office/Easy/P006",
  "office/office/Hard/P000",
  # "office/office/Hard/P001",
  # "office/office/Hard/P002",
  # "office/office/Hard/P003",
  # "office/office/Hard/P004",
  # "office/office/Hard/P005",
  # "office/office/Hard/P006",
  # "office/office/Hard/P007",
  "office2/office2/Easy/P000",
  # "office2/office2/Easy/P003",
  # "office2/office2/Easy/P004",
  # "office2/office2/Easy/P005",
  # "office2/office2/Easy/P006",
  # "office2/office2/Easy/P007",
  # "office2/office2/Easy/P008",
  # "office2/office2/Easy/P009",
  # "office2/office2/Easy/P010",
  # "office2/office2/Easy/P011",
  "office2/office2/Hard/P000",
  # "office2/office2/Hard/P001",
  # "office2/office2/Hard/P002",
  # "office2/office2/Hard/P003",
  # "office2/office2/Hard/P004",
  # "office2/office2/Hard/P005",
  # "office2/office2/Hard/P006",
  # "office2/office2/Hard/P007",
  # "office2/office2/Hard/P008",
  # "office2/office2/Hard/P009",
  # "office2/office2/Hard/P010",
  "oldtown/oldtown/Easy/P000",
  # "oldtown/oldtown/Easy/P001",
  # "oldtown/oldtown/Easy/P002",
  # "oldtown/oldtown/Easy/P004",
  # "oldtown/oldtown/Easy/P005",
  # "oldtown/oldtown/Easy/P007",
  "oldtown/oldtown/Hard/P000",
  # "oldtown/oldtown/Hard/P001",
  # "oldtown/oldtown/Hard/P002",
  # "oldtown/oldtown/Hard/P003",
  # "oldtown/oldtown/Hard/P004",
  # "oldtown/oldtown/Hard/P005",
  # "oldtown/oldtown/Hard/P006",
  # "oldtown/oldtown/Hard/P007",
  # "oldtown/oldtown/Hard/P008",
  "seasidetown/seasidetown/Easy/P000",
  # "seasidetown/seasidetown/Easy/P001",
  # "seasidetown/seasidetown/Easy/P002",
  # "seasidetown/seasidetown/Easy/P003",
  # "seasidetown/seasidetown/Easy/P004",
  # "seasidetown/seasidetown/Easy/P005",
  # "seasidetown/seasidetown/Easy/P006",
  # "seasidetown/seasidetown/Easy/P007",
  # "seasidetown/seasidetown/Easy/P008",
  # "seasidetown/seasidetown/Easy/P009",
  "seasidetown/seasidetown/Hard/P000",
  # "seasidetown/seasidetown/Hard/P001",
  # "seasidetown/seasidetown/Hard/P002",
  # "seasidetown/seasidetown/Hard/P003",
  # "seasidetown/seasidetown/Hard/P004",
  "seasonsforest/seasonsforest/Easy/P001",
  # "seasonsforest/seasonsforest/Easy/P002",
  # "seasonsforest/seasonsforest/Easy/P003",
  # "seasonsforest/seasonsforest/Easy/P004",
  # "seasonsforest/seasonsforest/Easy/P005",
  # "seasonsforest/seasonsforest/Easy/P007",
  # "seasonsforest/seasonsforest/Easy/P008",
  # "seasonsforest/seasonsforest/Easy/P009",
  # "seasonsforest/seasonsforest/Easy/P010",
  # "seasonsforest/seasonsforest/Easy/P011",
  "seasonsforest/seasonsforest/Hard/P001",
  # "seasonsforest/seasonsforest/Hard/P002",
  # "seasonsforest/seasonsforest/Hard/P004",
  # "seasonsforest/seasonsforest/Hard/P005",
  # "seasonsforest/seasonsforest/Hard/P006",
  "seasonsforest_winter/seasonsforest_winter/Easy/P000",
  # "seasonsforest_winter/seasonsforest_winter/Easy/P001",
  # "seasonsforest_winter/seasonsforest_winter/Easy/P002",
  # "seasonsforest_winter/seasonsforest_winter/Easy/P003",
  # "seasonsforest_winter/seasonsforest_winter/Easy/P004",
  # "seasonsforest_winter/seasonsforest_winter/Easy/P005",
  # "seasonsforest_winter/seasonsforest_winter/Easy/P006",
  # "seasonsforest_winter/seasonsforest_winter/Easy/P007",
  # "seasonsforest_winter/seasonsforest_winter/Easy/P008",
  # "seasonsforest_winter/seasonsforest_winter/Easy/P009",
  "seasonsforest_winter/seasonsforest_winter/Hard/P010",
  # "seasonsforest_winter/seasonsforest_winter/Hard/P011",
  # "seasonsforest_winter/seasonsforest_winter/Hard/P012",
  # "seasonsforest_winter/seasonsforest_winter/Hard/P013",
  # "seasonsforest_winter/seasonsforest_winter/Hard/P014",
  # "seasonsforest_winter/seasonsforest_winter/Hard/P015",
  # "seasonsforest_winter/seasonsforest_winter/Hard/P016",
  # "seasonsforest_winter/seasonsforest_winter/Hard/P017",
  # "seasonsforest_winter/seasonsforest_winter/Hard/P018",
  "soulcity/soulcity/Easy/P000",
  # "soulcity/soulcity/Easy/P001",
  # "soulcity/soulcity/Easy/P002",
  # "soulcity/soulcity/Easy/P003",
  # "soulcity/soulcity/Easy/P004",
  # "soulcity/soulcity/Easy/P005",
  # "soulcity/soulcity/Easy/P006",
  # "soulcity/soulcity/Easy/P007",
  # "soulcity/soulcity/Easy/P008",
  # "soulcity/soulcity/Easy/P009",
  # "soulcity/soulcity/Easy/P010",
  # "soulcity/soulcity/Easy/P011",
  # "soulcity/soulcity/Easy/P012",
  "soulcity/soulcity/Hard/P000",
  # "soulcity/soulcity/Hard/P001",
  # "soulcity/soulcity/Hard/P002",
  # "soulcity/soulcity/Hard/P003",
  # "soulcity/soulcity/Hard/P004",
  # "soulcity/soulcity/Hard/P005",
  # "soulcity/soulcity/Hard/P008",
  # "soulcity/soulcity/Hard/P009",
  "westerndesert/westerndesert/Easy/P001",
  # "westerndesert/westerndesert/Easy/P002",
  # "westerndesert/westerndesert/Easy/P004",
  # "westerndesert/westerndesert/Easy/P005",
  # "westerndesert/westerndesert/Easy/P006",
  # "westerndesert/westerndesert/Easy/P007",
  # "westerndesert/westerndesert/Easy/P008",
  # "westerndesert/westerndesert/Easy/P009",
  # "westerndesert/westerndesert/Easy/P010",
  # "westerndesert/westerndesert/Easy/P011",
  # "westerndesert/westerndesert/Easy/P012",
  # "westerndesert/westerndesert/Easy/P013",
  "westerndesert/westerndesert/Hard/P000",
  # "westerndesert/westerndesert/Hard/P001",
  # "westerndesert/westerndesert/Hard/P002",
  # "westerndesert/westerndesert/Hard/P003",
  # "westerndesert/westerndesert/Hard/P004",
  # "westerndesert/westerndesert/Hard/P005",
  # "westerndesert/westerndesert/Hard/P006",
  # "westerndesert/westerndesert/Hard/P007"
]

dynamic_log_path = '/pool0/piaodeng/adaptive_DPVO/dynamic_slam_log/logs/'
#dynamic_log_path = '/pool0/piaodeng/dynamic_dpvo_selector/data_log/'

model_path = "trained_weights/confidence_to_difficulty_model_fixed_agline_all_works.pth"

data_set_path = '/pool0/piaodeng/DROID-SLAM/datasets/EuRoC/'

def test_model(newest_model = None):
      
      if newest_model != None:
         model_path = newest_model

      for scene_name in tartan_scenes:
        print(scene_name)
        for trails_idx in range(1, 2):
          #current_input_log = dynamic_log_path + 'dynamic_slam_log_confidence_array_'+scene_name+'_'+str(num_patches)+'_patches_'+str(num_frame)+'_frames_trials_'+str(trails_idx)+'.pickle'
          leagal_scene_name = scene_name.replace("/", "_")
          current_input_log = dynamic_log_path + 'dynamic_slam_log_gt_difficulty_validate_'+leagal_scene_name+'_trials_'+str(trails_idx)+'.pickle'
          current_saved_label_data = 'test_processed_data/dynamic_slam_log_confidence_array_'+leagal_scene_name+'_'+str(trails_idx)+'_label.pth'
          current_saved_confidence_data = 'test_processed_data/dynamic_slam_log_confidence_array_'+leagal_scene_name+'_trials_'+str(trails_idx)+'_confidence.pth'

          if os.path.exists(current_saved_confidence_data):
              #load processed data
              confidence_data = torch.load(current_saved_confidence_data)
              difficulty_labels = torch.load(current_saved_label_data)
          else:
              #load from otiginal data
              dynamic_log_picke_file = open(current_input_log, 'rb')
              logged_data = pickle.load(dynamic_log_picke_file)

              # imu_data_path = data_set_path + scene_name + "/mav0/imu0/data.csv"
              # imu_data = pd.read_csv(imu_data_path)

              imu_data = None

              confidence_data_1_test,  difficulty_labels_1_test= aglin_and_preprocess_log_data(logged_data, imu_data)

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
          criterion = nn.MSELoss()
          eva_loss = 0.0
          with torch.no_grad():
            for i, (inputs, targets) in enumerate(progress_bar):
                # Move inputs and targets to device (e.g., GPU if available)
                inputs, targets = inputs.to('cuda'), targets.to('cuda').unsqueeze(1)

                # Forward pass: Compute predicted difficulty
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                eva_loss += loss
                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

          print(f'eva Loss: {eva_loss/len(data_loader)}')
          tstamp = []

          for i in range(0, len(all_predictions)):
            tstamp.append(i)

          fig, (ax1) = plt.subplots(figsize=(40, 28))

          min_plot = min(min(all_predictions), min(all_targets))
          max_plot = max(max(all_predictions), max(all_targets))

          #min_plot = 0
          #max_plot = 1

          ax1.plot(tstamp, all_predictions, 'c-', label='prediction')
          ax1.set_ylabel('prediction', color='c')
          ax1.set_xlabel('time stamp')
          ax1.set_ylim(min_plot, max_plot)
          ax1.tick_params(axis='y', labelcolor='c')

          ax3 = ax1.twinx()
          ax3.plot(tstamp, all_targets, 'b-', label='ground truth')
          ax3.set_ylabel('ground truth', color='b')
          ax3.set_ylim(min_plot, max_plot)
          ax3.tick_params(axis='y', labelcolor='b')

          plt.title(scene_name+' ' + str(trails_idx) + 'trail' )

          output_folder = "plot/dynamic_slam_prediction_"+scene_name+"_trail_"+str(trails_idx)+"/"
          os.makedirs(output_folder, exist_ok=True)

          if os.path.exists(output_folder+"100_input.png"):
            os.remove(output_folder+"100_input.png")

          plt.savefig(output_folder+"100_input.png", dpi=100, bbox_inches='tight')
          plt.close('all')

                      
                
    # Save the model's state dictionary

if __name__ == '__main__':
  test_model(model_path)