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



def aglin_and_preprocess_log_data(logged_data,):
    confidence_array  = logged_data['conf_array']

    error_tstamp      = logged_data['result'].np_arrays['timestamps']
    est_xyz           = logged_data['result'].trajectories['traj'].positions_xyz
    ref_xyz           = logged_data['result'].trajectories['reference'].positions_xyz
    confidence        = logged_data['confidence']
    est_pose          = logged_data['result'].trajectories['traj'].poses_se3
    tstamp            = logged_data['tstamp']
    delta             = logged_data['delta']
    distance_f_S      = logged_data['result'].np_arrays['distances_from_start']



    for start_idx in range(0, len(tstamp)):
      if tstamp[start_idx] == None:
        continue
      else:
        tstamp            = tstamp[start_idx:]
        confidence_array  = confidence_array[start_idx:]
        confidence        = confidence[start_idx:]
        delta             = delta[start_idx:]
        break

    if tstamp[0] < error_tstamp[0]:
      #tstamp start earlier
      for start_idx in range(0, len(tstamp)):
        if tstamp[start_idx] == error_tstamp[0]:
          tstamp            = tstamp[start_idx:]
          confidence_array  = confidence_array[start_idx:]
          confidence        = confidence[start_idx:]
          delta             = delta[start_idx:]
          break
    else:
      #ref start earlier
      for start_idx in range(0, len(error_tstamp)):
        if tstamp[0] == error_tstamp[start_idx]:
          error_tstamp = error_tstamp[start_idx:]
          break

    if tstamp[-1] > error_tstamp[-1]:
      #tstamp end later
      for end_idx in range(len(tstamp) - 1, 0, -1):
        if tstamp[end_idx] == error_tstamp[-1]:
          tstamp            = tstamp[:end_idx+1]
          confidence_array  = confidence_array[:end_idx+1]
          confidence        = confidence[:end_idx+1]
          delta             = delta[:end_idx+1]
          break
    else:
      #ref ends later
      for end_idx in range(len(error_tstamp) - 1, 0 ,-1):
        if tstamp[-1] == error_tstamp[end_idx]:
          error_tstamp = error_tstamp[:end_idx+1]
          break

    #compute the distance traveled in the past 100 frames
    distance_100_frame = []
    for idx in range(0, len(distance_f_S)):
      if idx < 100:
        distance_100_frame.append(0)
      else:
        distance_100_frame.append(distance_f_S[idx]-distance_f_S[idx-100])

    #clip the initialization
    initialization_clip = 100

    error_tstamp            = error_tstamp[initialization_clip:]
    est_xyz                 = est_xyz[initialization_clip:]
    ref_xyz                 = ref_xyz[initialization_clip:]
    est_pose                = est_pose[initialization_clip:]
    tstamp                  = tstamp[initialization_clip:]
    confidence              = confidence[initialization_clip:]
    delta                   = delta[initialization_clip:]
    distance_100_frame      = distance_100_frame[initialization_clip:]
    
    #compute change of drift
    change_of_drift = []
    error_vec_log = []

    for idx in range(0, len(est_xyz)):

      error_vec = est_xyz[idx] - ref_xyz[idx]

      error_vec_log.append(error_vec)

      if idx == 0:
        change_of_drift.append(0)
        continue

      change_of_error_vec = error_vec - error_vec_log[idx - 1]

      change_of_error = math.sqrt(change_of_error_vec[0]**2 + change_of_error_vec[1]**2 + change_of_error_vec[2]**2)

      change_of_drift.append(change_of_error)


    #smooth drift by finding the max value over past 10 frame
    smooth_len = 10
    smoothed_drift = []
    for idx in range(0, len(change_of_drift)):
      if idx < smooth_len:
        smoothed_drift.append(0)
        continue

      smoothed_drift.append(max(change_of_drift[idx-smooth_len:idx]))


    #compute movement
    
    rotation_log = []
    translation_log = []
    for idx in range(0, len(est_pose)):
        if idx == 0:
          rotation_log.append(0)
          translation_log.append(0)
          continue

        #translation
        last_coor = est_xyz[idx-1]
        curr_coor = est_xyz[idx]

        diff_coor = last_coor - curr_coor

        translation_movement = math.sqrt(diff_coor[0]**2 + diff_coor[1]**2 + diff_coor[2]**2)

        translation_log.append(translation_movement)

        #rotation

        last_rotation = np.array(est_pose[idx - 1][:3, :3])
        current_totation = np.array(est_pose[idx][:3, :3])

        R_relative = np.dot(last_rotation.T, current_totation)

        trace = np.trace(R_relative)

        cos_theta = (trace - 1) / 2
        cos_theta = np.clip(cos_theta, -1, 1)
        theta = np.arccos(cos_theta)

        rotation_log.append(theta)

    translation_scale_factor      = 0.5
    confidence_scale_factor       = 800
    rotation_scale_factor         = 0.5
    smoothed_drift_scale_facotr   = 0.02
    # distance_100_frame_factor     = 20
    # delta_scale_factor            = 25


    confidence          = [x / confidence_scale_factor for x in confidence]
    rotation_log        = [x / rotation_scale_factor for x in rotation_log]
    translation_log     = [x / translation_scale_factor for x in translation_log]
    smoothed_drift      = [x / smoothed_drift_scale_facotr for x in smoothed_drift]
    # distance_100_frame  = [x / distance_100_frame_factor for x in distance_100_frame]
    # delta               = [x / delta_scale_factor for x in delta]

    #clip
    confidence          = [min(x, 1) for x in confidence]
    rotation_log        = [min(x, 1) for x in rotation_log]
    translation_log     = [min(x, 1) for x in translation_log]
    smoothed_drift      = [min(x, 1) for x in smoothed_drift]
    # distance_100_frame  = [min(x, 1) for x in distance_100_frame]
    # delta               = [min(x, 1) for x in delta]

    #rearrange the data
    history_window = 100 #give history_window len of past data
    #                                   frame,        ,  history window,  confidence+translation+rotation+last 100 frame distance+delta
    model_input_data = np.zeros([len(confidence)-history_window, history_window,            3], dtype=float) # skip the first 100 frame for complete data

    for frame_idx in range(history_window, len(confidence)):
       for histroy_idx in range(0, history_window):
         history_frame_pointer = frame_idx - histroy_idx - 1
         if (history_frame_pointer <0 ):
           break
         model_input_data[frame_idx-history_window, histroy_idx] = [confidence[history_frame_pointer], translation_log[history_frame_pointer], rotation_log[history_frame_pointer]]
         #model_input_data[frame_idx-history_window, histroy_idx] = [confidence[history_frame_pointer], translation_log[history_frame_pointer], rotation_log[history_frame_pointer], distance_100_frame[history_frame_pointer], delta[history_frame_pointer]]


    label = smoothed_drift[history_window:]
    return model_input_data, label