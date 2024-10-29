import torch
import torch.nn as nn
from tqdm import tqdm
import pickle
import sys
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from ahrs.filters import Madgwick
from scipy.spatial.transform import Rotation as R

def debug_plot_raw_data(data1, data2, postfix):
    plot_path = 'plot/data_comp'

    if len(data1) != len(data2):
      raise("non identical data length to compare")
    
    x_axis = []
    for i in range(0, len(data1)):
      x_axis.append(i)

    fig, (ax1) = plt.subplots(1, 1, figsize=(40, 8))

    min_plot = min(min(data1), min(data2))
    max_plot = max(max(data1), max(data2))

    
    ax1.plot(x_axis, data1, 'c-', label='data1', linewidth=7)
    ax1.set_ylim(min_plot, max_plot)
    ax1.tick_params(axis='y', labelcolor='c')
    
    #confidence = [x / num_patch for x in confidence]
    ax3 = ax1.twinx()
    ax3.plot(x_axis, data2, 'b-', label='data2')
    ax3.set_ylim(min_plot, max_plot)
    ax3.tick_params(axis='y', labelcolor='b')

    if os.path.exists(plot_path+postfix):
      os.remove(plot_path+postfix)

    plt.savefig(plot_path+postfix, dpi=100, bbox_inches='tight')
    plt.close('all')

def aglin_and_preprocess_log_data(logged_data, imu_data):


    #collected data from dpvo
    error_tstamp      = logged_data['result'].np_arrays['timestamps']
    est_xyz           = logged_data['result'].trajectories['traj'].positions_xyz
    ref_xyz           = logged_data['result'].trajectories['reference'].positions_xyz
    confidence        = logged_data['confidence']
    est_pose          = logged_data['result'].trajectories['traj'].poses_se3
    tstamp            = logged_data['tstamp']
    delta             = logged_data['delta']
    est_translation   = logged_data['translation']
    #est_acc           = logged_data['acce']
    #raw_traj          = logged_data['raw_xyz']

    raltime_diffc     = logged_data['diffculty_log']

    lin_imu_acc = []
    if imu_data != None:
      #data from imu in dataset
      imu_tstamp  = imu_data['#timestamp [ns]'].tolist()
      imu_x_rotat = imu_data['w_RS_S_x [rad s^-1]'].tolist()
      imu_y_rotat = imu_data['w_RS_S_y [rad s^-1]'].tolist()
      imu_z_rotat = imu_data['w_RS_S_z [rad s^-1]'].tolist()
      imu_x_acc   = imu_data['a_RS_S_x [m s^-2]'].tolist()
      imu_y_acc   = imu_data['a_RS_S_y [m s^-2]'].tolist()
      imu_z_acc   = imu_data['a_RS_S_z [m s^-2]'].tolist()

      #remove the gravity from the imu acceleration
      gravity_vector = np.array([9.81, 0, 0])
      
      q = np.array([1.0, 0.0, 0.0, 0.0])
      madgwick_filter = Madgwick()

      for idx in range(0, len(imu_tstamp)):
        
        gyro_data = np.array([imu_x_rotat[idx], imu_y_rotat[idx], imu_z_rotat[idx]], dtype=np.float32)
        acc_data  = np.array([imu_x_acc[idx], imu_y_acc[idx], imu_z_acc[idx]], dtype=np.float32)

        q = madgwick_filter.updateIMU(q = q, gyr=gyro_data, acc=acc_data)

        rotation = R.from_quat([q[1], q[2], q[3], q[0]])

        # Rotate accelerometer data to world frame
        acc_world = rotation.apply(acc_data)

        # Remove gravity to get linear acceleration
        linear_acc = acc_world - gravity_vector

        lin_imu_acc.append(linear_acc)




    for start_idx in range(0, len(tstamp)):
      if tstamp[start_idx] == None:
        continue
      else:
        tstamp            = tstamp[start_idx:]
        est_translation   = est_translation[start_idx:]
        #est_acc           = est_acc[start_idx:]
        confidence        = confidence[start_idx:]
        raltime_diffc     = raltime_diffc[start_idx:]
        #raw_traj          = raw_traj[start_idx:]
        delta             = delta[start_idx:]
        break

    if tstamp[0] < error_tstamp[0]:
      #tstamp start earlier
      for start_idx in range(0, len(tstamp)):
        if tstamp[start_idx] == error_tstamp[0]:
          tstamp            = tstamp[start_idx:]
          est_translation   = est_translation[start_idx:]
          #est_acc           = est_acc[start_idx:]
          confidence        = confidence[start_idx:]
          raltime_diffc     = raltime_diffc[start_idx:]
          #raw_traj          = raw_traj[start_idx:]
          delta             = delta[start_idx:]
          break
    else:
      #ref start earlier
      for start_idx in range(0, len(error_tstamp)):
        if tstamp[0] == error_tstamp[start_idx]:
          error_tstamp = error_tstamp[start_idx:]
          est_xyz      = est_xyz[start_idx:]
          ref_xyz      = ref_xyz[start_idx:]
          est_pose     = est_pose[start_idx:]
          break

    if tstamp[-1] > error_tstamp[-1]:
      #tstamp end later
      for end_idx in range(len(tstamp) - 1, 0, -1):
        if tstamp[end_idx] == error_tstamp[-1]:
          tstamp            = tstamp[:end_idx+1]
          est_translation   = est_translation[:end_idx+1]
          #est_acc           = est_acc[:end_idx+1]
          confidence        = confidence[:end_idx+1]
          raltime_diffc     = raltime_diffc[:end_idx+1]
          #raw_traj          = raw_traj[:end_idx+1]
          delta             = delta[:end_idx+1]
          break
    else:
      #ref ends later
      for end_idx in range(len(error_tstamp) - 1, 0 ,-1):
        if tstamp[-1] == error_tstamp[end_idx]:
          error_tstamp = error_tstamp[:end_idx+1]
          est_xyz      = est_xyz[:end_idx+1]
          ref_xyz      = ref_xyz[:end_idx+1]
          est_pose     = est_pose[:end_idx+1]
          break


    #smooth the realtime diffc
    smoothed_realtime_diffc = []
    smooth_realtime_diffc_len = 10
    for i in range(0, len(raltime_diffc)):
      if i < smooth_realtime_diffc_len:
        smoothed_realtime_diffc.append(0)
        continue

      smoothed_realtime_diffc.append(max(raltime_diffc[i-smooth_realtime_diffc_len:i]))

    #clip the initialization
    initialization_clip = 150

    error_tstamp            = error_tstamp[initialization_clip:]
    est_xyz                 = est_xyz[initialization_clip:]
    ref_xyz                 = ref_xyz[initialization_clip:]
    est_pose                = est_pose[initialization_clip:]
    tstamp                  = tstamp[initialization_clip:]
    confidence              = confidence[initialization_clip:]
    raltime_diffc           = raltime_diffc[initialization_clip:]
    smoothed_realtime_diffc = smoothed_realtime_diffc[initialization_clip:]
    #raw_traj                = raw_traj[initialization_clip:]
    est_translation         = est_translation[initialization_clip:]
    #est_acc                 = est_acc[initialization_clip:]
    delta                   = delta[initialization_clip:]

    if imu_data != None:
      #algin the collected data with data from imu
      imu_rotation_aglined = []
      imu_acc_aglined = []

      imu_current_x_rotat_agligned = []
      imu_current_y_rotat_agligned = []
      imu_current_z_rotat_agligned = []

      imu_current_x_acc_agligned = []
      imu_current_y_acc_agligned = []
      imu_current_z_acc_agligned = []

      imu_start_ptr = 0

      imu_current_x_rotat = 0
      imu_current_y_rotat = 0
      imu_current_z_rotat = 0

      imu_current_x_acc = 0
      imu_current_y_acc = 0
      imu_current_z_acc = 0

      imu_linear_acc = [0.0, 0.0, 0.0]
      for idx in range(0, len(error_tstamp)):

        for imu_ptr in range(imu_start_ptr, len(imu_tstamp)):

          imu_current_x_rotat = imu_current_x_rotat + imu_x_rotat[imu_ptr]
          imu_current_y_rotat = imu_current_y_rotat + imu_y_rotat[imu_ptr]
          imu_current_z_rotat = imu_current_z_rotat + imu_z_rotat[imu_ptr]

          imu_current_x_acc = imu_current_x_acc + imu_x_acc[imu_ptr]
          imu_current_y_acc = imu_current_y_acc + imu_y_acc[imu_ptr]
          imu_current_z_acc = imu_current_z_acc + imu_z_acc[imu_ptr]

          imu_linear_acc[0] = imu_linear_acc[0] + lin_imu_acc[imu_ptr][0]
          imu_linear_acc[1] = imu_linear_acc[1] + lin_imu_acc[imu_ptr][1]
          imu_linear_acc[2] = imu_linear_acc[2] + lin_imu_acc[imu_ptr][2]

          if error_tstamp[idx] == imu_tstamp[imu_ptr]:
            #find the tstamp in imu data

            imu_current_x_rotat_agligned.append(imu_current_x_rotat)
            imu_current_y_rotat_agligned.append(imu_current_y_rotat)
            imu_current_z_rotat_agligned.append(imu_current_z_rotat)

            imu_current_rotat = math.sqrt(imu_current_x_rotat ** 2 + imu_current_y_rotat ** 2 + imu_current_z_rotat ** 2)
            imu_rotation_aglined.append(imu_current_rotat)
            
            imu_current_x_acc_agligned.append(imu_current_x_acc)
            imu_current_y_acc_agligned.append(imu_current_y_acc)
            imu_current_z_acc_agligned.append(imu_current_z_acc)

            imu_acc_aglined.append(imu_linear_acc[0] ** 2 + imu_linear_acc[1] ** 2 + imu_linear_acc[2] ** 2)

            imu_current_x_rotat = 0
            imu_current_y_rotat = 0
            imu_current_z_rotat = 0

            imu_current_x_acc = 0
            imu_current_y_acc = 0
            imu_current_z_acc = 0

            imu_linear_acc = [0.0, 0.0, 0.0]

            imu_start_ptr = imu_ptr
            break

      
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
    acceleration_log = []
    for idx in range(0, len(est_pose)):
        if idx == 0:
          rotation_log.append(0)
          translation_log.append(0)
          acceleration_log.append(0)
          continue

        if idx == 1:
          acceleration_log.append(0)

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






    #scale the est translation with imu acc

    #find the scaling between imu acc and est acc
    #sampled_est_acc = sum(est_acc[100:200])
    #sampled_imu_acc = sum(imu_acc_aglined[100:200])

    #translation_real_sacle = sampled_imu_acc / sampled_est_acc

    #est_translation_scale         = 1 / translation_real_sacle
    est_acc_scale_factor          = 1
    translation_scale_factor      = 0.8
    confidence_scale_factor       = 800
    rotation_scale_factor         = 0.5

    delta_scale_factor            = 30
    imu_rotat_scale_factor        = 6.8
    imu_acc_scale_factor          = 400

    smoothed_drift_scale_facotr   = 0.02
    raltime_diffc_scale_factor    = 1


    confidence            = [x / confidence_scale_factor for x in confidence]
    rotation_log          = [x / rotation_scale_factor for x in rotation_log]
    translation_log       = [x / translation_scale_factor for x in translation_log]
    delta                 = [x / delta_scale_factor for x in delta]
    #imu_rotation_aglined  = [x / imu_rotat_scale_factor for x in imu_rotation_aglined]
    #imu_acc_aglined       = [x / imu_acc_scale_factor for x in imu_acc_aglined]
    #est_translation       = [x / est_translation_scale for x in est_translation]
    smoothed_realtime_diffc = [x / raltime_diffc_scale_factor for x in smoothed_realtime_diffc]

    #est_acc               = [x / est_acc_scale_factor for x in est_acc]

    smoothed_drift        = [x / smoothed_drift_scale_facotr for x in smoothed_drift]

    #clip
    # confidence          = [min(x, 1) for x in confidence]
    # rotation_log        = [min(x, 1) for x in rotation_log]
    # translation_log     = [min(x, 1) for x in translation_log]
    smoothed_drift                = [min(x, 3) for x in smoothed_drift]
    smoothed_realtime_diffc       = [min(x, 2) for x in smoothed_realtime_diffc]
    # delta               = [min(x, 1) for x in delta]

    #debug compare data
    #debug_plot_raw_data(est_translation, translation_log, 'trans.png')
    #debug_plot_raw_data(imu_rotation_aglined, rotation_log, 'rotat.png')
    #debug_plot_raw_data(imu_acc_aglined[100:], est_acc[100:], 'acc.png')

    #rearrange the data
    history_window = 100 #give history_window len of past data
    #                                   frame,        ,  history window,  confidence+translation+rotation+last 100 frame distance+delta
    model_input_data = np.zeros([len(confidence)-history_window, history_window,            3], dtype=float) # skip the first 100 frame for complete data

    for frame_idx in range(history_window, len(confidence)):
       for histroy_idx in range(0, history_window):
         history_frame_pointer = frame_idx - histroy_idx - 1
         if (history_frame_pointer <0 ):
           break
         model_input_data[frame_idx-history_window, histroy_idx] = [confidence[history_frame_pointer], rotation_log[history_frame_pointer] ,translation_log[history_frame_pointer]]

    label = smoothed_realtime_diffc[history_window:]
    return model_input_data, label