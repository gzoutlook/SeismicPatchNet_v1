# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 15:46:21 2019  @author: Geng

Note: 
    process and analysis the random search results
    
"""

import os
import sys
path = '.\libs'
sys.path.append(path)
import helper

# ------- cnn graph ------- #
models_path = r".\libs\CNN_Models"
sys.path.append(models_path)
import SeismicPatchNet

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
import shutil
import time



#%% ========================================================================
#                       Global setting: analysis dir
# ==========================================================================

scan_dir = r".\results\synthetic_data\search 1"   # the directory contains searched results (many model folders)


#%% ========================================================================
#                       Step 1:  scan and delete/remove 
#                   failed/incomplete  jobs/model folder
# ==========================================================================

# log
removed_models = pd.DataFrame()
scan_count = 0

for obj in os.listdir(scan_dir):
    # only folders name with "Model"
    if os.path.isdir(os.path.join(scan_dir, obj)) and obj.find("Model") != -1:
        scan_count += 1
        if len(os.listdir(os.path.join(scan_dir, obj))) < 2:
            removed_models = removed_models.append(pd.DataFrame(data=[obj], columns=["model name"]), ignore_index=True)
            # delete incomplete folder
            shutil.rmtree(os.path.join(scan_dir, obj),True)
#        else:
#            # delete irrelevant files
#            for file in os.listdir(os.path.join(scan_dir, obj)):
#                if (file.endswith(".csv") or file.endswith(".txt")) is False:
#                    os.remove(os.path.join(scan_dir, obj, file))
            
# save log
removed_models.to_csv(os.path.join(scan_dir, "removed_models scan_{}_{}.csv".format(scan_count, time.strftime("%Y%m%d%H%M%S"))), index=None)       



#%% ========================================================================
#           Step 2:  summarize statistics of architectures' performance
# ==========================================================================

## 'mean top5': sort validation acc. and select the highest 5 value, then average it

## log
#models_performance_columns = ['model name', 'ramdom scales of perception layer', 'mean top5', 'mean top3', 'largest accuracy', 'trainable para. (M)', 'Speed'] # 
#models_performance = pd.DataFrame()
#
#print("\n Scanning and analyzing ...")
#scan_list = os.listdir(scan_dir)
#for i in tqdm(np.arange(len(scan_list)), position=1):
#    obj = scan_list[i]
#    # only folders name with "Model"
#    if os.path.isdir(os.path.join(scan_dir, obj)) and obj.find("Model") != -1:
#        for file in os.listdir(os.path.join(scan_dir, obj)):
#            # analyze csv file
#            if file.endswith(".csv"):
#                # --- csv ---
#                result_file = pd.read_csv(os.path.join(scan_dir, obj, file))
#                top5 = np.sort(result_file['Val_accuracy'])[-5:]
#                mean_top5 = top5.mean()
#                top3 = np.sort(result_file['Val_accuracy'])[-3:]
#                mean_top3 = top3.mean()
#                largest_acc = top5.max()
#                # --- txt ---
#                f1=open(os.path.join(scan_dir, obj, "Log_training.txt"),'r')
#                txt_file=f1.readlines()
#                f1.close()
#                # num of trainable paras
#                trainable_para = 0
#                for line in txt_file:
#                    target_start = line.find("trainable paras: ")
#                    if target_start != -1:
#                        trainable_para = float(line[target_start+len("trainable paras: "):].split(" ")[0])      
#                        break
#                # perception type
#                ramdom_perception_layer = "NA"
#                for line in txt_file:
#                    target_start = line.find("Random the type of perception layers: ")
#                    if target_start != -1:
#                        ramdom_perception_layer = line[target_start+len("Random the type of perception layers: "):].split(" ")[0]  
#                        break
#                # speed
#                speed = 0
#                speed_count = 0
#                for line in txt_file[-10:]:
#                    target_start = line.find("Speed: ")
#                    if target_start != -1:
#                        speed += float(line[target_start+len("Speed: "):].split(" ")[0])
#                        speed_count += 1
#                speed /= speed_count
#                                 
#                # build log
#                models_performance = models_performance.append(pd.DataFrame(data=[[obj, ramdom_perception_layer, mean_top5, mean_top3, largest_acc,
#                                                                                  trainable_para, speed]],
#                                                                            columns=models_performance_columns), ignore_index=True)
#    
#            
## format final statistics
#models_performance = models_performance.sort_values(['mean top5'], ascending=False)
#models_performance.reset_index(drop=True, inplace=True)
#models_performance = models_performance.round(decimals=6)
#
## save that larger than threshold
#threshold = 0.996
##save_name_threshold = "good_models {}larger_{}.csv".format(threshold, time.strftime("%Y%m%d%H%M%S"))
##good_models[good_models['mean top5']>threshold].to_csv(os.path.join(scan_dir, save_name_threshold), index=None)       
#
#save_name_all = "models_performance N{}_{}larger{}_{}.csv".format(
#                                    len(models_performance), threshold, 
#                                    len(models_performance[models_performance['mean top5']>threshold]),
#                                    time.strftime("%Y%m%d%H%M%S"))
#models_performance.to_csv(os.path.join(scan_dir, save_name_all), index=None)
 
    


#%% ==========================================================================
#                   (optional) Step 3: merge statistics results
#       if there are several groups of search results from distributed GPUs
# ============================================================================

#statistics_base_dir = r".\results\synthetic_data\merge dir"
#key_table_name = "models_performance"
#threshold = 0.875
#
#merge_table = pd.DataFrame()
#
#for obj in os.listdir(statistics_base_dir):
#    if obj.endswith(".csv") and obj.find(key_table_name)!= -1:
#        temp_table = pd.read_csv(os.path.join(statistics_base_dir, obj))
#        worker_name = obj.split(" ")[0]
#        models_dir = helper.Get_key_folder_name(statistics_base_dir, worker_name)
#        temp_table.insert(0, "worker_name", str(worker_name))
#        temp_table.insert(1, "models_dir", str(models_dir))
#        merge_table = pd.concat([temp_table, merge_table], axis=0)
#merge_table = merge_table.reset_index(drop=True)
#    
## add structure dict
#structure_dict = {"conv0 output size": None, "conv1 output size": None, "conv2 output size": None, 
#                  
#                  "perception_3 output size": None, "perception_3/conv_MxM": None, "perception_3/conv_C1": None,
#                  "perception_3/conv_C2": None, "perception_3/pool1": None, "perception_3/pool2": None,
#                  
#                  "perception_4 output size": None, "perception_4/conv_MxM": None, "perception_4/conv_C1": None,
#                  "perception_4/conv_C2": None, "perception_4/pool1": None, "perception_4/pool2": None,
#                  
#                  "perception_5 output size": None, "perception_5/conv_MxM": None, "perception_5/conv_C1": None,
#                  "perception_5/conv_C2": None, "perception_5/pool1": None, "perception_5/pool2": None
#                  }
#structure_df = pd.DataFrame(columns=['model structure'])
#print("\n\n extracting models structures ...")    
#for i in tqdm(np.arange(len(merge_table))):
#    record = merge_table.iloc[i]
#    target_model_dir = os.path.join(statistics_base_dir, record['models_dir'], record['model name'])
#    model_log_path = os.path.join(target_model_dir, "Log_training.txt")
#    structure_temp = SeismicPatchNet.Log_Restore_SeismicPatchNet_topologyV1(model_log_path, structure_dict)
#    structure_df = structure_df.append({"model structure": structure_temp}, ignore_index=True)
#
#merge_table = pd.concat([merge_table, structure_df], axis=1)
#
## sorting
#merge_table = merge_table.sort_values(['mean top5'], ascending=False)
#merge_table.reset_index(drop=True, inplace=True)
#merge_table = merge_table.round(decimals=6)
#        
## save 
#save_name_all = "compare_performance N{}_{}larger{}_{}.csv".format(
#                                    len(merge_table), threshold, 
#                                    len(merge_table[merge_table['mean top5']>=threshold]),
#                                    time.strftime("%Y%m%d%H%M%S"))
#merge_table.to_csv(os.path.join(statistics_base_dir, save_name_all), index=None)



#%% ==========================================================================
#                       Step 4: find good architectures
#               an simple example of selecting by high mean acc.
# ============================================================================

#threshold = 0.75         # validation score (mean top5). just for debugging here, remember to replace it with reasonable value (e.g. 0.95)
#performance_table_path = r".\results\synthetic_data\search 1\models_performance N16_0.996larger0_20200508213047.csv"  # just for debugging here, remember to replace it with your file
#models_save_to_dir = r".\results\synthetic_data\search 1\good ones"
#
#performace_table = pd.read_csv(performance_table_path)
#good_table = performace_table[performace_table["mean top5"]>=threshold]
#models_base_dir = os.path.abspath(os.path.join(performance_table_path, ".."))
#
## find good models folders
#for index, record in good_table.iterrows():
#    try:
#        target_model_dir = os.path.join(models_base_dir, record['models_dir'], record['model name'])
#    except:
#        target_model_dir = os.path.join(models_base_dir, record['model name'])
#    save_destination = os.path.join(models_save_to_dir, record['model name'])
#    if os.path.exists(save_destination):
#        shutil.rmtree(save_destination)
#    shutil.copytree(target_model_dir, save_destination)
 


#%% ==========================================================================
#                       Some other suggestions:
# always pay attention to quality control
# plot training processes of trained models, select models with reliable training
# ============================================================================


