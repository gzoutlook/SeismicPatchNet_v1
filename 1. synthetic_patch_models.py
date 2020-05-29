# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 15:46:21 2019  @author: Geng

generate synthetic dataset for research

Note that the synthetic dataset is only used to simulate features with polarization, 
which are not real waveforms signals. More complicated dataset can be made by 
multiplying reflection fators with various wavelets.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import os

path = '.\libs'
import sys
sys.path.append(path)
import helper


#----------------------------samples making fuction----------------------------#
    
def _rotate(  
    img,  #image matrix  
    angle #angle of rotation  
    ):  

    height = img.shape[0]  
    width = img.shape[1]  

    if angle%180 == 0:  
        scale = 1  
    elif angle%90 == 0:  
        scale = float(max(height, width))/min(height, width)  
    else:  
        scale = np.sqrt(np.power(height,2)+np.power(width,2))/np.min([height, width])  

    #print 'scale %f\n' %scale  

    rotateMat = cv2.getRotationMatrix2D((width/2, height/2), angle, scale)  
    rotateImg = cv2.warpAffine(img, rotateMat, (width, height))  
    return rotateImg


def Background(patch_height, target_thickness_range=None, target_positive_value_range=[0.2, 0.99], type="zero", rotate_angle_range=[-30, 30]):
    def _indices_to_split(patch_height):
        lithofacies_list = []
        num_lines = 0
        while num_lines<patch_height: 
            facies_thickness = np.rint(np.random.uniform(1,target_thickness_range[0]))
            num_lines += facies_thickness
            if num_lines < patch_height:
                lithofacies_list.append(int(num_lines))
        return lithofacies_list
    
    if type=="zero":
        background_patch = np.zeros([patch_height,patch_height])   
    elif type=="environmental":
        target_height = np.copy(patch_height)
        if rotate_angle_range is not None:
            patch_height = int(np.ceil(1.414*patch_height))
        depth_index = np.arange(patch_height)
        indices_to_split = _indices_to_split(patch_height)
        facies_array = np.array_split(depth_index, indices_to_split)
        background_patch = np.zeros([patch_height,patch_height])
        background_patch[:facies_array[0][0]] = np.random.uniform(-target_positive_value_range[0]*1.5, target_positive_value_range[0]*1.2)
        for facies in facies_array:
            background_patch[facies[0]-1:facies[-1]] = np.random.uniform(-target_positive_value_range[0]*1.5, target_positive_value_range[0]*1.2)
        background_patch[-1] = np.random.uniform(-target_positive_value_range[0]*1.5, target_positive_value_range[0]*1.5)
        if rotate_angle_range is not None:
            patch_rotated = _rotate(background_patch, np.random.uniform(rotate_angle_range[0], rotate_angle_range[1]))
            background_patch = patch_rotated[int((patch_height-target_height)/2):int((patch_height+target_height)/2), 
                                             int((patch_height-target_height)/2):int((patch_height+target_height)/2)]
    
    return background_patch
    
    
def Seabed(target_height, target_center_position, target_thickness_frac, target_negative_range, target_positive_range, rotate_angle_range=[-8,8]):
    big_size = int(np.ceil(1.414*target_height))
    patch = Background(big_size, type="zero")
    background_big = Background(big_size, type="environmental", target_thickness_range=[4,8], 
                              target_positive_value_range=[0.2, 0.99], rotate_angle_range=rotate_angle_range)
    # seabed reflection  
    target_positive = np.random.uniform(target_positive_range[0], target_positive_range[1])
    target_negative = np.random.uniform(target_negative_range[0], target_negative_range[1])
    target_center_ind = int(target_center_position*big_size)
    target_thickness_num = int(target_thickness_frac*big_size)
    patch[target_center_ind-int(target_thickness_num/2):target_center_ind] = target_positive
    patch[target_center_ind:target_center_ind+int(target_thickness_num/2)] = target_negative
    patch[target_center_ind+int(target_thickness_num/2):] = background_big[target_center_ind+int(target_thickness_num/2):]
    patch_rotated = _rotate(patch, np.random.uniform(rotate_angle_range[0], rotate_angle_range[1]))
    
    target_patch = patch_rotated[int((big_size-target_height)/2):int((big_size+target_height)/2), 
                                 int((big_size-target_height)/2):int((big_size+target_height)/2)]
    label = 0
    return target_patch, label    
    
    
def BSR_like(target_height, target_center_position, target_thickness_frac, target_negative_range, target_positive_range, type="BSR_positive", rotate_angle_range=[-15,15]):
    """
    type="BSR_positive": real bsr like
    type="BSR_negative": real seabed like
    """
    big_size = int(np.ceil(1.414*patch_height))
    # background reflection
    background_big = Background(big_size, type="environmental", target_thickness_range=[4,8], 
                              target_positive_value_range=[0.2, 0.99], rotate_angle_range=[-20, 20])
    # target reflection
    patch = np.zeros([big_size, big_size])
    target_positive = np.random.uniform(target_positive_range[0], target_positive_range[1])
    target_negative = np.random.uniform(target_negative_range[0], target_negative_range[1])
    target_center_ind = int(target_center_position*big_size)
    target_thickness_num = int(target_thickness_frac*big_size)
    
    if type=="BSR_positive":
        patch[target_center_ind-int(target_thickness_num/2):target_center_ind] = target_negative
        patch[target_center_ind:target_center_ind+int(target_thickness_num/2)] = target_positive
        label = 1
    elif type=="BSR_negative":
        patch[target_center_ind-int(target_thickness_num/2):target_center_ind] = target_positive
        patch[target_center_ind:target_center_ind+int(target_thickness_num/2)] = target_negative
        label = 0
        
    patch_rotated = _rotate(patch, np.random.uniform(rotate_angle_range[0], rotate_angle_range[1]))
    
    # merge patch
    patch_rotated = np.where(patch_rotated==0, background_big, patch_rotated)
    target_patch = patch_rotated[int((big_size-target_height)/2):int((big_size+target_height)/2), 
                                     int((big_size-target_height)/2):int((big_size+target_height)/2)]

    return target_patch, label

def generate_patch(num_samples, type="BSR_positive", patch_size=112):
    num_samples = int(np.round(num_samples))
    target_center_position = np.random.normal(0.5, 0.05, num_samples)
    target_thickness = np.random.uniform(0.06, 0.1, num_samples)
    
    target_positive_value_range = np.array([0.2, 0.99])
    target_negative_value_range = -target_positive_value_range
    
    patch_list = []
    label_list = []
    print("\nGenerating {} patches ...".format(type))
    # BSR_like: positive
    if type == "BSR_positive":
        for i in tqdm(np.arange(num_samples)):
            patch, label = BSR_like(patch_height, target_center_position[i], target_thickness[i], 
                                    target_negative_value_range, target_positive_value_range, 
                                    type="BSR_positive")
            label_list.append(label)
            patch_list.append(patch)
            
    # BSR_like: negative
    if type == "BSR_negative":
        for i in tqdm(np.arange(num_samples)):
            patch, label = BSR_like(patch_height, target_center_position[i], target_thickness[i], 
                                    target_negative_value_range, target_positive_value_range, 
                                    type="BSR_negative")
            label_list.append(label)
            patch_list.append(patch)    
    
    # sea bed
    elif type == "seabed":
        for i in tqdm(np.arange(num_samples)):
            patch, label = Seabed(patch_height, target_center_position[i], target_thickness[i], target_negative_value_range, target_positive_value_range)
            label_list.append(label)
            patch_list.append(patch)
    # environments
    elif type == "environmental":
        for i in tqdm(np.arange(num_samples)):
            patch= Background(patch_height, type="environmental", target_thickness_range=[4,8], 
                                  target_positive_value_range=[0.2, 0.99], rotate_angle_range=[-20, 20])
            label = 0
            label_list.append(label)
            patch_list.append(patch)
            
    return patch_list, label_list
 

def make_samples(patches_list, labels_list, guidance, num_classes=2, shuffle=True, seismic_standardize=True): 
    import imgaug.augmenters as iaa
    samples_list = []
    label_list = []
    label_note_list = []
    # just for using tqdm to monitor the process. WASTE Memory.
    for i in np.arange(len(patches_list)):
        samples_list.extend(patches_list[i])
        label_list.extend(labels_list[i])
        
    #----------------------------sample corrupt--------------------------------#
    # do not use iaa.Sometimes, cause wanna to record augmenter status mannually
  
    num_patch = len(samples_list)
    patch_size = samples_list[0].shape[0]
    patches_tensor = np.zeros((num_patch, patch_size, patch_size))
    labels_tensor = np.zeros((num_patch, num_classes))
 
    print("\nProcessing samples ...")
    for i in tqdm(np.arange(num_patch)):
        """
        back ground noise
        """
        temp_array = samples_list[i]
        # brightness
        temp_array = temp_array*np.random.uniform(low=0.2, high=1) # brightness
        label_note = "Bright; "
        
        for aug in guidance:
            if aug[0] >= np.random.random():
                temp_array = aug[-1](images=np.float32(temp_array))
                label_note += aug[1]+"; "   
        
        patches_tensor[i,:] = temp_array
        labels_tensor[i,:] = helper.one_hot_encode(label_list[i], num_classes=num_classes)
        label_note_list.append(label_note)

    #--------------------------seismic standardize----------------------------#    
    if seismic_standardize:
        patches_tensor = helper.Standardize_Seismic(patches_tensor, method="Keep Polarity", feature_range=(-1, 1), SourceData_Bound=None)

    #-------------------------------shuffle-----------------------------------#    
    if shuffle:
        state = np.random.get_state()
        np.random.shuffle(patches_tensor)
        np.random.set_state(state)
        np.random.shuffle(labels_tensor)
        np.random.set_state(state)
        np.random.shuffle(label_note_list)
        
    return patches_tensor, labels_tensor, label_note_list


def generate_samples(num_samples, patch_height, samples_category, tag_type_list, corrupt_list, samples_save_dir):
    patches_list = []
    labels_list = []
    
    # generate patches
    for tag_type in tag_type_list:
        patch_list, patch_labels = generate_patch(num_samples*tag_type[0], type=tag_type[1], patch_size=patch_height)
        # merge patches
        patches_list.append(patch_list)
        labels_list.append(patch_labels)
                 
    # make samples
    print("\n\n Processing {} samples, patch size: {} {}".format(num_samples, patch_height, " for {}".format(samples_category)))
    patches_tensor, labels_tensor, label_note_list = make_samples(patches_list, labels_list, corrupt_list, 
                                                                  shuffle=True, seismic_standardize=True)
    
    #----------- plot samples -----------#
    num_image = 32
    helper.plot_images(num_image, patches_tensor, labels_tensor, labels_note=label_note_list, fig_title="Raw samples - {}".format(samples_category))
    
    #----------- save samples -----------#
    import time
    import pickle
    print("\nSaving samples ...")
    pickle.dump((patches_tensor, labels_tensor, label_note_list), open(os.path.join(samples_save_dir, "Patches_samples _n{}s{} {} {}.sp".format(num_samples, patch_height, time.strftime("%Y%m%d%H%M"), samples_category)), 'wb'), protocol=4)  # save samples
    # note for samples
    note_file = os.path.join(samples_save_dir, "Note for samples _n{}s{} {}, {}.txt".format(num_samples, patch_height, time.strftime("%Y%m%d%H%M"), samples_category))
    with open(note_file,"w") as log:   
        log.write("{} note for samples \n\n".format(time.strftime("%Y%m%d%H%M")))
        log.write("Tag type:\n")
        for tag_type in tag_type_list:
            log.write(" {}: {}\n".format(tag_type[1], tag_type[0]))
        log.write("\nNumber of samples: {}; patch height (size): {}\n\n".format(num_samples, patch_height))
        log.write("\ncorrupt info.:\n\n")
        for aug in corrupt_list:
            log.write(" {}\n\n".format(str(aug)))


def make_tensors(samples_dir, tensor_size=112):
    import pickle
    num_plot = 32
    # load samples
    listFilesInDir = os.listdir(samples_dir)
    for files in listFilesInDir:
        if files.endswith(".sp"):                       # find samples file
            file_path = os.path.join(samples_dir, files)  
            _, tempfilename = os.path.split(file_path)
            filename, _ = os.path.splitext(tempfilename)
            samples_category = filename.split(" ")[-1]
            # read
            patches_tensor, labels_tensor, label_note_list = pickle.load(open(file_path, mode='rb'))
            # plot samples
            mask = helper.plot_images(num_plot, patches_tensor, labels_tensor, labels_note=label_note_list, data_scale=1, img_aspect=1,
                                       Square_sampling = False, fig_title="Raw samples - {}".format(samples_category))
            
            #------- make learning tensors ------#
            
            learning_tensor = np.zeros((len(patches_tensor), tensor_size, tensor_size))
            print("\n Making tensor data set ... : {}".format(learning_tensor.shape))
            for i in tqdm(np.arange(len(patches_tensor))):    
                learning_tensor[i,:] = helper.Square_Sampling(patches_tensor[i,:], square_size=tensor_size)   # resampling
            # reshape to learning dataset
            learning_tensor = learning_tensor.reshape(-1, tensor_size, tensor_size, 1)
            # plot learning tensors
            helper.plot_images(num_plot, learning_tensor, labels_tensor, labels_note=label_note_list, data_scale=1, img_aspect=1,
                                Square_sampling = False, fig_title="Learning Tensors - {}".format(samples_category), mask=mask)

            #----------- save tensors -----------#  
            print("\nSaving learning tensor from: {}".format(filename))
            save_tensor_name = "{} tensor{} - {}.ds".format(samples_category, tensor_size, filename)
            pickle.dump((learning_tensor, labels_tensor), open(os.path.join(samples_dir, save_tensor_name), 'wb'), protocol=4)  # train set
          
    
    
if __name__ == "__main__":

#%% # ========================================================================
    #                              Global setting
    # ========================================================================    
    
    # -------- samples settings ------- #
    samples_category = "Test"  # "Train" or "Test"
    
    total_num_samples = 2000     # 20000, the script automaticly split the samples
     
    patch_height = 48
    test_size = 0.15             # split fraction, 15% of 20000 samples are test samples
    samples_save_dir = r".\results\synthetic_data"
    
    # (percent, tag type)
    tag_type_list = [
            [0.5, "BSR_positive"],
            [0.38, "BSR_negative"],
            [0.02, "seabed"],
            [0.1, "environmental"],
            ]
      
    #---------- corrupt methond ---------#
    import imgaug.augmenters as iaa
    # train data corrupt
    corrupt_list_train = [
            [0.4, "Blur", iaa.GaussianBlur(np.random.uniform(low=0.05, high=2))],
            [0.5, "Gau.Noise", iaa.AdditiveGaussianNoise(scale=0.1*np.random.uniform(low=0.05, high=0.5))],
            [0.35, "Elas.Trans.", iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=(0.1, 2))],
            [0.4, "Freq.Noise", iaa.FrequencyNoiseAlpha(exponent=-2, first=iaa.EdgeDetect(1.0), size_px_max=np.random.randint(1,patch_height))],
            [0.35, "Persp.Tras.", iaa.PerspectiveTransform(scale=(0.01, 0.1))],
            [0.35, "Cors.Drop.", iaa.CoarseDropout(p=np.random.uniform(low=0.05, high=0.2), size_percent=(np.random.uniform(low=0.05, high=0.2), np.random.uniform(low=0.05, high=0.2)))]
            ]
      
    # test data corrupt
    corrupt_list_test = [
            [0.5, "Blur", iaa.GaussianBlur(np.random.uniform(low=0.05, high=3))],
            [0.5, "Gau.Noise", iaa.AdditiveGaussianNoise(scale=0.1*np.random.uniform(low=0.01, high=1.5))],
            [0.45, "Elas.Trans.", iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=(0.01, 3))],
            [0.5, "Freq.Noise", iaa.FrequencyNoiseAlpha(exponent=-2, first=iaa.EdgeDetect(1.0), size_px_max=np.random.randint(1,patch_height))],
            [0.45, "Persp.Tras.", iaa.PerspectiveTransform(scale=(0.01, 0.2))],
            [0.4, "Cors.Drop.", iaa.CoarseDropout(p=np.random.uniform(low=0.05, high=0.3), size_percent=(np.random.uniform(low=0.01, high=0.3), np.random.uniform(low=0.01, high=0.3)))]
            ]
    
    
#%% # ========================================================================
    #                          Step 1:  make samples
    # ========================================================================
    
    if samples_category == "Train":
        num_samples = int(total_num_samples*(1-test_size))
        corrupt_list = corrupt_list_train
    else:
        num_samples = int(total_num_samples*test_size)
        corrupt_list = corrupt_list_test
    
    generate_samples(num_samples, patch_height, samples_category, tag_type_list, corrupt_list, samples_save_dir)
    

    # ========================================================================
    #                           Step 2:  make tensor
    # ========================================================================
    
    samples_dir = samples_save_dir
    Tensor_Size = 112  # 224, 227
    
    make_tensors(samples_save_dir, tensor_size=Tensor_Size)
    
    
    
    
    
    
    
    
    
    
    