# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 15:46:21 2019  @author: Geng

some shared functions 

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import os
import scipy.signal
import ast

#-------------------------------helper fuction---------------------------------#

def Statistics_Data_Scale(input_data, scale = 0.99, Zero_Centralization=True):
    """Determine data scale of overwhelming majority data """
    input_data = np.array(input_data)
    # Viz X Percentage of the data:
    delta = 1 - scale
    lower, upper = np.percentile(input_data, [100*delta*0.5, 100*(1-delta*0.5)])
    filtered_data = input_data[(input_data >= lower) & (input_data <= upper)]
    minLUT = filtered_data.min()
    maxLUT = filtered_data.max()
    # Viz 0-Centralization:
    if Zero_Centralization:
        if (np.abs(maxLUT) > np.abs(minLUT)) and minLUT*maxLUT<0:
            minLUT = -maxLUT     # larger scale the better
#            maxLUT = -minLUT    # smaller scale the better
        if (np.abs(maxLUT) < np.abs(minLUT)) and minLUT*maxLUT<0:
            maxLUT = -minLUT
#            minLUT = -maxLUT
    return minLUT, maxLUT


def statistics_Keys_value_range(keys_list, dict_str):
    
    Keys_value_range_dict = {key: [1e5, 0] for key in keys_list}
    for i in np.arange(len(dict_str)):
        dict_item = dict_str[i]
        dict_item = ast.literal_eval(dict_item)
        
        for key in keys_list:
            if Keys_value_range_dict[key][0] > dict_item[key]:
                Keys_value_range_dict[key][0] = dict_item[key]
            if Keys_value_range_dict[key][1] < dict_item[key]:
                Keys_value_range_dict[key][1] = dict_item[key] 
    
    return Keys_value_range_dict


def Smooth(y, box_pts):

    '''smooth the data'''
#    box = np.ones(box_pts)/box_pts
#    y_smooth = np.convolve(y, box, mode='same')
    
    y_smooth = scipy.signal.savgol_filter(y, box_pts, 1)
    
    return y_smooth


def Get_key_folder_name(base_folder, key_name):
    folder_name = ""
    for obj in os.listdir(base_folder):
        if os.path.isdir(os.path.join(base_folder, obj)) and obj.find(key_name)!=-1:
            folder_name = obj
            break
    return folder_name     



def Standardize_Seismic(source_data, method="Keep Polarity", feature_range=(-1, 1), SourceData_Bound=None):
    """Standardize seismic data.
    
    Args:
        source_data: numpy array.
        
        feature_range: the maximum bound of data.
        
        method: "Keep Polarity" or "Normalization".
        
    Returns:
        standardized numpy array   
    """
    seismic_data = source_data.copy()
    if SourceData_Bound is not None:
        min_value, max_value = SourceData_Bound
        seismic_data[seismic_data>max_value] = max_value
        seismic_data[seismic_data<min_value] = min_value  
    else:     
#        max_value, min_value = seismic_data.max(), seismic_data.min()
        min_value, max_value = Statistics_Data_Scale(seismic_data, scale = 0.99, Zero_Centralization=True)
        seismic_data[seismic_data>max_value] = max_value
        seismic_data[seismic_data<min_value] = min_value  
        
    symmetric_bound = 0
    # Senario 1: the data is polarity
    if method=="Keep Polarity" and max_value*min_value < 0:   
        scale_factor = feature_range[1]
        # determine the symmetric boundary:
        if max_value >= -min_value:
            symmetric_bound = max_value
        else:
            symmetric_bound = -min_value
        # standardization
        seismic_scaled = seismic_data / symmetric_bound * scale_factor
    # Senario 2: conventional normalization
    else:
        print("\nCaution: the seismic data is normalized between feature range: {}".format(feature_range))
        seismic_std = (seismic_data - min_value) / (max_value - min_value)
        seismic_scaled = seismic_std * (max(feature_range) - min(feature_range)) + min(feature_range)
    
    return seismic_scaled


def one_hot_encode(list_of_labels, num_classes=2):
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(categories=[range(num_classes)]) # # sklearn function: OneHotEncoder()
    input_list_of_labels = np.array(list_of_labels).reshape(-1, 1) # resize the list_of_labels
    one_hot_encoded_targets = encoder.fit_transform(input_list_of_labels)
    return one_hot_encoded_targets.toarray()


def Square_Sampling(ndarray, square_size=None):
    """ resampling "ndarray" (not 1d) to (square_size x square_size) """
    from skimage.transform import resize
    if square_size is None:
        square_size = np.max(ndarray.shape)
    new_ndarray = resize(ndarray, (square_size, square_size),mode='reflect', anti_aliasing=True, preserve_range=True)
    return new_ndarray   


def plot_images(num_image, patches_tensor, labels_tensor, labels_note=None, data_scale=0.98, img_aspect=None,
                fig_title="", mask=None, Square_sampling = True):
    """
    patches_tensor: numpy array (-1, tensor_size, tensor_size, channel)
    """
    Tensor_Size = patches_tensor.shape[1]
    if num_image>patches_tensor.size:
        num_image = patches_tensor.size
    if mask is None:
        mask = np.random.choice(len(patches_tensor), num_image, replace=False)
    imshow_patches = patches_tensor[mask,:].reshape(-1, Tensor_Size, Tensor_Size)
    imshow_labels = labels_tensor[mask,:]
    
    # color table
    cmapFile = "D:\My tools\ColorTable\Jason Used\Red_white_dkblue.rgb"
    num_of_img = num_image
    if num_of_img <=8:
        Shape = [1, 8]
    elif num_of_img>8 and num_of_img<=40:
        Shape = [num_of_img//8+1, 8]
    elif num_of_img>40 and num_of_img<=108:
        Shape = [num_of_img//12+1, 12]
    else:
        Shape = [num_of_img//24+1, 24]
        
    print("\nPlotting images: {} ...".format(fig_title))
    if labels_note is not None:
        imshow_labels_note = [labels_note[item] for item in mask]
    else:
        imshow_labels_note = None
    imshow_grid(imshow_patches, imshow_labels, cmapFile, shape=Shape, Note=imshow_labels_note, data_scale=data_scale,
                img_aspect=img_aspect, Square_sampling = Square_sampling, windowTitle="{}, size: {}".format(fig_title, (Tensor_Size,Tensor_Size)), figWidth=22)
    return mask


def imshow_grid(image_list, label_list, cmapFile, X_label=None , Y_label=None, Note=None, shape=[2, 8], data_scale=0.98, fontsize = 6,
                img_aspect=None, depth_interval=2, trace_Space=12.5, Height_Scale=2, Square_sampling = False, windowTitle="", figWidth=18):
    """Plot images in a grid of a given shape."""
    from mpl_toolkits.axes_grid1 import ImageGrid
    interval_velocity = 3    # km/s
    Aspect_Fig = depth_interval*(image_list[0].shape[0]-1)/2*interval_velocity*Height_Scale/((image_list[0].shape[1]-1)*trace_Space)
    if img_aspect is None:
        Aspect_img = depth_interval/2*interval_velocity*Height_Scale/trace_Space # aspect of pixel element
    else:
        Aspect_img = img_aspect
    fig, ax = subPlotDefi(windowTitle=windowTitle, figSizeCM=(figWidth, figWidth*Aspect_Fig),figShape=(1, 1), subAdjust=(0.125, 0.9, 0.05, 0.95), axLabelSize=12, axTickSize=12)
    grid = ImageGrid(fig, 111, nrows_ncols=shape, axes_pad=0.07, 
                     cbar_location="right", cbar_mode='edge', cbar_size="15%", cbar_pad=0.35)
    ax.axis('off')
    ax.set_title(windowTitle, fontsize=fontsize)
    
    # colortable
    cmapArray = pd.read_csv(cmapFile,header=None)
    myCmap = array2cmap(cmapArray)  
    # color scale
    minLUT, maxLUT = Statistics_Data_Scale(image_list, scale = data_scale, Zero_Centralization=True)
    # Plot image
    size = len(image_list)
    for i in range(size):
        image = image_list[i].copy()
        # if Square is True: resampling the matrix to Square
        if Square_sampling is True:
            image = Square_Sampling(image)
            Aspect_img = 1
            print("Attention: The images are resampled to square.")
        grid[i].axis('off')
        if X_label and Y_label and Note is not None:
            grid[i].set_title("No.{0}, Lb:{1}\n X:{2}, Y:{3}\n {4}".format(str(i+1), label_list[i], X_label[i], Y_label[i], Note[i]), fontsize=fontsize, pad=-22)
        elif X_label and Y_label is not None: 
            grid[i].set_title("No.{0}, Lb:{1}\n X:{2}, Y:{3}".format(str(i+1), label_list[i], X_label[i], Y_label[i]), fontsize=fontsize, pad=-15)
        elif Note is not None: 
            grid[i].set_title("No.{0}, Lb:{1}\n Note: {2}".format(str(i+1), label_list[i], Note[i]), fontsize=fontsize-1, pad=-15)
        else:
            grid[i].set_title("Lb:{}".format(label_list[i]), fontsize=fontsize, pad=-8)  # No.{0},   str(i+1), 
        im = grid[i].imshow(image, cmap=myCmap, vmax=maxLUT, vmin=minLUT, aspect=Aspect_img, interpolation='kaiser')  #
        if i % 2:
            grid.cbar_axes[i//2].colorbar(im)
            

def subPlotDefi(windowTitle='', figSizeCM=(8,5), figDpi=185, figShape=(1, 1), 
                subAdjust=(0.175, 0.825, 0.15, 0.85), subSpace=(0.25,0.25), 
                axLabelSize=12, axTickSize=12, shareY=False, constrained_layout=None): 
    # figDpi=216, 185 24"4K

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams['axes.titlesize'] = axLabelSize    
    mpl.rcParams['xtick.labelsize'] = axTickSize   
    mpl.rcParams['ytick.labelsize'] = axTickSize
    mpl.rcParams['axes.labelsize'] = axLabelSize   
    mpl.rcParams['axes.labelpad'] = 2               
    mpl.rcParams['legend.fontsize'] = axLabelSize  
    mpl.rcParams['savefig.dpi'] = 300              
#    mpl.rcParams['text.usetex'] = True              
#    mpl.rcParams['lines.linewidth'] = 2            
#    mpl.rcParams['lines.color'] = 'r'              
    figSize = (figSizeCM[0]/2.54,figSizeCM[1]/2.54) 
    fig, axes = plt.subplots(nrows=figShape[0], ncols=figShape[1], figsize=figSize,  
                             dpi=figDpi, facecolor='white', sharey=shareY, constrained_layout=constrained_layout)
    fig.canvas.set_window_title(windowTitle)        
    fig.subplots_adjust(wspace=subSpace[0],hspace=subSpace[1])   
    fig.subplots_adjust(left=subAdjust[0], right=subAdjust[1],
                        bottom=subAdjust[2], top=subAdjust[3])   
    return    fig, axes    

            
def array2cmap(X):
    from matplotlib import colors
    # Assuming array is Nx3, where x3 gives RGB values
    # Append 1's for the alpha channel, to make X Nx4
    X = np.c_[X,np.ones(len(X))]

    return colors.LinearSegmentedColormap.from_list('my_colormap', X)


def lowerleft_corner_of_box (center_xy, box_width, box_height):
    """
    center_xy: (x, y)
    """
    x = center_xy[0] - box_width/2.0
    y = center_xy[1] - box_height/2.0
    return (x, y)   


def Plot_SeismicPatchNet_topologyV1_evolution(num_models_plot, performance_table, 
                                              fig_width=11, fig_height=12, font_size=12, 
                                              fig_title="", window_title="Evolution"):
    import matplotlib.patches as mpatches
    from matplotlib.collections import PatchCollection
    from matplotlib.collections import LineCollection
    
    # data
    step = int(len(performance_table)/num_models_plot)
    select_indexes = np.arange(0, len(performance_table)-1, step)
    performance_table = performance_table.loc[select_indexes,:]
    performance_table = performance_table.sort_values(['mean top5'], ascending=True) # the last one is the BEST
    performance_table = performance_table.reset_index(drop=True)
    num_of_item = len(performance_table)
    
    # geometry
    unit_width_range = [50, 150]
    unit_height_range = [50, 150]
    unit_keys = ["conv0 output size", "conv1 output size", "conv2 output size", 
                 "perception_3 output size", "perception_4 output size", "perception_5 output size"]   
    center_xy_start = [0, 0]
    y_step = 250
    
    # analysis
    print("\n\n Analyzing data ...")
    keys_value_range_dict = statistics_Keys_value_range(unit_keys, performance_table["model structure"].to_list())
    structure_table = pd.DataFrame() # unit sizes to be plotted
    for i in tqdm(np.arange(num_of_item)):
        models_structure = performance_table.loc[i, "model structure"]
        models_structure = ast.literal_eval(models_structure)
        plot_dict = {key: models_structure[key] for key in unit_keys}
        structure_table = structure_table.append(plot_dict, ignore_index=True)
    structure_table.insert(0, 'mean top5', performance_table['mean top5'])
        
    # plot status   
    print("\n\n Ploting models evolution ...")
    # color scheme
    if num_of_item==1: num_of_color = 2
    else: num_of_color = num_of_item
    cmap = plt.get_cmap("rainbow") # "rainbow", 'Reds'
    colors = [cmap(i) for i in np.linspace(0, 1, num_of_color)]
    
    
#    #--------------------- generate figure: unit shape -----------------------#
    fig_shape, axis_shape = subPlotDefi(windowTitle="{}: shape".format(window_title), figSizeCM=(fig_width, fig_height), 
                             subAdjust=(0.15, 0.85, 0.1, 0.85), axLabelSize=font_size, axTickSize=font_size)   
    # evolution
    x_max = 0
    for i in tqdm(np.arange(num_of_item)):
        plot_structure = structure_table.iloc[i]
        
        # if it is the best one
        if i==num_of_item-1:
            Linewidth = 2 
            Edgecolor = 'black' # colors[i]
            Alpha = 0.9
        else:
            Linewidth = 1.8    
            Edgecolor=colors[i]
            Alpha = 0.7
            
        key_ix=0
        for key in unit_keys:
            value = plot_structure[key]
            box_width = ((value - min(keys_value_range_dict[key])) / (max(keys_value_range_dict[key]) - min(keys_value_range_dict[key]))) * (max(unit_width_range) - min(unit_width_range)) + min(unit_width_range)
            box_height = ((value - min(keys_value_range_dict[key])) / (max(keys_value_range_dict[key]) - min(keys_value_range_dict[key]))) * (max(unit_height_range) - min(unit_height_range)) + min(unit_height_range)
            if x_max < box_width:
                x_max = box_width
            fancybox = mpatches.FancyBboxPatch(
                    lowerleft_corner_of_box((center_xy_start[0], center_xy_start[1]+key_ix*y_step), box_width, box_height),
                    box_width, box_height, facecolor="None", edgecolor=Edgecolor, linewidth=Linewidth, alpha=Alpha,
                    boxstyle=mpatches.BoxStyle("Round", pad=8), zorder=i)
            key_ix += 1
            axis_shape.add_patch(fancybox)
            
    
    axis_shape.set_xlim(xmin = -x_max*1.2/2, xmax = x_max*1.2/2)
    axis_shape.set_ylim(bottom = -max(unit_height_range), top = y_step*len(keys_value_range_dict)*0.92)
    axis_shape.set_xticks([]) 
    axis_shape.set_yticks([])
    axis_shape.set_title(fig_title)
    # Colorbar
    vmin = performance_table["mean top5"].min()    # colorbar lower limit
    vmax = performance_table["mean top5"].max()   # colorbar upper limit
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    cbar = plt.colorbar(sm, ticks=np.round(np.linspace(vmin, vmax, 5), 4))
    cbar.ax.set_ylabel('acc.: mean top5')
    
    
    #--------------------- generate figure: size curve -----------------------#
    fig_curve, axis_curve = subPlotDefi(windowTitle="{}: curve".format(window_title), figSizeCM=(fig_width, fig_height), 
                             figShape=(len(unit_keys)+1, 1), subAdjust=(0.15, 0.85, 0.1, 0.85), 
                             subSpace=(0.25,1), axLabelSize=font_size, axTickSize=font_size)   
    # lines
    x = structure_table["mean top5"].values  
    for i in np.arange(len(unit_keys)):
        y = structure_table[unit_keys[i]].values
        subfig_pos = len(unit_keys)- 1 - i
        axis_curve[subfig_pos].plot(x, y, color="#80DEEA", alpha=0.7, lw=1, zorder=1)
    # colorful scatter
    for i in np.arange(num_of_item):
        scatter_x = structure_table.loc[i, "mean top5"]
        sizes = structure_table.loc[i, unit_keys].values
        for j in np.arange(len(unit_keys)):
            y = sizes[j]
            subfig_pos = len(unit_keys)- 1 - j
            axis_curve[subfig_pos].scatter(scatter_x, y, color=colors[i], s=10, alpha=1, zorder=2)
    # histogram
    Bins=int(len(x)/5)
    if len(x)<25:
        Bins=int(len(x)/2)
    axis_curve[-1].hist(x, bins=Bins, color="#26C6DA")
    # final format
    best_size = structure_table.iloc[-1][unit_keys].values   # best result   
    for k in np.arange(len(unit_keys)):
        subfig_pos = len(unit_keys)- 1 - k
        axis_curve[subfig_pos].scatter(structure_table["mean top5"].values[-1], 
                                      best_size[k], 
                                      edgecolor=Edgecolor, color="None", s=11, linewidths=1.2, alpha=Alpha, zorder=3)  
        axis_curve[subfig_pos].set_title(unit_keys[k], color='black')
#        axis_curve[subfig_pos].xaxis.set_tick_params(labelsize=10)     
        axis_curve[subfig_pos].set_xticks([])
    axis_curve[-1].set_xlabel("mean acc.")
    axis_curve[-1].xaxis.set_tick_params(labelsize=font_size)
    axis_curve[-1].set_title("performance distribution", color='black')
    fig_curve.text(0.02, 0.5, 'size', fontsize=font_size, va='center', rotation='vertical')   
    fig_curve.suptitle(fig_title)   

    return fig_shape, fig_curve 



