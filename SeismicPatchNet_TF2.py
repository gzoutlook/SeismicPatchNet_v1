# -*- coding: utf-8 -*-
'''
Created on Wed Nov 4 16:28:02 2020 @author: Geng
Projects based on TensorFlow v2.3.x

Reference:
    Geng, Z., Wang, Y. Automated design of a convolutional neural network 
    with multi-scale filters for cost-efficient seismic data classification.
    Nat Commun 11, 3311 (2020). https://doi.org/10.1038/s41467-020-17123-6    
    
version feature:
    basic implementation
    
'''
import os
import numpy as np

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers

from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.regularizers as regularizers





def Build_SPN_TF2(architecture_var=None, num_of_class=2,
                  dropout_rate=0.3, is_training=True,
                  model_name="SPN", saveDir=""):
    """
    Build TensorFlow v2.x model for SPN
    
    architecture_var: dict. If it is None, then using defaults.

    Returns
    -------
    TensorFlow v2.x model

    """
    
    def fusion_layer_template_tf2(x, output_size,
                                        convMM_size, convMM_num, 
                                        num_of_c11_1, conv1_size, conv1_num, 
                                        num_of_c11_2, conv2_size, conv2_num, 
                                        pool1_size, num_of_pool1c1_max, num_of_pool1c1_min,
                                        pool2_size, num_of_pool2c1_max, num_of_pool2c1_min,
                                        name=""):

        convMM = layers.Conv2D(convMM_num, convMM_size, padding='same', name=name+"_conv_MxM")(x) 

        conv1_11 = layers.Conv2D(num_of_c11_1, [1, 1], padding='same')(x) 
        conv1 = layers.Conv2D(conv1_num, [conv1_size, 1], padding='same')(conv1_11)
        conv1 = layers.Conv2D(conv1_num, [1, conv1_size], padding='same', name=name+"_conv_C1")(conv1)

        conv2_11 = layers.Conv2D(num_of_c11_2, [1, 1], padding='same')(x) 
        conv2 = layers.Conv2D(conv2_num, [conv2_size, 1], padding='same')(conv2_11)
        conv2 = layers.Conv2D(conv2_num, [1, conv2_size], padding='same', name=name+"_conv_C2")(conv2)            

        pool_1_max = layers.MaxPool2D(pool_size=[pool1_size, pool1_size], strides=1, padding='same')(x)
        pool_11_max = layers.Conv2D(num_of_pool1c1_max, [1, 1], padding='same')(pool_1_max)
        pool_1_min = - layers.MaxPool2D(pool_size=[pool1_size, pool1_size], strides=1, padding='same')(-x)
        pool_11_min = layers.Conv2D(num_of_pool1c1_min, [1, 1], padding='same', name=name+"_pool1")(pool_1_min)

        pool_2_max = layers.MaxPool2D(pool_size=[pool2_size, pool2_size], strides=1, padding='same')(x)
        pool_21_max = layers.Conv2D(num_of_pool2c1_max, [1, 1], padding='same')(pool_2_max)
        pool_2_min = - layers.MaxPool2D(pool_size=[pool2_size, pool2_size], strides=1, padding='same')(-x)
        pool_21_min = layers.Conv2D(num_of_pool2c1_min, [1, 1], padding='same', name=name+"_pool2")(pool_2_min)        
        
        fused = layers.Concatenate(axis=-1)([convMM, conv1, conv2, pool_11_max, pool_11_min, pool_21_max, pool_21_min])
        
        return fused

        
        
    if architecture_var is None:
        archi_dict = {"input shape": [112, 112, 1],
                      "conv0 output size": 137, "conv1 output size": 137, "conv2 output size": 144, 
                      
                      "perception_3 output size": 133, "perception_3/conv_MxM": [2, 83], "perception_3/conv_C1": [104, 3, 2],
                      "perception_3/conv_C2": [112, 7, 16], "perception_3/pool1": [2, 12, 2, 11], "perception_3/pool2": [3, 4, 3, 5],
                      
                      "perception_4 output size": 151, "perception_4/conv_MxM": [2, 75], "perception_4/conv_C1": [280, 5, 5],
                      "perception_4/conv_C2": [382, 7, 7], "perception_4/pool1": [2, 20, 2, 21], "perception_4/pool2": [5, 12, 5, 11],
                      
                      "perception_5 output size": 459, "perception_5/conv_MxM": [2, 14], "perception_5/conv_C1": [46, 3, 25],
                      "perception_5/conv_C2": [170, 5, 116], "perception_5/pool1": [4, 107, 4, 107], "perception_5/pool2": [5, 45, 5, 45],
                      }
    
    
    inputs = layers.Input(shape=archi_dict["input shape"], name='input')
    x = inputs
    
    # 3 conv layers
    x = layers.Conv2D(filters=archi_dict["conv0 output size"], kernel_size=[7, 7],
               activation='elu', strides=2, padding='same', name='conv_0')(x)
    
    x = layers.MaxPool2D(pool_size=[3, 3], strides=2, padding='same')(x)       

    x = layers.Conv2D(filters=archi_dict["conv1 output size"], kernel_size=[1, 1],
               activation='elu', strides=2, padding='same', name='conv_1')(x)
    x = layers.Conv2D(filters=archi_dict["conv2 output size"], kernel_size=[3, 3],
               activation='elu', strides=1, padding='same', name='conv_2')(x)

    x = layers.MaxPool2D(pool_size=[3, 3], strides=2, padding='same', name='pooling_0')(x)
    
    # Fusin layer 1
    x = fusion_layer_template_tf2(x, archi_dict["perception_3 output size"],
                                     archi_dict["perception_3/conv_MxM"][0], archi_dict["perception_3/conv_MxM"][1],
                                     archi_dict["perception_3/conv_C1"][0], archi_dict["perception_3/conv_C1"][1], archi_dict["perception_3/conv_C1"][2],
                                     archi_dict["perception_3/conv_C2"][0], archi_dict["perception_3/conv_C2"][1], archi_dict["perception_3/conv_C2"][2],
                                     archi_dict["perception_3/pool1"][0], archi_dict["perception_3/pool1"][1], archi_dict["perception_3/pool1"][3],
                                     archi_dict["perception_3/pool2"][0], archi_dict["perception_3/pool2"][1], archi_dict["perception_3/pool2"][3],
                                     name='fusion_0')
    
    # Fusin layer 2
    x = fusion_layer_template_tf2(x, archi_dict["perception_4 output size"],
                                     archi_dict["perception_4/conv_MxM"][0], archi_dict["perception_4/conv_MxM"][1],
                                     archi_dict["perception_4/conv_C1"][0], archi_dict["perception_4/conv_C1"][1], archi_dict["perception_4/conv_C1"][2],
                                     archi_dict["perception_4/conv_C2"][0], archi_dict["perception_4/conv_C2"][1], archi_dict["perception_4/conv_C2"][2],
                                     archi_dict["perception_4/pool1"][0], archi_dict["perception_4/pool1"][1], archi_dict["perception_4/pool1"][3],
                                     archi_dict["perception_4/pool2"][0], archi_dict["perception_4/pool2"][1], archi_dict["perception_4/pool2"][3],
                                     name='fusion_1')    
    # Fusin layer 3
    x = fusion_layer_template_tf2(x, archi_dict["perception_5 output size"],
                                     archi_dict["perception_5/conv_MxM"][0], archi_dict["perception_5/conv_MxM"][1],
                                     archi_dict["perception_5/conv_C1"][0], archi_dict["perception_5/conv_C1"][1], archi_dict["perception_5/conv_C1"][2],
                                     archi_dict["perception_5/conv_C2"][0], archi_dict["perception_5/conv_C2"][1], archi_dict["perception_5/conv_C2"][2],
                                     archi_dict["perception_5/pool1"][0], archi_dict["perception_5/pool1"][1], archi_dict["perception_5/pool1"][3],
                                     archi_dict["perception_5/pool2"][0], archi_dict["perception_5/pool2"][1], archi_dict["perception_5/pool2"][3],
                                     name='fusion_2')    
    
    # Average pooling layer
    x = layers.AveragePooling2D(pool_size=[7, 7], strides=1, padding='valid', name='pooling_1')(x)
    
    # Reshape to [-1, 1]
    x = layers.Flatten()(x)
    # x = layers.Reshape([-1, archi_dict["perception_5 output size"]])(x)
    
    # Dropout
    x = layers.Dropout(dropout_rate)(x, training=is_training)
    
    # Dense: activation - softmax
    x = layers.Dense(num_of_class, activation='elu', name='output')(x)
    
    # Build model
    model = keras.Model(inputs, x, name='Model_SPN')
    model.summary()
    
    
    # plot and save graph
    if saveDir != "":
        # graph_dir = os.path.join(saveDir, "{} graph".format(model_name))
        graph_dir = saveDir
        os.makedirs(graph_dir, exist_ok=True)
        keras.utils.plot_model(model, to_file=os.path.join(graph_dir, "{} graph.jpg".format(model_name)),
                               show_shapes=True, dpi=300)    
    
    
    return model
    
    
    
    