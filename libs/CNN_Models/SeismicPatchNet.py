
'''
https://github.com/da-steve101/googlenet/blob/master/googlenet_model.py

G Modified tf.__version__:'1.13.0-rc2'
'''

import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.framework as ops

#import tensorflow.compat.v1.layers as layers
#import tensorflow.python.framework.ops as ops

import numpy as np
import itertools
import random
import re

#--------------------------- Restore methods ---------------------------------#
def Log_Restore_SeismicPatchNet_topologyV1(model_log_path, structure_dict):
    """
    structure_dict:
        "conv0 output size"
        "conv1 output size"
        "conv2 output size"
   
        "perception_3 output size"
        "perception_3/conv_MxM"
        "perception_3/conv_C1"
        "perception_3/conv_C2"
        "perception_3/pool1"
        "perception_3/pool2"
        
        ...
        
    """
    structure_dict = structure_dict.copy()
    # --- read log txt ---
    txt_file = open(model_log_path,'r').readlines()
    # search paras.
    for key in structure_dict.keys():
        for line in txt_file:
            if line.find(key) != -1:
                if line.find("=") != -1:
                   p = re.compile('(?<==)\s*\d+')
                   structure_dict[key] = list(map(int,p.findall(line)))     
                else:
                   p = re.compile('(?<=:)\s*\d+')
                   structure_dict[key] = list(map(int,p.findall(line)))[0]     
                break
    
    return structure_dict


def Find_L1_L2_lamda(txt_path):
    txt_file = open(txt_path,'r').readlines()
    p_left, p_right = 0, 0
    lamda_list = ""
    # search paras.
    for line in txt_file:
        if line.find("Regularization") and line.find("[") and line.find("]") != -1:
            p_left, p_right = line.find("["), line.find("]")  
            lamda_list = line[p_left+1: p_right]
            break
    lamda_list = lamda_list.split(",")
    lamda_list = list(map(float, lamda_list))
    return lamda_list


#-------------------------- Prototype design ---------------------------------#

def perception_layer( inputs, convMM_size, convB3_11_size, convB3_size,
                         convB5_11_size, convB5_size, poolP_size ):
    with tf.variable_scope("conv_MxM"):
        conv11 = layers.conv2d( inputs, convMM_size, [ 1, 1 ], activation_fn = tf.nn.leaky_relu)
    with tf.variable_scope("conv_B3x3"):
        conv33_11 = layers.conv2d( inputs, convB3_11_size, [ 1, 1 ], activation_fn = tf.nn.leaky_relu)
        conv33 = layers.conv2d( conv33_11, convB3_size, [ 3, 1 ], activation_fn = tf.nn.leaky_relu)
        conv33 = layers.conv2d( conv33, convB3_size, [ 1, 3 ], activation_fn = tf.nn.leaky_relu)
    with tf.variable_scope("conv_B5x5"):
        conv55_11 = layers.conv2d( inputs, convB5_11_size, [ 1, 1 ], activation_fn = tf.nn.leaky_relu)
        conv55 = layers.conv2d( conv55_11, convB5_size, [ 5, 1 ], activation_fn = tf.nn.leaky_relu)
        conv55 = layers.conv2d( conv55, convB5_size, [ 1, 5 ], activation_fn = tf.nn.leaky_relu)
    with tf.variable_scope("pool_max"):
        pool_max = layers.max_pool2d( inputs, [ 3, 3 ], stride = 1 )
        pool11_max = layers.conv2d( pool_max, int(poolP_size/2), [ 1, 1 ] )
    with tf.variable_scope("pool_min"):
        pool_min = - layers.max_pool2d(-inputs, [ 3, 3 ], stride = 1 )
        pool11_min = layers.conv2d( pool_min, poolP_size - int(poolP_size/2), [ 1, 1 ] )
    if tf.__version__ == '0.11.0rc0':
        return tf.concat(3, [conv11, conv33, conv55, pool11_max, pool11_min])
    return tf.concat([conv11, conv33, conv55, pool11_max, pool11_min], 3)


def perception_layer_2scale( inputs, convMM_size, convB3_11_size, convB3_size,
                         convB5_11_size, convB5_size, poolP_size ):
    with tf.variable_scope("conv_MxM"):
        conv11 = layers.conv2d( inputs, convMM_size, [ 1, 1 ] )
    with tf.variable_scope("conv_B3x3"):
        conv33_11 = layers.conv2d( inputs, convB3_11_size, [ 1, 1 ] )
        conv33 = layers.conv2d( conv33_11, convB3_size, [ 3, 1 ] )
        conv33 = layers.conv2d( conv33, convB3_size, [ 1, 3 ] )
    with tf.variable_scope("conv_B5x5"):
        conv55_11 = layers.conv2d( inputs, convB5_11_size, [ 1, 1 ] )
        conv55 = layers.conv2d( conv55_11, convB5_size, [ 5, 1 ] )
        conv55 = layers.conv2d( conv55, convB5_size, [ 1, 5 ] )
    with tf.variable_scope("pool33_max"):
        pool33_max = layers.max_pool2d( inputs, [ 3, 3 ], stride = 1 )
        pool33_max11 = layers.conv2d( pool33_max, int(poolP_size/4), [ 1, 1 ] )
    with tf.variable_scope("pool55_max"):
        pool_max55 = layers.max_pool2d( inputs, [ 5, 5 ], stride = 1 )
        pool55_max11 = layers.conv2d( pool_max55, int(poolP_size/4), [ 1, 1 ] )        
    with tf.variable_scope("pool33_min"):
        pool_min33 = - layers.max_pool2d(-inputs, [ 3, 3 ], stride = 1 )
        pool33_min11 = layers.conv2d( pool_min33, int(poolP_size/4), [ 1, 1 ] )
    with tf.variable_scope("pool55_min"):
        pool_min55 = - layers.max_pool2d(-inputs, [ 5, 5 ], stride = 1 )
        pool55_min11 = layers.conv2d( pool_min55, int(poolP_size/4), [ 1, 1 ] )        
    if tf.__version__ == '0.11.0rc0':
        return tf.concat(3, [conv11, conv33, conv55, pool33_max11, pool55_max11, pool33_min11, pool55_min11])
    return tf.concat([conv11, conv33, conv55, pool33_max11, pool55_max11, pool33_min11, pool55_min11], 3)



#-------------------------- Topology design ---------------------------------#

def perception_layer_topologyV1( inputs, output_size, convMM_size, conv1_size, conv2_size, pool1_size, pool2_size, name=""):
    """
    5 unit in the layer
    """
    # hyperparameters
    c11_num_range = [16,384]
    num_of_unit = 5
    
    # Architecture
    perception_structure = ""
    # generate proportion of units
    unit_proporsion = np.round(np.random.dirichlet(np.ones(num_of_unit),size=1)* output_size)[0].astype(int)
    # sum = output_size
    if np.sum(unit_proporsion) < output_size:
        unit_proporsion[np.argmin(unit_proporsion)] += output_size - np.sum(unit_proporsion)
    elif np.sum(unit_proporsion) > output_size:
        unit_proporsion[np.argmax(unit_proporsion)] -= np.sum(unit_proporsion) - output_size
    # minimum >= 2
    num_below =  len(unit_proporsion[unit_proporsion<2])
    if num_below > 0:
        for i in np.arange(num_below):
           unit_proporsion[np.argmax(unit_proporsion)] = unit_proporsion.max()- (2-unit_proporsion.min())
           unit_proporsion[np.argmin(unit_proporsion)] = 2 
    
    num_of_c11_1 = np.random.randint(c11_num_range[0], c11_num_range[1])
    num_of_c11_2 = np.random.randint(c11_num_range[0], c11_num_range[1])
        
    # build graph    
    with tf.variable_scope("conv_MxM"):
        convMM = layers.conv2d( inputs, unit_proporsion[0], [ convMM_size, convMM_size ] )
    with tf.variable_scope("conv_C1"):
        conv1_11 = layers.conv2d( inputs, num_of_c11_1, [ 1, 1 ] )
        conv1 = layers.conv2d( conv1_11, unit_proporsion[1], [ conv1_size, 1 ] )
        conv1 = layers.conv2d( conv1, unit_proporsion[1], [ 1, conv1_size ] )
    with tf.variable_scope("conv_C2"):
        conv2_11 = layers.conv2d( inputs, num_of_c11_2, [ 1, 1 ] )
        conv2 = layers.conv2d( conv2_11, unit_proporsion[2], [ conv2_size, 1 ] )
        conv2 = layers.conv2d( conv2, unit_proporsion[2], [ 1, conv2_size ] )
    with tf.variable_scope("pool1"):
        pool1_max = layers.max_pool2d( inputs, [ pool1_size, pool1_size ], stride = 1 )
        num_of_pool1c1_max = np.round(unit_proporsion[3]*0.5).astype(int)
        pool11_max = layers.conv2d( pool1_max, num_of_pool1c1_max, [ 1, 1 ] )
        pool1_min = - layers.max_pool2d(-inputs, [ pool1_size, pool1_size ], stride = 1 )
        pool11_min = layers.conv2d( pool1_min, unit_proporsion[3] - num_of_pool1c1_max, [ 1, 1 ] )
    with tf.variable_scope("pool2"):
        pool2_max = layers.max_pool2d( inputs, [ pool2_size, pool2_size ], stride = 1 )
        num_of_pool2c1_max = np.round(unit_proporsion[4]*0.5).astype(int)
        pool21_max = layers.conv2d( pool2_max, num_of_pool2c1_max, [ 1, 1 ] )
        pool2_min = - layers.max_pool2d(-inputs, [ pool2_size, pool2_size ], stride = 1 )
        pool21_min = layers.conv2d( pool2_min, unit_proporsion[4] - num_of_pool2c1_max, [ 1, 1 ] )


    # record model structure    
    perception_structure += "{}/conv_MxM: convMM_size={}, num={}\n".format(name, convMM_size, unit_proporsion[0])
    perception_structure += "{}/conv_C1: num_of_c11_1={}; conv1_size={}, num={} \n".format(name, num_of_c11_1, conv1_size, unit_proporsion[1])
    perception_structure += "{}/conv_C2: num_of_c11_2={}; conv2_size={}, num={} \n".format(name, num_of_c11_2, conv2_size, unit_proporsion[2])
    perception_structure += "{}/pool1: pool1_max_size={}, num_of_pool1c1_max={}; pool1_min_size={}, num_of_pool1c1_min={} \n"\
                            .format(name, pool1_size, num_of_pool1c1_max, pool1_size, unit_proporsion[3] - num_of_pool1c1_max)
    perception_structure += "{}/pool2: pool2_max_size={}, num_of_pool2c1_max={}; pool2_min_size={}, num_of_pool2c1_min={} \n"\
                            .format(name, pool2_size, num_of_pool2c1_max, pool2_size, unit_proporsion[4] - num_of_pool2c1_max)    
        
    return tf.concat([convMM, conv1, conv2, pool11_max, pool11_min, pool21_max, pool21_min], 3), perception_structure


def perception_layer_topologyV1_template( inputs, output_size, convMM_size, convMM_num, 
                                         num_of_c11_1, conv1_size, conv1_num, 
                                         num_of_c11_2, conv2_size, conv2_num, 
                                         pool1_size, num_of_pool1c1_max, num_of_pool1c1_min,
                                         pool2_size, num_of_pool2c1_max, num_of_pool2c1_min,
                                         name=""):
    """
    5 unit in the layer
    """  
    # Architecture
    perception_structure = ""
        
    # build graph    
    with tf.variable_scope("conv_MxM"):
        convMM = layers.conv2d( inputs, convMM_num, [ convMM_size, convMM_size ] )
    with tf.variable_scope("conv_C1"):
        conv1_11 = layers.conv2d( inputs, num_of_c11_1, [ 1, 1 ] )
        conv1 = layers.conv2d( conv1_11, conv1_num, [ conv1_size, 1 ] )
        conv1 = layers.conv2d( conv1, conv1_num, [ 1, conv1_size ] )
    with tf.variable_scope("conv_C2"):
        conv2_11 = layers.conv2d( inputs, num_of_c11_2, [ 1, 1 ] )
        conv2 = layers.conv2d( conv2_11, conv2_num, [ conv2_size, 1 ] )
        conv2 = layers.conv2d( conv2, conv2_num, [ 1, conv2_size ] )
    with tf.variable_scope("pool1"):
        pool1_max = layers.max_pool2d( inputs, [ pool1_size, pool1_size ], stride = 1 )
        pool11_max = layers.conv2d( pool1_max, num_of_pool1c1_max, [ 1, 1 ] )
        pool1_min = - layers.max_pool2d(-inputs, [ pool1_size, pool1_size ], stride = 1 )
        pool11_min = layers.conv2d( pool1_min, num_of_pool1c1_min, [ 1, 1 ] )
    with tf.variable_scope("pool2"):
        pool2_max = layers.max_pool2d( inputs, [ pool2_size, pool2_size ], stride = 1 )
        pool21_max = layers.conv2d( pool2_max, num_of_pool2c1_max, [ 1, 1 ] )
        pool2_min = - layers.max_pool2d(-inputs, [ pool2_size, pool2_size ], stride = 1 )
        pool21_min = layers.conv2d( pool2_min, num_of_pool2c1_min, [ 1, 1 ] )


    # record model structure    
    perception_structure += "{}/conv_MxM: convMM_size={}, num={}\n".format(name, convMM_size, convMM_num)
    perception_structure += "{}/conv_C1: num_of_c11_1={}; conv1_size={}, num={} \n".format(name, num_of_c11_1, conv1_size, conv1_num)
    perception_structure += "{}/conv_C2: num_of_c11_2={}; conv2_size={}, num={} \n".format(name, num_of_c11_2, conv2_size, conv2_num)
    perception_structure += "{}/pool1: pool1_max_size={}, num_of_pool1c1_max={}; pool1_min_size={}, num_of_pool1c1_min={} \n"\
                            .format(name, pool1_size, num_of_pool1c1_max, pool1_size, num_of_pool1c1_min)
    perception_structure += "{}/pool2: pool2_max_size={}, num_of_pool2c1_max={}; pool2_min_size={}, num_of_pool2c1_min={} \n"\
                            .format(name, pool2_size, num_of_pool2c1_max, pool2_size, num_of_pool2c1_min)    
        
    return tf.concat([convMM, conv1, conv2, pool11_max, pool11_min, pool21_max, pool21_min], 3), perception_structure



def SeismicPatchNet_topologyV1p2(inputs, 
                                 convMM_range = [1,2],
                                 convCX_range = [3,4,5,6,7],
                                 pool_range = [2,3,4,5],
                                 random_perception = 0.7,
                                 output_size_conv0 = 64,
                                 output_size_conv2 = 192,
                                 output_size_perception_3=256,
                                 output_size_perception_4=448,
                                 output_size_perception_5=512,
                              dropout_keep_prob=0.6,
                              num_classes=64,
                              is_training=True,
                              restore_logits = None,
                              scope=''):
    '''
    SeismicPatchNet - topologyV1p2
    '''
    def _generate_units_scale(convMM_range, convCX_range, pool_range):
        convMM_sample = random.sample(convMM_range, 1)[0]  
        convCX_sample = random.sample(list(itertools.combinations(convCX_range,2)), 1)[0] 
        pool_sample = random.sample(list(itertools.combinations(pool_range,2)), 1)[0]
        return convMM_sample, convCX_sample, pool_sample
    
    
    model_speci = "Name: SeismicPatchNet - topologyV1p2\n"
    model_speci += "Probability to random the type of perception layers: {}\n".format(random_perception)
        
    end_points = {}
    with tf.name_scope( scope, "SeismicPatchNet", [inputs] ):
        with ops.arg_scope( [ layers.max_pool2d ], padding = 'SAME' ):
            # fixed conv layers      
            end_points['conv0'] = layers.conv2d( inputs, output_size_conv0, [ 7, 7 ], stride = 2, scope = 'conv0' )
            end_points['pool0'] = layers.max_pool2d(end_points['conv0'], [3, 3], stride=2, scope='pool0')
            model_speci += "conv0 output size: {}\n".format(output_size_conv0)
          
            end_points['conv1'] = layers.conv2d( end_points['pool0'], output_size_conv0, [ 1, 1 ], scope = 'conv1_a' )
            model_speci += "conv1 output size: {}\n".format(output_size_conv0)
            
            end_points['conv2'] = layers.conv2d( end_points['conv1'], output_size_conv2, [ 3, 3 ], scope = 'conv1_b' )
            end_points['pool1'] = layers.max_pool2d(end_points['conv2'], [ 3, 3 ], stride=2, scope='pool1')
            model_speci += "conv2 output size: {}\n".format(output_size_conv2)
            
            # perception layers
            # random perception layers
            if np.random.random() <= random_perception:
                model_speci += "Random the type of perception layers: Yes \n"
                with tf.variable_scope("perception_3"):
                    convMM_sample, convCX_sample, pool_sample = _generate_units_scale(convMM_range, convCX_range, pool_range)
                    end_points['perception_3'], perception_structure = perception_layer_topologyV1( end_points['pool1'], output_size_perception_3, 
                              convMM_sample, convCX_sample[0], convCX_sample[1], pool_sample[0], pool_sample[1], name="perception_3")
                model_speci += "perception_3 output size: {}\n".format(output_size_perception_3) + perception_structure
                
                with tf.variable_scope("perception_4"):
                    convMM_sample, convCX_sample, pool_sample = _generate_units_scale(convMM_range, convCX_range, pool_range)
                    end_points['perception_4'], perception_structure = perception_layer_topologyV1( end_points['perception_3'], output_size_perception_4, 
                              convMM_sample, convCX_sample[0], convCX_sample[1], pool_sample[0], pool_sample[1], name="perception_4")
                model_speci += "perception_4 output size: {}\n".format(output_size_perception_4) + perception_structure
                
                end_points['pool2'] = layers.max_pool2d(end_points['perception_4'], [ 3, 3 ], stride=2, scope='pool2')
                
                with tf.variable_scope("perception_5"):
                    convMM_sample, convCX_sample, pool_sample = _generate_units_scale(convMM_range, convCX_range, pool_range)
                    end_points['perception_5'], perception_structure = perception_layer_topologyV1( end_points['pool2'], output_size_perception_5,
                              convMM_sample, convCX_sample[0], convCX_sample[1], pool_sample[0], pool_sample[1], name="perception_5")
                model_speci += "perception_5 output size: {}\n".format(output_size_perception_5) + perception_structure
                
            # same type perception layer
            else:
                model_speci += "Random the type of perception layers: No \n"
                convMM_sample, convCX_sample, pool_sample = _generate_units_scale(convMM_range, convCX_range, pool_range)
                
                with tf.variable_scope("perception_3"):
                    end_points['perception_3'], perception_structure = perception_layer_topologyV1( end_points['pool1'], output_size_perception_3,
                              convMM_sample, convCX_sample[0], convCX_sample[1], pool_sample[0], pool_sample[1], name="perception_3")
                model_speci += "perception_3 output size: {}\n".format(output_size_perception_3) + perception_structure
                
                with tf.variable_scope("perception_4"):
                    end_points['perception_4'], perception_structure = perception_layer_topologyV1( end_points['perception_3'], output_size_perception_4, 
                              convMM_sample, convCX_sample[0], convCX_sample[1], pool_sample[0], pool_sample[1], name="perception_4")
                model_speci += "perception_4 output size: {}\n".format(output_size_perception_4) + perception_structure
                
                end_points['pool2'] = layers.max_pool2d(end_points['perception_4'], [ 3, 3 ], stride=2, scope='pool2')
                
                with tf.variable_scope("perception_5"):
                    end_points['perception_5'], perception_structure = perception_layer_topologyV1( end_points['pool2'], output_size_perception_5, 
                              convMM_sample, convCX_sample[0], convCX_sample[1], pool_sample[0], pool_sample[1], name="perception_5")
                model_speci += "perception_5 output size: {}\n".format(output_size_perception_5) + perception_structure                


            end_points['pool3'] = layers.avg_pool2d(end_points['perception_5'], [ 7, 7 ], stride = 1, scope='pool3')

            end_points['reshape'] = tf.reshape( end_points['pool3'], [-1, output_size_perception_5] )
            
            end_points['dropout'] = layers.dropout( end_points['reshape'], dropout_keep_prob, is_training = is_training )

            end_points['logits_modelbuild'] = layers.fully_connected( end_points['dropout'], num_classes, activation_fn=None, scope='logits_modelbuild')

#            end_points['predictions'] = tf.nn.softmax(end_points['logits'], name='predictions')

    return end_points['logits_modelbuild'], end_points, model_speci


def SeismicPatchNet_topologyV1_restore(model_log_path, inputs, 
                              dropout_keep_prob=0.5,
                              num_classes=64,
                              is_training=False,
                              scope=''):
    '''
    SeismicPatchNet - topologyV1 template
    '''    
    model_speci = "Name: SeismicPatchNet - topologyV1 restored\n"

    
    structure_dict = {"conv0 output size": None, "conv1 output size": None, "conv2 output size": None, 
                      
                      "perception_3 output size": None, "perception_3/conv_MxM": None, "perception_3/conv_C1": None,
                      "perception_3/conv_C2": None, "perception_3/pool1": None, "perception_3/pool2": None,
                      
                      "perception_4 output size": None, "perception_4/conv_MxM": None, "perception_4/conv_C1": None,
                      "perception_4/conv_C2": None, "perception_4/pool1": None, "perception_4/pool2": None,
                      
                      "perception_5 output size": None, "perception_5/conv_MxM": None, "perception_5/conv_C1": None,
                      "perception_5/conv_C2": None, "perception_5/pool1": None, "perception_5/pool2": None
                      }
    
    structure_dict = Log_Restore_SeismicPatchNet_topologyV1(model_log_path, structure_dict)
    
    output_size_conv0 = structure_dict["conv0 output size"]
    output_size_conv2 = structure_dict["conv2 output size"]
    
    output_size_perception_3 = structure_dict["perception_3 output size"]
    perception_3_convMM = structure_dict["perception_3/conv_MxM"]
    perception_3_convC1 = structure_dict["perception_3/conv_C1"]
    perception_3_convC2 = structure_dict["perception_3/conv_C2"]
    perception_3_pool1 = structure_dict["perception_3/pool1"]
    perception_3_pool2 = structure_dict["perception_3/pool2"]
    
    output_size_perception_4 = structure_dict["perception_4 output size"]
    perception_4_convMM = structure_dict["perception_4/conv_MxM"]
    perception_4_convC1 = structure_dict["perception_4/conv_C1"]
    perception_4_convC2 = structure_dict["perception_4/conv_C2"]
    perception_4_pool1 = structure_dict["perception_4/pool1"]
    perception_4_pool2 = structure_dict["perception_4/pool2"]
    
    output_size_perception_5 = structure_dict["perception_5 output size"]
    perception_5_convMM = structure_dict["perception_5/conv_MxM"]
    perception_5_convC1 = structure_dict["perception_5/conv_C1"]
    perception_5_convC2 = structure_dict["perception_5/conv_C2"]
    perception_5_pool1 = structure_dict["perception_5/pool1"]
    perception_5_pool2 = structure_dict["perception_5/pool2"]
    
          
    end_points = {}
    with tf.name_scope( scope, "SeismicPatchNet", [inputs] ):
        with ops.arg_scope( [ layers.max_pool2d ], padding = 'SAME' ):
            # fixed conv layers      
            end_points['conv0'] = layers.conv2d( inputs, output_size_conv0, [ 7, 7 ], stride = 2, scope = 'conv0' )
            end_points['pool0'] = layers.max_pool2d(end_points['conv0'], [3, 3], stride=2, scope='pool0')
            model_speci += "conv0 output size: {}\n".format(output_size_conv0)
          
            end_points['conv1'] = layers.conv2d( end_points['pool0'], output_size_conv0, [ 1, 1 ], scope = 'conv1_a' )
            model_speci += "conv1 output size: {}\n".format(output_size_conv0)
            
            end_points['conv2'] = layers.conv2d( end_points['conv1'], output_size_conv2, [ 3, 3 ], scope = 'conv1_b' )
            end_points['pool1'] = layers.max_pool2d(end_points['conv2'], [ 3, 3 ], stride=2, scope='pool1')
            model_speci += "conv2 output size: {}\n".format(output_size_conv2)
            
            # perception layers
            model_speci += "Random the type of perception layers: Yes \n"
            with tf.variable_scope("perception_3"):
                end_points['perception_3'], perception_structure = perception_layer_topologyV1_template( end_points['pool1'], output_size_perception_3, 
                          perception_3_convMM[0], perception_3_convMM[1],
                          perception_3_convC1[0], perception_3_convC1[1], perception_3_convC1[2],
                          perception_3_convC2[0], perception_3_convC2[1], perception_3_convC2[2],
                          perception_3_pool1[0], perception_3_pool1[1], perception_3_pool1[3],
                          perception_3_pool2[0], perception_3_pool2[1], perception_3_pool2[3],
                          name="perception_3")
            model_speci += "perception_3 output size: {}\n".format(output_size_perception_3) + perception_structure
            
            with tf.variable_scope("perception_4"):
                end_points['perception_4'], perception_structure = perception_layer_topologyV1_template( end_points['perception_3'], output_size_perception_4,
                          perception_4_convMM[0], perception_4_convMM[1],
                          perception_4_convC1[0], perception_4_convC1[1], perception_4_convC1[2],
                          perception_4_convC2[0], perception_4_convC2[1], perception_4_convC2[2],
                          perception_4_pool1[0], perception_4_pool1[1], perception_4_pool1[3],
                          perception_4_pool2[0], perception_4_pool2[1], perception_4_pool2[3],
                          name="perception_4")
            model_speci += "perception_4 output size: {}\n".format(output_size_perception_4) + perception_structure
            
            end_points['pool2'] = layers.max_pool2d(end_points['perception_4'], [ 3, 3 ], stride=2, scope='pool2')
            
            with tf.variable_scope("perception_5"):
                end_points['perception_5'], perception_structure = perception_layer_topologyV1_template( end_points['pool2'], output_size_perception_5,
                          perception_5_convMM[0], perception_5_convMM[1],
                          perception_5_convC1[0], perception_5_convC1[1], perception_5_convC1[2],
                          perception_5_convC2[0], perception_5_convC2[1], perception_5_convC2[2],
                          perception_5_pool1[0], perception_5_pool1[1], perception_5_pool1[3],
                          perception_5_pool2[0], perception_5_pool2[1], perception_5_pool2[3],
                          name="perception_5")
            model_speci += "perception_5 output size: {}\n".format(output_size_perception_5) + perception_structure        


            end_points['pool3'] = layers.avg_pool2d(end_points['perception_5'], [ 7, 7 ], stride = 1, scope='pool3')

            end_points['reshape'] = tf.reshape( end_points['pool3'], [-1, output_size_perception_5] )
            
            end_points['dropout'] = layers.dropout( end_points['reshape'], dropout_keep_prob, is_training = is_training )

            end_points['logits_modelbuild'] = layers.fully_connected( end_points['dropout'], num_classes, activation_fn=None, scope='logits_modelbuild')

#            end_points['predictions'] = tf.nn.softmax(end_points['logits'], name='predictions')

    return end_points['logits_modelbuild'], end_points, model_speci


