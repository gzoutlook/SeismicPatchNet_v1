# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 15:46:21 2019  @author: Geng

Random search SeismicPatchNet_topology series

Note:
    searching results are saved under predifined train_data_path

"""

import sys
import os
path = '.\libs'
sys.path.append(path)
import helper


import time
import pickle
from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import numpy as np
import pandas as pd
import random

import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer

import tensorflow as tf
    
    
# Defining the model inputs
def images_input(img_shape):
    return tf.compat.v1.placeholder(tf.float32, (None,) + img_shape, name="input_images")

def target_input(num_classes):
    target_input = tf.compat.v1.placeholder(tf.int32, (None, num_classes), name="input_images_label")
    return target_input

# define a function for the dropout layer keep probability
def keep_prob_input():
    return tf.compat.v1.placeholder(tf.float32, name="keep_prob")


#Define a helper function for kicking off the training process
def train(session, model_optimizer, keep_probability, in_feature_batch, target_batch):
    session.run(model_optimizer, feed_dict={input_images: in_feature_batch, input_images_label: target_batch, keep_prob: keep_probability})


# Defining a helper funcitno for print information about the model accuracy and it's validation accuracy as well
def print_batch_stats(session, input_feature_batch, target_label_batch, model_cost, model_accuracy):
    batch_loss = session.run(model_cost,
                                  feed_dict={input_images: input_feature_batch, input_images_label: target_label_batch,
                                             keep_prob: 1.0})
    batch_accuracy = session.run(model_accuracy, feed_dict={input_images: input_feature_batch,
                                                                 input_images_label: target_label_batch,
                                                                 keep_prob: 1.0})
    info_loss = "Batch loss: %0.6f" % (batch_loss)
    print(info_loss, end="; ")
    info_accuracy = "Batch accuracy: %0.6f" % (batch_accuracy)
    print(info_accuracy)
    return info_loss, info_accuracy

def validation_stats(session, validation_feature, validation_label, batch_size, model_cost_test, model_accuracy_test):
    '''metrics of validation dataset '''
    test_acc = 0.
    test_loss =0.
    test_count = 0
    # Training cycle
    num_samples = len(validation_feature)
    start_index = np.arange(0, num_samples, batch_size)
    end_index = np.arange(batch_size, num_samples, batch_size)
    start_time = time.time()
    for i in np.arange(len(end_index)):
        # Test by batch
        start, end = start_index[i], end_index[i]
        feature_batch, label_batch = validation_feature[start:end].astype(np.float32), validation_label[start:end].astype(np.float32)
        
        batch_loss = session.run(model_cost_test,
                                  feed_dict={input_images: feature_batch, input_images_label: label_batch,
                                             keep_prob: 1.0})
        batch_accuracy = session.run(model_accuracy_test, feed_dict={input_images: feature_batch,
                                                                 input_images_label: label_batch,
                                                                 keep_prob: 1.0})
        test_loss += batch_loss  
        test_acc += batch_accuracy
        test_count += 1
    end_time = time.time()
    test_acc /= test_count
    test_loss /= test_count
    sample_per_sec = batch_size * test_count / (end_time - start_time)
    return test_loss, test_acc, sample_per_sec
        
        
# Splitting the dataset features and labels to batches
def batch_split_features_labels(input_features, target_labels, batch_size=128):
    batch_data_sets = []
    batch_target_labels = [] 
    for start in range(0, len(input_features), batch_size):
        end = min(start + batch_size, len(input_features))
        batch_data_sets.append(input_features[start:end])
        batch_target_labels.append(target_labels[start:end])
    return zip(batch_data_sets, batch_target_labels)


        
def _compute_gradients(tensor, var_list):
    # Get gradients of all trainable variables
    grads = tf.gradients(tensor, var_list)
    return [grad if grad is not None else tf.zeros_like(var) 
            for var, grad in zip(var_list, grads)]

# Defining a helper function for testing the trained model
def test_classification_model(input_test_features, target_test_labels, model_path, cmapFile, Plot=True, num_samples = 6):
    
    tf.reset_default_graph()
    loaded_graph = tf.Graph()  
      
    with tf.Session(graph=loaded_graph) as sess:   
#        from tensorflow.python.tools import inspect_checkpoint as chkp
#        chkp.print_tensors_in_checkpoint_file(model_path, tensor_name='', all_tensors=True)
      
        # print ops in the graph
        for op in loaded_graph.get_operations():
            print(op.name)
            
        # loading the trained model
        model = tf.train.import_meta_graph(model_path + '.meta')
        model.restore(sess, model_path)
        # Getting some input and output Tensors from loaded model
        model_input_values = loaded_graph.get_tensor_by_name('input_images:0')
        model_target = loaded_graph.get_tensor_by_name('input_images_label:0')
        model_keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
        
        model_logits = loaded_graph.get_tensor_by_name('logits:0')
        
#        model_accuracy = loaded_graph.get_tensor_by_name('accuracy:0')
        # Evaluation op: Accuracy of the model
        correct_prediction = tf.equal(tf.argmax(model_logits, 1), tf.argmax(model_target, 1))
        model_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Testing the trained model on the test set batches
        test_batch_accuracy_total = 0
        test_batch_count = 0

        for input_test_feature_batch, input_test_label_batch in batch_split_features_labels(input_test_features,
                                                                                            target_test_labels,
                                                                                            batch_size=128):
            test_batch_accuracy_total += sess.run(
                model_accuracy,
                feed_dict={model_input_values: input_test_feature_batch, model_target: input_test_label_batch,
                           model_keep_prob: 1.0})
            test_batch_count += 1

        print('Test set accuracy: {}\n'.format(test_batch_accuracy_total / test_batch_count))
        
        # print some random images and their corresponding predictions from the test set results
        if Plot:
            num_samples = num_samples
            top_n_predictions = 2
            random_input_test_features, random_test_target_labels = tuple(
                zip(*random.sample(list(zip(input_test_features, target_test_labels)), num_samples)))
    
            random_test_predictions = sess.run(
                tf.nn.top_k(tf.nn.softmax(model_logits), top_n_predictions),
                feed_dict={model_input_values: random_input_test_features, model_target: random_test_target_labels,
                           model_keep_prob: 1.0})
            # visualization data range
            minLUT, maxLUT = helper.Statistics_Data_Scale(random_input_test_features, scale = 0.99, Zero_Centralization=True)
            visualize_samples_predictions(random_input_test_features, random_test_target_labels, random_test_predictions, 
                                          cmapFile, feature_scale=(minLUT, maxLUT), window_title="Tests")

    return random_input_test_features

    
# A helper function to visualize some samples and their corresponding predictions
def visualize_samples_predictions(input_features, target_labels, samples_predictions, cmapFile, feature_scale=(-1,1), window_title=""):
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.gridspec as gridspec
    num_classes = 2
    class_names = ['Negative', 'BSR']
    
    # Color map
    cmapArray = pd.read_csv(cmapFile,header=None)
    myCmap = array2cmap(cmapArray)

    label_binarizer = LabelBinarizer()
    label_binarizer.fit(range(num_classes))
    label_inds = label_binarizer.inverse_transform(np.array(target_labels))

    fig, axies = subPlotDefi(windowTitle=window_title, figSizeCM=(18, 25),figShape=(len(input_features), 3), 
                             subAdjust=(0.05, 0.85, 0.05, 0.95), subSpace=(0.5,0.5), axLabelSize=12, axTickSize=12)
#    fig.tight_layout()
    grid_spec = gridspec.GridSpec(len(input_features), 3)
    
    fig.suptitle('Predictions', fontsize=20, y=1.1)

    num_predictions = 2
    margin = 0.05
    ind = np.arange(num_predictions)
    width = (1. - 2. * margin) / num_predictions
#    image_list = []
    for image_ind, (feature, label_ind, prediction_indicies, prediction_values) in enumerate(
            zip(input_features, label_inds, samples_predictions.indices, samples_predictions.values)):
        prediction_names = [class_names[pred_i] for pred_i in prediction_indicies]
        correct_name = class_names[label_ind]
         
        # Density image
        image = feature.reshape((feature.shape[1],feature.shape[1]))
        im = axies[image_ind][0].imshow(image, cmap=myCmap, vmax=max(feature_scale), vmin=min(feature_scale))
        axies[image_ind][0].set_title(correct_name)
        axies[image_ind][0].set_axis_off()
        # colorbar
        # Create divider for existing axes instance
        divider = make_axes_locatable(axies[image_ind][0])
        # Append axes to the right of ax3, with 20% width of ax3
        cax = divider.append_axes("right", size="20%", pad=0.05)
        # Create colorbar in the appended axes. Tick locations can be set with the kwarg `ticks`, and the format of the ticklabels with kwarg `format`
        cbar = plt.colorbar(im, cax=cax, format="%.2f")
        
        # Wiggle trace
        num_of_trace_in_window = 10
        skip = int(image.shape[1]/num_of_trace_in_window)
        perc = 99.0
        gain = 30.0
        rgb = (0, 0, 0)
        alpha = 1
        lw = 0.5    
        rgba = list(rgb) + [alpha]
        sc = 0.4 # np.percentile(image, perc)  # Normalization factor
        wigdata = image[:, ::skip]
        xpos = np.arange(image.shape[1])[::skip]

        for x, trace in zip(xpos, wigdata.T):
            # Compute high resolution trace.
            amp = gain * trace / sc + x
            t = np.arange(wigdata.shape[0])
            hypertime = np.linspace(t[0], t[-1], 10 * t.size)
            hyperamp = np.interp(hypertime, t, amp)

            # Plot the line, then the fill.
            axies[image_ind][1].set_title(correct_name)
            axies[image_ind][1].plot(hyperamp, hypertime, 'k', lw=lw)
            axies[image_ind][1].fill_betweenx(hypertime, hyperamp, x,
                             where=hyperamp > x,
                             facecolor=rgba,
                             lw=0)
        axies[image_ind][1].invert_yaxis()
        axies[image_ind][1].set_axis_off()
        axies[image_ind][1].set_aspect(aspect=1)
        
        
        # Prediction bar
        axies[image_ind][2].barh(ind + margin, prediction_values[::-1], width)
        axies[image_ind][2].yaxis.tick_right()
        axies[image_ind][2].set_yticks(ind + margin)
        axies[image_ind][2].set_yticklabels(prediction_names[::-1])
        axies[image_ind][2].set_xticks([0, 0.5, 1.0])


def Plot_Validation(Item_list, loss_max=None, smooth_pts=9, alpha=0.2, label_x='Iteration', label_y_left="Loss", label_y_right="Accuracy", fig_title="Validation score", window_title="Validation", fig_size=(22 ,16)):
    """ Plot charts of validation data
    Item_list: [ [ndarray_x1, ndarray_y1_left, ndarray_y1_right, label_item1],
                [ndarray_x2, ndarray_y2_left, ndarray_y2_right, label_item1],
                [ndarray_x3, ndarray_y3_left, ndarray_y3_right], label_item1,]
    """
    num_of_item = len(Item_list)
    # 颜色配置
    if num_of_item==1: num_of_color = 2
    else: num_of_color = num_of_item
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, num_of_color)]
    # smooth line style
    linewidth_smooth = 1.5
    dashes_smooth = (5, 5)
    # 图
    fig, axis = subPlotDefi(windowTitle=window_title, figSizeCM=fig_size)
    axis_right = axis.twinx()
    legend_list = []
    for i in range(num_of_item):
        x, y_left, y_right, label = Item_list[i][0], Item_list[i][1], Item_list[i][2], Item_list[i][3] 
        y_left_smooth = helper.Smooth(y_left, smooth_pts)
        y_right_smooth = helper.Smooth(y_right, smooth_pts)
        axis.plot(x, y_left, color=colors[i], linewidth=1, alpha=alpha)
        plot1 = axis.plot(x, y_left_smooth, color=colors[i], linewidth=linewidth_smooth, label="{}: loss".format(label))
        if num_of_item==1: colors[i] = colors[-1]
        axis_right.plot(x, y_right, color=colors[i], linewidth=1, alpha=alpha, linestyle='--')
        plot2 = axis_right.plot(x, y_right_smooth, color=colors[i], linewidth=linewidth_smooth, linestyle='--', dashes=dashes_smooth, label="{}: accuracy".format(label))
        legend_list.extend(plot1)
        legend_list.extend(plot2)
    # merge legends
    labs = [l.get_label() for l in legend_list]
    lgnd = axis.legend(legend_list, labs, loc='upper left', prop={'size':11})
    lgnd.get_frame().set_alpha(0.5) # 图例背景半透明
    
    axis.set_title("Validation scores")
    axis.set_xlabel(label_x)
    axis.set_ylabel(label_y_left)
    axis_right.set_ylabel(label_y_right)
    axis.set_ylim(bottom=0, top=loss_max)
    axis_right.set_ylim(bottom=0.5, top=1.1)
    
    return fig


def save_training(sess, temp_dir, model_path, validation_record, training_status=None):
#    if training_status is None:
#        save_path = saver.save(sess, model_path)
#    else:
#        save_path = saver.save(sess, model_path + "_{}".format(training_status))
        
    # save records
    record_name = "Validation records _{} {}.csv".format(training_status, os.path.split(model_path)[-1])
    record_path = os.path.join(temp_dir, os.path.abspath(os.path.join(model_path,os.path.pardir)), record_name)
    validation_record.to_csv(path_or_buf=record_path, index=False, index_label=False)  
    
    info_finish = "\n{} Training finished, model saved.".format(time.strftime("%Y-%m-%d %H:%M:%S"))
    log.write(info_finish)
    print(info_finish)       


if __name__ == "__main__":

#%% # ========================================================================
    #                          some settings
    # ========================================================================    
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"       # 1st GPU starts from 0
    
    """
    Train setting
    """
    # Operation settings
    # be careful that the controlling params are samples dependent. make sure the number of samples is large enough.
    num_epochs = 60                                 # maximum searching epoch
    batch_size = 64
    display_status_step = 10                        # how often (num_of_batches) to print training status
    require_improve_epoch_ratio = 3               # 0.7 batch_size=64, the number of epoch to interrupt the training without loss improvements
    minimum_relative_loss_improvement = 0.005       # 0.01, stop the training if the relative loss change is smaller
    
    save_tensor_summary = False                 # monitor and save the tensorboard summary
    save_checkpoint_epoch = 10                  # how often to save summary & checkpoint data to disk
    open_monitor = False                        # monitor plot

    # training hyperparameters
    class_Num = 2
    learning_rate = 1e-4        # 1e-4 (general), 1e-5 (VGG)
    keep_probability = 0.5
     
    # Note
    Note = "{} ramdom search SeismicPatchNet_v1p1".format(time.strftime("%Y/%m/%d-%H:%M:%S"))
    model_speci = ""

    """
    Load models & data
    """
    models_path = r".\libs\CNN_Models"
    sys.path.append(models_path)
  
    from SeismicPatchNet import SeismicPatchNet_topologyV1p2
    model_name = "SeismicPatchNet_topologyV1p2"
     
    print("\nLoading data for {} ...".format(model_name))
    train_data_path = r".\results\synthetic_data\Train tensor112 - Patches_samples _n1700s48 202005081500 Train.ds"  # for debuging here, remember to replace it
    test_data_path = r".\results\synthetic_data\Test tensor112 - Patches_samples _n300s48 202005081501 Test.ds"      # for debuging here, remember to replace it
    train_data_name = os.path.split(train_data_path)[1].split(".")[0]
    train_set, train_labels = pickle.load(open(train_data_path, mode='rb'))
    test_set, test_labels = pickle.load(open(test_data_path, mode='rb'))
    
    # half the data size
    train_set, train_labels = train_set[:int(len(train_set)/2)], train_labels[:int(len(train_set)/2)]
    test_set, test_labels = test_set[:int(len(test_set)/2)], test_labels[:int(len(test_set)/2)]
    
    print("Data for {} is loaded.".format(model_name))
    
    
    """
    Search settings
    """

    random_times = 1870         # 1 random_time: ~[0.77, 0.9, 1.39] min, 4 high end GPUs: 6250 models/day (distribute training)
    random_perception = 0.5     # probability to random the type of perception layers
    
    # scale size
    convMM_range = [1,2]
    convCX_range = [3,4,5,6,7]
    pool_range = [2,3,4,5]
    
    # layers size
    output_size_conv0_range = [32,256]
    output_size_conv2_range = [32,256]
    output_size_perception_3_range = [128,640]
    output_size_perception_4_range = [128,640]
    output_size_perception_5_range = [128,640]
    
    
#%% # ========================================================================
    #                          automatic search
    # ========================================================================
    
   
    print("\nRandom searching graphs ...\n")
    for m in tqdm(np.arange(random_times)):
        
        search_process_info = "\n\n Searching to {}/{} ...\n".format(m+1, random_times)
        print(search_process_info)
        
        # generate layers size
        output_size_conv0 = np.random.randint(output_size_conv0_range[0],output_size_conv0_range[1])
        output_size_conv2 = np.random.randint(output_size_conv2_range[0],output_size_conv2_range[1])
        output_size_perception_3 = np.random.randint(output_size_perception_3_range[0],output_size_perception_3_range[1])
        output_size_perception_4 = np.random.randint(output_size_perception_4_range[0],output_size_perception_4_range[1])
        output_size_perception_5 = np.random.randint(output_size_perception_5_range[0],output_size_perception_5_range[1])
        
        # Remove all the previous inputs, weights, biases form the previous runs
        tf.compat.v1.reset_default_graph()
        # Defining the input placeholders to the convolution neural network
        input_images_label = target_input(2)
        input_images = images_input((112, 112, 1))
        keep_prob = keep_prob_input()
       
        # SeismicPatchNet
        logits, _, model_speci = SeismicPatchNet_topologyV1p2(input_images, 
                                                              convMM_range = convMM_range,
                                                              convCX_range = convCX_range,
                                                              random_perception = random_perception,
                                                              pool_range = pool_range,
                                                              output_size_conv0 = output_size_conv0,
                                                              output_size_conv2 = output_size_conv2,
                                                              output_size_perception_3=output_size_perception_3,
                                                              output_size_perception_4=output_size_perception_4,
                                                              output_size_perception_5=output_size_perception_5,
                                                        dropout_keep_prob=keep_probability, num_classes=class_Num)
     
        ''' defining operations '''
        
        # Name logits Tensor, so that is can be loaded from disk after training
        logits_values = tf.identity(logits, name='logits') 
    #    logits_test_values = tf.identity(logits_test, name='logits_test') 
        
        # save model path
        temp_dir = os.path.abspath(os.path.join(train_data_path,os.path.pardir))
        model_id = "Model_{0}_{1}".format(model_name, time.strftime("%Y%m%d%H%M%S"))
        model_path = os.path.join(temp_dir, model_id, model_id)
        ckpt_and_summary_dir = os.path.join(temp_dir, model_id, "Checkpoint and Summary")
        # log file path
        log_file = os.path.join(temp_dir, os.path.abspath(os.path.join(model_path,os.path.pardir)), "Log_training.txt")
        
        # Defining the model loss
        with tf.name_scope("cross_entropy"):
            model_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_values, labels=input_images_label))

        # Train op 
        var_list = [v for v in  tf.compat.v1.trainable_variables()] # List of trainable variables of the layers (we want to train)  #  if v.name.split('/')[0] in train_layers
        with tf.name_scope("train"):
            # Define an optimizer
            model_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)                 # VGGNet - lr:1e-5
            
            # Compute gradients
            gradients = tf.gradients(model_cost, var_list, unconnected_gradients='zero') # Get gradients of all trainable variables 
            grads_and_vars = list(zip(gradients, var_list))
            grads_and_vars = [(tf.clip_by_value(g, -10., 10.), v) for g, v in grads_and_vars]   # prevent weights explosion
    
            # Train operation by applying gradients
            train_op = model_optimizer.apply_gradients(grads_and_vars=grads_and_vars)
    
        # Add gradients to summary
        for gradient, var in grads_and_vars:
            tf.compat.v1.summary.histogram(var.name.replace(':','_') + '/gradient', gradient)
        # Add the variables we train to the summary
        for var in var_list:
            tf.compat.v1.summary.histogram(var.name.replace(':','_'), var)
        # Add logits to summary
        tf.compat.v1.summary.histogram('logit_values', logits_values)
        # Add the loss to summary
        tf.compat.v1.summary.scalar('cross_entropy', model_cost)
    
        # Evaluation op: Accuracy of the model
        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(tf.argmax(logits_values, 1), tf.argmax(input_images_label, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
     
        # Add the accuracy to the summary
        tf.compat.v1.summary.scalar('accuracy', accuracy)
        
        # Merge all summaries together
        merged_summary = tf.compat.v1.summary.merge_all()
 
        
        ''' Formal training on all the data '''
        # monitor
        if open_monitor:
            figure = plt.figure()
            ax = figure.add_subplot(1, 1, 1)
            figure.canvas.set_window_title('Training Monitor')
            ax.set_title('loss evo.: {}'.format(model_name),fontsize=11)
            plt.xticks(fontsize=10)
            plt.xlabel('steps',fontsize=11)
            plt.ylabel('metrics', fontsize=11)
            plt.ion()
            plt.show()
    
        
        # start training
        start_timing = time.time()
        print('\n{} Start training the network...'.format(time.strftime("%Y-%m-%d %H:%M:%S")))
        totoal_paras = np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()])
        print("Total number of trainable paras: %.2f M." % (totoal_paras/1e6))
        print("{} Tensorboard is activated.".format(time.strftime("%H:%M:%S")))
        config = tf.compat.v1.ConfigProto()
    #    config.gpu_options.allocator_type ='BFC'
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        config.gpu_options.allow_growth = True
        with tf.compat.v1.Session(config=config) as sess:
            # Initialize all variables
            sess.run(tf.compat.v1.global_variables_initializer())
            # Initialize an saver for store model checkpoints
#                saver = tf.compat.v1.train.Saver(max_to_keep=int(num_epochs/save_checkpoint_epoch))
            if save_tensor_summary:
                # Initialize the FileWriter
                writer = tf.summary.FileWriter(ckpt_and_summary_dir)
                # Add the model graph to TensorBoard
                writer.add_graph(sess.graph)
        
            # Training cycle
            train_count = len(train_set)
            start_index = np.arange(0, train_count-batch_size, batch_size)
            end_index = np.arange(batch_size, train_count, batch_size)
            # create log file if not exist
            if not os.path.isdir(os.path.abspath(os.path.join(log_file, ".."))):
                os.mkdir(os.path.abspath(os.path.join(log_file, "..")))
            # create metrics data records to be saved
            validation_record = pd.DataFrame()
            with open(log_file,"w") as log:   # recording
    
                # save training setting
                info_model = "# Model info:\n  Model name: {0}\n  Total number of trainable paras: {1} M.".format(model_id, totoal_paras/1e6)
                info_setting = "# Data name:\n {0} \n# Operation settings:\n  Number of epochs: {1}\n  Batch size: {2}\n  Ratio of epoch to improve (for early stop): {3}\n  Minimum loss change (relative, for early stop): {4}\n  Summary/status step: {5}\n  Save the summary: {6}\n  Checkpoint step: {7}".format(
                    train_data_name, num_epochs, batch_size, require_improve_epoch_ratio, minimum_relative_loss_improvement, 
                    display_status_step, save_tensor_summary, save_checkpoint_epoch)
                info_hyper = "# Hyperparameters:\n  Number of class: {0}\n  Optimizer: {1}\n  Initial learning rate: {2}\n  Keep probability: {3}".format(
                    class_Num, model_optimizer.get_name(), learning_rate, keep_probability)
                
                log.write(Note+'\n')
                log.write(search_process_info+'\n')
                log.write(model_speci+'\n\n')
                log.write(info_model+'\n')
                log.write(info_setting+'\n')
                log.write(info_hyper+'\n')
                
                #------------------------ traing process -------------------------#
                
                # for early stopping 
                best_loss_avg = 1e6 
                stop_training = False
                require_improve_step = int(require_improve_epoch_ratio*len(start_index)/display_status_step)
                last_improved_step = 0
                tiny_improved_step = 0
#                save_sess = None    # forget it. 
                for i in tqdm(np.arange(num_epochs), position=1): # just for using tqdm showing rate of progress
                    info = "\nEpoch number: {}, batch training ... ({}):".format(i+1, time.strftime("%Y-%m-%d %H:%M:%S"))
                    print(info)
                    
                    # debug
                    print(model_speci)
                    
                    log.write(info+'\n')
                    for j in np.arange(len(end_index)):
                        #------------------ Train by batch -------------------#
                        start, end = start_index[j], end_index[j]
  
                        train(sess, train_op, keep_probability, train_set[start:end].astype(np.float32), train_labels[start:end].astype(np.float32))

                        #-------------------- Validation ---------------------#
                        
                        # print & record status ...
                        if end_index[j] % (batch_size*display_status_step) == 0:
                            # Metrics on single batch of training data, in fact, Done by merged_summary
    #                        batch_loss, batch_accuracy = print_batch_stats(sess, train_set[start:end].astype(np.float32), train_labels[start:end].astype(np.float32), model_cost, accuracy)
                           
                            # Generate summary with the current batch of data and write to file
                            if save_tensor_summary:
                                print("{} Merge summary and saving ...".format(time.strftime("%H:%M:%S")))
                                temp_summary = sess.run(merged_summary, feed_dict={input_images: train_set[start:end].astype(np.float32),
                                                                        input_images_label: train_labels[start:end].astype(np.float32),
                                                                        keep_prob: 1.})
                                writer.add_summary(temp_summary, i*len(end_index) + (j+1))
                            
                            # Metrics on total Validation data
                            valid_loss, valid_accuracy, samples_per_sec = validation_stats(sess, test_set, test_labels, batch_size, model_cost, accuracy)
                            info_loss = "Val_loss: %0.5f" % (valid_loss)
                            info_accuracy = "Val_accuracy: %0.5f" % (valid_accuracy)
                            info_speed = "Speed: %0.2f sam./sec." % (samples_per_sec)
                            print("{0}; {1}; {2}".format(info_loss, info_accuracy, info_speed))
                            log.write("{0}; {1}; {2} \n".format(info_loss, info_accuracy, info_speed))  
                            # save record
                            record = {'Epoch': [i+1], 'Batch num': [j+1], 'Iter_num': [i*len(end_index) + (j+1)], 
                                      'Val_loss': [valid_loss], 'Val_accuracy': [valid_accuracy]}
                            record_df = pd.DataFrame(data=record)
                            validation_record = validation_record.append(record_df, ignore_index=True)
                            
                            # monitor: plot 
                            if open_monitor:
                                try:
                                    ax.lines = []
                                except Exception:
                                    pass
                                smooth_pts = 17
                                if len(validation_record['Iter_num'].values)<smooth_pts:
                                    pass
                                else:
                                    x_raw = validation_record['Iter_num'].values
                                    y_raw = validation_record['Val_loss'].values
                                    y_smooth = helper.Smooth(y_raw, smooth_pts)
                                    plt.plot(x_raw, y_raw, 'b', linewidth=2, alpha=0.3)
                                    plt.plot(x_raw, y_smooth, 'r-', lw=2)
                                    plt.pause(0.1)  # for show timely
                                    
                            # early stopping            
                            if stop_training is False:
                                epoch_data = validation_record["Val_loss"][validation_record["Epoch"]==(i+1)]
                                val_epoch_loss = epoch_data.mean()
                                loss_relative_change = np.abs((best_loss_avg-val_epoch_loss)/best_loss_avg)
                            
                                if val_epoch_loss < best_loss_avg:
#                                    save_sess = sess # record session, pass by reference? forget it. 
                                    best_loss_avg = val_epoch_loss
                                    last_improved_step = 0
                                elif epoch_data.min() < best_loss_avg:
#                                    save_sess = sess # record session, pass by reference? forget it. 
                                    best_loss_avg = epoch_data.min()
                                    last_improved_step = 0
                                else:
                                    if (val_epoch_loss-best_loss_avg)/best_loss_avg > minimum_relative_loss_improvement*5:  # increase the tolerance of loss recovery
                                        last_improved_step +=1
                                if loss_relative_change>=minimum_relative_loss_improvement:
                                    tiny_improved_step = 0
                                else:
                                    tiny_improved_step += 1
                                    
                                if last_improved_step > require_improve_step or tiny_improved_step > require_improve_step:
                                    end_timing = time.time()
                                    info_stop = "\n No improvement found during the {} last validation, stopped at epoch: {}. Time elapse: {} min.".format(require_improve_step, i+1, '%.2f' % np.float((end_timing-start_timing)/60))
                                    print(info_stop)
                                    log.write(info_stop+'\n')
                                    
                                    # Break out from the loop.
                                    stop_training = True
#                                    sess = save_sess        # forget it. pass by reference.   
                                    save_training(sess, temp_dir, model_path, validation_record, training_status="EarlyStop")
                                    if open_monitor:
                                        figure.savefig(os.path.join(temp_dir, model_id, "Training monitor.png"), dpi=300)
                                    break
                    else:
                        continue
                    break
                
                    # save checkpoints
                    if (i+1) % save_checkpoint_epoch == 0:
                        checkpoint_name = os.path.join(ckpt_and_summary_dir, 'model_epoch'+str(i+1)+'.ckpt')
#                            save_path = saver.save(sess, checkpoint_name)
                        info_checkpoint = "{} Training checkpoint saved.\n".format(time.strftime("%H:%M:%S"))
                        print(info_checkpoint)
                        log.write(info_checkpoint)
                        
                #-------------------- save the trained model ---------------------#
                if stop_training is False:
                    save_training(sess, temp_dir, model_path, validation_record)
                    if open_monitor:
                        figure.savefig(os.path.join(temp_dir, model_id, "Training monitor.png"), dpi=300)
                                                  
    
    