# SeismicPatchNet_v1
1. Demonstration of automated searching of a neural network architecture for cost-efficient seismic data classification (e.g. gas hydrates reflections). (Sep. 2019)
Ref: https://doi.org/10.1038/s41467-020-17123-6

2. A basic implementation of SPN_v1 using TensorFlow v2.3.1 and a corrpesonding graph visualization.

## *** System requirements ***

* Python 3.7.4.1
* Tensorflow-gpu 1.14
* Nvidia CUDA 10
* Nvidia cuDNN v7.5.0 (Feb 21, 2019), for CUDA 10.0


## *** Instructions for use ***

Functional programming scripts. 😃


1. Run "1. synthetic_patch_models.py" to generate synthetic data as much as possible (e.g. more than 20,000 samples) for architecture searching.

> It is strongly recommended to run the script step by step. For example, comment code block of:

>> "Step 2:  summarize statistics of architectures' performance"

> before run:

>> "Step 1:  scan and delete/remove failed/incomplete jobs/model folder"


2. Run "2. random_search_SeismicPatchNet_v1p2.py" to start searching, which can be deployed on multiple GPUs by replications
 and specify a GPU ID as suggested in the script.


3. Run "3. random_search_Analysis.py" to sort the searching results and find the most qualified architecture.

More details were presented in our article (https://doi.org/10.1038/s41467-020-17123-6).


Some other suggestions:

* Always pay attention to quality control for training, depending on hardwares, libraries, and settings.
* Plot training processes of trained models, making sure that the training is reliable/reasonable.
* As the size of SeismicPatchNet_v1 is very small, all trainable parameters matter, the training curve might be volatile. Fine-tuned regularization would stabilize the training process.


## *** Results reproducibility  ***
Edge computing platform: Raspberry Pi 4 Model B (4GB RAM)
OS: Raspberry Pi OS (August 2020)
* Framework: Tensorflow v2.3.0
* Model: trained without tuning (Nvidia GPU), deployed without pruning (Rasberry Pi 4B)
Results for Blake Ridge Line 88 (article: https://doi.org/10.1038/s41467-020-17123-6), using a more coarse grid (lower resolution):
![image](https://github.com/gzoutlook/SeismicPatchNet_v1/Raspberry Pi 4 inference.png)
