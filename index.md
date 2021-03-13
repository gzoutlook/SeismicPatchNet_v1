## A Practice of Tiny Machine Learning for Earth Science


(Sep. 2019, ref article: [https://doi.org/10.1038/s41467-020-17123-6](https://doi.org/10.1038/s41467-020-17123-6)

Automated design of a convolutional neural network with multi-scale filters for cost-efficient seismic data classification.

Edge computing platform: Raspberry Pi 4 Model B (4GB RAM)

OS: Raspberry Pi OS (August 2020)

- Framework: Tensorflow v2.3.0
- Model: trained without tuning (Nvidia GPU), deployed without pruning (Rasberry Pi 4B)

Results for predicting natural gas hydrates indicators on Blake Ridge Line 88 (see Figure 6s in [the article](https://doi.org/10.1038/s41467-020-17123-6)), using a more coarse grid (lower resolution):

<img src="https://github.com/gzoutlook/SeismicPatchNet_v1/blob/master/Raspberry%20Pi%204%20inference.png" style="display: block; margin: auto;" />
