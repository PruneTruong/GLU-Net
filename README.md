# Official implementation of GLUNet

This is the official implementation of our paper : 

**GLU-Net: Global-Local Universal Network for dense flow and correspondences (CVPR 2020-Oral).**

Authors: [Prune Truong](https://prunetruong.com/), [Martin Danelljan](https://martin-danelljan.github.io/) and Radu Timofte <br />
\[[Paper](https://arxiv.org/abs/1912.05524)\]
\[[Website](https://prunetruong.com/research/glu-net)\]
\[[Poster](https://drive.google.com/file/d/1pS_OMZ83EG-oalD-30vDa3Ru49GWi-Ky/view?usp=sharing)\]
\[[Oral Video](https://www.youtube.com/watch?v=xB2gNx8f8Xc&feature=emb_title)\]
\[[Teaser Video](https://www.youtube.com/watch?v=s5OUdkM9QLo)\]



Check out our related publication [**GOCor**](https://arxiv.org/abs/2009.07823) ([website](https://prunetruong.com/research/gocor)) 
and corresponding code [here](https://github.com/PruneTruong/GOCor) ! 

For an improved version of a dense correspondence network, also predicting a confidence mask, check out [**PDCNet**](https://arxiv.org/abs/2101.01710) ([website](https://prunetruong.com/research/pdcnet)) and code [here](https://github.com/PruneTruong/DenseMatching). 
<br /><br /><br />



For any questions, issues or recommendations, please contact Prune at prune.truong@vision.ee.ethz.ch <br />
If our project is helpful for your research, please consider citing :
```bash
@inproceedings{GLUNet_Truong_2020,
      title = {{GLU-Net}: Global-Local Universal Network for dense flow and correspondences},
      author    = {Prune Truong and
                   Martin Danelljan and
                   Radu Timofte},
      year = {2020},
      booktitle = {2020 {IEEE} Conference on Computer Vision and Pattern Recognition, {CVPR} 2020}
}
```

# Network

Our model GLU-Net is illustrated below:
![alt text](/images/glunet.png)



The models, evaluation and training codes for Local-Net (a 3 level pyramidal network with only local correlations), Global-Net (a 3 levels pyramidal network with a single 
global correlation followed by concatenation of feature maps) and GLOCAL-Net (a combination of the two, a 3 levels pyramidel network 
with a single global correlation followed by two local correlation layers) are also available for reference. 
They are all illustrated below:
![alt text](/images/nets.png)


For more details, refer to our [paper](https://arxiv.org/abs/1912.05524)


# Table of Content

1. [Installation](#Installation)
2. [Test on your own image pairs !](#test_pairs)
3. [Datasets downloading](#download_dataset)
    1. [Training datasets](#training_dataset)
    2. [Testing datasets](#testing_dataset)
4. [Training](#training)
5. [Evaluation](#evaluation)
    1. [Performance on geometric matching dataset](#geometric_matching)
    2. [Performance on semantic matching dataset](#semantic_matching)
    3. [Performance on optical flow dataset](#OF)
6. [Acknowledgement](#acknowledgement)
7. [Changelog](#changelog)


# 

# 1. Installation <a name="Installation"></a>

Note that the models were trained with **torch 1.0**. Torch versions up to 1.7 were tested for inference on testing datasets but NOT for training, 
so I cannot guarantee that the models train smoothly for higher torch versions. 


* Create and activate conda environment with Python 3.x

```bash
conda create -n GLUNet_env python=3.7
conda activate GLUNet_env
```

* Install all dependencies (except for cupy, see below) by running the following command:

```bash
pip install -r requirements.txt
```

**ATTENTION**, CUDA is required to run the code. Indeed, the correlation layer is implemented in CUDA using CuPy, 
which is why CuPy is a required dependency. It can be installed using pip install cupy or alternatively using one of the 
provided binary packages as outlined in the CuPy repository. The code was developed using Python 3.7 & PyTorch 1.0 & CUDA 9.0, 
which is why I installed cupy for cuda90. For another CUDA version, change accordingly. 

```bash
pip install cupy-cuda90==7.8.0 --no-cache-dir 
```

There are some issues with latest versions of cupy. So for all cuda, install cupy version 7.8.0. For example, on cuda10, 
```bash
pip install cupy-cuda100==7.8.0 --no-cache-dir 
```

* **Download an archive with pre-trained models [click](https://drive.google.com/open?id=15yXIi8kJbCyXCAHzg-UbMQ6RD2lJ1nOi) and extract it to the project folder**                                                


# 2. Test on your own image pairs ! <a name="test_pairs"></a>

One can test GLU-Net on a pair of images using test_GLUNet.py and the provided trained model weights. 
The inputs are the paths to the source and target images. They are then passed
to the network which outputs the corresponding flow field relating the target to the source image. The source is then warped according to
the estimated flow, and a figure is saved. 

For this pair of images (provided to check that the code is working properly), the output is:

```bash
python test_GLUNet.py --path_source_image images/yosemite_source.png --path_target_image images/yosemite_target.png --write_dir evaluation/

additional optional arguments:
--pre_trained_models_dir (default is pre_trained_models/)
--pre_trained_model (default is DPED_CityScape_ADE)
```
![alt text](/images/yosemite_test_output.png)

Another example and output (attention large images):
```bash
python test_GLUNet.py --path_source_image images/hp_source.png --path_target_image images/hp_target.png --write_dir evaluation/

optional arguments:
* --pre_trained_models_dir : Directory containing the pre-trained-models (default is pre_trained_models/)
* --pre_trained_model: Name of the pre-trained-model (default= DPED_CityScape_ADE )
```
![alt text](/images/hp_test_output.png)


# 3. Datasets downloading <a name="download_dataset"></a>

## 3.1. Training datasets <a name="training_dataset"></a>
For the training, we use a combination of the DPED, CityScapes and ADE-20K datasets. 
The DPED training dataset is composed of only approximately 5000 sets of images taken by four different cameras. 
We use the images from two cameras, resulting in around  10,000 images. 
CityScapes additionally adds about 23,000 images. 
We complement with a random sample of ADE-20K images with a minimum resolution of 750 x 750. 
It results in 40.000 original images, used to create pairs of training images by applying geometric transformations to them. 
The path to the original images as well as the geometric transformation parameters are given in the csv files
'datasets/csv_files/homo_aff_tps_train_DPED_CityScape_ADE.csv' and 'datasets/csv_files/homo_aff_tps_test_DPED_CityScape_ADE.csv'.

**Apparently, the structure of the ADE-20K dataset has changed and the provided paths in the csv files are not valid anymore**. 
I am working on a fix for the ADE-20K images. 
In the meantime, use 'datasets/csv_files/homo_aff_tps_train_DPED_CityScape.csv' and 'datasets/csv_files/homo_aff_tps_test_DPED_CityScape.csv' to exclude the ADE images (you don't need to download the ADE-20K dataset in that case). 
Resulting training data is 31K images. Performance of the resulting trained model might be a bit different. 

 

* Download the [DPED dataset](http://people.ee.ethz.ch/~ihnatova/) (54 GB) ==> images are created in original_images/
* Download the [CityScapes dataset](https://www.cityscapes-dataset.com/)
    - download 'leftImg8bit_trainvaltest.zip' (11GB, left 8-bit images - train, val, and test sets', 5000 images) ==> images are created in CityScape/
    - download leftImg8bit_trainextra.zip (44GB, left 8-bit images - trainextra set, 19998 images) ==> images are created in CityScape_extra/

* Download the [ADE-20K dataset](https://groups.csail.mit.edu/vision/datasets/ADE20K/) (3.8 GB, 20.210 images) ==> images are created in ADE20K_2016_07_26/


Put all the datasets in the same directory. 
As illustration, your root training directory should be organised as follows:
<pre>
/training_datasets/
                   original_images/
                   CityScape/
                   CityScape_extra/
                   ADE20K_2016_07_26/
</pre>

**Optional: To save the synthetic image pairs and flows to disk**                   
During training, from this set of original images, the pairs of synthetic images are created on the fly at each epoch. 
However, this dataset generation takes time and since no augmentation is applied at each epoch, one can also create the dataset in advance
and save it to disk. During training, the image pairs composing the training datasets are then just loaded from the disk 
before passing through the network, which is a lot faster. 
To generate the training dataset and save it to disk: 

```bash
python save_training_dataset_to_disk.py --image_data_path /directory/to/original/training_datasets/ 
--csv_path datasets/csv_files/homo_aff_tps_train_DPED_CityScape_ADE.csv --save_dir /path/to/save_dir --plot True
```    
It will create the images pairs and corresponding flow fields in save_dir/images and save_dir/flow respectively.

## 3.2. Testing datasets <a name="testing_dataset"></a>

The testing datasets are available at the following links:
* HPatches dataset: Download HPatches dataset (Full image sequences). The dataset is available [here](https://github.com/hpatches/hpatches-dataset) at the end of the page.
The corresponding csv files for each viewpoint ID with the path to the images and the homography parameters relating the pairs are listed in /datasets/csv_files/

* TSS dataset: The dataset is available [here](https://taniai.space/projects/cvpr16_dccs/) (1.2 GB)

* KITTI datasets: Both KITTI-2012 and KITTI-2015 are available [here](http://www.cvlibs.net/datasets/kitti/eval_flow.php)

* ETH3D dataset: See below in section 5.1


# 4.0 Training <a name="training"></a>

Training files for GLUNet (and its variants, including Semantic-GLU-Net), GLOCAL-Net, LocalNet and GlobalNet are available. 

**This will create the synthetic training and evaluation pairs along with the ground-truth on the fly !**
```bash 
python train_GLUNet.py --name_exp GLUNet_train --training_data_dir /path/to/directory/original_images-for-training/ --evaluation_data_dir /path/to/directory/original_images-for-evaluation/

if the network is already pretrained and the user wants to start the training from an old weight file
* --pretrained /path/to/pretrained_file.pth
```

**To load the pre-saved synthetic training and evaluation image pairs and ground truth flow fields instead (created earlier and saved to disk):**
```bash 
python train_GLUNet.py --name_exp GLUNet_train --pre_loaded_training_dataset True --training_data_dir /path/to/directory/synthetic_training_image_pairs_and_flows/
--evaluation_data_dir /path/to/directory/synthetic_validation_image_pairs_and_flows/

if the network is already pretrained and the user wants to start the training from an old weight file
* --pretrained /path/to/pretrained_file.pth
```

In the training files, one can modify all the parameters of the network. The default ones are for GLU-Net. 


# 5. Evaluation  <a name="evaluation"></a>

# 5.1. Performance on geometric matching dataset <a name="geometric_matching"></a>

In the case of geometric matching, pairs of images present different viewpoints of the same scene. 


## HPATCHES (original size and resized to 240x240)


To test on the HPatches dataset, HP-240 (images and flow rescaled to 240x240) and HP (original)
```bash
python eval.py --model GLUNet --pre_trained_models DPED_CityScape_ADE --dataset HPatchesdataset --data_dir /directory/to/hpatches --save_dir /directory/to/save_dir

optional argument: 
* --hpatches_original_size To test on the original image size, True or False (default to False) 
```

Out of the 120 sequences of HPatches, we only evaluate on the 59 sequences in HP labelled with v_X, which have viewpoint changes, 
thus excluding the ones labelled i_X, which only have illumination changes. 

<br /><br />
AEPE on the different viewpoints of HP-240 and HP:
| Method |  HP-240 I  | HP-240 II |  HP-240 III |  HP-240 IV |  HP-240 V | HP-240 All    |  HP I |  HP II |  HP III |  HP IV |  HP V | HP All    |
| -------------------- |:-----------:| ------------:| ------------- | ------------ | ----------- | ------ |:-----------:| ------------:| ------------- | ------------ | ----------- | ------ |
| PWC-Net              | 5.74        | 17.69        | 20.46         | 27.61        | 36.97       | 21.68  | 23.93   |76.33    | 91.30   | 124.22 | 164.91 | 96.14 |
| LiteFlowNet          | 6.99        | 16.78        | 19.13         | 25.27        | 28.89       | 19.41  |36.69    | 102.17    | 113.58   |  154.97 | 186.82 | 118.85 |
| DGC-Net (paper)      | 1.55 	     | 5.53 	    | 8.98 	        | 11.66 	   | 16.70       | 8.88   | - | - | - | - | - | - |
| DGC-Net (repo)       | 1.74 	     | 5.88 	    | 9.07 	        | 12.14 	   | 16.50       | 9.07   | 5.71   | 20.48    | 34.15   | 43.94 | 62.01 | 33.26 |
| **GLU-Net (Ours)**       | **0.59**        | **4.05**         | **7.64**         | **9.82**         | **14.89**       | **7.40**   | **1.55**  | **12.66**   | **27.54** | **32.04**  | **51.47**  | **25.05** |
| **GLU-Net (this repo)**       | **0.59**        | **4.05**         | **7.64**         | **9.82**         | **14.89**       | **7.40**   | **1.53**  | **12.62**   | **27.38** | **31.95**  | **51.11**  | **24.91** |


Illustration on two examples of pairs of HP:
![alt text](/images/hp.jpg)

## ETH3D dataset 

**Data preparation**: execute 'bash download_ETH3D.sh', the file is stored in datasets/
It does the following: 
- Create your root directory ETH3D/, create two sub-directories multiview_testing/ and multiview_training/
- Download the "Low rew multi-view, training data, all distorted images" [here](https://www.eth3d.net/data/multi_view_training_rig.7z) and unzip them in multiview_training/
- Download the "Low rew multi-view, testing data, all undistorted images" [here](https://www.eth3d.net/data/multi_view_test_rig_undistorted.7z) and unzip them in multiview_testing/
- We directly provide correspondences for pairs of images taken at different intervals. There is one bundle file for each dataset and each rate of interval, for example "lakeside_every_5_rate_of_3". 
This means that we sampled the source images every 5 images and the target image is taken at a particular rate from each source image. Download all these files [here](https://drive.google.com/file/d/1Okqs5QYetgVu_HERS88DuvsABGak08iN/view?usp=sharing) and unzip them. 

As illustration, your root ETH3D directory should be organised as follows:
<pre>
/ETH3D/
       multiview_testing/
                        lakeside/
                        sand_box/
                        storage_room/
                        storage_room_2/
                        tunnel/
       multiview_training/
                        delivery_area/
                        electro/
                        forest/
                        playground/
                        terrains/
        info_ETH3D_files/
</pre>
The organisation of your directories is important, since the bundle files contain the relative paths to the images, from the ETH3D root folder. 

<br /><br />
**Evaluation**: for each interval rate (3,5,7,9,11,13,15), we compute the metrics for each of the sub-datasets (lakeside, delivery area and so on). The final metrics are the average over all datasets for each rate. 
```bash
python eval_ETH3D.py --model GLUNet --pre_trained_models DPED_CityScape_ADE --data_dir /directory/to/ETH3D --save_dir /directory/to/save_dir 
```

<br />
AEPE for different rates of intervals between image pairs (corresponds to Fig. 5 of the main paper and Tab. 9 in supplementary). 
PCK results are also provided in the paper. 
These results are computed with a slightly more precise ground-truth than in the paper.

| Method               | rate=3   | rate=5 |   rate=7  |  rate=9   |  rate=11  | rate=13   |rate=15  |
| -------------------- |:----:| ---:| ---- | ------------ | ----------- | ------ |:-----------:| 
| PWC-Net              | **1.75** | **2.10** | **3.21**  | 5.59 |  14.35 | 27.49 |  43.41  |
| LiteFlowNet          | 1.66 | 2.58 | 6.05  | 12.95| 29.67  | 52.41 | 74.96 |
| DGC-Net              | 2.49 | 3.28 | 4.18  | 5.35 | 6.78  | 9.02  |  12.23 |
| **GLU-Net (Ours)**   | 1.98 | 2.54 | 3.49  | **4.24** |  **5.61** |  **7.55** | **10.78**  |


## Qualitative examples on the testing set of DPED

We tested our network GLU-Net as well as DGC-Net on a the testing image pairs of the DPED dataset. A few examples are presented below. 
No ground-truth flow field between the image pairs are available, therefore those are only qualitative results. 
![alt text](/images/DPED.jpg)


# 5.2. Performance on semantic matching dataset <a name="semantic_matching"></a>


In the case of semantic matching, pairs of images show two instances of the same object or scene category. 

## TSS dataset = only dataset with dense ground truth on foreground objects 

To test on TSS
```bash
python eval.py --model GLUNet --flipping_condition True --pre_trained_models DPED_CityScape_ADE --dataset TSS --data_dir /directory/to/TSS/DJOBS --save_dir /directory/to/save_dir 

optional arguments:
* --flipping condition True or False, for TSS recommanded
```

or for the custom network:
```bash
python eval.py --model SemanticGLUNet --flipping_condition True --pre_trained_models DPED_CityScape_ADE --dataset TSS --data_dir /directory/to/TSS/JOBS --save_dir /directory/to/save_dir 

optional arguments:
* --flipping condition True or False, for TSS recommanded
```

<br />
PCK [%] obtained on TSS for the task of semantic matching

| Method | FG3DCar | JODS  |  PASCAL | avg.|
| -------------------- |:----:| ----:| ---  | ---- |
| PARN (VGG-16)        | 87.6 | 71.6 | 68.8 | 76.0 |
| NC-Net               | 94.5 | 81.4 | 57.1 | 77.7 |
| DCCNet               | 93.5 | 82.6 | 57.6 | 77.9 |
| SAM-Net              | **96.1** | 82.2 | 67.2 | 81.8 |
| **GLU-Net (Ours)**   | 93.2 | 73.3 | 71.1 | 79.2 |
| **GLU-Net (this repo)**   | 93.2 | 73.7 | 71.1 | 79.3 |
| **Semantic-GLU-Net (Ours)**     | 94.4 | 75.5 | **78.3** | **82.8** |
| **Semantic-GLU-Net (this repo)**     | 94.4 | 75.7 | **78.3** | **82.8** |


Illustration on examples of the TSS dataset
![alt text](/images/TSS-more.jpg)


## Qualitative examples of day/night, seasonnal changes

Scenarios such as day/night, seasonnal changes and can be considered as borderline between geometric matching and semantic matching tasks
since such pairs of images depict the same scenes but the appearance variations are so drastic that those images can be associated to
semantic matching tasks. We qualitatively tested our network GLU-Net and Semantic-GLU-Net on examples of such cases and compared them to DGC-Net. 
The corresponding figures are presented below:

Day/Night changes:
![alt text](/images/day-night.jpg)


Seasonnal changes:
![alt text](/images/seasonnal_changes.jpg)

Some of the images are from the WxBS dataset [WxBS: Wide Baseline Stereo Generalizations. D. Mishkin et al. Proceedings of the British Machine Vision Conference. 2015. ]

# 5.3. Performance on optical flow dataset <a name="OF"></a>


In the case of optical flow dataset, pairs of images show two consecutive images of a sequence or video. 

## KITTI 2012 and 2015

To test on KITTI datasets
* for evaluation on all pixels, including the occluded ones (without the invalid ones of course): --dataset KITTI_occ
* for evaluation on only non occluded pixels: --dataset KITTI_noc

```bash
python eval.py --model GLUNet --pre_trained_models DPED_CityScape_ADE --dataset KITTI_occ --data_dir /directory/to/KITTI/training/ --save_dir /directory/to/save_dir 

```

|             | KITTI-2012 | KITTI-2015 |         |
|-------------|------------|------------|---------|
|             | AEPE       | AEPE       | F1  [%] |
| PWC-Net (flying-chairs ft 3d-Things)    | 4.14       | 10.35      | 33.67   |
| LiteFlowNet (flying-chairs ft 3d-Things| 4.0        | 10.39      | 28.50   |
| DGC-Net  (tokyo)   | 8.50       | 14.97      | 50.98   |
| **GLU-Net**  (CityScape-DPED-ADE)   | 3.34       | 9.79       | 37.52   |
| **GLU-Net**  (CityScape-DPED-ADE), this repo   | 3.33       | 9.79       | 37.77   |

Quantitative  results  on  optical  flow  KITTI  training datasets.  Fl-all: Percentage of outliers averaged over all pixels. 
Inliers are defined as AEPE < 3 pixels or < 5 %. Lower F1 and AEPE are best.

# 6. Acknowledgement <a name="Acknowledgement"></a>

We borrow code from public projects, such as [DGC-Net](https://github.com/AaltoVision/DGC-Net), [PWC-Net](https://github.com/NVlabs/PWC-Net), [NC-Net](https://github.com/ignacio-rocco/ncnet), [Flow-Net-Pytorch](https://github.com/ClementPinard/FlowNetPytorch)...

# 7. Changelog <a name="changelog"></a>

* Remove useless parts + rename some functions / parameters
* Modified READme to make it clearer
* Added website + evaluation code for ETH3D
* Modif of the READme file
* Added reference to WxBS dataset and csv files excluding ADE dataset