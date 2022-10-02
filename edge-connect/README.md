## Generative Image Inpainting using Edge Learning wihout a Mask
[Presentation](https://github.com/harsh-agar/Image-Inpainting-without-Mask/blob/Image-Inpainting-with-Edge-Connect/edge-connect/Presentation__Image-Inpainting_with_Edge-Connect.pdf) 
| [Report](https://github.com/harsh-agar/Image-Inpainting-without-Mask/blob/Image-Inpainting-with-Edge-Connect/edge-connect/Report__Image-Inpainting_with_Edge-Connect.pdf)

### Introduction:
Deep learning approaches have resulted in considerable improvements in image inpainting during the last few years. However, having accurate masks is difficult in practice in a
variety of settings. For example, naturally occurring Image deformations are random and do not have a defined mask. Passing an explicit mask to solve the image inpainting problem is therefore tricky. 

This work proposes a novel method to solve image inpainting task without the need for any explicit mask. As our network architecture, we proposed a robust pipeline. Initially, an "Autoencoder" Mask Prediction network gets an incomplete color image as input and generates masks. The resulting mask will then be used to predict the complete edge map using "Gated Convolutions". This predicted edge map as well as an incomplete color image are sent to the refinement network for Image Inpainting. As a result, semantically realistic and visually appealing image is generated. 

<p align='center'>  
  <img src='examples\2022-10-03_00h33_33.png' width='400'/>
</p>
(a) Input images with missing regions. The missing regions are depicted in Black. 

(b) Predicted Masks. Region shown in white are generated (for the masked regions) using a Mask Predictor Network. 

(c) Computed edge masks. Edges drawn in black are computed (for the available regions) using Canny edge detector; whereas edges shown in blue are predicted by the Edge Generator Network. 

(d) Image inpainting results of the proposed approach.

## Prerequisites
- Python 3
- PyTorch 1.0
- NVIDIA GPU + CUDA cuDNN

## Installation
- Clone this repo:
```bash
git clone https://github.com/harsh-agar/Image-Inpainting-without-Mask.git
cd edge-connect
```
- Install PyTorch and dependencies from http://pytorch.org
- Install python requirements:
```bash
pip install -r requirements.txt
```

## Datasets
### 1) Images
 Use [Places2](http://places2.csail.mit.edu), [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) or [Paris Street-View](https://github.com/pathak22/context-encoder) datasets. To train a model on the full dataset, download datasets from official websites. 

After downloading, run [`scripts/flist.py`](scripts/flist.py) to generate train, test and validation set file lists. For example, to generate the training set file list on Places2 dataset run:
```bash
mkdir datasets
python ./scripts/flist.py --path path_to_places2_train_set --output ./datasets/places_train.flist
```

### 2) Irregular Masks
Our model is trained on the irregular mask dataset provided by [Liu et al.](https://arxiv.org/abs/1804.07723). You can download publically available Irregular Mask Dataset from [their website](http://masc.cs.gmu.edu/wiki/partialconv).

Alternatively, you can download [Quick Draw Irregular Mask Dataset](https://github.com/karfly/qd-imd) by Karim Iskakov which is combination of 50 million strokes drawn by human hand.

Please use [`scripts/flist.py`](scripts/flist.py) to generate train, test and validation set masks file lists as explained above.

### 1) Training
To train the model, create a `config.yaml` file similar to the [example config file](https://github.com/harsh-agar/Image-Inpainting-without-Mask.git/edge-connect/blob/master/config.yml.example) and copy it under your checkpoints directory. Read the [configuration](#model-configuration) guide for more information on model configuration.

EdgeConnect and Inpainting Module together can be trained in three stages: 1) training the edge model, 2) training the inpaint model and 3) training the joint model. To train the model:
```bash
python train.py --model [stage] --checkpoints [path to checkpoints]
```

For example to train just the edge model on Places2 dataset under `./checkpoints/places2` directory:
```bash
python train.py --model 1 --checkpoints ./checkpoints/places2
```

### 2) Testing
To test the model, create a `config.yaml` file similar to the [example config file](config.yml.example) and copy it under your checkpoints directory. Read the [configuration](#model-configuration) guide for more information on model configuration.

You can test the model on all three stages: 1) edge model, 2) inpaint model and 3) joint model. In each case, you need to provide an input image (image with a mask on it). To test the model:
```bash
python test.py \
  --model [stage] \
  --checkpoints [path to checkpoints] \
  --input [path to input directory or file] \
  --mask [path to masks directory or mask file] \
  --output [path to the output directory]
```

By default `test.py` script is run on stage 3 (`--model=3`).

### 3) Evaluating
To evaluate the model, you need to first run the model in [test mode](#testing) against your validation set and save the results on disk. We provide a utility [`./scripts/metrics.py`](scripts/metrics.py) to evaluate the model using PSNR, SSIM and Mean Absolute Error:

```bash
python ./scripts/metrics.py --data-path [path to validation set] --output-path [path to model output]
```

To measure the Fréchet Inception Distance (FID score) run [`./scripts/fid_score.py`](scripts/fid_score.py). We utilize the PyTorch implementation of FID [from here](https://github.com/mseitzer/pytorch-fid) which uses the pretrained weights from PyTorch's Inception model.

```bash
python ./scripts/fid_score.py --path [path to validation, path to model output] --gpu [GPU id to use]
```

### Alternative Edge Detection
By default, we use Canny edge detector to extract edge information from the input images. If you want to train the model with an external edge detection ([Holistically-Nested Edge Detection](https://github.com/s9xie/hed) for example), you need to generate edge maps for the entire training/test sets as a pre-processing and their corresponding file lists using [`scripts/flist.py`](scripts/flist.py) as explained above. Please make sure the file names and directory structure match your training/test sets. You can switch to external edge detection by specifying `EDGE=2` in the config file.

### Model Configuration

The model configuration is stored in a [`config.yaml`](config.yml.example) file under your checkpoints directory. The following tables provide the documentation for all the options available in the configuration file:

#### General Model Configurations

Option          | Description
----------------| -----------
MODE            | 1: train, 2: test, 3: eval
MODEL           | 1: edge model, 2: inpaint model, 3: edge-inpaint model, 4: joint model
MASK            | 1: random block, 2: half, 3: external, 4: external + random block, 5: external + random block + half
EDGE            | 1: canny, 2: external
NMS             | 0: no non-max-suppression, 1: non-max-suppression on the external edges
SEED            | random number generator seed
GPU             | list of gpu ids, comma separated list e.g. [0,1]
DEBUG           | 0: no debug, 1: debugging mode
VERBOSE         | 0: no verbose, 1: output detailed statistics in the output console

#### Loading Train, Test and Validation Sets Configurations

Option          | Description
----------------| -----------
TRAIN_FLIST     | text file containing training set files list
VAL_FLIST       | text file containing validation set files list
TEST_FLIST      | text file containing test set files list
TRAIN_EDGE_FLIST| text file containing training set external edges files list (only with EDGE=2)
VAL_EDGE_FLIST  | text file containing validation set external edges files list (only with EDGE=2)
TEST_EDGE_FLIST | text file containing test set external edges files list (only with EDGE=2)
TRAIN_MASK_FLIST| text file containing training set masks files list (only with MASK=3, 4, 5)
VAL_MASK_FLIST  | text file containing validation set masks files list (only with MASK=3, 4, 5)
TEST_MASK_FLIST | text file containing test set masks files list (only with MASK=3, 4, 5)

#### Training Mode Configurations

Option                 |Default| Description
-----------------------|-------|------------
LR                     | 0.0001| learning rate
D2G_LR                 | 0.1   | discriminator/generator learning rate ratio
BETA1                  | 0.0   | adam optimizer beta1
BETA2                  | 0.9   | adam optimizer beta2
BATCH_SIZE             | 8     | input batch size 
INPUT_SIZE             | 256   | input image size for training. (0 for original size)
SIGMA                  | 2     | standard deviation of the Gaussian filter used in Canny edge detector </br>(0: random, -1: no edge)
MAX_ITERS              | 2e6   | maximum number of iterations to train the model
EDGE_THRESHOLD         | 0.5   | edge detection threshold (0-1)
L1_LOSS_WEIGHT         | 1     | l1 loss weight
FM_LOSS_WEIGHT         | 10    | feature-matching loss weight
STYLE_LOSS_WEIGHT      | 1     | style loss weight
CONTENT_LOSS_WEIGHT    | 1     | perceptual loss weight
INPAINT_ADV_LOSS_WEIGHT| 0.01  | adversarial loss weight
GAN_LOSS               | nsgan | **nsgan**: non-saturating gan, **lsgan**: least squares GAN, **hinge**: hinge loss GAN
GAN_POOL_SIZE          | 0     | fake images pool size
SAVE_INTERVAL          | 1000  | how many iterations to wait before saving model (0: never)
EVAL_INTERVAL          | 0     | how many iterations to wait before evaluating the model (0: never)
LOG_INTERVAL           | 10    | how many iterations to wait before logging training loss (0: never)
SAMPLE_INTERVAL        | 1000  | how many iterations to wait before saving sample (0: never)
SAMPLE_SIZE            | 12    | number of images to sample on each samling interval

## License
Licensed under a [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/).

Except where otherwise noted, this content is published under a [CC BY-NC](https://creativecommons.org/licenses/by-nc/4.0/) license, which means that you can copy, remix, transform and build upon the content as long as you do not use the material for commercial purposes and give appropriate credit and provide a link to the license.

## References