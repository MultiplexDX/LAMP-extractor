# COVID-19 LAMP plate samples extraction 

# Installation instructions
0. Install conda distribution [Link](https://www.anaconda.com/products/distribution)
1. Prepare conda enviroment and activate
```
conda create -n lamp_extractor python=3.8; 
conda activate lamp_extractor;
```
2. Install this lamp-extractor package
```
cd lamp-extractor
pip install .;
```
3. Run API to extract lamp test
```
python -m lamp_extractor.apis.rest.api;
```
4. Upload image
```
curl --location --request POST 'http://127.0.0.1:8081/lamp-extractor/extract/' \
--header 'Content-Type: multipart/form-data' \
--form 'image=@"image_path_sars.jpg"'
```

# Analysis
[MultiplexDX LAMP-tests (Loop-mediated isothermal amplification)](https://www.multiplexdx.com/post/novy-lamp-test-prinasa-presnost-pcr-testov)

As a follow up to developing Vivid COVID-19 LAMP, in conjunction with engineers and a software development company we designed and created a mobile LAMP testing laboratory for medium - to high-throughput LAMP testing in the field (Supplementary Figure 6). 

The laboratory is fully contained in a commercially available van divided in 2 distinct sections, for sample inactivation and for LAMP, respectively (Supplementary Figure 6A). 

The process is streamlined (Supplementary Figure 6B, Supplementary Video 1) and involves first pairing incoming samples with inactivation tubes pre-filled with our inactivation buffer by the means of machine-readable codes on both. Thereafter, samples are processed as in a lab setting and inactivated samples are put into barcoded 96-well racks with open bottoms. Filled racks are then scanned in the LAMP section of the van and a semi-automated 96-well pipettor is used to transfer inactivated sample supernatants into pre-filled LAMP plates which are then incubated and left to amplify. 

. The outlined procedure allows anonymized and automated sample ID and position tracking, all tied together by a central database.

Finished plates are then scanned with a tablet running custom-designed software for sample-level colorimetric LAMP analysis.  

**This system was designed to greatly reduce the hands-on time necessary to individually classify final reaction colors.**

## 2 types of plates
- RNaseP - To identify poorly collected gargle specimens, presence of human genomic material testing in parallel (4µl inactivated sample input, 25 µl reaction volume, targeting RPP30, Supplementary Table 2) was introduced as well
- SARS - Plates to indetify presence of COVID. Reaction volume is 50µl

## Requirements
- Human validation of results
- Extracting result from regular photo captured on tablet
- Response should not takes more than 1s
- Be able to run offline

# Proposed solution
<img width="500" alt="assets/sample.jpg" src="assets/0_pipeline.png">

## Base algorithm
Our approach is based on traditional computer vision techniques combined with neural network landmark detection.

As an input system expects photo of the plate and gives output matrix in which are result map position to each sample position.

In the first part system performs four corners localization to crop and align plate in the image to get sample id and position.

In the second part system segments and clasify each sample. The segmentation is based on color saturation to get rid of uninteresting pixels, which could have negative impact for further classification process. The classification is based on hue value frequency for each pixel in the sample. To be able to assign hue pixel value to specific class we defined hue color values range (fig. color.png).

As a result is matrix with resolution 8x12 in which are shown LAMP test results.

## Errors
    - System NEGATIVE, Human operator POSITIVE, INCONCLUSIVE 
    - System INCONCLUSIVE, Human operator NEGATIVE, POSITIVE
    - System POSITIVE, Human operator NEGATIVE, INCONCLUSIVE 

## Discussion
During solution proposition we considered following problems:
- Low amount of data before pilot release
- Different lightning conditions
- Different plate position and rotation in the photo
- Be able to run without specialized hardware

We did not have data from real testing before pilot release so we've decided to use traditional computer vision techniques [Niall O’ Mahony, 2019](https://arxiv.org/pdf/1910.13796.pdf) to extract plate and result from the photo based on manual feature engineering.

To tackle different lightning conditions we designed and 3D printed box with LED lights. This help us create stable enviroment in which differences in samples color are related to chemical reaction itself and not to enviroment in which is photo taken. 

Stable enviroment helped us defined color features for each class and segmentation algorithm based on HSV color space.

To ensure same plate position and rotation we added drawer to the box. Human operator need to put plate in a drawer to be able to capture photo. 

We expected discrepancies in plate positioning caused by human factor so we decided utilized neural network KeypointsNetwork (KpsNet) which looks for 4 corners of the plate in the photo. This approach was inspired by work which was done on face landmark detection ([Feng-Ju Chang et All, 2017](https://arxiv.org/pdf/1708.07517v2.pdf), [Yue Wu et All, 2018](https://arxiv.org/pdf/1805.05563.pdf)). With this approach system is also able to localize 4 corners of a plate outside of the box. Subsequentely, plate can be cropped and aligned from the photo with overlayed result for human operator to do manual validation. 

To be able to train neural network we prepared custom dataset consiting of video frames in which is plate from various viewpoints and in various positions. We manually annotated 4 corner of a plate in each frames (approx. 1800 frames). For annotations we used open-source annotation tools ([CVAT](https://zenodo.org/record/4009388/export/hx#.Y01pC-xBxCN)).

To ensure realtime experience we chose pretrained MobileNet V2 as a backbone ([Mark Sandler et All, 2018](https://arxiv.org/abs/1801.04381)) which has significantly fewer parameters and smaller computational complexity then other popular backbones as AlexNet, VGG, ResNet, Inception.

## KpsNet architecture
```
======================================================================
Layer (type:depth-idx)                        Param #
======================================================================
├─MobileNetV2: 1-1                            --
├─Linear: 1-2                                 10,248
======================================================================
Total params: 2,243,528
Trainable params: 2,243,528
Non-trainable params: 0
======================================================================
```

## Training paramaters
- number of photos: 1800
- number of epochs: 680
- backbone: MobileNet V2 pretrained
- learning_rate: 0.001
- loss: MSE 
- train set augmentation: 
    - Resize: 352x352
    - Vertical Horizontal 
    - ShiftScaleRotate, 
    - RandomBrightnessContrast, 
    - ISONois, 
    - MotionBlur, 
    - ImageCompression, 
    - GaussNoise, 
    - Blur, 
    - ImageCompression
    - net tranforms:
    - Resize 352x352
    - Normalize (mean 127, 125, 127), (std: 40, 41, 43)

# Future work
Currently our segmentation and classification algorithm highly depends on stable enviroment provided by the box. But human is able to classify results on plate also outside of the box in various enviroments. We found process of 3D printed box costy and time consuming. In future work we highly reccomend do automatic analysis without stable enviroment provided by box and make system decisions closer to human operator. Thanks to collected dataset from national testing we see potential to utilize neural network for sample-level localization and classification. In this case neural network should be able to automatically extract features requierd to infer position and class for each sample without stable enviroment.


# Licenses
```
+---------------------------------+----------------------------+
|             Package             |          License           |
+---------------------------------+----------------------------+
|       albumentations 1.1.0      |            MIT             |
|        atomicwrites 1.4.0       |            MIT             |
|           attrs 21.2.0          |            MIT             |
|           build 0.7.0           |            MIT             |
|          fastapi 0.63.0         |            MIT             |
|          imageio 2.13.1         |        BSD-2-Clause        |
|         imantics 0.1.12         |            MIT             |
|          imutils 0.5.4          |            MIT             |
|       lamp-extractor 3.0.1      |        Common Clause       |
|           loguru 0.5.3          |            MIT             |
|         matplotlib 3.3.3        |            PSF             |
|           numpy 1.21.4          |            BSD             |
| opencv-python-headless 4.5.1.48 |            MIT             |
|           pandas 1.2.3          |            BSD             |
|        prettytable 2.4.0        |       BSD (3 clause)       |
|           pytest 6.2.5          |            MIT             |
|        pytest-mock 3.6.1        |            MIT             |
|      python-multipart 0.0.5     |           Apache           |
|           PyYAML 5.3.1          |            MIT             |
|         requests 2.25.1         |         Apache 2.0         |
|       scikit-image 0.18.1       |        Modified BSD        |
|       scikit-learn 0.24.0       |          new BSD           |
|           scipy 1.6.0           |            BSD             |
|         starlette 0.13.6        |            BSD             |
|           toml 0.10.2           |            MIT             |
|           torch 1.10.0          |           BSD-3            |
|        torchvision 0.11.1       |            BSD             |
|           tqdm 4.56.0           |   MPLv2.0, MIT Licences    |
|           wheel 0.37.0          |            MIT             |
+---------------------------------+----------------------------+
```
