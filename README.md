# BAND AWARE DOMAIN GENERALIZATION FOR CROSS-COUNTRY MULTISPECTRAL REMOTE SENSING SCENE CLASSIFICATION
This repository contains the code for the paper titled "BAND AWARE DOMAIN GENERALIZATION FOR CROSS-COUNTRY MULTISPECTRAL REMOTE SENSING SCENE CLASSIFICATION" accepted for WHISPERS 2024.

[📄 Paper (WHISPERS 2024)](https://drive.google.com/file/d/1a6brgiiAcjsOqDEPwXQ4skIHFmOvXZxW/view?usp=sharing)\
[📄 Poster Presentation (WHISPERS 2024)](https://drive.google.com/file/d/1smTxD6jWeBtHzfGdHrxgrp6ZHm4duMvN/view?usp=sharing)

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
- [Results](#results)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)

## Introduction

### **Abstract** 
> *Domain shift refers to the overall distribution differences between data used for model development and post-deployment. If not addressed, it can typically lead to performance degradation in operational settings. It is especially emphasized in the context of remote sensing, where scenes commonly capture large areas with significant geographical, temporal, and sensor variations. Domain generalization is a type of transfer learning for addressing this issue, often through feature alignment, that on the contrary of domain adaptation, assumes access to neither target domain labels nor to target domain data. In this study, we explore the progressive alignment of an image's spectral bands, instead of handling them collectively and concurrently. Experiments have been conducted with Sentinel-2 multi-spectral images, with six European countries denoting the domains, using various contemporary domain generalization techniques, and it is shown that a gradual alignment of spectral bands leads to consistent performance improvements.*

### **Proposed Iterative Fine-tuning Scheme** 

![Overview](/images/design.png)

## Requirements
- Python 3.11
- PyTorch 2.5.0
- torchvision 0.20.1
- CUDA 11.8


## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/efkandurakli/DG-SC-RS.git
   cd DG-SC-RS
   ```
   
2. Create a conda virtual environment using `environment.yml` under the root folder and activate it
   ```
   conda env export --no-builds > environment.yml
   conda activate dgscrs
   ```

## Dataset

We have employed a subset of the multi-label BigEarth. Since BigEarth does not provide country labels, the GPS coordinates associated with each sample have been exploited in order to determine its country of origin. The extracted country information can be found in the `data/bigearthnet.csv` file.

### Folder Structure

```md
data/
├── bigearthnet/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    ├── test/
    │   ├── images/
    │   └── labels/
    ├── test-10%/
    │   ├── images/
    │   └── labels/
    └── test-90%/
        ├── images/
        └── labels/
```

## Usage

### Training

#### Run All Experiments
```
./train.sh
```
### Inference

#### Evaluate All Models
```
./test.sh
```

## Results

### Table: Scene classification performances across the target domains in terms of kappa score and overall accuracy for all the tested setups. Best results are in bold. For more detailed results, please refer to the full paper.

![Results](/images/results.png)

## Citation
```
@inproceedings{durakkli2024,
  title={Band aware domain generalization for cross-country multispectral remote sensing scene classification},
  author={E.~Durakli and D.E.~Turan and M.~Thota and P.~Bosilj and E.Aptoula},
  booktitle={14th IEEE Workshop on Hyperspectral Image and Signal Processing (IEEE WHISPERS)},
  year={2024}
}
```

## Acknowledgement

This study was supported by The Scientific and Technological Research Council of Turkiye (TUBITAK) under the Grant Number 123R108 and by the GEBIP Award program (2021) of the Turkish Academy of Sciences.
  
