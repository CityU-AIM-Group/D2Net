# D2-Net: Dual Disentanglement Network for Brain Tumor Segmentation with Missing Modalities

This repository is the official **PyTorch** implementation of [D2-Net: Dual Disentanglement Network for Brain Tumor Segmentation with Missing Modalities](https://ieeexplore.ieee.org/document/9775681) published on IEEE TMI 2022.

## Overview
![image](https://github.com/CityU-AIM-Group/D2Net/blob/main/figs/D2Net.png)

We propose a Dual Disentanglement Network (D2-Net) for brain tumor segmentation with missing modalities, which consists of a modality disentanglement stage (MD-Stage) and a tumor-region disentanglement stage (TDStage). In the MD-Stage, a spatial-frequency joint modality contrastive learning scheme is designed to directly decouple the modality-specific information from MRI data. To decompose tumor-specific representations and extract discriminative holistic features, we propose an affinity-guided dense tumor-region knowledge distillation mechanism in the TD-Stage through aligning the features of a disentangled binary teacher network with a holistic student network. By explicitly discovering relations among modalities and tumor regions, our model can learn sufficient information for segmentation even if some modalities are missing.

## Requirements
All experiments use the PyTorch library. We recommend installing the following package versions:

* &nbsp;&nbsp; python=3.7 

* &nbsp;&nbsp; pytorch=1.6.0

* &nbsp;&nbsp; torchvision=0.7.0

Dependency packages can be installed using following command:
```
pip install -r requirements.txt
```

## Dateset
### Download
We use multimodal brain tumor dataset (BraTS 2018) in our experiments. [Download](https://www.med.upenn.edu/sbia/brats2018.html) the BraTS2018 dataset and change the path:

```
experiments/PATH.yaml
```

### Data preprocess
Convert the .nii files as .pkl files. Normalization with zero-mean and unit variance . 

```
python preprocess.py
```

(Optional) Split the training set into k-fold for the **cross-validation** experiment.

```
python split.py
```

## Training
D2-Net can be trained by running following command:

```
sh tool/train.sh
```
In our experiments, D2-Net is built on the lightweight backbone [U2-Net](https://arxiv.org/abs/1909.06012). Training D2-Net requires at least one V100 GPU with 32G memory. The defualt hyperparameters are set in train.py. Running the training code will generate logs files and saved models in a directory name logs and ckpts, respectively.

## Inference
D2-Net can be tested with a saved model using following command:
```
sh tool/inference.sh
```
The inference code will test all 15 cases with missing modalities together.

## Citation
```
@article{yang2022d2,
  title={D2-Net: Dual Disentanglement Network for Brain Tumor Segmentation with Missing Modalities},
  author={Yang, Qiushi and Guo, Xiaoqing and Chen, Zhen and Woo, Peter YM and Yuan, Yixuan},
  journal={IEEE Transactions on Medical Imaging},
  year={2022},
  publisher={IEEE}
}
```

## Acknowledgement
1. The implementation is based on the repo: [DMFNet](https://github.com/China-LiuXiaopeng/BraTS-DMFNet)
