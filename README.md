# D2-Net: Dual Disentanglement Network for Brain Tumor Segmentation with Missing Modalities

This repository is the work of "D2-Net: Dual Disentanglement Network for Brain Tumor Segmentation with Missing Modalities" based on **pytorch** implementation. The multimodal brain tumor dataset (BraTS 2018) could be acquired from [here](https://www.med.upenn.edu/sbia/brats2018.html).


## Requirements
```
pip install -r requirements.txt
```

## Usage

Download the BraTS2018 dataset and change the path:

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

### Training

```
sh tool/train.sh
```

### Inference
```
sh tool/inference.sh
```



## Acknowledge
1. The implementation is based on the repo: [DMFNet](https://github.com/China-LiuXiaopeng/BraTS-DMFNet)
