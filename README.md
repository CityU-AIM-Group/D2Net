# D2-Net: Dual Disentanglement Network for Brain Tumor Segmentation with Missing Modalities

This repository is the work of "D2-Net: Dual Disentanglement Network for Brain Tumor Segmentation with Missing Modalities" based on **pytorch** implementation. The multimodal brain tumor dataset (BraTS 2018) could be acquired from [here](https://www.med.upenn.edu/sbia/brats2018.html).


## Requirements
* python 3.7
* pytorch 1.6.0
* nibabel
* pickle 

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

```python
python train.py  --gpu=0 --net=DisenNet --batch_size=1 \
    --valid_freq=5 --output_set=train_val \
    --DisenNet_indim=8 --AuxDec_dim=4 \
    --miss_modal=True --use_Bernoulli_train=True \
    --use_contrast=True --use_freq_contrast=True \
    --use_distill=True --use_kd=True --affinity_kd=True \
    --setting=D2Net 
```



## Acknowledge
1. The implementation is based on the repo: [DMFNet](https://github.com/China-LiuXiaopeng/BraTS-DMFNet)
