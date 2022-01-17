import os
import torch
from torch.utils.data import Dataset

from .rand import Uniform
from .transforms import Rot90, Flip, Identity, Compose
from .transforms import GaussianBlur, Noise, Normalize, RandSelect
from .transforms import RandCrop, CenterCrop, Pad,RandCrop3D,RandomRotion,RandomFlip,RandomIntensityChange
from .transforms import NumpyType
from .transforms import *
from .data_utils import pkload

import numpy as np
import glob
import cv2


class BraTSDataset(Dataset):
    def __init__(self, list_file, root='', for_train=False, transforms='', semantic_edge=False, canny=False, true_valid_data=False):
        paths, names = [], []
        with open(list_file) as f:
            for line in f:
                line = line.strip()
                name = line.split('/')[-1]
                names.append(name)
                path = os.path.join(root, name + '_') ##
                paths.append(path)

        self.names = names
        self.paths = paths

        self.transforms = eval(transforms or 'Identity()')
        self.list_file = list_file
        self.for_train = for_train
        self.semantic_edge = semantic_edge
        self.canny = canny
        self.true_valid_data = true_valid_data

    def __getitem__(self, index):
        path = self.paths[index].split('/')[-1]
        if self.list_file.split('/')[-2] == 'train':
            save_path = ".data/BraTS_2018/pkl_file/MICCAI_BraTS2018_TrainingData/"
        if self.list_file.split('/')[-2] == 'valid':
            save_path = "./data/BraTS_2018/pkl_file/MICCAI_BraTS2018_ValidationData/"
        if self.list_file.split('/')[-2] == 'test':
            save_path = "./data/BraTS_2018/pkl_file/MICCAI_BraTS2018_TestingData/"
        x, y = pkload(save_path + path + 'data_f32.pkl')
        
        x, y = x[None, ...], y[None, ...]
        
        x,y = self.transforms([x, y])


        if self.semantic_edge and  (not self.true_valid_data):
            edge_label = self._mask2mask_semantic(y)  # [1,4,128,128,128]
        elif (not self.semantic_edge) and (not self.true_valid_data):
            edge_label = self._mask2maskb(y)          # [1,128,128,128]
        else:
            edge_label = np.array(0.01)

        if self.for_train :  # Operation only for training data      and not (self.true_valid_data):
            y[y == 4] = 3 # For the loss calculation

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))# [Bsize,channels,Height,Width,Depth]
        y = np.ascontiguousarray(y)

        x, y = torch.from_numpy(x), torch.from_numpy(y)
       
        if (not self.true_valid_data):
            return x, y, torch.from_numpy(edge_label)  # For training data and train_val data
        else:
            return x, y   # Only for true_validation data

    def __len__(self):
        return len(self.names)

    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]
    
    
    def _mask2maskb(self, mask): # mask := ori-label
        maskb = np.array(mask).astype('int32')
        b,h,w,d = maskb.shape
        maskb [maskb == 255] = -1
        maskb_ = np.array(mask).astype('float32')

        if self.canny:
            mask_tmp = np.zeros((b,h,w,d),np.float32)
            for i in range(d):
                mask_tmp[:,:,:,i] = cv2.Canny(np.uint8(maskb_[:,:,:,i]), 0, 0.001)
            # mask_tmp = mask_tmp > 0
            mask_tmp[mask_tmp > 0] = 1
            # mask_tmp = torch.from_numpy(mask_tmp).cuda().float() 
        else:
            kernel = np.ones((2,2),np.float32)/4
            mask_tmp = np.zeros((b,h,w,d),np.float32)
            for i in range(d):
                mask_tmp[:,:,:,i] = cv2.filter2D(maskb_[:,:,:,i],-1, kernel)
            mask_tmp = abs(mask_tmp - maskb_)
            mask_tmp[mask_tmp > 0.005] = 1

        return mask_tmp #mask_tmp          # [1,128,128,128]
    

    def _mask2mask_semantic(self, mask):
        _mask = np.array(mask).astype('float32')
        b,h,w,d = _mask.shape
        mask_tmp = np.zeros((b,3,h,w,d),np.float32) # 4
        mask_tmp[:,0,:,:,:] = (_mask==1)
        mask_tmp[:,1,:,:,:] = (_mask==2)
        mask_tmp[:,2,:,:,:] = (_mask==4)
        
        if self.canny:
            semantic_mask = np.zeros((b,3,h,w,d),np.float32)
            for n in range(3):
                for i in range(d):
                    semantic_mask[:,n,:,:,i] = cv2.Canny(np.uint8(mask_tmp[:,n,:,:,i]), 0, 0.001)
            semantic_mask[semantic_mask > 0] = 1
        else:
            kernel = np.ones((9,9),np.float32)/81
            semantic_mask = np.zeros((b,3,h,w,d),np.float32)
            for n in range(3):
                for i in range(d):
                    semantic_mask[:,n,:,:,i] = cv2.filter2D(mask_tmp[:,n,:,:,i],-1, kernel)
            semantic_mask = abs(semantic_mask - mask_tmp)  # smoothing edge label: (0-1)
            semantic_mask[mask_tmp > 0.005] = 1  # hard edge label: [0,1]

        return semantic_mask       # [1,3,128,128,128]
