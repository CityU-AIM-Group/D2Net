#coding=utf-8
import argparse
import os
import time
import logging
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms

import numpy as np
import models
from data import datasets
from data.sampler import CycleSampler
from data.data_utils import 
from utils import Parser,criterions
from utils import lovasz_loss as lovasz
from predict import validate_softmax, AverageMeter

import sys
import ast
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from models import config
from loss import Losses, Loss_Region
import math
import gc
from data.transforms import *


parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default='DMFNet_GDL_all', type=str,
                    help='Your detailed configuration of the network')
parser.add_argument('--gpu', default='0,1', type=str, required=True,
                    help='Supprot one GPU & multiple GPUs.')
parser.add_argument('--seed', default='1024', type=int)
parser.add_argument('--restore', default='model_last.pth', type=str)

parser.add_argument('--train_data_dir', default='./data/MICCAI_BraTS2018_TrainingData_gz', type=str)
parser.add_argument('--valid_data_dir', default='./data/MICCAI_BraTS2018_ValidationData_gz', type=str)
parser.add_argument('--test_data_dir', default='./data/MICCAI_BraTS2018_TestingData_gz', type=str)
parser.add_argument('--train_list', default=['train_0.txt', 'train_1.txt', 'train_2.txt'], type=list)
parser.add_argument('--train_valid_list', default=['valid_0.txt', 'valid_1.txt', 'valid_2.txt'], type=list)
parser.add_argument('--valid_list', default=['valid.txt'], type=list)

# Training hyper-parameters:
parser.add_argument('--criterion', choices=['sigmoid_dice_loss', 'softmax_dice_loss', 'FocalLoss'], default='sigmoid_dice_loss', type=str) 
parser.add_argument('--num_epochs', default='400', type=float)
parser.add_argument('--valid_freq', default='5', type=float)
parser.add_argument('--save_freq', default='20', type=float)
parser.add_argument('--start_iter', default='0', type=int)
parser.add_argument('--workers', default=8, type=int)

parser.add_argument('--output_set', choices=['train_val','val','test'], default='val', type=str) # [train_val,val,test] as output of submission
parser.add_argument('--batch_size', default=2, type=int, help='Batch size')
parser.add_argument('--opt', default='Adam', type=str)
parser.add_argument('--lr', default='3e-4', type=float) # 1e-3
parser.add_argument('--warmup_epoch', default='20', type=float) # Warm-up for learning rate
parser.add_argument('--weight_decay', default='1e-5', type=float)
parser.add_argument('--amsgrad', default=True, type=bool)
parser.add_argument('--weight_type', default='square', type=str)
parser.add_argument('--eps', default=1e-5, type=float)
parser.add_argument('--dataset', choices=['BraTSDataset'], default='BraTSDataset', type=str)   # RandCrop3D((128,128,128)), \  112,112,112
parser.add_argument('--train_transforms', default='Compose([ \
                                                    RandCrop3D((128,128,128)), \
                                                    RandomRotion(10),  \
                                                    RandomIntensityChange((0.1,0.1)), \
                                                    RandomFlip(0), \
                                                    NumpyType((np.float32, np.int64)), \
                                                    ])', type=str)
                                                    
parser.add_argument('--test_transforms', default='Compose([ \
                                                    Pad((0, 16, 16, 5, 0)), \
                                                    NumpyType((np.float32, np.int64)), \
                                                    ])', type=str)    ### Only if args.net=='U2net' and config.num_pool_per_axis=[5,5,5], Pad-16

parser.add_argument('--setting', default='None', type=str) # summary of all setting you want to record, to save as the name of logs

# Hyper-parameters of the Networks:
parser.add_argument('--net', default='DisenNet', choices=['DMFNet','Unet','U2net3d','DisenNet'], type=str) # name of the used networks
parser.add_argument('--in_channels', default='4', type=int)
parser.add_argument('--channels1', default='32', type=int)
parser.add_argument('--channels2', default='128', type=int) # 128
parser.add_argument('--groups', default='16', type=int) # 16
parser.add_argument('--norm', default='sync_bn', type=str)
parser.add_argument('--num_classes', default='4', type=int)

parser.add_argument('--unet_filter_num_list', default=[8,16,32,48,64], type=list) #ori:[16,32,48,64,96] or small:[8,16,32,48,64]

# Hyper-parameters for U2Net:
parser.add_argument('--use_lovasz', default=False, type=ast.literal_eval) #bool
parser.add_argument('--use_snapshot_ensemble', default=False, type=ast.literal_eval) #bool
parser.add_argument('--smooth_label', default=False, type=ast.literal_eval) #bool
parser.add_argument('--use_focal_loss', default=False, type=ast.literal_eval) #bool
parser.add_argument('--vis_badcase', default=False, type=ast.literal_eval) #bool
parser.add_argument('--R_loss', default=False, type=ast.literal_eval) #bool
parser.add_argument('--loss_balance', default=False, type=ast.literal_eval) #bool
parser.add_argument('--duration', default='20', type=int) #bool
parser.add_argument('--u2net_inchann', default=8, type=int) # Only for U2net3d, the in_channel_num

parser.add_argument('--valid_submission_only', default=False, type=ast.literal_eval) #bool

# Hyper-paremeters for DisenNet:
parser.add_argument('--DisenNet_indim', default=2, type=int) # 2/4
parser.add_argument('--AuxDec_dim', default=2, type=int)     # 2/1
parser.add_argument('--recon_w', default=1, type=float) # [2, 1, 0.5, 0.1]
parser.add_argument('--kl_w', default=1, type=float)    # [2, 1, 0.5, 0.1]
parser.add_argument('--use_distill', default=True, type=ast.literal_eval) #bool
parser.add_argument('--use_contrast', default=True, type=ast.literal_eval) #bool
parser.add_argument('--contrast_w', default=1, type=float)   # [2, 1, 0.5, 0.1]
parser.add_argument('--use_style_map', default=True, type=ast.literal_eval) #bool
parser.add_argument('--style_dim', default=16, type=int) # 8, dim of style vector

# Missing Modality Setting:
parser.add_argument('--miss_modal', default=False, type=ast.literal_eval) #bool
parser.add_argument('--use_Bernoulli_train', default=False, type=ast.literal_eval) #bool
parser.add_argument('--use_kd', default=False, type=ast.literal_eval) # Use knowledge distillation (Fea+Logit KD)
parser.add_argument('--fea_dim', default='8', type=int) #  dim of feature of Decoder to be distilled
parser.add_argument('--kd_logit_w', default=1, type=float)  # [30, 10, 1, 0.1]
parser.add_argument('--kd_fea_w', default=1, type=float)    # [2, 1, 0.5, 0.1]
parser.add_argument('--kd_fea_channel_w', default=1, type=float)   # # [2, 1, 0.5, 0.1]
parser.add_argument('--kd_channel_attn', default=False, type=ast.literal_eval) #bool  Like the paper by chunhua shen
parser.add_argument('--kd_dense_fea_attn', default=False, type=ast.literal_eval) #bool  our novel calculation on each feature map's dense channel
parser.add_argument('--affinity_kd', default=False, type=ast.literal_eval) #bool the affinity kd loss
parser.add_argument('--self_distill', default=False, type=ast.literal_eval) #bool
parser.add_argument('--self_distill_logit_w', default=1, type=float)  # [5, 1, 0.5, 0.1]
parser.add_argument('--self_distill_fea_w', default=1, type=float)  # [2, 1, 0.5, 0.1]

parser.add_argument('--use_freq_map', default=False, type=ast.literal_eval) # Frequency
parser.add_argument('--use_freq_channel', default=False, type=ast.literal_eval) # Frequency, band-pass filter
parser.add_argument('--freq_w', default=1, type=float)  # # [2, 1, 0.5, 0.1]
parser.add_argument('--use_freq_contrast', default=False, type=ast.literal_eval) # Frequency (simple), as part of constrastive loss

parser.add_argument('--saveroot', default='./fig/seg_result_save', type=str)# root_path of saving logs

path = os.path.dirname(__file__)
args = parser.parse_args()
args = Parser(args.cfg, log='train').add_args(args)
ckpts = args.makedir()
args.resume = None

val_log_savepath = os.path.join(args.saveroot,'trainval_log_'+args.net+'_'+args.criterion+'_'+args.output_set+'_output_'+args.setting+'.txt')
val_submission_savepath = os.path.join(args.saveroot,'submission/'+args.net+'_'+args.criterion+'_'+args.output_set+'_output_'+args.setting)
val_model_savepath = './ckpts/'+args.net


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    Network = getattr(models, args.net)

    if args.net == 'DMFNet':
        model = Network(in_channels=args.in_channels, channels1=args.channels1, channels2=args.channels2, groups=args.groups, 
                        norm=args.norm, num_classes=args.num_classes)
    elif args.net == 'Unet':
        model = Network(filter_num_list=args.unet_filter_num_list, \
                        )
    elif args.net == 'U2net3d':
        model = Network(inChans_list=[4], base_outChans=args.u2net_inchann, num_class_list=[4], args=args)
    elif args.net == 'DisenNet':
        model = Network(base_outChans=args.DisenNet_indim, args=args)
    else:
        print ('Error: This network has not been implemented!')

    model = torch.nn.DataParallel(model).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr*args.batch_size, weight_decay=args.weight_decay) # 
    criterion = getattr(criterions, args.criterion)
    

    if args.dataset == 'BraTSDataset':
        num_fold = 3

    for cv_idx in range(num_fold):  # Cross-Validation with 3-fold
        if args.net == 'U2net3d':
            if args.dataset == 'BraTSDataset':
                model = Network(inChans_list=[4], base_outChans=args.u2net_inchann, num_class_list=[4], args=args)
        elif args.net == 'DisenNet':
            if args.dataset == 'BraTSDataset':
                model = Network(inChans_list=[4], base_outChans=args.DisenNet_indim, num_class_list=[4], args=args)

        model = torch.nn.DataParallel(model).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr*args.batch_size, weight_decay=args.weight_decay) # 
        
        with open(val_log_savepath,'a') as f:
            f.write('Start time is: %s \n' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
            f.write('---------------------Structure of Network------------------------\n')
            f.write(str(model))
            f.write('\n') 
            f.write('---------------------Setting of the model------------------------\n')
            f.write(str(args))
            f.write('\n') 
            f.write('---------------------------------------------------------------\n')

        msg = ' \n'


        if args.resume and args.valid_submission_only:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                args.start_iter = checkpoint['iter']

                net_dict = model.state_dict()
                pretrain_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in net_dict.keys()}
                net_dict.update(pretrain_dict)
                model.load_state_dict(net_dict)

                msg = ("=> loaded checkpoint '{}' (iter {})"
                    .format(args.resume, checkpoint['iter']))
            else:
                msg = "=> no checkpoint found at '{}'".format(args.resume)
        else:
            msg = '-------------- New training session ----------------\n'

        msg += str(args)
        logging.info(msg)

        Dataset = getattr(datasets, args.dataset) #
        para_num = count_parameters(model)
        print ('The number of parameters of the model are: %f' % para_num)


        print ('-----------The %d-th iterations of the Cross-Validation training starts--------\n'% (cv_idx+1))
        if args.dataset == 'BraTSDataset':
            train_list = os.path.join('./data/MICCAI_BraTS2018_txt/train/', args.train_list[cv_idx])
            train_set = Dataset(train_list, root=args.train_data_dir, for_train=True,
              transforms=args.train_transforms)

        print ('Length of training sets:')
        print (len(train_set))
        num_iters = (len(train_set) * args.num_epochs) // args.batch_size
        num_iters -= args.start_iter
        train_sampler = CycleSampler(len(train_set), num_iters*args.batch_size)
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            collate_fn=train_set.collate, sampler=train_sampler,
            num_workers=args.workers, pin_memory=True, worker_init_fn=init_fn)
        

        if args.train_valid_list:
            if args.dataset == 'BraTSDataset':
                train_valid_list = os.path.join('./data/MICCAI_BraTS2018_txt/train/', args.train_valid_list[cv_idx])
                train_valid_set = Dataset(train_valid_list,
                                root=args.train_data_dir,
                                for_train=False,
                                transforms=args.test_transforms)
            train_valid_loader = DataLoader(
                train_valid_set,
                batch_size=1,
                shuffle=False,
                collate_fn=train_valid_set.collate,
                num_workers=4,
                pin_memory=True)
            valid_loader = None
            valid_loader = train_valid_loader

        if args.valid_list and args.dataset == 'BraTSDataset':
            train_valid_list = os.path.join('./data/MICCAI_BraTS2018_txt/train', args.train_valid_list[cv_idx])
            valid_list = './data/MICCAI_BraTS2018_txt/valid/valid.txt'
            test_list = './data/MICCAI_BraTS2018_txt/test/test.txt'
            if args.output_set == 'train_val':
                data_list = train_valid_list
                input_data_dir = args.train_data_dir
            if args.output_set == 'val':
                data_list = valid_list
                input_data_dir = args.valid_data_dir
            if args.output_set == 'test':
                data_list = test_list
                input_data_dir = args.test_data_dir
            
            valid_set = Dataset(data_list, # [train_valid_list, test_list, valid_list]
                            root=input_data_dir, # [train_data_dir, valid_data_dir, test_data_dir]
                            for_train=False,
                            transforms=args.test_transforms, true_valid_data=True) 
            valid_loader = DataLoader(
                valid_set,
                batch_size=1,
                shuffle=False,
                collate_fn=valid_set.collate,
                num_workers=4,
                pin_memory=True)
        
        start = time.time()

        enum_batches = len(train_set)/float(args.batch_size) # nums_batch per epoch
        args.schedule   = {int(k*enum_batches): v for k, v in args.schedule.items()} # 17100
        args.save_freq  = int(enum_batches * args.save_freq)

        losses = AverageMeter()
        torch.set_grad_enabled(True)
        avg_cost = np.zeros([int(num_iters//args.duration)+1, 2])  # duration=30, For balancing two losses of edge and seg branch

        for i, data in enumerate(train_loader, args.start_iter):
            gc.collect()
            torch.cuda.empty_cache()
            elapsed_bsize = int( i / enum_batches)+1
            epoch = int((i + 1) / enum_batches)
          
            if args.use_snapshot_ensemble:
                snapshot_ensemble_lr(optimizer,epoch)
            else:
                adjust_learning_rate(optimizer, epoch, args.num_epochs, args.lr*args.batch_size, warmup_epoch=args.warmup_epoch, lr_min=1e-8, _type='exp')  # Or original: 'exp' or 'CosineAnnealing'
              
            data = [t.cuda(non_blocking=True) for t in data]
            x, target = data[:2]
            edge_label = None

            if args.net == 'U2net3d':
                if config.trainMode in ["universal"]:
                    output, share_map, para_map, deep_sup_fea = model(x)
                    if config.deep_supervision:
                        feature_maps = [deep_sup_fea[i1] for i1 in range(config.num_pool_per_axis[0]-2)]
                        feature_maps.append(output)
                    else:
                        feature_maps = [output]
                else:
                    output = model(x)

                total_seg_loss = 0.0
                weights = np.array([1 / (2 ** i2) for i2 in range(4)]) # weights=[1/2, 1/4, 1/8, 1/16] according to the size of 4 feature maps
                weights = weights / weights.sum()
                for iii, feature_map in enumerate(feature_maps):
                    b,c,h,w,d = feature_map.shape
                    b1,h1,w1,d1 = target.shape
                    if d != d1:
                        _target = nn.MaxPool3d(int(h1/h), stride=int(h1/h))(target.float())
                        _target = _target.long()
                    else:
                        _target = target

                    _loss = Losses()
                    ori_loss = _loss(feature_map, _target, datasets=args.dataset, use_dice=True, ce=True, focal=True, iou=False, use_lovasz=False, use_TverskyLoss=False)
                    if args.R_loss:
                        _region_loss = Loss_Region()
                        region_loss = _region_loss(feature_map, _target, use_dice=True, bce=True)
                        sum_loss = ori_loss + region_loss
                    else:
                        sum_loss = ori_loss
                    total_seg_loss += sum_loss * weights[-1-iii]
                
                loss = total_seg_loss
                print ('Total loss: %f'%loss.item())
                 
            elif args.net == 'DisenNet':                
                if args.miss_modal == True and args.use_Bernoulli_train == True:
                    random_miss = np.random.binomial(n=1,p=0.5,size=4)
                    miss_list = [x[:,i,...]*random_miss[i] for i in range(4)]
                    
                    # reconstruction for all modalities MRI:
                    complete_x = x
                    x = torch.cat([torch.unsqueeze(miss_list[0],1),torch.unsqueeze(miss_list[1],1),torch.unsqueeze(miss_list[2],1),torch.unsqueeze(miss_list[3],1)],1)

                seg_out, binary_seg_out_all, deep_sup_fea, weight_recon_loss, weight_kl_loss, weight_recon_c_loss, weight_recon_s_loss, distill_loss, kd_loss, contrastive_loss, freq_loss, seg_aux = model(x,complete_x,is_test=False)  ### deep_sup_fea: [bs,4,16/32/64,...], len=3
                
                if config.deep_supervision:
                    feature_maps = [deep_sup_fea[i1] for i1 in range(0,config.num_pool_per_axis[0]-2)] # 1
                    feature_maps.append(binary_seg_out_all)
                    feature_maps.append(seg_aux)
                    feature_maps.append(seg_out)
                else:
                    feature_maps = [seg_aux]
                    feature_maps.append(binary_seg_out_all)
                    feature_maps.append(seg_out)
                total_seg_loss = 0.0
                
                if config.deep_supervision:
                    weights = np.array([1,0.5,0.5])
                    weights = np.append(weights,[1 / (2 ** (i2+1)) for i2 in range(config.num_pool_per_axis[0]-2-1, -1, -1)])
                    weights = weights / weights.sum()
                else:
                    weights = np.array([1,0.5,0.5])
                    weights = weights / weights.sum()
              

                for iii, feature_map in enumerate(feature_maps):
                    b,c,h,w,d = feature_map.shape
                    b1,h1,w1,d1 = target.shape
                    if d != d1:
                        _target = nn.MaxPool3d(int(h1/h), stride=int(h1/h))(target.float())
                        _target = _target.long()
                    else:
                        _target = target

                    _loss = Losses()
                    ori_loss = _loss(feature_map, _target, datasets=args.dataset, use_dice=True, ce=True, focal=False, iou=False, use_lovasz=False, use_TverskyLoss=False)
                    if args.R_loss:
                        _region_loss = Loss_Region()
                        region_loss = _region_loss(feature_map, _target, use_dice=True, bce=True)
                        sum_loss = ori_loss + region_loss
                    else:
                        sum_loss = ori_loss
                        print ('Aux/Binary/Main Seg loss: %f'%((sum_loss.item() * weights[-1-iii])))
                    total_seg_loss += sum_loss * weights[-1-iii]

                kd_logit_loss = kd_loss[0]
                kd_fea_loss = kd_loss[1]
                
                loss = total_seg_loss + weight_recon_loss.mean() + weight_kl_loss.mean() + distill_loss.mean() + kd_logit_loss.mean() + kd_fea_loss.mean() + contrastive_loss.mean() + freq_loss.mean()
                print ('Seg loss: %f, Recon loss: %f, KL loss: %f, Distill loss: %f, KD_logit_loss: %f, KD_fea_loss: %f, Contrast loss: %f, Freq loss: %f, Total loss: %f'%(total_seg_loss.item(),weight_recon_loss.mean().item(),weight_kl_loss.mean().item(), distill_loss.mean().item(), kd_logit_loss.mean().item(), kd_fea_loss.mean().item(), contrastive_loss.mean().item(),freq_loss.mean(),loss.item()))
            
            else:
                output = model(x)
                _loss = Losses()
                loss = _loss(output, target, use_dice=True, ce=True, focal=True, iou=False, use_lovasz=False, use_TverskyLoss=False)
                print ('Loss is: %f'%loss.item())
                

            if not args.weight_type: # compatible for the old version
                args.weight_type = 'square'

            # measure accuracy and record loss
            losses.update(loss.item(), target.numel())

            # compute gradient and do SGD step
            if args.valid_submission_only:
                loss = loss * 0
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            # validation and save, when get the best sum(dice-score) among before
            if epoch < 300:
                if args.dataset == 'BraTSDataset':
                    _valid_freq = args.valid_freq * 4
                else:
                    _valid_freq = args.valid_freq
            else:
                _valid_freq = args.valid_freq
            
            if  (i+1) % int(enum_batches * _valid_freq) == 0\
                or (i+1) % int(enum_batches * (args.num_epochs -1))==0\
                or (i+1) % int(enum_batches * (args.num_epochs -2))==0\
                or (i+1) % int(enum_batches * (args.num_epochs -3))==0\
                or (i+1) % int(enum_batches * (args.num_epochs -4))==0:

                logging.info('-'*50)
                msg  =  'Validation: Iter {}, Epoch {:.4f}, {}'.format(i, i/enum_batches, 'validation')
                logging.info(msg)

                del data, target, loss
                torch.cuda.empty_cache()


                with torch.no_grad():
                    if (i+1) // int(enum_batches * _valid_freq) == 1 or args.valid_submission_only == True:
                        torch.cuda.empty_cache()
                        best_dice_score_sum = 0.0
                        vals_dice_scores = validate_softmax(
                                        train_valid_loader,
                                        valid_loader,
                                        model, #cpu_model,
                                        net=args.net,
                                        args=args, ##
                                        log_savepath=val_log_savepath, #args.savepath,
                                        submission_savepath=val_submission_savepath,
                                        names=train_valid_set.names,
                                        scoring=True,
                                        verbose=False, #False
                                        use_TTA=True, #False,
                                        save_format='nii',
                                        snapshot=False,  # False
                                        postprocess=True, #False,
                                        cpu_only=False,
                                        epoch_id=int((i + 1) / enum_batches),
                                        best_dice_score_sum=0.0
                                        )
                    else:
                        if vals_dice_scores.sum() > best_dice_score_sum:
                            best_dice_score_sum = vals_dice_scores.sum()

                        vals_dice_scores = validate_softmax(
                                        train_valid_loader,
                                        valid_loader,
                                        model,
                                        net=args.net,
                                        args=args,
                                        log_savepath=val_log_savepath, # args.savepath,
                                        submission_savepath=val_submission_savepath,
                                        names=train_valid_set.names,
                                        scoring=True,
                                        verbose=False, # False
                                        use_TTA=True, # False,
                                        save_format='nii',
                                        snapshot=False,  # False
                                        postprocess=True, # False,
                                        cpu_only=False,
                                        epoch_id=int(i/enum_batches),
                                        best_dice_score_sum=best_dice_score_sum # vals_dice_scores.sum()
                                        )

                        if vals_dice_scores.sum() > best_dice_score_sum and int(i/enum_batches) > 50: # vals_dice_scores.mean() > 0.83: #2.0:
                            file_name = os.path.join(val_model_savepath, args.setting+'_'+str(vals_dice_scores.mean())+'_model_epoch_{}.pth'.format(epoch))
                            torch.save({
                                'iter': i+1,
                                'state_dict': model.state_dict(),
                                'optim_dict': optimizer.state_dict(),
                                },
                                file_name)
            if i % 100 == 0:
                msg = 'Iter {0:}, Epoch {1:.4f}, Loss {2:.7f}'.format(
                        i, (i+1)/enum_batches, losses.avg)
                logging.info(msg)
            losses.reset()

        i = num_iters + args.start_iter
        file_name = os.path.join(val_model_savepath, args.setting+'model_last.pth')
        torch.save({
            'iter': i,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            },
            file_name)

        msg = 'total time: {:.4f} minutes'.format((time.time() - start)/60)
        logging.info(msg)


def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9, warmup_epoch=20, lr_min=1e-8, _type='exp', use_warmup=False): 
    lr_max = INIT_LR
    
    if use_warmup:
        if _type == 'exp':
            if epoch >= warmup_epoch:
                lr = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)
            else:
                lr = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8) * epoch / warmup_epoch
        else:
            if epoch >= warmup_epoch:
                lr = lr_min + (lr_max - lr_min) * (1 + math.cos(math.pi * epoch / MAX_EPOCHES)) / 2       # Cosine Annealing
            else:
                lr = lr_min + (lr_max - lr_min) * (1 + math.cos(math.pi * epoch / MAX_EPOCHES)) / 2 * epoch / warmup_epoch
    else:
        if _type == 'exp':
            lr = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)
        else:
            lr = lr_min + (lr_max - lr_min) * (1 + math.cos(math.pi * epoch / MAX_EPOCHES)) / 2

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def snapshot_ensemble_lr(optimizer, epoch, CYCLE=4, LR_INIT=0.001, LR_MIN=0.0001):
    scheduler = lambda x: ((LR_INIT-LR_MIN)/2)*(np.cos(np.pi*(np.mod(x-1,CYCLE)/(CYCLE)))+1)+LR_MIN
    for param_group in optimizer.param_groups:
        param_group['lr'] = scheduler(epoch)


def bce_loss(prediction, label, smooth_label=False):
    label = label.long()
    mask = label.float()
    
    if smooth_label:
        num_positive = torch.sum((mask!=0).float()).float()
        num_negative = torch.sum((mask==0).float()).float()
        mask[mask > 0] = 1.0 * num_negative / (num_positive + num_negative)
        mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
    else:
        num_positive = torch.sum((mask==1).float()).float()
        num_negative = torch.sum((mask==0).float()).float()
        mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
        mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
        mask[mask == 2] = 0
    cost = torch.nn.functional.binary_cross_entropy(prediction.float(),label.float(), weight=mask, reduce=False)
    
    return torch.sum(cost) / (num_positive+1e-6)


def weighted_mse_loss(prediction, label, smooth_label=True):
    mask = label.float()
    num_positive = torch.sum((mask!=0).float()).float()
    num_negative = torch.sum((mask==0).float()).float()
    mask[mask != 0] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
    loss = torch.nn.functional.mse_loss(prediction.float(),label.float(),reduce=False, size_average=False)
    
    return torch.sum(loss) / (num_positive+1e-4)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    main()
