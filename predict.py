import os
import time
import logging
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import sys
import nibabel as nib
import scipy.misc
import re

import torchvision
from medpy.metric import dc, hd95

cudnn.benchmark = True
path = os.path.dirname(__file__)

def compute_BraTS_HD95(ref, pred):
    """
    ref and gt are binary integer numpy.ndarray s
    spacing is assumed to be (1, 1, 1)
    :param ref:
    :param pred:
    :return:
    """
    num_ref = np.sum(ref)
    num_pred = np.sum(pred)
    if num_ref == 0:
        if num_pred == 0:
            return 0
        else:
            return 373.12866
    elif num_pred == 0 and num_ref != 0:
        return 373.12866
    else:
        return hd95(pred, ref, (1, 1, 1))

def cal_hd95(output, target):
     # whole tumor
    mask_gt = (target != 0).astype(int)
    mask_pred = (output != 0).astype(int)
    hd95_whole = compute_BraTS_HD95(mask_gt, mask_pred)
    del mask_gt, mask_pred

    # tumor core
    mask_gt = (target > 1).astype(int)
    mask_pred = (output > 1).astype(int)
    hd95_core = compute_BraTS_HD95(mask_gt, mask_pred)
    del mask_gt, mask_pred

    # enhancing
    mask_gt = (target == 4).astype(int)
    mask_pred = (output == 3).astype(int)
    hd95_enh = compute_BraTS_HD95(mask_gt, mask_pred)
    del mask_gt, mask_pred

    return hd95_whole, hd95_core, hd95_enh

# dice socre is equal to f1 score
def dice_score(o, t,eps = 1e-8):
    num = 2*(o*t).sum() + eps #
    den = o.sum() + t.sum() + eps # eps
    return num/den

def prostate_dice(output, target, ignore_pixel=255):
    ret = []
    # whole
    if (target==ignore_pixel).sum() > 0:
        target[target==ignore_pixel] = 0
    o = output > 0; t = (target > 0)
    ret += dice_score(o, t),
    # 1
    o = (output==1) 
    t = (target==1)
    ret += dice_score(o , t),
    # 2
    o = (output==2); t = (target==2)
    ret += dice_score(o , t),
    return ret

def softmax_output_dice(output, target):
    ret = []
    # whole
    o = output > 0; t = target > 0 # ce
    ret += dice_score(o, t),
    # core
    o = (output==1) | (output==3)
    t = (target==1) | (target==4)
    ret += dice_score(o , t),
    # active
    o = (output==3); t = (target==4)
    ret += dice_score(o , t),
    return ret

keys = 'WT', 'TC', 'ET', 'loss'
keys_hd95 = 'WT', 'TC', 'ET'

def validate_softmax(
        train_valid_loader,
        valid_loader,
        model,
        net='',
        args=None, 
        log_savepath='', # when in validation set, you must specify the path to save the 'nii' segmentation results here
        submission_savepath='',
        names=None, # The names of the patients orderly!
        scoring=True, # If true, print the dice score.
        verbose=False,
        use_TTA=False, # Test time augmentation, False as default!
        save_format=None, # ['nii','npy'], use 'nii' as default. Its purpose is for submission.
        snapshot=False, # for visualization. Default false. It is recommended to generate the visualized figures.
        postprocess=False, # Defualt False, when use postprocess, the score of dice_ET would be changed.
        cpu_only=False,
        epoch_id=False,
        best_dice_score_sum=0.0
        ):

    # assert cfg is not None
    H, W, T = 240, 240, 155
    model.eval()
    runtimes = []
    vals = AverageMeter()
    vals_hd95 = AverageMeter()
    
    vals_miss1 = AverageMeter()
    vals_miss2 = AverageMeter()
    vals_miss3 = AverageMeter()
    vals_miss4 = AverageMeter()
    vals_miss5 = AverageMeter()
    vals_miss6 = AverageMeter()
    vals_miss7 = AverageMeter()
    vals_miss8 = AverageMeter()
    vals_miss9 = AverageMeter()
    vals_miss10 = AverageMeter()
    vals_miss11 = AverageMeter()
    vals_miss12 = AverageMeter()
    vals_miss13 = AverageMeter()
    vals_miss14 = AverageMeter()
    vals_miss15 = AverageMeter()

    if not args.valid_submission_only:
        for i, data in enumerate(train_valid_loader):
            if args.net == 'ClsTransformer' or args.net == 'T2t_vit' or args.dataset == 'ProstateDataset' or args.dataset == 'ProstateDataset2D':
                target_cpu = data[1][0].numpy() if scoring else None 
            else:
                target_cpu = data[1][0, :H, :W, :T].numpy() if scoring else None # when validing, make sure that argument 'scoring' must be false, else it raise a error!
            
            if cpu_only == False:
                data = [t.cuda(non_blocking=True) for t in data]
            
            x, target = data[:2]

            # compute output
            if not use_TTA:
                start_time = time.time()
                if args.net == 'Unet':
                    logit, _ = model(x)
                else:
                    logit = model(x)
                
                elapsed_time = time.time() - start_time
                runtimes.append(elapsed_time)
                output = F.softmax(logit,dim=1)

            else:
                if args.net == 'Unet' or args.net == 'U2net3d' or args.net == 'DisenNet':
                    if args.miss_modal == True:  # Modality order：[x_flair, x_t1ce，x_t1, x_t2]
                        mri_full = x

                        mri_tmp = x.clone()
                        mri_tmp[:,0,...] = 0.0
                        mri_missF = mri_tmp

                        mri_tmp = x.clone()
                        mri_tmp[:,1,...] = 0.0
                        mri_missT1ce = mri_tmp

                        mri_tmp = x.clone()
                        mri_tmp[:,2,...] = 0.0
                        mri_missT1 = mri_tmp

                        mri_tmp = x.clone()
                        mri_tmp[:,3,...] = 0.0
                        mri_missT2 = mri_tmp

                        mri_tmp = x.clone()
                        mri_tmp[:,0,...] = 0.0
                        mri_tmp[:,1,...] = 0.0
                        mri_missF_T1ce = mri_tmp

                        mri_tmp = x.clone()
                        mri_tmp[:,0,...] = 0.0
                        mri_tmp[:,2,...] = 0.0
                        mri_missF_T1 = mri_tmp

                        mri_tmp = x.clone()
                        mri_tmp[:,0,...] = 0.0
                        mri_tmp[:,3,...] = 0.0
                        mri_missF_T2 = mri_tmp

                        mri_tmp = x.clone()
                        mri_tmp[:,1,...] = 0.0
                        mri_tmp[:,2,...] = 0.0
                        mri_missT1ce_T1 = mri_tmp

                        mri_tmp = x.clone()
                        mri_tmp[:,1,...] = 0.0
                        mri_tmp[:,3,...] = 0.0
                        mri_missT1ce_T2 = mri_tmp

                        mri_tmp = x.clone()
                        mri_tmp[:,2,...] = 0.0
                        mri_tmp[:,3,...] = 0.0
                        mri_missT1_T2 = mri_tmp

                        mri_tmp = x.clone()
                        mri_tmp[:,0,...] = 0.0
                        mri_tmp[:,1,...] = 0.0
                        mri_tmp[:,2,...] = 0.0
                        mri_missF_T1ce_T1 = mri_tmp

                        mri_tmp = x.clone()
                        mri_tmp[:,0,...] = 0.0
                        mri_tmp[:,1,...] = 0.0
                        mri_tmp[:,3,...] = 0.0
                        mri_missF_T1ce_T2 = mri_tmp

                        mri_tmp = x.clone()
                        mri_tmp[:,0,...] = 0.0
                        mri_tmp[:,2,...] = 0.0
                        mri_tmp[:,3,...] = 0.0
                        mri_missF_T1_T2 = mri_tmp

                        mri_tmp = x.clone()
                        mri_tmp[:,1,...] = 0.0
                        mri_tmp[:,2,...] = 0.0
                        mri_tmp[:,3,...] = 0.0
                        mri_missT1ce_T1_T2 = mri_tmp

                        mri = [mri_full, mri_missF, mri_missT1ce, mri_missT1, mri_missT2, mri_missF_T1ce, mri_missF_T1, mri_missF_T2,
                               mri_missT1ce_T1, mri_missT1ce_T2, mri_missT1_T2, mri_missF_T1ce_T1, mri_missF_T1ce_T2, mri_missF_T1_T2,
                               mri_missT1ce_T1_T2]
                        output = []

                        for idx in range(15):
                            logit = F.softmax(model(mri[idx])[0] ,1)
                            logit += F.softmax(model(mri[idx].flip(dims=(2,)))[0].flip(dims=(2,)),1)
                            logit += F.softmax(model(mri[idx].flip(dims=(3,)))[0].flip(dims=(3,) ),1)
                            logit += F.softmax(model(mri[idx].flip(dims=(4,)))[0].flip(dims=(4,)),1)
                            logit += F.softmax(model(mri[idx].flip(dims=(2,3)))[0].flip(dims=(2,3) ),1)
                            logit += F.softmax(model(mri[idx].flip(dims=(2,4)))[0].flip(dims=(2,4)),1)
                            logit += F.softmax(model(mri[idx].flip(dims=(3,4)))[0].flip(dims=(3,4)),1)
                            logit += F.softmax(model(mri[idx].flip(dims=(2,3,4)))[0].flip(dims=(2,3,4)),1)
                            output.append(logit / 8.0) # mean
                        del mri, mri_full, mri_missF, mri_missT1ce, mri_missT1, mri_missT2, mri_missF_T1ce, mri_missF_T1, mri_missF_T2, \
                            mri_missT1ce_T1, mri_missT1ce_T2, mri_missT1_T2, mri_missF_T1ce_T1, mri_missF_T1ce_T2, mri_missF_T1_T2, \
                            mri_missT1ce_T1_T2

                    else:
                        logit = F.softmax(model(x)[0] ,1)
                        logit += F.softmax(model(x.flip(dims=(2,)))[0].flip(dims=(2,)),1)
                        logit += F.softmax(model(x.flip(dims=(3,)))[0].flip(dims=(3,) ),1)
                        logit += F.softmax(model(x.flip(dims=(4,)))[0].flip(dims=(4,)),1)
                        logit += F.softmax(model(x.flip(dims=(2,3)))[0].flip(dims=(2,3) ),1)
                        logit += F.softmax(model(x.flip(dims=(2,4)))[0].flip(dims=(2,4)),1)
                        logit += F.softmax(model(x.flip(dims=(3,4)))[0].flip(dims=(3,4)),1)
                        logit += F.softmax(model(x.flip(dims=(2,3,4)))[0].flip(dims=(2,3,4)),1)
                        output = logit / 8.0 # mean
                else:
                    logit = F.softmax(model(x) ,1)
                    logit += F.softmax(model(x.flip(dims=(2,))).flip(dims=(2,)),1)
                    logit += F.softmax(model(x.flip(dims=(3,))).flip(dims=(3,) ),1)
                    logit += F.softmax(model(x.flip(dims=(4,))).flip(dims=(4,)),1)
                    logit += F.softmax(model(x.flip(dims=(2,3))).flip(dims=(2,3) ),1)
                    logit += F.softmax(model(x.flip(dims=(2,4))).flip(dims=(2,4)),1)
                    logit += F.softmax(model(x.flip(dims=(3,4))).flip(dims=(3,4)),1)
                    logit += F.softmax(model(x.flip(dims=(2,3,4))).flip(dims=(2,3,4)),1)
                    output = logit / 8.0 # mean
            
            if args.miss_modal:
                for _i in range(len(output)):
                    output[_i] = output[_i][0, :, :H, :W, :T].cpu().numpy()
                    output[_i] = output[_i].argmax(0) # (channels,height,width,depth)
            else:
                output = output[0, :, :H, :W, :T].cpu().numpy()
                output = output.argmax(0) # (channels,height,width,depth)

            if postprocess == True:
                if args.miss_modal:
                    for _i in range(len(output)):
                        ET_voxels = (output[_i] == 3).sum()
                        if ET_voxels < 500:
                            output[_i][np.where(output[_i] == 3)] = 1
                else:
                    ET_voxels = (output == 3).sum()
                    if ET_voxels < 500:
                        output[np.where(output == 3)] = 1

            msg = 'Subject {}/{}, '.format(i+1, len(train_valid_loader))
            name = str(i)
            if names:
                name = names[i]
                msg += '{:>20}, '.format(name)
        
            if scoring:
                if args.dataset == 'BraTSDataset':
                    if args.miss_modal:
                        keys = 'WT', 'TC', 'ET', 'loss'
                        keys_miss = ['mri_full', 'mri_missF', 'mri_missT1ce', 'mri_missT1', 'mri_missT2', 'mri_missF_T1ce', 'mri_missF_T1', 'mri_missF_T2',
                                    'mri_missT1ce_T1', 'mri_missT1ce_T2', 'mri_missT1_T2', 'mri_missF_T1ce_T1', 'mri_missF_T1ce_T2', 'mri_missF_T1_T2',
                                    'mri_missT1ce_T1_T2']
                        vals_miss = [vals_miss1,vals_miss2,vals_miss3,vals_miss4,vals_miss5,vals_miss6,vals_miss7,vals_miss8,vals_miss9,
                                    vals_miss10,vals_miss11,vals_miss12,vals_miss13,vals_miss14,vals_miss15]
                        for _i in range(len(output)):
                            scores = softmax_output_dice(output[_i], target_cpu)
                            vals_miss[_i].update(np.array(scores))
                            msg += ' | ' + keys_miss[_i] + ': '
                            msg += 'Dice Score: '
                            msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(keys, scores)])
                    
                    else:
                        keys = 'WT', 'TC', 'ET', 'loss'
                        scores = softmax_output_dice(output, target_cpu)
                        vals.update(np.array(scores))
                        msg += 'Dice Score: '
                        msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(keys, scores)])

                        msg += ' | HD95 Score: '
                        hd95_score = cal_hd95(output, target_cpu)
                        vals_hd95.update(np.array(hd95_score))
                        msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(keys_hd95, hd95_score)])

                if snapshot:
                    # red: (255,0,0) green:(0,255,0) blue:(0,0,255) 1 for NCR & NET, 2 for ED, 4 for ET, and 0 for everything else.
                    gap_width = 2 # boundary width = 2
                    Snapshot_img = np.zeros(shape=(H, W*2+gap_width,3,T), dtype=np.uint8)
                    Snapshot_img[:,W:W+gap_width,:] = 255 # white boundary

                    empty_fig = np.zeros(shape=(H, W, T), dtype=np.uint8)
                    empty_fig[np.where(output == 1)] = 255
                    Snapshot_img[:,:W,0,:] = empty_fig
                    empty_fig = np.zeros(shape=(H, W, T), dtype=np.uint8)
                    empty_fig[np.where(target_cpu == 1)] = 255
                    Snapshot_img[:, W+gap_width:, 0, :] = empty_fig

                    empty_fig = np.zeros(shape=(H, W, T), dtype=np.uint8)
                    empty_fig[np.where(output == 2)] = 255
                    Snapshot_img[:,:W,1,:] = empty_fig
                    empty_fig = np.zeros(shape=(H, W, T), dtype=np.uint8)
                    empty_fig[np.where(target_cpu == 2)] = 255
                    Snapshot_img[:, W+gap_width:, 1, :] = empty_fig

                    empty_fig = np.zeros(shape=(H, W, T), dtype=np.uint8)
                    empty_fig[np.where(output == 3)] = 255
                    Snapshot_img[:,:W,2,:] = empty_fig
                    empty_fig = np.zeros(shape=(H, W, T), dtype=np.uint8)
                    empty_fig[np.where(target_cpu == 4)] = 255
                    Snapshot_img[:, W+gap_width:,2, :] = empty_fig

                    for frame in range(T):
                        os.makedirs(os.path.join( 'snapshot',net, name), exist_ok=True)
                        scipy.misc.imsave(os.path.join('snapshot',net, name, str(frame) + '.png'), Snapshot_img[:,:,:,frame])
            logging.info(msg)

        if scoring:
            if args.dataset == 'BraTSDataset':
                if args.miss_modal:
                    keys = 'WT', 'TC', 'ET', 'loss'
                    keys_miss = ['mri_full', 'mri_missF', 'mri_missT1ce', 'mri_missT1', 'mri_missT2', 'mri_missF_T1ce', 'mri_missF_T1', 'mri_missF_T2',
                                'mri_missT1ce_T1', 'mri_missT1ce_T2', 'mri_missT1_T2', 'mri_missF_T1ce_T1', 'mri_missF_T1ce_T2', 'mri_missF_T1_T2',
                                'mri_missT1ce_T1_T2']
                    
                    msg = str(epoch_id+1)+': '
                    msg += 'Average scores:' + str(np.mean([vals_miss[i].avg for i in range(len(vals_miss))],0)) + '(/' + str(len(keys_miss)) + ')'
                    for _i in range(len(vals_miss)):
                        msg += ' | ' + keys_miss[_i] + ': '
                        msg += ', '.join(['{}: {:.5f}'.format(k, v) for k, v in zip(keys, vals_miss[_i].avg)])

                    logging.info(msg)
                    with open(log_savepath,'a') as f:
                        f.write(str(msg)) 
                        f.write('\n')  
                    
                else:
                    msg = str(epoch_id+1)+': '
                    msg += 'Average scores:'
                    msg += ', '.join(['{}: {:.5f}'.format(k, v) for k, v in zip(keys, vals.avg)])

                    msg += ' | Average HD95 scores:' 
                    msg += ', '.join(['{}: {:.5f}'.format(k, v) for k, v in zip(keys_hd95, vals_hd95.avg)])
                    logging.info(msg)
                    with open(log_savepath,'a') as f:
                        f.write(str(msg)) 
                        f.write('\n')  
            
            if args.miss_modal == True:
                cal_validation = False
            else:
                if vals.avg.sum() > best_dice_score_sum and best_dice_score_sum > 0 and vals.avg.mean() > 0.84 and epoch_id>250: #3.0:  # 2.0, 3.0
                    cal_validation = True 
                else:
                    cal_validation = False

    # computational_runtime(runtimes)
    if args.valid_submission_only:
        cal_validation = True

    if valid_loader and submission_savepath and cal_validation and args.dataset == 'BraTSDataset': # cal the true validation-set and save the predictions as submission:
        if args.valid_submission_only:
            submission_savepath = submission_savepath + '_' + args.resume.split('/')[-1].split('_')[-4] + '_' + args.setting.split('_')[-2]
        else:
            submission_savepath = submission_savepath + '_' + str(vals.avg.mean())
        for i, data in enumerate(valid_loader):
            torch.cuda.empty_cache() 
            target_cpu = data[1][0, :H, :W, :T].numpy() if not scoring else None # when validing, make sure that argument 'scoring' must be false, else it raise a error!

            if cpu_only == False:
                data = [t.cuda(non_blocking=True) for t in data]
            x, target = data[:2]

            # Modality order：x_flair, x_t1ce，x_t1, x_t2
            if args.miss_modal == True and args.setting.split('_')[-2] == 'missF':
                x[:,0,...] = 0.0
            elif args.miss_modal == True and args.setting.split('_')[-2] == 'missT1':  # missT1ce
                x[:,1,...] = 0.0
            elif args.miss_modal == True and args.setting.split('_')[-2] == 'missT1ce': # missT1
                x[:,2,...] = 0.0
            elif args.miss_modal == True and args.setting.split('_')[-2] == 'missT2':
                x[:,3,...] = 0.0
                
            elif args.miss_modal == True and args.setting.split('_')[-2] == 'missF+T1':
                x[:,0,...] = 0.0
                x[:,1,...] = 0.0
            elif args.miss_modal == True and args.setting.split('_')[-2] == 'missF+T2':
                x[:,0,...] = 0.0
                x[:,3,...] = 0.0
            elif args.miss_modal == True and args.setting.split('_')[-2] == 'missF+T1ce':
                x[:,0,...] = 0.0
                x[:,2,...] = 0.0
            elif args.miss_modal == True and args.setting.split('_')[-2] == 'missT1+T2':
                x[:,1,...] = 0.0
                x[:,3,...] = 0.0
            elif args.miss_modal == True and args.setting.split('_')[-2] == 'missT1+T1ce':
                x[:,1,...] = 0.0
                x[:,2,...] = 0.0
                
            elif args.miss_modal == True and args.setting.split('_')[-2] == 'missT2+T1ce':
                x[:,2,...] = 0.0
                x[:,3,...] = 0.0
            

            elif args.miss_modal == True and args.setting.split('_')[-2] == 'missF+T1+T1ce':
                x[:,0,...] = 0.0
                x[:,1,...] = 0.0
                x[:,2,...] = 0.0
            elif args.miss_modal == True and args.setting.split('_')[-2] == 'missF+T1+T2':
                x[:,0,...] = 0.0
                x[:,1,...] = 0.0
                x[:,3,...] = 0.0
            elif args.miss_modal == True and args.setting.split('_')[-2] == 'missF+T1ce+T2':
                x[:,0,...] = 0.0
                x[:,2,...] = 0.0
                x[:,3,...] = 0.0
            elif args.miss_modal == True and args.setting.split('_')[-2] == 'missT1+T1ce+T2':
                x[:,1,...] = 0.0
                x[:,2,...] = 0.0
                x[:,3,...] = 0.0


            # compute output
            if not use_TTA:
                start_time = time.time()
                if args.net == 'Unet':
                    logit = model(x)[0]
                else:
                    logit = model(x)
                elapsed_time = time.time() - start_time
                runtimes.append(elapsed_time)

                output = F.softmax(logit,dim=1)
                del x, logit
                torch.cuda.empty_cache()
            
            else:
                if args.net == 'Unet' or args.net == 'U2net3d' or args.net == 'DisenNet':
                    logit = F.softmax(model(x)[0] ,1) 
                    logit += F.softmax(model(x.flip(dims=(2,)))[0].flip(dims=(2,)),1)
                    logit += F.softmax(model(x.flip(dims=(3,)))[0].flip(dims=(3,) ),1)
                    logit += F.softmax(model(x.flip(dims=(4,)))[0].flip(dims=(4,)),1)
                    logit += F.softmax(model(x.flip(dims=(2,3)))[0].flip(dims=(2,3) ),1)
                    logit += F.softmax(model(x.flip(dims=(2,4)))[0].flip(dims=(2,4)),1)
                    logit += F.softmax(model(x.flip(dims=(3,4)))[0].flip(dims=(3,4)),1)
                    logit += F.softmax(model(x.flip(dims=(2,3,4)))[0].flip(dims=(2,3,4)),1)
                    output = logit / 8.0 # mean
                else:
                    logit = F.softmax(model(x) ,1) 
                    logit += F.softmax(model(x.flip(dims=(2,))).flip(dims=(2,)),1)
                    logit += F.softmax(model(x.flip(dims=(3,))).flip(dims=(3,) ),1)
                    logit += F.softmax(model(x.flip(dims=(4,))).flip(dims=(4,)),1)
                    logit += F.softmax(model(x.flip(dims=(2,3))).flip(dims=(2,3) ),1)
                    logit += F.softmax(model(x.flip(dims=(2,4))).flip(dims=(2,4)),1)
                    logit += F.softmax(model(x.flip(dims=(3,4))).flip(dims=(3,4)),1)
                    logit += F.softmax(model(x.flip(dims=(2,3,4))).flip(dims=(2,3,4)),1)
                    output = logit / 8.0 # mean

            if len(output.shape) != 5:
                output = torch.unsqueeze(output,0)
            output = output[0, :, :H, :W, :T].cpu().numpy()

            output = output.argmax(0) # (channels,height,width,depth)

            if postprocess == True:
                ET_voxels = (output == 3).sum()
                if ET_voxels < 500:
                    output[np.where(output == 3)] = 1
            
            # Save the prediciton of validation-set as submission:
            # .npy for farthur model ensemble
            # .nii for directly model submission
            name = str(i+1)

            assert save_format in ['npy','nii']
            if save_format == 'npy':
                np.save(os.path.join(submission_savepath, name + '_preds'), output)
            if save_format == 'nii':
                if not os.path.exists(submission_savepath):
                    os.makedirs(submission_savepath)
                oname = os.path.join(submission_savepath, 'BraTS18_Validation_'+name.zfill(3)+'.nii.gz')
                seg_img = np.zeros(shape=(H,W,T),dtype=np.uint8)

                seg_img[np.where(output==1)] = 1
                seg_img[np.where(output==2)] = 2
                seg_img[np.where(output==3)] = 4
                nib.save(nib.Nifti1Image(seg_img),oname)
                print ('Finishing the %d-th valid submission result.'%(i+1))

                if snapshot:
                    """ --- grey figure---"""
                    # Snapshot_img = np.zeros(shape=(H,W,T),dtype=np.uint8)
                    # Snapshot_img[np.where(output[1,:,:,:]==1)] = 64
                    # Snapshot_img[np.where(output[2,:,:,:]==1)] = 160
                    # Snapshot_img[np.where(output[3,:,:,:]==1)] = 255
                    """ --- colorful figure--- """
                    Snapshot_img = np.zeros(shape=(H, W, 3, T), dtype=np.uint8)
                    Snapshot_img[:, :, 0, :][np.where(output == 1)] = 255
                    Snapshot_img[:, :, 1, :][np.where(output == 2)] = 255
                    Snapshot_img[:, :, 2, :][np.where(output == 3)] = 255

                    for frame in range(T):
                        os.makedirs(os.path.join(log_savepath, args.output_set+'_affine_snapshot','BraTS18_Testing_'+name.zfill(3)),exist_ok=True)
                        scipy.misc.imsave(os.path.join(log_savepath, args.output_set+'_affine_snapshot','BraTS18_Testing_'+name.zfill(3),str(frame)+'.png'), Snapshot_img[:,:,:,frame])
        if args.valid_submission_only:
            assert 1==2
        del output
        torch.cuda.empty_cache() 
    
    model.train()
    
    if args.miss_modal:
        return np.mean([vals_miss[i].avg for i in range(len(vals_miss))],0)
    else:
        return vals.avg


def computational_runtime(runtimes):
    # remove the maximal value and minimal value
    runtimes = np.array(runtimes)
    maxvalue = np.max(runtimes)
    minvalue = np.min(runtimes)
    nums = runtimes.shape[0] - 2
    meanTime = (np.sum(runtimes) - maxvalue - minvalue ) / nums
    fps = 1 / meanTime
    print('mean runtime:',meanTime,'fps:',fps)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
