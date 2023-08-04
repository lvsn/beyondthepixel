
import numpy as np
import HDR.metrics
import torch
from torch import nn
from options.options import Options
from LightningModule.LitUnet import LitFixupUnet, buildUnet
from HDR.visualization import visual_divide_x, luminanceFalseColors, distributionBoxPlot, scatterPlot, scatterPlotMultiple
from util.saveImages import Tensor2Numpy, saveImage
from HDR.tonemap import  make_tonemap_HDR, linear_clip
from HDR.LDR_from_HDR import LDRfromHDR, unNormalizeScale, unNormalizeIlluminance, normalizeIlluminance
from dataManagement.dataset import Dataset, find_size
import os
from omegaconf import DictConfig, OmegaConf
from glob import glob
from natsort import natsorted
from HDR.metrics import *
import csv

from os import environ
environ["OPENCV_IO_ENABLE_OPENEXR"] = "true"
import cv2


if __name__ ==  '__main__':
    opt = Options().parse()
    opt.phase = 'test'

    test_dataset = Dataset(opt, phase='test')
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=1, shuffle=False)

    sizey, sizex = find_size(test_data_loader)
    opt.size_y = sizey
    opt.size_x = sizex

    Unet_network = buildUnet(opt)

    # load checkpoint
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    if opt.version == 'None':
        checkpoints = glob(save_dir + '/**/epoch*.ckpt', recursive=True)
        if not checkpoints:
            raise ValueError("No checkpoint found at path: "+ save_dir)
        checkpoint = natsorted(checkpoints)[-1]
    else:
        checkpoints = glob(save_dir + "/lightning_logs/version_" + opt.version + "/**/epoch*.ckpt", recursive=True)
        if not checkpoints:
            raise ValueError("No checkpoint found at path: "+ save_dir + "/lightning_logs/version_" + opt.version)
        checkpoint = natsorted(checkpoints)[-1]
    
    print("Using network weights from: ", checkpoint)
    LitUnet = LitFixupUnet.load_from_checkpoint(checkpoint, opt=opt, Unet_network=Unet_network)
    Unet_network = LitUnet.Unet_network

    default_root_dir = os.path.join(opt.checkpoints_dir, opt.name)
    out_dir = os.path.join(default_root_dir, 'test_res')
    if opt.existing_in:
        out_dir = out_dir + '_existing_in'
    if opt.version != 'None':
        out_dir = out_dir+ '_' + opt.version
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    

    out_dir_ldr = os.path.join(out_dir, 'LDR')
    if not os.path.exists(out_dir_ldr):
        os.makedirs(out_dir_ldr)

    if opt.mode in ['luminance','temperature']:
        out_dir_hdr = os.path.join(out_dir, 'HDR')
        if not os.path.exists(out_dir_hdr):
            os.makedirs(out_dir_hdr)

        out_dir_hdr_pred = os.path.join(out_dir, 'HDR_pred')
        if not os.path.exists(out_dir_hdr_pred):
            os.makedirs(out_dir_hdr_pred)
            
        out_dir_hdr_vdp = os.path.join(out_dir, 'HDR_vdp')
        if not os.path.exists(out_dir_hdr_vdp):
            os.makedirs(out_dir_hdr_vdp)

        out_dir_hdr_pred_vdp = os.path.join(out_dir, 'HDR_pred_vdp')
        if not os.path.exists(out_dir_hdr_pred_vdp):
            os.makedirs(out_dir_hdr_pred_vdp)
    elif opt.mode == 'illuminance':
        out_dir_illuminance = os.path.join(out_dir, 'illuminance')
        if not os.path.exists(out_dir_illuminance):
            os.makedirs(out_dir_illuminance)
        out_dir_hemi = os.path.join(out_dir, 'hemi')
        if not os.path.exists(out_dir_hemi):
            os.makedirs(out_dir_hemi)

    HDR2LDR = LDRfromHDR()

    count = 0
    for data in test_data_loader:
        count += 1
        print(count,'/',len(test_data_loader))
        
        LDR = data['LDR']
        target = data['target']
        scale = data['scale'].float()
        scale = torch.unsqueeze(scale,1)

        with torch.no_grad():

            if opt.mode == 'luminance':
                pred, scale_pred = Unet_network(LDR)
            else: # temperature or illuminance
                pred = Unet_network(LDR)

            tonemap = make_tonemap_HDR(opt)

            LDR_numpy = Tensor2Numpy(LDR)
            img_path_LDR = os.path.join(out_dir_ldr, data['img_path'][0][:-4]+'.exr')
            saveImage(LDR_numpy, img_path_LDR)


            if opt.mode in ['temperature']:
                target_numpy = Tensor2Numpy(target)
                target_untonemapped = tonemap.inv_process(target_numpy)

                pred_numpy = Tensor2Numpy(pred)
                pred_untonemapped = tonemap.inv_process(pred_numpy)

                saveImage(target_untonemapped, os.path.join(out_dir_hdr, data['img_path'][0][:-4]+'.exr'))
                saveImage(pred_untonemapped, os.path.join(out_dir_hdr_pred, data['img_path'][0][:-4]+'.exr'))


            if opt.mode == 'luminance':
                target_numpy = Tensor2Numpy(target)
                target_untonemapped = tonemap.inv_process(target_numpy)

                pred_numpy = Tensor2Numpy(pred)
                pred_untonemapped = tonemap.inv_process(pred_numpy)
                
                scale_numpy = scale.cpu().float().numpy()
                scale_pred_numpy = scale_pred.cpu().float().numpy()
                scale_numpy = unNormalizeScale(scale_numpy)
                scale_pred_numpy = unNormalizeScale(scale_pred_numpy)
                HDR_untonemapped = target_untonemapped * scale_numpy[0]
                HDR_pred_untonemapped = pred_untonemapped * scale_pred_numpy[0,0]

                saveImage(HDR_untonemapped, os.path.join(out_dir_hdr, data['img_path'][0][:-4]+'.exr'))
                saveImage(HDR_pred_untonemapped, os.path.join(out_dir_hdr_pred, data['img_path'][0][:-4]+'.exr'))
                #Save .hdr for HDR-VDP-3

                saveImage(HDR_untonemapped, os.path.join(out_dir_hdr_vdp, data['img_path'][0][:-4]+'.hdr'))
                saveImage(HDR_pred_untonemapped, os.path.join(out_dir_hdr_pred_vdp, data['img_path'][0][:-4]+'.hdr'))
    
            if opt.mode == 'illuminance':
                hemi_numpy = Tensor2Numpy(data['source'])
                img_path_hemi = os.path.join(out_dir_hemi, data['img_path'][0][:-4]+'.exr')
                saveImage(hemi_numpy, img_path_hemi)

                target_numpy = target.cpu().float().numpy()
                pred_numpy = pred.cpu().float().numpy()
                target_numpy = unNormalizeIlluminance(target_numpy)
                pred_numpy = unNormalizeIlluminance(pred_numpy)
                with open(os.path.join(out_dir_illuminance, data['img_path'][0][:-4]+'.csv'), 'w') as f:
                    writer = csv.writer(f, lineterminator="\n")
                    writer.writerow(["gt", target_numpy[0]])
                    writer.writerow(["pred", pred_numpy[0][0]])


    if opt.mode in ['luminance']:
        matlab_command = "matlab -nodisplay -nosplash -nodesktop -r \"run_hdrvdp('."+out_dir_hdr_vdp+"', '."+out_dir_hdr_pred_vdp+"', '."+out_dir+"/hdr-vdp.csv');exit\""

        print("run following command in HDR folder:")
        print(matlab_command)
