## python3 test.py --dataroot dataset/calibrated64 --checkpoints_dir '/Volumes/M E L V I N/Maitrise/Code/Archives/UnetVersions/RescaleHDR+RescaleClipLDR/UNetHDR/checkpoints' --name calibrated64 --tonemap_LDR rescale --rescale_HDR True

import numpy as np
import HDR.metrics
from options.options import Options
from LightningModule.LitUnet import LitFixupUnet, buildUnet
from HDR.visualization import visual_divide_x, luminanceFalseColors, distributionBoxPlot, temperatureFalseColors, temperatureMap, relativeErrorMap, luminanceMap
from util.saveImages import Tensor2Numpy, saveImage
from HDR.tonemap import  make_tonemap_HDR
from HDR.LDR_from_HDR import LDRfromHDR, torchnormalizeEV, torchnormalizeEV0, unNormalizeScale
from dataManagement.dataset import Dataset
import os
from omegaconf import DictConfig, OmegaConf
from glob import glob
from natsort import natsorted
from HDR.metrics import *
import csv
import cv2
import scipy


if __name__ ==  '__main__':
    opt = Options().parse()
    opt.phase = 'test'

    #metrics lists
    wrMSE_list = []
    si_wrMSE_list = []
    HDR_VDP_list = []
    rel_err_list = []
    illum_targets = []
    illum_preds = []
    scale_targets = []
    scale_preds = []
    names = []

    default_root_dir = os.path.join(opt.checkpoints_dir, opt.name)
    out_dir = os.path.join(default_root_dir, 'test_res')
    if opt.existing_in:
        out_dir = out_dir + '_existing_in'
    if opt.version != 'None':
        out_dir = out_dir+ '_' + opt.version

    out_dir_hdr = os.path.join(out_dir, 'HDR')

    out_dir_hdr_pred = os.path.join(out_dir, 'HDR_pred')

    out_dir_ldr = os.path.join(out_dir, 'LDR')

    out_dir_csv = os.path.join(out_dir, 'illuminance')

    out_dir_scale = os.path.join(out_dir, 'scale')

    if os.path.isfile(out_dir+'/wrMSE.csv'):
        os.remove(out_dir+'/wrMSE.csv')
    if os.path.isfile(out_dir+'/si_wrMSE.csv'):
        os.remove(out_dir+'/si_wrMSE.csv')
    if os.path.isfile(out_dir+'/r2.csv'):
        os.remove(out_dir+'/r2.csv')
    if os.path.isfile(out_dir+'/relerr.csv'):
        os.remove(out_dir+'/relerr.csv')

    LDRs = glob(out_dir_ldr + '/*.exr')

    HDR2LDR = LDRfromHDR()
    LitUnet = LitFixupUnet(None, opt)

    if opt.mode in ['luminance']:
        with open(out_dir+'/hdr-vdp.csv', 'r') as file:
            reader = csv.reader(file)
            count = 0
            for row in reader:
                if count > 0:
                    HDR_VDP_list.append(float(row[1]))
                count += 1

    count = 0
    for LDR_path in LDRs:
        print(count,'/',len(LDRs))

        name = os.path.basename(LDR_path)[:-4]
        pred = LDR_path.replace(out_dir_ldr, out_dir_hdr_pred)
        HDR_path = LDR_path.replace(out_dir_ldr, out_dir_hdr)
        csv_path = LDR_path.replace(out_dir_ldr, out_dir_csv)
        csv_path = csv_path.replace(".exr",".csv")
        scale_csv_path = csv_path.replace(out_dir_csv, out_dir_scale)

        LDR_BGR = cv2.imread(LDR_path,cv2.IMREAD_UNCHANGED)
        LDR = cv2.cvtColor(LDR_BGR, cv2.COLOR_BGR2RGB)
        LDR[LDR<0] = 0

        if opt.mode in ['luminance','temperature']:
            target_BGR = cv2.imread(HDR_path,cv2.IMREAD_UNCHANGED)
            target = cv2.cvtColor(target_BGR, cv2.COLOR_BGR2RGB)
            target[target<0] = 0

            HDR_pred_BGR = cv2.imread(pred,cv2.IMREAD_UNCHANGED)
            HDR_pred = cv2.cvtColor(HDR_pred_BGR, cv2.COLOR_BGR2RGB)
            HDR_pred[HDR_pred<0] = 0
        elif opt.mode in ['illuminance']: 
            with open(csv_path) as f:
                rows = list(csv.reader(f))
                target = float(rows[0][1])
                illum_pred = float(rows[1][1])
                illum_preds.append(illum_pred)
                illum_targets.append(target)

        
        #Compute metrics
        if opt.mode in ['luminance','temperature']:
            wrMSE_list.append(wrmse(target, HDR_pred, solid_angles_map=LitUnet.solid_angles_map_np))
            with open(out_dir+'/wrMSE.csv', 'a') as file:
                writer_object = csv.writer(file)
                writer_object.writerow([name, wrMSE_list[-1]])
        if opt.mode in ['luminance']:
            si_wrMSE_list.append(si_wrmse(target, HDR_pred, solid_angles_map=LitUnet.solid_angles_map_np))
            with open(out_dir+'/si_wrMSE.csv', 'a') as file:
                writer_object = csv.writer(file)
                writer_object.writerow([name, si_wrMSE_list[-1]])
        if opt.mode in ['illuminance']:
            wrMSE_list.append(wrmse(np.array([target]), np.array([illum_pred])))
            with open(out_dir+'/wrMSE.csv', 'a') as file:
                writer_object = csv.writer(file)
                writer_object.writerow([name, wrMSE_list[-1]])
        if opt.mode in ['luminance','temperature']:
            rel_err_map = np.minimum(np.abs(target-HDR_pred)/target, 100)
            rel_err_map = np.nan_to_num(rel_err_map)
            rel_err = np.mean(rel_err_map)
            rel_err_list.append(rel_err)
            if np.isnan(rel_err_map).any():
                print(np.min(rel_err_map), np.max(rel_err_map))
            with open(out_dir+'/relerr.csv', 'a') as file:
                writer_object = csv.writer(file)
                writer_object.writerow([name, rel_err_list[-1]])

        if opt.mode in ['illuminance']:
            rel_err_map = np.abs(np.array([target])-np.array([illum_pred]))/np.array([target])
            rel_err = np.mean(rel_err_map)
            rel_err_list.append(rel_err)
            with open(out_dir+'/relerr.csv', 'a') as file:
                writer_object = csv.writer(file)
                writer_object.writerow([name, rel_err_list[-1]])
        names.append(name)
            

        if opt.mode in ['luminance']:
            #Save visualisation /10
            factor = HDR2LDR.rescaleAlpha(target)
            out_dir_hdr_vis = os.path.join(out_dir, 'divide_10')
            if not os.path.exists(out_dir_hdr_vis):
                os.makedirs(out_dir_hdr_vis)
            title = "wrMSE: " + str(wrMSE_list[-1]) + ", si_wrMSE: " + str(si_wrMSE_list[-1]) + ", HDR-VDP: " + str(HDR_VDP_list[count]) + ", rel_err: " + str(rel_err_list[-1])
            visual_divide_x(LDR, HDR_pred*factor, target*factor, 10, title=title, output=os.path.join(out_dir_hdr_vis, name+'.pdf'))

        if opt.mode in ['luminance']:
            #Save false colors
            out_dir_hdr_falsecol = os.path.join(out_dir, 'false_colors')
            if not os.path.exists(out_dir_hdr_falsecol):
                os.makedirs(out_dir_hdr_falsecol)
            luminanceFalseColors(LDR,HDR_pred, target,output=os.path.join(out_dir_hdr_falsecol, name+'.pdf'))

            out_dir_hdr_lumpred = os.path.join(out_dir, 'luminance_pred')
            if not os.path.exists(out_dir_hdr_lumpred):
                os.makedirs(out_dir_hdr_lumpred)
            luminanceMap(HDR_pred, output=os.path.join(out_dir_hdr_lumpred, name+'.png'))

            out_dir_hdr_lumtarget = os.path.join(out_dir, 'luminance_target')
            if not os.path.exists(out_dir_hdr_lumtarget):
                os.makedirs(out_dir_hdr_lumtarget)
            luminanceMap(target, output=os.path.join(out_dir_hdr_lumtarget, name+'.png'))

            out_dir_hdr_relerr = os.path.join(out_dir, 'relative_error')
            if not os.path.exists(out_dir_hdr_relerr):
                os.makedirs(out_dir_hdr_relerr)
            relativeErrorMap(rel_err_map, output=os.path.join(out_dir_hdr_relerr, name+'.png'))
        
        if opt.mode == 'temperature':
            out_dir_hdr_falsecol = os.path.join(out_dir, 'false_colors')
            if not os.path.exists(out_dir_hdr_falsecol):
                os.makedirs(out_dir_hdr_falsecol)
            title = "wrMSE: " + str(wrMSE_list[-1]) + ", rel_err: " + str(rel_err_list[-1])
            temperatureFalseColors(LDR,HDR_pred, target, rel_err_map, output=os.path.join(out_dir_hdr_falsecol, name+'.pdf'), title=title)

            out_dir_hdr_temppred = os.path.join(out_dir, 'temperature_pred')
            if not os.path.exists(out_dir_hdr_temppred):
                os.makedirs(out_dir_hdr_temppred)
            temperatureMap(HDR_pred, output=os.path.join(out_dir_hdr_temppred, name+'.png'))

            out_dir_hdr_temptarget = os.path.join(out_dir, 'temperature_target')
            if not os.path.exists(out_dir_hdr_temptarget):
                os.makedirs(out_dir_hdr_temptarget)
            temperatureMap(target, output=os.path.join(out_dir_hdr_temptarget, name+'.png'))

            out_dir_hdr_relerr = os.path.join(out_dir, 'relative_error')
            if not os.path.exists(out_dir_hdr_relerr):
                os.makedirs(out_dir_hdr_relerr)
            relativeErrorMap(rel_err_map, output=os.path.join(out_dir_hdr_relerr, name+'.png'))


        count += 1

    np_HDR_VDP = np.array(HDR_VDP_list)
    np_wrMSE = np.array(wrMSE_list)
    np_si_wrMSE = np.array(si_wrMSE_list)

    #distribution plots
    if opt.mode in ['luminance']:
        distributionBoxPlot(HDR_VDP_list, title='HDR_VDP',text=names,output=os.path.join(out_dir, 'HDR_VDP_distribution.html'))
        distributionBoxPlot(si_wrMSE_list, title='si_wrMSE',text=names,output=os.path.join(out_dir, 'si_wrMSE_distribution.html'))
        print("HDR-VDP: ", np.mean(np_HDR_VDP))
        print("si_wrMSE: ", np.mean(np_si_wrMSE))
    if opt.mode in ["illuminance"]:
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(illum_preds, illum_targets)
        print("r2: ",r_value)
    if opt.mode in ['temperature','illuminance']:
        distributionBoxPlot(rel_err_list, title='rel_err',text=names,output=os.path.join(out_dir, 'rel_err_distribution.html'))
        print("rel_err: ", np.mean(np.array(rel_err_list)))
    
    distributionBoxPlot(wrMSE_list, title='wrMSE',text=names,output=os.path.join(out_dir, 'wrMSE_distribution.html'))
    
    print("wrMSE: ", np.mean(np_wrMSE))
    