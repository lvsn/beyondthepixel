#Encapsuling of the network into a pytorch lightning module
#
#
#

import tensorflow
import os
import numpy as np
from torchvision import models
from collections import OrderedDict 
import pytorch_lightning as pl
import torch
from torch import nn
from util.saveImages import Tensor2Numpy, saveImage
from HDR.tonemap import  make_tonemap_HDR
from dataManagement.DatasetUtils import loadSolidAnglesMap, createSolidAnglesMap
from omegaconf import DictConfig, OmegaConf
from fixupunet.network import FixUpUnet, FixUpUnetInject, FixUpUnetScale, FixUpUnetChopped, FixUpUnetChoppedBins
from HDR.LDR_from_HDR import unNormalizeScale, unNormalizeIlluminance


def buildUnet(opt):

    default_root_dir = os.path.join(opt.checkpoints_dir, opt.name)

    if opt.mode == 'temperature':
        opt.out_feat = 1
    else:
        opt.out_feat = 3

    opt.out_act_fn = 'tanh'
    
    cfg = DictConfig(
        {
            "feat": opt.feat,
            "in_feat": 3,
            "out_feat": opt.out_feat,
            "down_layers": opt.down_layers,
            "identity_layers": opt.identity_layers,
            "bottleneck_layers": opt.bottleneck_layers,
            "skips": opt.skips,
            "act_fn": opt.act_fn,
            "out_act_fn": opt.out_act_fn,
            "max_feat": opt.max_feat,
            "script_submodules": opt.script_submodules,
            "input_sizey": opt.size_y,
            "input_sizex": opt.size_x,
        }
    )
    
    if opt.mode == 'luminance':
        Unet_network = FixUpUnetScale(cfg)
    elif opt.mode == 'temperature':
        Unet_network = FixUpUnet(cfg)
    elif opt.mode == 'illuminance':
        Unet_network = FixUpUnetChopped(cfg)
    else:
        raise(ValueError, 'Invalid mode')

    return Unet_network

class LitFixupUnet(pl.LightningModule):
    def __init__(self, Unet_network, opt):
        super().__init__()
        self.mode = opt.mode
        self.Unet_network = Unet_network
        self.tonemap = make_tonemap_HDR(opt)
        self.learning_rate = opt.learning_rate
        self.scale_loss_factor = opt.scale_loss_factor
        self.batch_size = opt.batch_size
        self.save_val_img = opt.save_val_img

        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.image_dir = os.path.join(self.save_dir, 'images')

        if opt.phase == 'train':
            if not os.path.exists(self.image_dir):
                os.makedirs(self.image_dir)
        
        self.use_solid_angles_map = opt.use_solid_angles_map
        
        self.solid_angles_map = None
        if self.use_solid_angles_map:
            if opt.solid_angles_map == "none":
                createSolidAnglesMap(opt)
            self.solid_angles_map_np = loadSolidAnglesMap(opt.solid_angles_map)
            self.solid_angles_map_np = self.solid_angles_map_np / np.max(self.solid_angles_map_np) #To have values not so small
            self.solid_angles_map = torch.from_numpy(self.solid_angles_map_np)


    def training_step(self, batch, batch_idx):
        LDR = batch['LDR']
        target = batch['target']
        scale = batch['scale'].float()
        scale = torch.unsqueeze(scale,1)


        if self.mode in ['illuminance']:
            target = target.float()
            pred = self.Unet_network(LDR)
            target = torch.unsqueeze(target,1)
            loss = nn.functional.mse_loss(pred, target)

            self.log("train_loss", loss, on_epoch=True)
            return loss

        elif self.mode == 'luminance':
            pred, scale_pred = self.Unet_network(LDR)
        else: #temperature
            pred = self.Unet_network(LDR)


        if self.use_solid_angles_map:
            weighted_pred = pred * self.solid_angles_map.to(self.device)
            weighted_target = target * self.solid_angles_map.to(self.device)
        else:
            weighted_target = target
            weighted_pred = pred
            
        if self.mode == 'luminance':
            loss_target = nn.functional.mse_loss(weighted_pred, weighted_target)
            loss_scale = nn.functional.l1_loss(scale_pred, scale)
            loss = loss_target + self.scale_loss_factor*loss_scale
        else:
            loss = nn.functional.mse_loss(weighted_pred, weighted_target)

        # Logging to TensorBoard
        self.log("train_loss", loss, on_epoch=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        LDR = batch['LDR']
        target = batch['target']
        scale = batch['scale'].float()
        scale = torch.unsqueeze(scale,1)

        if self.mode in ['illuminance']:
            pred = self.Unet_network(LDR)
        if self.mode == 'luminance':
            pred, scale_pred = self.Unet_network(LDR)
        else: #temperature
            pred = self.Unet_network(LDR)

        if batch_idx == 0 and self.save_val_img:
            LDR_numpy = Tensor2Numpy(LDR)
            img_path_LDR = os.path.join(self.image_dir, 'epoch%.3d_LDR.exr' % (self.current_epoch))
            saveImage(LDR_numpy, img_path_LDR)

            if self.mode in ['luminance','temperature']:
                target_numpy = Tensor2Numpy(target)
                target_untonemapped = self.tonemap.inv_process(target_numpy)
                img_path_target = os.path.join(self.image_dir, 'epoch%.3d_target.exr' % (self.current_epoch))
                saveImage(target_untonemapped, img_path_target)

                pred_numpy = Tensor2Numpy(pred)
                pred_untonemapped = self.tonemap.inv_process(pred_numpy)
                img_path_pred = os.path.join(self.image_dir, 'epoch%.3d_pred.exr' % (self.current_epoch))
                saveImage(pred_untonemapped, img_path_pred)
            

            if self.mode == 'luminance':
                scale_numpy = scale.cpu().float().numpy()
                scale_pred_numpy = scale_pred.cpu().float().numpy()
                scale_numpy = unNormalizeScale(scale_numpy)
                scale_pred_numpy = unNormalizeScale(scale_pred_numpy)
                with open(os.path.join(self.image_dir, 'epoch%.3d_scale.txt' % (self.current_epoch)), 'w') as f:
                    f.write('gt: ' + str(scale_numpy[0]))
                    f.write('\npred: ' + str(scale_pred_numpy[0,0]))
            if self.mode in ['illuminance']:
                target_numpy = target.cpu().float().numpy()
                pred_numpy = pred.cpu().float().numpy()
                target_numpy = unNormalizeIlluminance(target_numpy)
                pred_numpy = unNormalizeIlluminance(pred_numpy)
                with open(os.path.join(self.image_dir, 'epoch%.3d_illuminance.txt' % (self.current_epoch)), 'w') as f:
                    f.write('gt: ' + str(target_numpy[0]))
                    f.write('\npred: ' + str(pred_numpy[0,0]))
        
        if self.mode in ['illuminance']:
            target = target.float()
            target = torch.unsqueeze(target,1)
            loss = nn.functional.mse_loss(pred, target)

            self.log("val_loss", loss)#, on_epoch=True)
            return loss

        if self.use_solid_angles_map:
            weighted_pred = pred * self.solid_angles_map.to(self.device)
            weighted_target = target * self.solid_angles_map.to(self.device)
        else:
            weighted_target = target
            weighted_pred = pred
            
        if self.mode == 'luminance':
            loss_target = nn.functional.mse_loss(weighted_pred, weighted_target)
            loss_scale = nn.functional.l1_loss(scale_pred, scale)
            loss = loss_target + self.scale_loss_factor*loss_scale
        else:
            loss = nn.functional.mse_loss(weighted_pred, weighted_target)

        # Logging to TensorBoard
        self.log("val_loss", loss, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

