#The pytorch dataset
#
#

import torch.utils.data as data
import torch
import numpy as np
from torch import from_numpy
import os
import pickle
from HDR.tonemap import  make_tonemap_HDR
from HDR.LDR_from_HDR import make_LDRfromHDR, LDRfromHDR, normalizeIlluminance
from dataManagement.DatasetUtils import loadSolidAnglesMap, createSolidAnglesMap
from envmap import EnvironmentMap, rotation_matrix
from HDR.white_balance.classes import WBsRGB as wb_srgb
from dataManagement.WBAugmenter import WBEmulator as wbAug

from os import environ
environ["OPENCV_IO_ENABLE_OPENEXR"] = "true"
import cv2

WB_num = 9

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff', '.exr'
]

def find_size(dataloader):
    batch = next(iter(dataloader))
    LDR = batch['LDR']
    sizey = LDR.size(dim=2)
    sizex = LDR.size(dim=3)
    return sizey, sizex 

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset_paths(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


class Dataset(data.Dataset):

    def __init__(self, opt, phase):
        self.opt = opt
        self.phase = phase
        self.root = opt.dataroot    

        self.dir = os.path.join(opt.dataroot, phase)
        self.paths = sorted(make_dataset_paths(self.dir))

        self.mode = opt.mode
        self.crop = opt.crop

        self.tonemap_LDR = opt.tonemap_LDR
        
        self.tonemap = make_tonemap_HDR(opt)
        self.LDRfromHDR = make_LDRfromHDR(opt)

        self.existing_in = opt.existing_in

        self.dataset_size = len(self.paths)

        self.augmentation = opt.augmentation

        self.WBAugmenter = opt.WBaugmenter



        if self.mode in ['illuminance'] or opt.use_solid_angles_map:
            createSolidAnglesMap(opt, self.mode in ['illuminance'])
            if self.mode in ['illuminance']:
                self.solid_angles_map, self.cos_map = loadSolidAnglesMap(opt.solid_angles_map, opt.cos_path)
            else:
                self.solid_angles_map = loadSolidAnglesMap(opt.solid_angles_map)
        
        if self.WBAugmenter:
            self.wb_color_aug = wbAug.WBEmulator()
            self.mapping = self.compute_mapping()

    "Computes the mapping functions for the WB augmenter"
    def compute_mapping(self):
        if os.path.exists(os.path.join(self.root, 'wb_mfs.pickle')):
          with open(os.path.join(self.root, 'wb_mfs.pickle'), 'rb') as handle:
            mapping_funcs = pickle.load(handle)
          return mapping_funcs

        mapping_funcs = []
        for idx in range(self.dataset_size):
            path = self.paths[idx]
            image_HDR_BGR = cv2.imread(path,cv2.IMREAD_UNCHANGED)
            image_HDR = cv2.cvtColor(image_HDR_BGR, cv2.COLOR_BGR2RGB)
            image_HDR[image_HDR<0] = 0

            LDR, scale = self.LDRfromHDR.process(image_HDR)

            mfs = self.wb_color_aug.computeMappingFunc(LDR)
            mapping_funcs.append(mfs)
        with open(os.path.join(self.root, 'wb_mfs.pickle'), 'wb') as handle:
          pickle.dump(mapping_funcs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        return mapping_funcs

    def __len__(self):
        return self.dataset_size


    def __getitem__(self, index):
        path = self.paths[index]

        image_HDR_BGR = cv2.imread(path,cv2.IMREAD_UNCHANGED)
        image_HDR = cv2.cvtColor(image_HDR_BGR, cv2.COLOR_BGR2RGB)
        image_HDR[image_HDR<0] = 0

        source = np.empty(image_HDR.shape)
        if self.opt.phase == 'test':
            source = image_HDR
            source = np.transpose(source, (2, 0, 1))

        #Augmentation rotation the pano on the azimuth
        if self.augmentation and self.opt.phase != 'test':
            shift = int(np.random.random_sample() * image_HDR.shape[1])
            image_HDR = np.roll(image_HDR, shift, axis=1)
        
        #Generate target
        if self.mode in ['luminance']:
            image_HDR_rescaled, _ = self.LDRfromHDR.rescale(image_HDR)
            target = self.tonemap.process(image_HDR_rescaled)
        elif self.mode == 'temperature':
            image_HDR_XYZ = cv2.cvtColor(image_HDR, cv2.COLOR_RGB2XYZ)
            XYZ_sum = np.sum(image_HDR_XYZ, axis=2)
            x = image_HDR_XYZ[:,:,0] / XYZ_sum
            y = image_HDR_XYZ[:,:,1] / XYZ_sum
            n = (x - 0.3320) / (0.1858 - y)
            target = (449 * np.power(n,3)) + (3525 * np.power(n,2)) + (6823.3 * n) + 5518.87
            target = np.expand_dims(target, axis=2)
            target[target<0] = 0
            target = self.tonemap.process(target)
        elif self.mode == 'illuminance':
            LDR, scale = self.LDRfromHDR.process(image_HDR)
            if self.crop != 0:
                if self.crop == -1:
                    fov = np.random.uniform(45,120,1)
                else:
                    fov = self.crop
                e = EnvironmentMap(LDR, 'latlong')
                dcm = rotation_matrix(azimuth=-np.pi/2,
                                    elevation=0.,
                                    roll=0.)
                proj = e.project(vfov=fov, # degrees
                                rotation_matrix=dcm,
                                ar=4./3.,
                                resolution=(160, 120),
                                projection="perspective",
                                mode="normal")
                proj = np.array(proj.data).astype(np.float32)
            #Comnpute illumiance
            LDR = LDR[:,:LDR.shape[1]//2,:]
            source = LDR
            source = np.transpose(source, (2, 0, 1))
            image_HDR = image_HDR[:,:image_HDR.shape[1]//2,:]
            illumiance_r = np.sum(self.solid_angles_map * self.cos_map * image_HDR[:,:,0])
            illumiance_g = np.sum(self.solid_angles_map * self.cos_map * image_HDR[:,:,1])
            illumiance_b = np.sum(self.solid_angles_map * self.cos_map * image_HDR[:,:,2])
            target = (0.212671 * illumiance_r + 0.715160 * illumiance_g + 0.072169 * illumiance_b)
            target = normalizeIlluminance(target)
            if self.crop != 0:
                LDR = proj

        else:
            raise(ValueError, 'invalid mode')
       
        if self.existing_in:
            scale = 0 #dummy
            path_LDR = path.replace(self.phase, self.phase+"_in")
            LDR_BGR = cv2.imread(path_LDR,cv2.IMREAD_UNCHANGED)
            LDR = cv2.cvtColor(LDR_BGR, cv2.COLOR_BGR2RGB)
            if self.augmentation and self.opt.phase != 'test':
                LDR = np.roll(LDR, shift, axis=1)
            if self.mode == 'illuminance':
                e = EnvironmentMap(LDR, 'latlong')
                dcm = rotation_matrix(azimuth=-np.pi/2,
                                    elevation=0.,
                                    roll=0.)
                proj = e.project(vfov=fov, # degrees
                                rotation_matrix=dcm,
                                ar=4./3.,
                                resolution=(160, 120),
                                projection="perspective",
                                mode="normal")
                proj = np.array(proj.data).astype(np.float32)
                if self.crop != 0:
                    LDR = proj
                else:
                    LDR = LDR[:,:LDR.shape[1]//2,:]
        elif self.mode != 'illuminance':
            LDR, scale = self.LDRfromHDR.process(image_HDR)
            if self.WBAugmenter:
                mfs = self.mapping[index]
                ind = np.random.randint(len(mfs))
                mf = mfs[WB_num]
                LDR = wbAug.changeWB(LDR, mf).astype(np.float32)


        #Permute image for CHW format
        if self.mode not in ['illuminance']:
            target = np.transpose(target, (2, 0, 1))  

        
        LDR = np.transpose(LDR, (2, 0, 1))

        return {'LDR': LDR, 'target': target, 'img_path': os.path.split(path)[1], 'scale': scale, 'source': source}
