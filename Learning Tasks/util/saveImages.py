import torch.utils.data as data
import torch
import numpy as np
from torch import from_numpy
import imageio
import os
from HDR.tonemap import  make_tonemap_HDR
from HDR.LDR_from_HDR import make_LDRfromHDR

from os import environ
environ["OPENCV_IO_ENABLE_OPENEXR"] = "true"
import cv2



def Tensor2Numpy(tensor):
    np_image = tensor.cpu().float().numpy()[0,:,:,:]
    np_image_transposed = np.transpose(np_image, (1, 2, 0))  
    return np_image_transposed

def saveImage(img, path):
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_bgr)