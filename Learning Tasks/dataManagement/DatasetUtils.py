from os import environ
environ["OPENCV_IO_ENABLE_OPENEXR"] = "true"
from glob import glob
import shutil
import random
from envmap import EnvironmentMap
import cv2
import numpy as np
import os

def extractSolidAnglesMap(img, dest_path, illuminance=False, cos_path=None):
    e = EnvironmentMap(img, 'latlong')
    e_sa = e.copy().solidAngles()
    e_sa_np = np.array(e_sa.data).astype(np.float32)

    if illuminance:
        vectors = e.worldCoordinates()
        vectors_x = vectors[0]
        vectors_y = vectors[1]
        vectors_z = vectors[2]
        n = np.array([-1,0,0]) #Front dir -X

        #Stack 3 direction vectors
        vectors = np.stack([vectors_x,vectors_y,vectors_z], axis=-1)

        #Compute cos angle between front dir and each pixel
        vectors_norm = np.linalg.norm(vectors, axis = 2)
        vectors_dot = np.dot(vectors, n)
        cos = vectors_dot/vectors_norm

        cos = cos[:,:cos.shape[1]//2]
        cv2.imwrite(cos_path, cos.astype(np.float32))

        e_sa_np = e_sa_np[:,:e_sa_np.shape[1]//2]

    cv2.imwrite(dest_path, e_sa_np)

def loadSolidAnglesMap(img_path, cos_path=None):
    e_sa = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if cos_path is not None:
        cos = cv2.imread(cos_path, cv2.IMREAD_UNCHANGED)
        return e_sa, cos
    return e_sa

def createSolidAnglesMap(opt, cos_map=False):
    dataset = opt.dataroot
    opt.solid_angles_map = opt.dataroot + '/sa_map.exr'

    imgs = glob(dataset + '/*/*.exr')

    image_HDR_BGR = cv2.imread(imgs[0],cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(image_HDR_BGR, cv2.COLOR_BGR2RGB)

    opt.cos_path = None
    if cos_map:
        opt.cos_path = opt.dataroot + '/cos_map.exr'

    extractSolidAnglesMap(img, opt.solid_angles_map, cos_map, opt.cos_path)

def createSolidAnglesMapHemisphere(opt):
    dataset = opt.dataroot
    opt.solid_angles_map = opt.dataroot + '/sa_map.exr'

    imgs = glob(dataset + '/**/*.exr', recursive=True)

    extractSolidAnglesMap(imgs[0], opt.solid_angles_map)

def duplicateImage(image, number):
    extention = image.split(".")[-1]
    length = len(extention) + 1
    for i in range(number):
        print(i, "/", range(number))
        shutil.copy(image, image[:-length]+str(i)+"."+extention)