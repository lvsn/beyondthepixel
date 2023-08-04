import cv2
import numpy as np

class tonemap():

    def __init__(self):
        pass

    def process(self, img):
        return img

    def inv_process(self, img):
        return img

#Log correction
class log_tonemap(tonemap):

    #Constructor
    #Base of log
    #Scale of tonemapped
    #Offset
    def __init__(self, base, scale = 1, offset = 1):
        self.base = base
        self.scale = scale
        self.offset = offset

    def process(self, img):
        tonemapped = (np.log(img + self.offset) / np.log(self.base)) * self.scale
        return tonemapped

    def inv_process(self, img):
        inverse_tonemapped = np.power(self.base, (img)/self.scale) - self.offset
        return inverse_tonemapped

class log_tonemap_clip(tonemap):

    #Constructor
    #Base of log
    #Scale of tonemapped
    #Offset
    def __init__(self, base, scale = 1, offset = 1):
        self.base = base
        self.scale = scale
        self.offset = offset

    def process(self, img):
        tonemapped = np.clip((np.log(img * self.scale + self.offset) / np.log(self.base)), 0, 2) - 1
        return tonemapped

    def inv_process(self, img):
        inverse_tonemapped = (np.power(self.base, (img + 1)) - self.offset) / self.scale
        return inverse_tonemapped


#Gamma Tonemap
class gamma_tonemap(tonemap):

    def __init__(self, gamma, ):
        self.gamma = gamma

    def process(self, img):
        tonemapped = np.power(img, 1/self.gamma)
        return tonemapped

    def inv_process(self, img):
        inverse_tonemapped = np.power(img, self.gamma)
        return inverse_tonemapped

class linear_clip(tonemap):

    def __init__(self, scale, mean):
        self.scale = scale
        self.mean = mean

    def process(self, img):
        tonemapped = np.clip((img - self.mean) / self.scale, -1, 1)
        return tonemapped
    
    def inv_process(self, img):
        inverse_tonemapped = img * self.scale + self.mean
        return inverse_tonemapped

def make_tonemap_HDR(opt):
    if opt.mode == 'luminance':
        res_tonemap = log_tonemap_clip(10, 1., 1.)
    else: #temperature
        res_tonemap = linear_clip(5000., 5000.)
    return res_tonemap
    