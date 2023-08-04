import cv2
import numpy as np
from HDR.tonemap import gamma_tonemap, log_tonemap
from matplotlib import colors
from HDR.white_balance.classes import WBsRGB as wb_srgb


class LDRfromHDR():

    def __init__(self, tonemap="none", orig_scale=False, clip=True, quantization=0, color_jitter=0, noise=0):
        self.tonemap_str = tonemap
        if tonemap=='gamma':
            self.tonemap = gamma_tonemap(2.2)
        elif tonemap=='log10':
            self.tonemap = log_tonemap(10)
        self.clip = clip
        self.orig_scale = orig_scale
        self.bits = quantization
        self.jitter = color_jitter
        self.noise = noise

        self.wbModel = None

    def process(self, HDR):
        LDR, normalized_scale = self.rescale(HDR)
        LDR = self.apply_clip(LDR)
        LDR = self.apply_scale(LDR,normalized_scale)
        LDR = self.apply_tonemap(LDR)
        LDR = self.colorJitter(LDR)
        LDR = self.gaussianNoise(LDR)
        LDR = self.quantize(LDR)
        LDR = self.apply_white_balance(LDR)
        return LDR, normalized_scale
        
    def rescale(self, img, percentile=90, max_mapping=0.8):
        r_percentile = np.percentile(img, percentile)
        alpha = max_mapping / (r_percentile + 1e-10)

        img_reexposed = img * alpha

        normalized_scale = normalizeScale(1/alpha)

        return img_reexposed, normalized_scale

    def rescaleAlpha(self, img, percentile=90, max_mapping=0.8):
        r_percentile = np.percentile(img, percentile)
        alpha = max_mapping / (r_percentile + 1e-10)

        return alpha

    def apply_clip(self,img):
        if self.clip:
            img = np.clip(img, 0, 1)
        return img

    def apply_scale(self, img, scale):
        if self.orig_scale:
            scale = unNormalizeScale(scale)
            img = img * scale
        return img

    def apply_tonemap(self, img):
        if self.tonemap_str == 'none':
            return img
        gammaed = self.tonemap.process(img)
        return gammaed

    def quantize(self, img):
        if self.bits == 0:
            return img
        max_val = np.power(2, self.bits)
        img = img * max_val
        img = np.floor(img)
        img = img / max_val
        return img

    def colorJitter(self, img):
        if self.jitter == 0:
            return img
        hsv = colors.rgb_to_hsv(img)
        hue_offset = np.random.normal(0, self.jitter, 1)
        hsv[:,:,0] = (hsv[:,:,0] + hue_offset) % 1.
        rgb = colors.hsv_to_rgb(hsv)
        return rgb

    def gaussianNoise(self, img):
        if self.noise == 0:
            return img
        noise_amount = np.random.uniform(0, self.noise, 1)
        noise_img = np.random.normal(0,noise_amount, img.shape)
        img = img + noise_img
        img = np.clip(img, 0, 1).astype(np.float32)
        return img

    def apply_white_balance(self, img):
        if self.wbModel is None:
            return img
        img = self.wbModel.correctImage(img)
        return img.copy()
    

def make_LDRfromHDR(opt):
    LDR_from_HDR = LDRfromHDR(opt.tonemap_LDR, opt.orig_scale, opt.clip, opt.quantization, opt.color_jitter, opt.noise)
    return LDR_from_HDR


def torchnormalizeEV(EV, mean=5.12, scale=6, clip=True):
    #Normalize based on the computed distribution between -1 1
    EV -= mean
    EV = EV / scale

    if clip:
        EV = torch.clip(EV, min=-1, max=1)

    return EV

def torchnormalizeEV0(EV, mean=5.12, scale=6, clip=True):
    #Normalize based on the computed distribution between 0 1
    EV -= mean
    EV = EV / scale

    if clip:
        EV = torch.clip(EV, min=-1, max=1)

    EV += 0.5
    EV = EV / 2

    return EV

def normalizeScale(x, scale=4):
    x = np.log10(x+1)

    x = x / (scale/2)
    x = x - 1

    return x

def unNormalizeScale(x, scale=4):
    x = x + 1
    x = x * (scale/2)

    x = np.power(10,x) - 1

    return x

def normalizeIlluminance(x, scale=5):
    x = np.log10(x+1)

    x = x / (scale/2)
    x = x - 1

    return x

def unNormalizeIlluminance(x, scale=5):
    x = x + 1
    x = x * (scale/2)

    x = np.power(10,x) - 1

    return x