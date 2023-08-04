import numpy as np
import torch
from math import log10

def meanSphericalIlluminance(pano, sa):
    illum_r = np.sum(pano[:,:,0]*sa)
    illum_g = np.sum(pano[:,:,1]*sa)
    illum_b = np.sum(pano[:,:,2]*sa)
    return 0.212671 * illum_r + 0.715160 * illum_g + 0.072169 * illum_b

def mse(gt, pred, solid_angles_map=None):
    if solid_angles_map is not None:
        gt = gt * np.expand_dims(solid_angles_map, axis=-1)
        pred = pred * np.expand_dims(solid_angles_map, axis=-1)
    return np.mean(np.power(gt - pred, 2))

def wrmse(gt, pred, mask=None, solid_angles_map=None):
    if solid_angles_map is not None:
        gt = gt * np.expand_dims(solid_angles_map, axis=-1)
        pred = pred * np.expand_dims(solid_angles_map, axis=-1)
    if mask is None:
        gt = gt.flatten()
        pred = pred.flatten()
    else:
        gt = gt[mask].flatten()
        pred = pred[mask].flatten()
    error = np.sqrt(np.mean(np.power(gt - pred, 2)))

    return error

def si_wrmse(gt, pred, mask=None, solid_angles_map=None):
    if solid_angles_map is not None:
        gt = gt * np.expand_dims(solid_angles_map, axis=-1)
        pred = pred * np.expand_dims(solid_angles_map, axis=-1)
    if mask is None:
        gt_c = gt.flatten()
        pred_c = pred.flatten()
    else:
        gt_c = gt[mask].flatten()
        pred_c = pred[mask].flatten()
    alpha = (np.dot(np.transpose(gt_c), pred_c)) / (np.dot(np.transpose(pred_c), pred_c))
    error = wrmse(gt, pred * alpha, mask)

    return error

def angular_error(gt_render, pred_render, mask=None, solid_angles_map=None):
    # The error need to be computed with the normalized rgb image.
    # Normalized RGB is r = R / (R+G+B), g = G / (R+G+B), b = B / (R+G+B)
    # The angular distance is the distance between pixel 1 and pixel 2.
    # It's computed with cos^-1(p1·p2 / ||p1||*||p2||)
    if solid_angles_map is not None:
        gt_render = gt_render * np.expand_dims(solid_angles_map, axis=-1)
        pred_render = pred_render * np.expand_dims(solid_angles_map, axis=-1)

    gt_norm = np.empty((gt_render.shape))
    pred_norm = np.empty(pred_render.shape)

    for i in range(3):
        gt_norm[:,:,i] = gt_render[:,:,i] / np.sum(gt_render, axis=2, keepdims=True)[:,:,0]
        pred_norm[:,:,i] = pred_render[:,:,i] / (np.sum(pred_render, axis=2, keepdims=True)[:,:,0] + 1e-8)

    angular_error_arr = np.arccos( np.sum(gt_norm*pred_norm, axis=2, keepdims=True)[:,:,0] / 
        ((np.sqrt(np.sum(gt_norm*gt_norm, axis=2, keepdims=True)[:,:,0])*np.sqrt(np.sum(pred_norm*pred_norm, axis=2, keepdims=True)[:,:,0]))) )

    if mask is not None:
        angular_error_arr = angular_error_arr[mask[:,:,0]]
    else:
        angular_error_arr = angular_error_arr.flatten()
    angular_error_arr = angular_error_arr[~np.isnan(angular_error_arr)]
    mean = np.mean(angular_error_arr)
    # convert to degree
    mean = mean * 180 / np.pi
    return mean

def psnr(original, compressed):
    mse = wrmse(original, compressed, None)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = max(original.max(), compressed.max())
    psnr = 20 * log10(max_pixel / mse)
    return psnr