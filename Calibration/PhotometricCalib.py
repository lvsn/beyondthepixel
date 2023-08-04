import argparse
from os import environ
environ["OPENCV_IO_ENABLE_OPENEXR"] = "true"
import os,sys
import cv2
import numpy as np
from glob import glob
import sys
import csv
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

from GeometricModule import GeometricCalib


def parse():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_rep', type=str, default='imgs', help='The path to the images to be used for photometric calibration in .exr format')
    parser.add_argument('--geometricCalib_file', type=str, default='geometricCalib.pkl', help='The file where the resulting calibration is saved')

    opt = parser.parse_args()
    return opt

def ComputeAverage(img,mask):
    mean = np.zeros((3))
    mean[0] = (img[:,:,0] * mask).sum() / mask.sum()
    mean[1] = (img[:,:,1] * mask).sum() / mask.sum()
    mean[2] = (img[:,:,2] * mask).sum() / mask.sum()

    integrated = np.pi * mean

    return integrated

if __name__ ==  '__main__':
    opt = parse()

    calib = GeometricCalib()
    calib.loadKD(opt.geometricCalib_file)
    
    input_imgs = glob(os.path.join(opt.input_rep, '*.exr'))

    R_illum = []
    G_illum = []
    B_illum = []

    R_cam = []
    G_cam = []
    B_cam = []

    for input_img in input_imgs:
        img = cv2.imread(input_img, cv2.IMREAD_UNCHANGED)[:,:,:3]
        cosCorrect = calib.createHemisphericalSelf(img)

        mask = np.zeros((cosCorrect.shape[0],cosCorrect.shape[1]))
        mask = cv2.circle(mask, (mask.shape[0]//2,mask.shape[1]//2), mask.shape[0]//2, 1, -1)
        mask = mask.astype(np.uint8)

        integrated = ComputeAverage(cosCorrect,mask)
        R_cam.append(integrated[0])
        G_cam.append(integrated[1])
        B_cam.append(integrated[2])

        with open(input_img[:-3]+"csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            XYZ = next(csv_reader)

        XYZ = np.array(XYZ).astype(np.float)
        RGB_illum = np.array([[3.240479,-1.53715,-0.498535],
                            [-0.969256,1.875991,0.041556],
                            [0.055648,-0.204043,1.057311]])@XYZ
        R_illum.append(RGB_illum[0])
        G_illum.append(RGB_illum[1])
        B_illum.append(RGB_illum[2])
        

    #Linear regressions
    fig, ax = plt.subplots(1, 1)

    R_illum = np.array(R_illum)
    G_illum = np.array(G_illum)
    B_illum = np.array(B_illum)

    R_cam = np.array(R_cam)
    G_cam = np.array(G_cam)
    B_cam = np.array(B_cam)

    ax.scatter(R_cam, R_illum, color="red")
    ax.scatter(G_cam, G_illum, color="green")
    ax.scatter(B_cam, B_illum, color="blue")

    regr_R = linear_model.LinearRegression(fit_intercept = False)
    regr_R.fit(R_cam.reshape(-1, 1), R_illum.reshape(-1, 1))
    R_pred = regr_R.predict(R_cam.reshape(-1, 1)).reshape(-1)
    ax.plot(R_cam, R_pred, color="r", linestyle='dotted')
    print("R_coeffs: ", regr_R.coef_, " + ", regr_R.intercept_)
    print("R_r2: ", r2_score(R_illum, R_pred))

    regr_G = linear_model.LinearRegression(fit_intercept = False)
    regr_G.fit(G_cam.reshape(-1, 1), G_illum.reshape(-1, 1))
    G_pred = regr_G.predict(G_cam.reshape(-1, 1)).reshape(-1)
    ax.plot(G_cam, G_pred, color="g", linestyle='dotted')
    print("G_coeffs: ", regr_G.coef_, " + ", regr_G.intercept_)
    print("G_r2: ", r2_score(G_illum, G_pred))

    regr_B = linear_model.LinearRegression(fit_intercept = False)
    regr_B.fit(B_cam.reshape(-1, 1), B_illum.reshape(-1, 1))
    B_pred = regr_B.predict(B_cam.reshape(-1, 1)).reshape(-1)
    ax.plot(B_cam, B_pred, color="b", linestyle='dotted')
    print("B_coeffs: ", regr_B.coef_, " + ", regr_B.intercept_)
    print("B_r2: ", r2_score(B_illum, B_pred))

    plt.show()