import argparse

from GeometricModule import GeometricCalib


def parse():
    parser = argparse.ArgumentParser()
    
    #Used for geometric calibration
    parser.add_argument('--input_rep', type=str, default='geometric_calib_imgs', help='The path to the checkboard images to be used for geometric calibration')
    parser.add_argument('--geometricCalib_file', type=str, default='geometricCalib.pkl', help='The file where the resulting calibration is saved')
    parser.add_argument('--checkboard_x', type=int, default=10, help='The number of checkboard corners in the x direction')
    parser.add_argument('--checkboard_y', type=int, default=7, help='The number of checkboard corners in the y direction')


    opt = parser.parse_args()
    return opt

if __name__ ==  '__main__':
    opt = parse()

    gc = GeometricCalib()

    gc.calibrate(opt.input_rep, opt.geometricCalib_file, (opt.checkboard_x, opt.checkboard_y))

