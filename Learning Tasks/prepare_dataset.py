from os import environ
environ["OPENCV_IO_ENABLE_OPENEXR"] = "true"
from envmap import EnvironmentMap
import numpy as np
import cv2
import argparse
import os
from glob import glob

def treatPano(pano_path, mode, sizes=[64,128], outpaths=["dataset/calibrated64_inpaint/","dataset/calibrated128_inpaint/"]):
    #load
    e = EnvironmentMap(pano_path, 'latlong')

    #inpaint
    mask = np.all(e.data == [0,0,0], axis=-1)
    first = np.argmax(mask, axis=0)
    tocopy = first - 4 # to avoid the anti aliased pixel
    values = e.data[tocopy, np.arange(e.data.shape[1])]
    for i in range(3):
        mask[first-i, np.arange(e.data.shape[1])] = True
    indexes = np.argwhere(mask == True)
    e.data[indexes[:,0], indexes[:,1]] = values[indexes[:,1]]

    #resize and save
    for i in range(len(sizes)):
        e_resized = e.copy().resize(sizes[i])

        parent = os.path.join(os.path.dirname(__file__), outpaths[i], mode)
        filename = os.path.join(parent, pano_path.split("/")[-1])
        os.makedirs(parent, exist_ok=True)
    
        img_correct_BGR = cv2.cvtColor(e_resized.data, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, img_correct_BGR)

def prepareDataset(path):

    train_path = os.path.join(os.path.dirname(__file__), "util/train.txt")
    test_path = os.path.join(os.path.dirname(__file__), "util/test.txt")
    val_path = os.path.join(os.path.dirname(__file__), "util/val.txt")

    count = 0

    #train
    with open(train_path, "r") as train_file:
        for line in train_file:
            count += 1
            print(str(count), "/2362")
            pano = line.rstrip()
            pano_path = os.path.join(path, pano) 
              
            treatPano(pano_path, 'train')

    #test
    with open(test_path, "r") as test_file:
        for line in test_file:
            count += 1
            print(str(count), "/2362")
            pano = line.rstrip()
            pano_path = os.path.join(path, pano)  
                
            treatPano(pano_path, 'test')

    #val
    with open(val_path, "r") as val_file:
        for line in val_file:
            count += 1
            print(str(count), "/2362")
            pano = line.rstrip()
            pano_path = os.path.join(path, pano)  
                
            treatPano(pano_path, 'val')

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='prepare_dataset.py',
        description='Prepares the dataset for learning tasks. Inpaints, rescale and split train/test/val.')

    parser.add_argument('path', type=str, help='Path to complete dataset')

    args = parser.parse_args()

    prepareDataset(args.path)