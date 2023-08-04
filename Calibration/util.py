import cv2
import numpy as np

def rescale(image, size):
    #size is the size of the larger side
    shape = image.shape
    if shape[0] > shape[1]:
        scale = size / shape[0]
        res = cv2.resize(image, (int(shape[1]*scale),size))
    else:
        scale = size / shape[1]
        res = cv2.resize(image, (size,int(shape[0]*scale)))
    
    return res, scale

def display(img):
    img_rescale, _ = rescale(img, 1000)
    cv2.imshow("hdr",img_rescale)
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()

def crop(img, mini, maxi, buffer):
    #buffer is num pixels bigger than rect
    mini = mini - buffer
    maxi = maxi + buffer
    mini = np.maximum(mini, [0,0])
    maxi = np.minimum(maxi, [img.shape[1],img.shape[0]])
    
    res = img[mini[1]:maxi[1],mini[0]:maxi[0]]
    
    return res, mini

def check_extension(list):
    extensions = [ '.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
    files = [ file for file in list if not file.endswith( extensions ) ]
    return files