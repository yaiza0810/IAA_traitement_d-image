# -*- coding: utf-8 -*-

import matplotlib as mat
import os, os.path

import numpy as np
#from PIL import Image
import matplotlib.pyplot as plt
#from skimage.feature import hog
#from skimage import data, exposure
#import random as rd


def to_blue_recursif(image):
    n = len(image)
    m = len(image[0])
    if n < 50 or m < 50:
        return image
    up = 0
    down = 0
    right = 0
    left = 0
    
    for i in range(m):
        up += image[0][i][2]
        down += image[n-1][i][2]    
    for i in range(n):
        right += image[i][m-1][2]
        left += image[i][0][2]
    
    a = up < down
    b = left < right
    return to_blue_recursif(image[0+b : m-2+b][0+a : n-2+a])


path = "Data/Resized/Mer"
valid_images = [".jpg", ".jpeg"]
for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    if ext.lower() in valid_images:
        image = mat.image.imread(path + "/" + f)
        image = image.tolist()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
        #
        ax1.axis('off')
        ax1.imshow(image, cmap=plt.cm.gray)
        ax1.set_title('Input image')
                
        ax2.axis('off')
        ax2.imshow(image, cmap=plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')
        plt.show()


