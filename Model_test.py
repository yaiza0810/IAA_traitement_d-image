import matplotlib as mat
import os, os.path
import cv2
import skimage
from PIL import Image
from skimage.feature import hog
import matplotlib.pyplot as plt
import numpy as np



def hist_blue(path):  # 0.72222 avec Gauss
    imgs = []
    for f in os.listdir(path):
        data = Image.open(path + "/" + f)
        r, g, b = data.split()
        maximum = max(b.histogram())
        list = []
        for nb in b.histogram():
            list.append(nb / maximum)
        imgs.append(list)
    return imgs


def comatrice(path):
    imgs = []
    # valid_images = [".jpg", ".jpeg"]
    for f in os.listdir(path):

        img = cv2.imread(path + "/" + f, 0)
        data = Image.open(path + "/" + f)
        r, g, b = data.split()
        maximum = max(b.histogram())
        list = []
        for nb in b.histogram():
            list.append(nb / maximum)
        glcmimg = skimage.feature.graycomatrix(img, distances=[10], angles=[0], levels=256,
                                               symmetric=True,
                                               normed=True)
        contrast = skimage.feature.greycoprops(glcmimg, 'contrast')
        homogeneite = skimage.feature.greycoprops(glcmimg, 'homogeneity')
        asm = skimage.feature.greycoprops(glcmimg, 'dissimilarity')

        for ligne in contrast:
            for pixel in ligne:
                list.append(pixel)

        for ligne in homogeneite:
            for pixel in ligne:
                list.append(pixel)

        for ligne in asm:
            for pixel in ligne:
                list.append(pixel)
        imgs.append(list)

    print("taille imgs", len(imgs))
    return imgs
