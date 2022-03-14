import os
import os.path
import random as rd

import matplotlib as mat
import numpy as np
import skimage
from PIL import Image
from skimage import data
from skimage.feature import hog


def rd_insert(imgs, y, desc, classe):
    rand = rd.randrange(2)
    if rand == 0:
        imgs.append(desc)
        y.append(classe)
    if rand == 1:
        imgs = [desc] + imgs
        y = [classe] + y
    return imgs, y


def resize(file):
    path = "Data/" + file
    for item in os.listdir(path):
        im = Image.open(path + "/" + item).convert('RGB')
        f, e = os.path.splitext(item)
        imResize = im.resize((30, 20), Image.ANTIALIAS)
        imResize.save('Data/SmallResized/' + file + '/' + f + 'SmallResized.jpg', 'JPEG', quality=95)


def gen_descr(fct):
    description = []
    y = []
    path = "Data/Resized/Mer"
    valid_images = [".jpg", ".jpeg"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() in valid_images:
            description.append(fct(path + "/" + f))
            y.append(1)
    path = "Data/Resized/Ailleurs"
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() in valid_images:
            description.append(fct(path + "/" + f))
            y.append(0)
    return description, y


def pixel_by_pixel(path):
    data = mat.image.imread(path)
    list_attr = []
    for ligne in data:
        for pixel in ligne:
            for couleur in pixel:
                list_attr.append(couleur)
    return list_attr

def blue_proportion_by_pix(path):
    data = mat.image.imread(path)
    list_attr = []
    for ligne in data:
        for pixel in ligne:
            if int(pixel[1]) + int(pixel[2]) + int(pixel[0]) == 0:
                couleur = 0
            else:
                couleur = int(pixel[2]) / (int(pixel[1]) + int(pixel[2]) + int(pixel[0]))
            list_attr.append(couleur)
    return list_attr


def histo(path):
    data = Image.open(path)
    return data.histogram()


def hist_blue(path):  # 0.72222 avec Gauss
    data = Image.open(path)
    r, g, b = data.split()
    return b.histogram()


def only_1_pix(path):
    data = mat.image.imread(path)
    return data[0][0]


def gradient(path):
    data = mat.image.imread(path)
    fd = hog(data, orientations=8, pixels_per_cell=(16, 16),
             cells_per_block=(1, 1), visualize=False, channel_axis=-1)
    x_gradient = fd.tolist()
    return x_gradient


def gradient_blue(path):
    data = Image.open(path)
    r, g, b = data.split()
    data = mat.image.imread(path)
    fd = hog(b, orientations=8, pixels_per_cell=(16, 16),
             cells_per_block=(1, 1), visualize=False, channel_axis=-1)
    x_gradient = fd.tolist()
    return x_gradient


def gradient_blue_hist(path):
    return gradient(path) + hist_blue(path)

def comatrice(path):
    data = mat.image.imread(path)
    data = np.array(data)
    data_reformed = []
    for i in range(len(data)):
        for j in range(len(data[0])):
            data_reformed.append([data[i][j][2], data[i][j][2], data[i][j][2]])
    glcmimg = skimage.feature.graycomatrix(data_reformed, distances=[10], angles=[0], levels=256,
                                           symmetric=True,
                                           normed=True)
    contrast = skimage.feature.graycoprops(glcmimg, 'contrast')
    homogeneite = skimage.feature.graycoprops(glcmimg, 'homogeneity')
    asm = skimage.feature.graycoprops(glcmimg, 'dissimilarity')
    list_attr = []
    for ligne in contrast:
        for pixel in ligne:
            list_attr.append(pixel)

    for ligne in homogeneite:
        for pixel in ligne:
            list_attr.append(pixel)

    for ligne in asm:
        for pixel in ligne:
            list_attr.append(pixel)
    return list_attr
