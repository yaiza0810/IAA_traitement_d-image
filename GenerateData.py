import matplotlib as mat
import os, os.path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure
import random as rd

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


def gen_data_all():
    imgs = []
    y = []
    path = "Data/Resized/Mer"
    valid_images = [".jpg", ".jpeg"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() in valid_images:
            data = mat.image.imread(path + "/" + f)
            imgsbis = []
            for ligne in data:
                for pixel in ligne:
                    for couleur in pixel:
                        imgsbis.append(couleur)
            rd_insert(imgs, y, imgsbis, 1)

    path = "Data/Resized/Ailleurs"
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() in valid_images:
            data = mat.image.imread(path + "/" + f)
            imgsbis = []
            for ligne in data:
                for pixel in ligne:
                    for couleur in pixel:
                        imgsbis.append(couleur)
            rd_insert(imgs, y, imgsbis, 0)

    return imgs, y

def gen_data_all_blue():
    imgs = []
    y = []
    path = "Data/Resized/Mer"
    valid_images = [".jpg", ".jpeg"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() in valid_images:
            data = mat.image.imread(path + "/" + f)
            imgsbis = []
            for ligne in data:
                for pixel in ligne:
                    # print((pixel[1]+pixel[2]+pixel[0]), "pix")
                    if int(pixel[1])+int(pixel[2])+int(pixel[0]) == 0:
                        couleur = 0
                    else:
                        couleur = int(pixel[2])/(int(pixel[1])+int(pixel[2])+int(pixel[0]))
                    imgsbis.append(couleur)
            rd_insert(imgs, y, imgsbis, 1)

    path = "Data//Resized/Ailleurs"
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() in valid_images:
            data = mat.image.imread(path + "/" + f)
            imgsbis = []
            for ligne in data:
                for pixel in ligne:
                    if int(pixel[1]) + int(pixel[2]) + int(pixel[0]) == 0:
                        couleur = 0
                    else:
                        couleur = int(pixel[2]) / (int(pixel[1]) + int(pixel[2]) + int(pixel[0]))
                    imgsbis.append(couleur)
            rd_insert(imgs, y, imgsbis, 0)

    return imgs, y

def gen_data_hist():
    imgs = []
    y = []
    path = "Data/Mer"
    valid_images = [".jpg", ".jpeg"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() in valid_images:
            data = Image.open(path + "/" + f)
            imgs.append(data.histogram())
            y.append(1)

    path = "Data/Ailleurs"
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() in valid_images:
            data = Image.open(path + "/" + f)
            imgs.append(data.histogram())
            y.append(0)

    return imgs, y


def hist_blue():  # 0.72222 avec Gauss
    imgs = []
    y = []
    count = 0
    path = "Data/Resized/Mer"
    valid_images = [".jpg", ".jpeg"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() in valid_images:
            data = Image.open(path + "/" + f)
            r, g, b = data.split()
            maximum = max(b.histogram())
            list = []
            for nb in b.histogram():
                list.append(nb / maximum)
            imgs.append(list)
            y.append(1)

    path = "Data/Resized/Ailleurs"
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() in valid_images:
            data = Image.open(path + "/" + f)
            r, g, b = data.split()
            maximum = max(b.histogram())
            list = []
            for nb in b.histogram():
                list.append(nb / maximum)
            imgs.append(list)
            y.append(0)
    # print(imgs)
    return imgs, y


def gen_data_1_px():
    imgs = []
    y = []
    path = "DATA/Mer"
    valid_images = [".jpg", ".jpeg"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() in valid_images:
            data = mat.image.imread(path + "/" + f)
            imgs.append(data[0][0])
            y.append(1)

    path = "DATA/Ailleurs"
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() in valid_images:
            data = mat.image.imread(path + "/" + f)
            imgs.append(data[0][0])
            y.append(0)

    return imgs, y


def gradient():
    imgs = []
    y = []
    path = "Data/Resized/Mer"
    valid_images = [".jpg", ".jpeg"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() in valid_images:
            data_img = mat.image.imread(path + "/" + f)
            imgs.append(data_img)
            y.append(1)

    path = "Data/Resized/Ailleurs"
    valid_images = [".jpg", ".jpeg"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() in valid_images:
            data_img = mat.image.imread(path + "/" + f)
            imgs.append(data_img)
            y.append(0)

    x_gradient = []
    for image in imgs:
        fd = hog(image, orientations=8, pixels_per_cell=(16, 16),
                 cells_per_block=(1, 1), visualize=False, channel_axis=-1)
        # print(type(fd))
        fdbis = fd.tolist()

        x_gradient.append(fdbis)
    # print(x_gradient, type(x_gradient))
    # print(y, len(y))

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    #
    # ax1.axis('off')
    # ax1.imshow(image, cmap=plt.cm.gray)
    # ax1.set_title('Input image')
    #
    # # Rescale histogram for better display
    # hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    #
    # ax2.axis('off')
    # ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    # ax2.set_title('Histogram of Oriented Gradients')
    # plt.show()

    return x_gradient, y


def gradient_blue():
    imgs = []
    y = []
    path = "Data/Resized/Mer"
    valid_images = [".jpg", ".jpeg"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() in valid_images:
            data_img = mat.image.imread(path + "/" + f)
            imgs.append(data_img)
            y.append(1)

    path = "Data/Resized/Ailleurs"
    valid_images = [".jpg", ".jpeg"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() in valid_images:
            data_img = mat.image.imread(path + "/" + f)
            imgs.append(data_img)
            y.append(0)

    x_gradient = []
    for image in imgs:
        fd = hog(image, orientations=8, pixels_per_cell=(16, 16),
                 cells_per_block=(1, 1), visualize=False, channel_axis=-1)
        # print(type(fd))
        fdbis = fd.tolist()
        for ligne in image:
            for pixel in ligne:
                # print((pixel[1]+pixel[2]+pixel[0]), "pix")
                if int(pixel[1]) + int(pixel[2]) + int(pixel[0]) == 0:
                    couleur = 0
                else:
                    couleur = int(pixel[2]) / (int(pixel[1]) + int(pixel[2]) + int(pixel[0]))
            fdbis.append(couleur)
        x_gradient.append(fdbis)
    print(x_gradient, len(x_gradient))
    print(y, len(y))
    return x_gradient, y

def gradient_blue_hist():
    imgs = []
    hist =[]
    y = []
    path = "Data/Resized/Mer"
    valid_images = [".jpg", ".jpeg"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() in valid_images:
            data_img = mat.image.imread(path + "/" + f)
            data = Image.open(path + "/" + f)
            # print(data.histogram())
            imgs.append(data_img)
            hist.append(data.histogram())
            y.append(1)

    path = "Data/Resized/Ailleurs"
    valid_images = [".jpg", ".jpeg"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() in valid_images:
            data_img = mat.image.imread(path + "/" + f)
            data = Image.open(path + "/" + f)
            imgs.append(data_img)
            hist.append(data.histogram())
            y.append(0)

    x_gradient = []
    for image in imgs:
        fd = hog(image, orientations=8, pixels_per_cell=(16, 16),
                 cells_per_block=(1, 1), visualize=False, channel_axis=-1)
        # print(type(fd))
        fdbis = fd.tolist()
        for ligne in image:
            for pixel in ligne:
                # print((pixel[1]+pixel[2]+pixel[0]), "pix")
                if int(pixel[1]) + int(pixel[2]) + int(pixel[0]) == 0:
                    couleur = 0
                else:
                    couleur = int(pixel[2]) / (int(pixel[1]) + int(pixel[2]) + int(pixel[0]))
            fdbis.append(couleur)
        for img in hist:
            for pix in img:
                fdbis.append(pix)
        x_gradient.append(fdbis)
    # print(x_gradient, len(x_gradient))
    # print(y, len(y))
    return x_gradient, y