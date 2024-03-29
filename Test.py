import matplotlib as mat
import os, os.path
import cv2
import skimage
from PIL import Image
from skimage.feature import hog
import matplotlib.pyplot as plt
import numpy as np


def gradient_blue_hist_test():
    imgs = []
    hist = []
    y = []
    path = "Data/Resized/Mer"
    valid_images = [".jpg", ".jpeg"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() in valid_images:
            data_img = mat.image.imread(path + "/" + f)
            data = Image.open(path + "/" + f)
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


def hist_blue():  # 0.72222 avec Gauss
    imgs = []
    y = []
    maximum = max_bleu()
    path = "Data/Resized/Mer"
    valid_images = [".jpg", ".jpeg"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() in valid_images:
            data = Image.open(path + "/" + f)
            r, g, b = data.split()
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
            list = []
            for nb in b.histogram():
                list.append(nb / maximum)
            imgs.append(list)
            y.append(0)
    return imgs, y


def gradient_hist():
    imgs = []
    y = []
    hist = []
    path = "Data/Resized/Mer"
    valid_images = [".jpg", ".jpeg"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() in valid_images:
            data_img = mat.image.imread(path + "/" + f)
            data = Image.open(path + "/" + f)
            r, g, b = data.split()
            maximum = max(b.histogram())
            list = []
            for nb in b.histogram():
                list.append(nb / maximum)
            hist.append(list)
            imgs.append(data_img)
            y.append(1)

    path = "Data/Resized/Ailleurs"
    valid_images = [".jpg", ".jpeg"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() in valid_images:
            data_img = mat.image.imread(path + "/" + f)
            data = Image.open(path + "/" + f)
            r, g, b = data.split()
            maximum = max(b.histogram())
            list = []
            for nb in b.histogram():
                list.append(nb / maximum)
            hist.append(list)
            imgs.append(data_img)
            y.append(0)

    x_gradient = []
    for image in imgs:
        fd = hog(image, orientations=8, pixels_per_cell=(16, 16),
                 cells_per_block=(1, 1), visualize=False, channel_axis=-1)
        # print(type(fd))
        fdbis = fd.tolist()
        for img in hist:
            for pix in img:
                fdbis.append(pix)
        x_gradient.append(fdbis)

    # print(x_gradient)
    return x_gradient, y


def comatrice_hist_bleu():
    def convert_glcm(glcm):
        cpy_glcm = []
        for kol in range(0, 256):
            cpy_glcm.append([])
            for kal in range(0, 256):
                cpy_glcm[kol].append(glcm[kol][kal][0][0])
        return np.array(cpy_glcm)

    maximum = max_bleu()
    y = []
    imgs = []
    path = "Data/Mer"
    valid_images = [".jpg", ".jpeg"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() in valid_images:
            img = cv2.imread(path + "/" + f, 0)
            data = Image.open(path + "/" + f)
            r, g, b = data.split()
            list = []
            for nb in b.histogram():
                list.append(nb / maximum)

            glcmimg = skimage.feature.graycomatrix(img, [1, 2], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                                                   symmetric=True,
                                                   normed=True)
            img_glcm = cv2.normalize(np.array(glcmimg), None, 0, 255, cv2.NORM_MINMAX,
                                     cv2.CV_8U)  # potentiellement inutile
            img_glcmc = convert_glcm(img_glcm)
            # contrast = skimage.feature.greycoprops(glcmimg,'contrast')

            for ligne in img_glcmc:
                for pixel in ligne:
                    list.append(pixel)
            imgs.append(list)

            y.append(1)
    path = "Data/Ailleurs"
    valid_images = [".jpg", ".jpeg"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() in valid_images:
            img = cv2.imread(path + "/" + f, 0)
            data = Image.open(path + "/" + f)
            r, g, b = data.split()
            list = []
            for nb in b.histogram():
                list.append(nb / maximum)

            glcmimg = skimage.feature.graycomatrix(img, [1, 2], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                                                   symmetric=True,
                                                   normed=True)
            img_glcm = cv2.normalize(np.array(glcmimg), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U) # potentiellement inutile
            img_glcmc = convert_glcm(img_glcm)
            # contrast = skimage.feature.greycoprops(glcmimg,'contrast')

            for ligne in img_glcmc:
                for pixel in ligne:
                    list.append(pixel)

            imgs.append(list)
            y.append(0)
            # cv2.imshow(f, img_glcm)
            # plt.title("hrllo")
            # plt.legend()
            # plt.show()
    print("taille imgs", len(imgs))
    # for ligne in imgs:
    #     print(len(ligne))
    return imgs, y


def max_bleu():
    maxi_bleu = 0
    path = "Data/Mer"
    valid_images = [".jpg", ".jpeg"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() in valid_images:
            data = Image.open(path + "/" + f)
            r, g, b = data.split()
            if maxi_bleu <= max(b.histogram()):
                maxi_bleu = max(b.histogram())

    path = "Data/Ailleurs"
    valid_images = [".jpg", ".jpeg"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() in valid_images:
            data = Image.open(path + "/" + f)
            r, g, b = data.split()
            if maxi_bleu <= max(b.histogram()):
                maxi_bleu = max(b.histogram())

    return maxi_bleu


def comatrice():
    y = []
    imgs = []
    path = "Data/Mer"
    valid_images = [".jpg", ".jpeg"]
    maximum = max_bleu()
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() in valid_images:
            img = cv2.imread(path + "/" + f, 0)
            data = Image.open(path + "/" + f)
            r, g, b = data.split()
            list = []
            for nb in b.histogram():
                list.append(nb /maximum)

            glcmimg = skimage.feature.graycomatrix(img, [1, 2], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                                                   symmetric=True,
                                                   normed=True)

            homogeneite = skimage.feature.graycoprops(glcmimg, 'homogeneity')
            dissimilarite = skimage.feature.graycoprops(glcmimg, 'dissimilarity')

            for ligne in homogeneite:
                for pixel in ligne:
                    list.append(pixel)

            for ligne in dissimilarite:
                for pixel in ligne:
                    list.append(pixel)
            imgs.append(list)
            y.append(1)

    path = "Data/Ailleurs"
    valid_images = [".jpg", ".jpeg"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() in valid_images:
            img = cv2.imread(path + "/" + f, 0)
            data = Image.open(path + "/" + f)
            r, g, b = data.split()
            list = []
            for nb in b.histogram():
                list.append(nb/maximum)

            glcmimg = skimage.feature.graycomatrix(img, [1, 2], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                                                   symmetric=True,
                                                   normed=True)

            homogeneite = skimage.feature.graycoprops(glcmimg, 'homogeneity')
            dissimilarite = skimage.feature.graycoprops(glcmimg,'dissimilarity')

            for ligne in homogeneite:
                for pixel in ligne:
                    list.append(pixel)

            for ligne in dissimilarite:
                for pixel in ligne:
                    list.append(pixel)
            imgs.append(list)
            y.append(0)
    # print(imgs)
    print("taille imgs", len(imgs))
    return imgs, y
