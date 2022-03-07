import numpy as np
from skimage.feature import greycomatrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import Classifieur
import GenerateData
import Model_test
import Test


def resize_all():
    path_Mer = "Mer"
    path_ailleurs = "Ailleurs"
    GenerateData.resize(path_Mer)
    GenerateData.resize(path_ailleurs)


# resize_all()
# imgbis, ybis = GenerateData.gen_data_all()
# GenerateData.gen_data_1_px()
# imgs,y,name = GenerateData.gen_data_hist_resized()
# im,y2 = GenerateData.gradient_blue_hist()
# imgs,y = GenerateData.gen_data_hist_resized()
# Classifieur.classifieur_gauss(imgs,y, name)
# Classifieur.cross_test_all(imgbis,ybis)
# Classifieur.cross_test(imgs,y,6)
# Classifieur.classifieur_gauss(im,y2)
# Classifieur.classifieur_gauss(imgbis,ybis)
# Classifieur.cross_test(imgs,y)
# print("4")
# Classifieur.cross_test(im,y2,4)
# print("normal")
# Classifieur.cross_test(im,y2,6)
# print("5")
# Classifieur.cross_test(im,y2,5)
# x, y = GenerateData.gradient()
# Classifieur.classifieur_gauss(x, y)
# Classifieur.cross_test(x,y)
# Classifieur.classifieur_QDA(imgs,y)

# image, y = Test.gradient_blue_hist_test()
# Classifieur.cross_test_all(image,y)
# imgs, y1 = GenerateData.gen_data_all()
# Classifieur.cross_test_all(imgs, y1)
#

# for distance in range(5, 20, 5):
# for angle in range(5,20, 5):
image, y = Test.comatrice()
# X,y = GenerateData.hist_blue()
# for max_d in range(1,10):
#     for estim in range(1,20):
#         for f in range(1,5):
#             print(max_d,estim,f)
#             X, y = shuffle(X, y, random_state=1)
#             RandomForestClassifier(max_depth=max_d, n_estimators=estim, max_features=f)
#             scores = cross_val_score(estimator= RandomForestClassifier(max_depth=max_d, n_estimators=estim, max_features=f), X=X, y=y, cv=10)
#             ecart_type, moyenne = np.std(scores), np.mean(scores)
#             print(moyenne)
Classifieur.cross_test_all(image, y)

# path = "Data/AllTest"
# # X_test = Model_test.comatrice(path)
# # X_train, y_train = Test.comatrice()
# # y_predit = Classifieur.cross_forest(X_train, y_train, X_test)
#
# # import numpy as np
# import os
#
#
# def compute_labels(rep):
#     labels = []
#     dico_labels = {}
#     for filename in os.listdir(rep)[0:]:
#         # print(filename)
#         y = 1
#         if (int(filename[0]) == 0):
#             y = 0
#         labels.append(y)
#         # dico_labels[filename] = y
#     return(np.array(labels))
#     # return (dico_labels)
#
#
# y_reel = compute_labels(path)
#
# # print(y_predit)
# # print(y_reel)
#
# # print(accuracy_score(y_reel, y_predit))
#
#
#
# # path = "Data/AllTest"
# X_test = Model_test.hist_blue(path)
# X_train, y_train = Test.hist_blue()
# y_predit = Classifieur.cross_forest(X_train,y_train,X_test)
#
# print(accuracy_score(y_reel, y_predit))
