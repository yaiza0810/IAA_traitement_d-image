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


image, y = Test.comatrice_hist_bleu()
# X,y = GenerateData.hist_blue()
Classifieur.cross_test_all(image, y)
# Classifieur.halving_grid(image,y)


# #Test sur le model de la prof
# path = "Data/AllTest"
# X_test = Model_test.comatrice_2(path)
# X_train, y_train = Test.comatrice()
# # y_predit = Classifieur.cross_mlp(X_train, y_train, X_test)
# y_predit = Classifieur.cross_forest(X_train, y_train, X_test)
#
#
# import numpy as np
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
# print(y_predit)
# print(y_reel)
#
# print(accuracy_score(y_reel, y_predit))
#
#
# # # path = "Data/AllTest"
# # X_test = Model_test.hist_blue(path)
# # X_train, y_train = Test.hist_blue()
# # y_predit = Classifieur.cross_forest(X_train,y_train,X_test)
#
# # print(accuracy_score(y_reel, y_predit))
