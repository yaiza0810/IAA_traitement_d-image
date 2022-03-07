import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mat
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier


def classifieur_gauss(x_array, y_array, name_array=[0 for i in range(414)]):
    imgs_array = np.array(x_array)
    y_array = np.array(y_array)

    X_train, X_test, y_train, y_test, name_train, name_test = train_test_split(imgs_array, y_array, name_array,
                                                                               test_size=0.20)
    classifieur = GaussianNB()
    classifieur.fit(X_train, y_train)
    y_predits = classifieur.predict(X_test)

    print("Les vraies classes :")
    print(y_test)
    print("Les classes prédites :")
    print(y_predits)
    for i in range(len(y_test)):
        if y_test[i] != y_predits[i]:
            plt.axis('off')
            plt.imshow(mat.image.imread(name_test[i]), cmap=plt.cm.gray)
            plt.title('réel' + str(y_test[i]) + 'prédit' + str(y_predits[i]))
            plt.show()

    print(accuracy_score(y_test, y_predits))
    print(classifieur.score(X_test, y_test))


def classifieur_QDA(x_array, y_array):
    imgs_array = np.array(x_array)
    y_array = np.array(y_array)

    X_train, X_test, y_train, y_test = train_test_split(imgs_array, y_array, test_size=0.20)
    classifieur = QuadraticDiscriminantAnalysis()
    classifieur.fit(X_train, y_train)
    y_predits = classifieur.predict(X_test)

    print(accuracy_score(y_test, y_predits))


def cross_test(X, y, class_type=0):
    X, y = shuffle(X, y, random_state=1)
    classifiers = [
        GaussianNB(),
        KNeighborsClassifier(3),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        QuadraticDiscriminantAnalysis(),
        SVC(kernel='poly'),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=4),
        MLPClassifier(random_state=1, max_iter=300)
    ]
    scores = cross_val_score(estimator=classifiers[class_type], X=X, y=y, cv=10)
    print(scores)
    ecart_type, moyenne = np.std(scores), np.mean(scores)
    print(ecart_type, moyenne)


def cross_test_all(X, y):
    X, y = shuffle(X, y, random_state=1)
    classifiers = [
        GaussianNB(),
        KNeighborsClassifier(3),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        QuadraticDiscriminantAnalysis(),
        SVC(kernel='poly'),
        # SVC(kernel="linear"),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(random_state=1, max_iter=300)
    ]
    for i in range(len(classifiers)):
        print(i)
        scores = cross_val_score(estimator=classifiers[i], X=X, y=y, cv=10)
        print(scores)
        ecart_type, moyenne = np.std(scores), np.mean(scores)
        print("ecart : ", ecart_type, " moyenne : ", moyenne)


def cross_grid(X, y, class_type=0):
    X, y = shuffle(X, y, random_state=1)
    classifiers = [
        GaussianNB(),
        KNeighborsClassifier(3),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        QuadraticDiscriminantAnalysis(),
        SVC(kernel='poly'),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(random_state=1, max_iter=300)
    ]
    scores = cross_val_score(estimator=classifiers[class_type], X=X, y=y, cv=10)
    # print(scores)
    ecart_type, moyenne = np.std(scores), np.mean(scores)
    # print(ecart_type, moyenne)
    return moyenne


def cross_forest(X_train,y_train,X_test):
    classifieur = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)

    classifieur.fit(X_train, y_train)
    y_predits = classifieur.predict(X_test)

    return y_predits
