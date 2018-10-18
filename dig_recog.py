from __future__ import division, print_function
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import scipy
from sklearn import decomposition
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from skimage import feature
from PIL import Image
from scipy.misc import imresize
import matplotlib.cm as cm
from scipy.interpolate import interp2d
from sklearn import svm 
import pattern_recog_func as prf
import matplotlib.gridspec as gridspec
import argparse


def plot_digits(X, Errors, nside = 5):
    gs = gridspec.GridSpec(1, nside)
    ax = [plt.subplot(gs[i]) for i in range(nside)]
    i = 0
    for i in range(len(ax)):
        ax[i].imshow(X[61 + Errors[i][0]].reshape(1,-1).reshape((8, 8)), cmap = "binary")
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        i += 1
    #plt.show()
    




dig_data = load_digits()
X = dig_data.data
y = dig_data.target
dig_img = dig_data.images

X_test = X[:60]
y_test = y[:60]

md_clf = prf.svm_train(X_test, y_test)
X_Validate = X[61:81]
y_validate = y[61:81]
Errors = []

for i in range(len(X_Validate)):
    y_pred = md_clf.predict(X_Validate[i].reshape(-1,64))
    if(y_validate[i] != y_pred[0]):
        Errors.append((i, y_validate[i], y_pred[0]))

plot_digits(dig_img,Errors, nside = len(Errors))
for i in range(len(Errors)):
    #PRINT PREDICTION THAT WAS INCORRECT
    print("index, actual digit, svm_prediction: {} {} {}".format(Errors[i][0], Errors[i][1], Errors[i][2]))
print("Success Rate: {}%".format((20-len(Errors))/20 * 100))


unseen = mpimg.imread("unseen_dig.png")
plt.imshow(dig_img[15], cmap = cm.Greys_r)
#plt.show() // Uncomment to see digit image without any changes.
unseen_not_rescaled = prf.interpol_im(unseen, dim1 = 8, dim2 = 8)
unseen = prf.rescale_pixel(X, unseen)
print("prediction for rescaled image!", md_clf.predict(unseen.reshape(1, -1))) #
print("prediction for not rescaled image!" ,md_clf.predict(unseen_not_rescaled.reshape(1,-1)))






