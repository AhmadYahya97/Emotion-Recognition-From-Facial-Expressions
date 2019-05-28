import math

import imutils
import numpy as np
import cv2 as cv2
import glob
import csv
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
# from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from scipy import interp
from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import SVC
import pickle


def returnLocalizedFace(img):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray =cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        # cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
    # print(faces)
    # cv2.imshow/('im',img)
    # cv2.waitKey(0)
    if len(faces) == 0:
        return img
    crop_img = img[y:y + h, x:x + w]
    return crop_img


def getImage(path):
    return cv2.imread(path)
def show(img):
    cv2.imshow('im', img)
    cv2.waitKey(0)

X = []
y = []
def read(imagesPath, label):

    for filename in glob.glob(imagesPath + '/*.*'):
        # print(filename.split(imagesPath + '/')[1])
        # print(filename)
        win_size = (64, 128)
        img = getImage(filename)

        win_size = (64, 128)

        img = cv2.resize(img, win_size)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        d = cv2.HOGDescriptor()
        hog = d.compute(img)
        X.append(hog.transpose()[0])
        y.append(label)

def fromIndexToFeatures(X, indecies):
    features = []
    for i in indecies:
        features.append(X[i])
    return np.asarray(features)

def fromIndexToLabels(y, indecies):
    labels = []
    for i in indecies:
        labels.append(y[i])
    return np.asarray(labels)
#
read('HAPPY',0)
read('CONTEMPT',1)
read('ANGER',2)
read('DISGUST',3)
read('FEAR',4)
read('SADNESS',5)
read('SURPRISE',6)
read('NEUTRAL',7)
classes = ["HAPPY", "CONTEMPT", "ANGER", "DISGUST", "FEAR", "SADNESS", "SURPRISE", "NEUTRAL"]
y = np.asarray(y)
X = np.asarray(X)
#
#
#
clf = OneVsRestClassifier(SVC(kernel='linear', probability=True, tol=1e-3))
clf.fit(X, y)
# #
filename = 'finalized_model.sav'
pickle.dump(clf, open(filename, 'wb'))
#
#
clf = pickle.load(open(filename, 'rb'))
#
img = getImage('testImages\surprise.png')
#
#
win_size = (64, 128)
#
img = cv2.resize(img, win_size)
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
d = cv2.HOGDescriptor()
hog = d.compute(img)
#
hog = hog.transpose()[0]
hog = np.asarray(hog)
print(hog)
print(clf.predict([hog]))
# # print(X)
# # print(y)
