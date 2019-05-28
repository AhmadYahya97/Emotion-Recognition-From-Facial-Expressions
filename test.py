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
        img = returnLocalizedFace(getImage(filename))

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


kf = KFold(n_splits=10,shuffle=True)
kf.get_n_splits(X)
clf = OneVsRestClassifier(SVC(kernel='linear', probability=True, tol=1e-3))

total_y_test = []
total_y_pred = []
test11 = []
test22 = []
for train_index, test_index in kf.split(X):
    # print(train_index,test_index)
    # print("_")
    train_X = fromIndexToFeatures(X, train_index)
    train_y = fromIndexToLabels(y, train_index)
    test_X = fromIndexToFeatures(X, test_index)
    test_y = fromIndexToLabels(y, test_index)

    test11.extend(test_y)

    clf.fit(train_X, train_y)

    score = clf.decision_function(test_X)
    for i in score:
        test22.append(i)

    y_pred = clf.predict(test_X)
    total_y_test.extend(test_y)
    total_y_pred.extend(y_pred)
    print('done')
    # print(train_X)
    # print()
test11 = np.asarray(test11)
test22 = np.asarray(test22)
test11 = label_binarize(y, classes=[0,1,2,3,4,5,6,7])

print(confusion_matrix(total_y_test, total_y_pred))
print(classification_report(total_y_test, total_y_pred))

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(8):
    fpr[i], tpr[i], _ = roc_curve(test11[i], test22[i])
    roc_auc[i] = auc(fpr[i], tpr[i])

lw = 2
# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(test11.ravel(), test22.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
# Plot ROC curves for the multiclass problem

# Compute macro-average ROC curve and ROC area
n_classes = 8
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue','blue','green','cyan','magenta','yellow'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(classes[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
# plt.show()
plt.savefig('roc for hog')