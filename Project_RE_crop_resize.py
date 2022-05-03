#######################################
#Project file for Ryan Elson
#
#File to crop images based on csv values, then resize
########################################
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import scipy.io as sio
import scipy.sparse.linalg as ll
import sklearn.preprocessing as skpp
from sklearn import decomposition
import copy
import time
from scipy.sparse import csc_matrix, find, csgraph
from scipy.spatial.distance import cdist
from scipy import ndimage
from PIL import Image
import os
import fnmatch
import csv

np.random.seed(123)

#read in the data with x, y values for cropping, and class labels
with open('cardatasettrain.csv', 'r') as file:
    reader = csv.reader(file)
    cartrain = list(reader)        #cartrain data with x,y data and classes
with open('cardatasettest.csv', 'r') as file:
    reader = csv.reader(file)
    cartest = list(reader)         #cartest data with x, y data. No classes here
with open('cartest_withlabel.csv', 'r') as file:
    reader = csv.reader(file)
    cartest_labels = list(reader)    #cartest data with x,y data and classes
with open('classlabel.csv', 'r') as file:
    reader = csv.reader(file)
    classlabel = list(reader)   #all 196 different possible classes

#set size that I will resize everything to
width = 224
height = 224

#For loop, to crop images (using provided x, y vals) and resize to same dims
for i in range(1, len(cartrain)):
    #read in crop locations, labels, image name
    x1 = int(cartrain[i][1])
    y1 = int(cartrain[i][2])
    x2 = int(cartrain[i][3])
    y2 = int(cartrain[i][4])
    label = cartrain[i][5]
    imagename = cartrain[i][6]

    #setimage path, crop image
    path = 'archive/cars_train/cars_train/' + imagename
    pathout = 'archive/cars_train_processed/' + imagename
    image = Image.open(path)
    smallimage = image.crop((x1, y1, x2, y2)).resize((width, height))
    im1 = smallimage.save(pathout)

#For loop, to crop images (using provided x, y vals) and resize to same dims
for i in range(1, len(cartest_labels)):
    #read in crop locations, labels, image name
    x1 = int(cartest_labels[i][1])
    y1 = int(cartest_labels[i][2])
    x2 = int(cartest_labels[i][3])
    y2 = int(cartest_labels[i][4])
    label = cartest_labels[i][5]
    imagename = cartest_labels[i][6]

    #setimage path, crop image
    path = 'archive/cars_test/cars_test/' + imagename
    pathout = 'archive/cars_test_processed/' + imagename
    image = Image.open(path)
    smallimage = image.crop((x1, y1, x2, y2)).resize((width, height))
    im1 = smallimage.save(pathout)

#imagename = cartest_labels[1][6]
#print(os.getcwd())

#image = Image.open(path)
#plt.imshow(image)
#plt.show()