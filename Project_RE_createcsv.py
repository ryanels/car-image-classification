#######################################
#Project file for Ryan Elson
#
#
# generate csv from mat file for car test data,
#
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

matFile = sio.loadmat('archive\cars_annos.mat')

#read in class descriptions, index
temp1 = []
for j in range(len(matFile['class_names'][0])):
    t1 = matFile['class_names'][0][j][0]
    temp1.append((j+1,t1))
headers = ['index', 'Class']
with open('classlabel.csv', 'w', newline = '') as f:
    write = csv.writer(f)
    write.writerow(headers)
    write.writerows(temp1)

#read in file for test data with correct labels
matFile = sio.loadmat('archive\cars_test_annos_withlabels.mat')
m, n = matFile['annotations'].shape  #1 x 8041

temp = []
for i in range(n):
    t1 = matFile['annotations'][0][i][0][0][0]  #first value
    t2 = matFile['annotations'][0][i][1][0][0]  #2nd value
    t3 = matFile['annotations'][0][i][2][0][0]  #3rd value
    t4 = matFile['annotations'][0][i][3][0][0]  #4th value
    t5 = matFile['annotations'][0][i][4][0][0]  #5th value
    t6 = matFile['annotations'][0][i][5][0]  #6th, image name
    temp.append((t1, t2, t3, t4, t5, t6))

#write car test data with labels to csv
headers = ['x1', 'y1', 'x2', 'y2', 'class', 'testimage']
with open('cartest_withlabel.csv', 'w', newline = '') as f:
    write = csv.writer(f)
    write.writerow(headers)
    write.writerows(temp)

#show a test image
imagename = matFile['annotations'][0][0][5][0]
print(os.getcwd())

path = 'archive/cars_test/cars_test/' + imagename

image = Image.open(path)
plt.imshow(image)
plt.show()