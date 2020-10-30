import os
import numpy as np
from cv2 import cv2
import tensorflow as tf
from utils import ignoreWarnings, useDevice, LoadModel, showImage
from dataprcs import InvertLabel, readImage
from settings import IMG_HEIGHT, IMG_WIDTH, CHANNELS, IMG_SIZE, CLASSES, numofSamp, RESIZE_HEIGHT, RESIZE_WIDTH

def predict(model, X_test):
    print('predicting...')
    result = model.predict(X_test, verbose = 1)
        
    result = result.reshape((RESIZE_HEIGHT, RESIZE_WIDTH, CLASSES))
    result = np.argmax(result, axis = 2)
    # result = np.round(result, 0)
    result = result.astype('int')

    return result


model_name = 'U-Net'
ignoreWarnings()
useDevice('CPU')

# time_stamp = input('enter time stamp: ')
time_stamp = '23-33-59'

model = LoadModel(time_stamp, model_name, weights = True, IMG_SIZE = IMG_SIZE)

impath = 'data/RGB/top_potsdam_3_10_RGB.tif'
label_impath = 'raw_data/Label/top_potsdam_3_10_label.tif.npy'

# X_test = np.load(impath)
# y_test = np.load(label_impath)
X_test = readImage(impath, IMG_SIZE)

result = predict(model, X_test)

del X_test, model

result = InvertLabel(result)

showImage(result, waitKey = 0)
