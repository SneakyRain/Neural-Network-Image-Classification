import os
import numpy as np
from cv2 import cv2
import tensorflow as tf
from utils import ignoreWarnings, useDevice, LoadModel, showImage
from dataprcs import InvertLabel

def predict(model, X_test):
    print('predicting...')
    result = model.predict(X_test, verbose = 1)
        
    result = result.reshape((2**12, 2**12, 6))
    result = np.argmax(result, axis = 2)
    # result = np.round(result, 0)
    result = result.astype('int')

    return result


model_name = 'U-Net'
ignoreWarnings()
useDevice('CPU')

IMG_SIZE = (2**8, 2**8, 3)
(IMG_HEIGHT, IMG_WIDTH, CHANNELS) = IMG_SIZE
numofSamp = int(2**24/(IMG_HEIGHT*IMG_WIDTH))

# time_stamp = input('enter time stamp: ')
time_stamp = '21-55-03'

model = LoadModel(time_stamp, model_name, weights = True, IMG_SIZE = IMG_SIZE)

impath = 'raw_data/RGB/top_potsdam_3_10_RGB.tif.npy'
label_impath = 'raw_data/Label/top_potsdam_3_10_label.tif.npy'

X_test = np.load(impath)
# y_test = np.load(label_impath)
X_test = X_test.reshape((numofSamp, 2**8, 2**8, 3))

result = predict(model, X_test)

del X_test, model

result = InvertLabel(result)

showImage(result, waitKey = 0)
