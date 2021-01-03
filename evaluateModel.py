import os
import numpy as np
import time
from tqdm import tqdm
from cv2 import cv2
import tensorflow as tf
from utils import ignoreWarnings, useDevice, LoadModel, showImage
from dataprcs import InvertLabel, readImage
from settings import IMG_HEIGHT, IMG_WIDTH, CHANNELS, IMG_SIZE, CLASSES, numofSamp, RESIZE_HEIGHT, RESIZE_WIDTH

def predict(model, X_test):
    numofSamp = len(X_test)
    pbar = tqdm(total = numofSamp, desc = 'Predicting', unit = 'batch')
    result = []
    for samp in range(numofSamp):
        test = X_test[samp]
        test = test.reshape((-1, IMG_HEIGHT, IMG_WIDTH, CHANNELS))
        prediction = model.predict(test)
        prediction = np.argmax(prediction, axis = 3)
        prediction = prediction.astype('int')
        result.append(prediction)
        time.sleep(0.0001)
        pbar.update(1)

    pbar.close()
    
    del test, prediction, X_test
    result = np.array(result)
    result = result.reshape((RESIZE_HEIGHT, RESIZE_WIDTH))
    # result = result.T
    # result = np.argmax(result, axis = 2)
    # result = result.astype('int')

    return result


model_name = 'U-Net'
ignoreWarnings()
useDevice('CPU')

# time_stamp = input('enter time stamp: ')
# time_stamp = '13-29-56'
time_stamp = '08-48-09'

model = LoadModel(time_stamp, model_name, weights = True, IMG_SIZE = IMG_SIZE)

impath = 'data/RGB/top_potsdam_5_12_RGB.tif'
label_impath = 'raw_data/Label/top_potsdam_3_10_label.tif.npy'

# X_test = np.load(impath)
# y_test = np.load(label_impath)
X_test = readImage(impath, IMG_SIZE)

result = predict(model, X_test)

del X_test, model

result = InvertLabel(result)

showImage(result, waitKey = 0)
