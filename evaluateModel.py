import os
import numpy as np
from cv2 import cv2
import tensorflow as tf
from utils import ignoreWarnings, useDevice, LoadModel, showImage
from dataprcs import InvertLabel

def predict(model, X_test):
    
    result = model.predict(X_test)
    
    result = result.reshape((2**12, 2**12))
    result = np.round(result, 0)
    result = result.astype('int')

    return result


ignoreWarnings()
useDevice('CPU')

time_stamp = input('enter time stamp: ')
# time_stamp = '00-38-10'
model = LoadModel(time_stamp)

impath = 'raw_data/RGB/top_potsdam_3_10_RGB.tif.npy'
label_impath = 'raw_data/Label/top_potsdam_3_10_label.tif.npy'

X_test = np.load(impath)
# y_test = np.load(label_impath)

result = predict(model, X_test)

del X_test, model

result = InvertLabel(result)

showImage(result, waitKey = 0)
