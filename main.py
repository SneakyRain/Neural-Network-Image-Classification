import os
import numpy as np
from cv2 import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from dataprcs import CreateLabel, InvertLabel, CreateData
from utils import ignoreWarnings, useDevice, crnt_time, SaveModel, showImage
from UNet import CreateUnet
from fcn8 import Createfcn_8

IMG_HEIGHT = 2**8
IMG_WIDTH = 2**8
CHANNELS = 3
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)

model_name = 'U-Net'

time_stamp = crnt_time()

if os.path.isdir('model') == False:
    os.mkdir('model')

if os.path.isdir(f'model/{time_stamp}') == False:
    os.mkdir(f'model/{time_stamp}')
    os.mkdir(f'model/{time_stamp}/Checkpoint')
    os.mkdir(f'model/{time_stamp}/Weights')

if os.path.isdir('logs') == False:
    os.mkdir('logs')

X_train, y_train = CreateData(IMG_SIZE)

# X_train, X_test, y_train, y_test = train_test_split(X_train, y_train)

ignoreWarnings()
useDevice('CPU')
model = CreateUnet(IMG_SIZE)

checkPoint = tf.keras.callbacks.ModelCheckpoint(
    filepath = f'model/{time_stamp}/Checkpoint/{model_name}.hdf5',
    monitor = 'val_acc',
    save_best_only = False,
    save_freq = 'epoch'
)

tensorBoard = tf.keras.callbacks.TensorBoard(log_dir = f'logs/')
# earlyStop = tf.keras.callbacks.EarlyStopping(monitor = 'val_acc', patience = 3)
    
model.fit(X_train, y_train, epochs = 50, shuffle = False, callbacks = [checkPoint, tensorBoard], validation_split = 0.3, use_multiprocessing = True)
SaveModel(model, time_stamp, model_name)
