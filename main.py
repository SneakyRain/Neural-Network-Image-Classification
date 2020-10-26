import os
import numpy as np
from cv2 import cv2
import tensorflow as tf
from dataprcs import CreateLabel, InvertLabel, CreateData
from utils import ignoreWarnings, useDevice, crnt_time, SaveModel, showImage
from UNet import CreateUnet

IMG_HEIGHT = 2**7
IMG_WIDTH = 2**7
CHANNELS = 3
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)

time_stamp = crnt_time()

if os.path.isdir('model') == False:
    os.mkdir('model')

if os.path.isdir(f'model/{time_stamp}') == False:
    os.mkdir(f'model/{time_stamp}')
    os.mkdir(f'model/{time_stamp}/Checkpoint')

if os.path.isdir('logs') == False:
    os.mkdir('logs')

X_train, y_train = CreateData(IMG_SIZE)

ignoreWarnings()
useDevice('GPU')
model = CreateUnet(IMG_SIZE)

checkPoint = tf.keras.callbacks.ModelCheckpoint(
    filepath = f'model/{time_stamp}/Checkpoint/U-Net.hdf5',
    monitor = 'val_acc',
    save_best_only = False,
    save_freq = 'epoch'
)

tensorBoard = tf.keras.callbacks.TensorBoard(log_dir = f'logs/')
earlyStop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 3)
    
model.fit(X_train, y_train, epochs = 100, batch_size = 10, callbacks = [checkPoint, tensorBoard, earlyStop], validation_split = 0.4, use_multiprocessing = True)
SaveModel(model, time_stamp)
