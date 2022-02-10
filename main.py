#!/usr/bin/env python3
# 
# Author:
#
# - Aditya Bhardwaj <adityabhardwaj727@gmail.com> -- Logic and Code
#
# This code is the main file which performs semantic segmentation 
# of satellite images. This work was done in partial fulfilment of the course
# CE F417 (Application of AI in Civil Engineering), Civil Engeenering Department,
# BITS Pilani, India
# 
# Data:
# 2D Semantic Labeling Contest, ISPRS (International Society for Photogrammetry and Remote Sensing)
# More details can be found at the link provided below
# https://www2.isprs.org/commissions/comm2/wg4/benchmark/semantic-labeling/

import os
import numpy as np
from cv2 import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from dataprcs import CreateLabel, InvertLabel, CreateData, data_stream
from utils import ignoreWarnings, useDevice, crnt_time, SaveModel, showImage
from UNet import CreateUnet
from fcn8 import Createfcn_8
from settings import IMG_SIZE

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

train_generator, test_generator, steps_per_epoch, validation_steps = data_stream()

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
    
model.fit_generator(train_generator, validation_data = test_generator, steps_per_epoch = steps_per_epoch, validation_steps = validation_steps, epochs = 50, shuffle = False, callbacks = [checkPoint, tensorBoard])
SaveModel(model, time_stamp, model_name)
