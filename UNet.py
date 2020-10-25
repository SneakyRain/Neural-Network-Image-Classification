# -*- coding: utf-8 -*-
"""Keras.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17uC0yIO3vAUUFpPrg6N-N73a1YkDmfnN
"""
import os
import datetime
import time
from tqdm.notebook import tqdm
import numpy as np
import tensorflow as tf
from cv2 import cv2
import warnings
from tensorflow.python.client import device_lib

def ignoreWarnings():
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
  # warnings.filterwarnings('ignore')

def useDevice(device = 'GPU'):
  print(f'using {device}')
  ignoreWarnings()
  if device == "GPU":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpu = device_lib.list_local_devices()[-1]
    print(gpu.physical_device_desc)
  elif device == "CPU":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    gpu = device_lib.list_local_devices()[1]
    print(gpu.physical_device_desc)
  else:
    raise Exception('enter a valid device name')

  
    # exit


def crnt_time():
  current_time = datetime.datetime.now()
  current_time = current_time.strftime('%H-%M-%S')
  return current_time

def SaveModel(model, time_stamp):
  savepath = f'model/{time_stamp}/U-Net.hdf5'
  model.save(savepath)
  print(f'time stamp is {time_stamp}, note it down as it may be used in future to recover models.')
  print(f'model is saved at /model/{time_stamp}')

def CreateUnet(IMG_SIZE):

  inputs = tf.keras.layers.Input(IMG_SIZE)
  s = tf.keras.layers.Lambda(lambda x: x/256)(inputs)

  # Contraction Path
  c1 = tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(s)
  c1 = tf.keras.layers.Dropout(0.1)(c1)
  c1 = tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c1)
  p1 = tf.keras.layers.MaxPooling2D((4, 4))(c1)

  c2 = tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(p1)
  c2 = tf.keras.layers.Dropout(0.1)(c2)
  c2 = tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c2)
  p2 = tf.keras.layers.MaxPooling2D((4, 4))(c2)

  c3 = tf.keras.layers.Conv2D(128, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(p2)
  c3 = tf.keras.layers.Dropout(0.2)(c3)
  c3 = tf.keras.layers.Conv2D(128, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c3)
  p3 = tf.keras.layers.MaxPooling2D((4, 4))(c3)

  c4 = tf.keras.layers.Conv2D(256, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(p3)
  c4 = tf.keras.layers.Dropout(0.2)(c4)
  c4 = tf.keras.layers.Conv2D(256, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c4)
  # p4 = tf.keras.layers.MaxPooling2D((4, 4))(c4)

  # c5 = tf.keras.layers.Conv2D(512, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(p4)
  # c5 = tf.keras.layers.Dropout(0.3)(c5)
  # c5 = tf.keras.layers.Conv2D(512, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c5)

  #Expansive Path

  # u6 = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides = (4, 4), padding = 'same')(c5)
  # u6 = tf.keras.layers.concatenate([u6, c4])
  # c6 = tf.keras.layers.Conv2D(256, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(u6)
  # c6 = tf.keras.layers.Dropout(0.2)(c6)
  # c6 = tf.keras.layers.Conv2D(256, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(u6)

  u7 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides = (4, 4), padding = 'same')(c4)#(c6)
  u7 = tf.keras.layers.concatenate([u7, c3])
  c7 = tf.keras.layers.Conv2D(128, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(u7)
  c7 = tf.keras.layers.Dropout(0.2)(c7)
  c7 = tf.keras.layers.Conv2D(128, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c7)

  u8 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides = (4, 4), padding = 'same')(c7)
  u8 = tf.keras.layers.concatenate([u8, c2])
  c8 = tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(u8)
  c8 = tf.keras.layers.Dropout(0.1)(c8)
  c8 = tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c8)

  u9 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides = (4, 4), padding = 'same')(c8)
  u9 = tf.keras.layers.concatenate([u9, c1], axis = 3)
  c9 = tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(u9)
  c9 = tf.keras.layers.Dropout(0.1)(c9)
  c9 = tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c9)

  outputs = tf.keras.layers.Conv2D(1, (1, 1), activation = 'sigmoid')(c9)

  model = tf.keras.Model(inputs = [inputs], outputs = [outputs])
  model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = [tf.keras.metrics.MeanIoU(num_classes = 6)])

  return model

def main():
  ignoreWarnings()
  useDevice('CPU')
  IMG_HEIGHT = 2**7
  IMG_WIDTH = 2**7
  CHANNELS = 3
  IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)
  model = CreateUnet(IMG_SIZE)
  model.summary()


if __name__ == '__main__':
  main()