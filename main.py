import os
import datetime
import time
from tqdm import tqdm
import numpy as np
from cv2 import cv2
import tensorflow as tf

from dataprcs import CreateLabel, InvertLabel, showImage
from UNet import ignoreWarnings, useDevice, crnt_time, SaveModel, CreateUnet

def main():

    IMG_HEIGHT = 2**7
    IMG_WIDTH = 2**7
    CHANNELS = 3
    IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)

    if os.path.isdir('raw_data') == False:
        os.mkdir('raw_data')
    
    if os.path.isdir('data') == False:
        os.mkdir('data')
    
    if os.path.isdir('model') == False:
        os.mkdir('model')

    if os.path.isdir('logs') == False:
        os.mkdir('logs')

    image_file = 'top_potsdam_3_10_RGB.tif'
    label_file = 'top_potsdam_3_10_label.tif'

    if not os.listdir('raw_data/'):
    
        image = cv2.imread(f'data/{image_file}')
        image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        np.save(f'raw_data/{image_file}.npy', image)
    
        label_image = cv2.imread(f'data/{label_file}')
        label_image = cv2.resize(label_image, (2**12, 2**12))
        label_image = cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB)
        label_image = CreateLabel(label_image)
        np.save(f'raw_data/{label_file}.npy', label_image) 
    
    image = np.load(f'raw_data/{image_file}.npy')
    label_image = np.load(f'raw_data/{label_file}.npy')
    
    image = image.reshape((1024, IMG_HEIGHT, IMG_WIDTH, CHANNELS))
    label_image = label_image.reshape((1024, IMG_HEIGHT, IMG_WIDTH))

    X_train = image
    y_train = label_image

    ignoreWarnings()
    # useDevice('CPU')
    useDevice('GPU')
    model = CreateUnet(IMG_SIZE)
    time_stamp = crnt_time()

    checkPoint = tf.keras.callbacks.ModelCheckpoint(
        filepath = f'model/{time_stamp}/Checkpoint/U-Net.h5',
        monitor = 'val_acc',
        save_best_only = True,
        save_freq = 'epoch'
    )

    tensorBoard = tf.keras.callbacks.TensorBoard(log_dir = f'logs/')
    earlyStop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 3)

    model.fit(X_train, y_train, epochs = 20, callbacks = [checkPoint, tensorBoard, earlyStop], validation_split = 0.3, use_multiprocessing = True)
    SaveModel(model, time_stamp)

if __name__ == '__main__':
    main()