import os
import datetime
import warnings
from cv2 import cv2
import tensorflow as tf
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

def crnt_time():
  
  current_time = datetime.datetime.now()
  current_time = current_time.strftime('%H-%M-%S')
  return current_time

def SaveModel(model, time_stamp):
  
  print('saving model ...')
  savepath = f'model/{time_stamp}/U-Net.hdf5'
  model.save(savepath)
  print(f'time stamp is {time_stamp}, note it down as it may be used in future to recover models.')
  print(f'model is saved at /model/{time_stamp}')

def LoadModel(time_stamp):
  
  print('loading model...')
  savepath = f'model/{time_stamp}/U-Net.hdf5'
  model = tf.keras.models.load_model(savepath)
  return model

def showImage(img, label = '', waitKey = 10**4):

  img = cv2.resize(img, (2**10, 2**10))
  img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
  print('press any button to close the window')
  
  if waitKey > 0:
    print(f'displaying image for {waitKey/1000} s')
  
  cv2.imshow(label, img)
  cv2.waitKey(waitKey)
  cv2.destroyAllWindows()
