import os
import datetime
import time
from tqdm import tqdm
import numpy as np
from cv2 import cv2

def CreateLabel(img):
  labeled_img = np.zeros((len(img), len(img[0,])))

  LB = [0, 255, 255] #Light Blue
  DB = [0, 0, 255] #Dark Blue
  G = [0, 255, 0] #Green
  W = [255, 255, 255] #White
  Y = [255, 255, 0] #Yellow
  R = [255, 0, 0] #Red

  pbar = tqdm(total = len(img), desc = 'Labelling Images', unit = 'pixels', unit_scale = 2**12, disable = False, leave = False)

  for x in range(len(img)):
    for y in range(len(img[0,])):
      
      # Check whether the pixel falls into which category
      vegetation = img[x, y, :] == LB
      building = img[x, y, :] == DB
      trees = img[x, y, :] == G
      imperv_surf = img[x, y, :] == W
      car = img[x, y, :] == Y
      clutter = img[x, y, :] == R

      # Label the image accordingly
      if np.prod(vegetation):
        labeled_img[x, y] =  0
      elif np.prod(building):
        labeled_img[x, y] =  1
      elif np.prod(trees):
        labeled_img[x, y] =  2
      elif np.prod(imperv_surf):
        labeled_img[x, y] =  3
      elif np.prod(car):
        labeled_img[x, y] =  4
      elif np.prod(clutter):
        labeled_img[x, y] =  5
    
    time.sleep(0.0001)
    pbar.update(1)
  
  pbar.close()

  return labeled_img

def InvertLabel(labeled_img):

  img = np.zeros((len(labeled_img), len(labeled_img[0,]), 3), dtype = np.uint8)

  LB = [0, 255, 255] #Light Blue
  DB = [0, 0, 255] #Dark Blue
  G = [0, 255, 9] #Green
  W = [255, 255, 255] #White
  Y = [255, 255, 0] #Yellow
  R = [255, 0, 0] #Red

  pbar = tqdm(total = len(img), desc = 'Inverting Labels', unit = 'pixels')

  for x in range(len(img)):
    for y in range(len(img[0,])):
      if labeled_img[x, y] ==  0:
        img[x, y, :] = LB
      elif labeled_img[x, y] ==  1:
        img[x, y, :] = DB 
      elif labeled_img[x, y] ==  2:
        img[x, y, :] = G 
      elif labeled_img[x, y] == 3:
        img[x, y, :] = W 
      elif labeled_img[x, y] ==  4:
        img[x, y, :] = Y 
      elif labeled_img[x, y] == 5:
        img[x, y, :] = R
    time.sleep(0.0001)
    pbar.update(1)
  
  pbar.close()

  return img

def CreateData():
  
  impath = '/media/aditya/DATA/Test/Satellite Images/'
  file_names = os.listdir(f'{impath}/Potsdam/Raw Data/LabelsForParticipants')

  X_train = np.zeros((len(file_names), 2**12, 2**12, 3))
  y_train = np.zeros((len(file_names), 2**12, 2**12))

  pbar = tqdm(total = len(file_names), desc = 'Reading Data', unit = 'files')

  for file, i in zip(file_names, range(len(file_names))):
    #Preparing the X_train
    image_file = file.split('_')
    image_file[-1] = 'IRRG.tif'
    image_file = ('_').join(image_file)
    

    if f'{image_file}.npy' in os.listdir(f'{impath}/data'):
      image = np.load(f'{impath}/data/{image_file}.npy')
    else:
      image = cv2.imread(f'{impath}/Potsdam/Raw Data/IRRG/{image_file}')
      image = cv2.resize(image, (2**12, 2**12))
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      np.save(f'{impath}/data/{image_file}.npy', image)

    if f'{file}.npy' in os.listdir(f'{impath}/data'):
      label_image = np.load(f'{impath}/data/{file}.npy')
    else:
      label_image = cv2.imread(f'{impath}/Potsdam/Raw Data/LabelsForParticipants/{file}')
      label_image = cv2.resize(label_image, (2**12, 2**12))
      label_image = cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB)
      label_image = CreateLabel(label_image)
      np.save(f'{impath}/data/{file}.npy', label_image)
    
    X_train[i] = image
    y_train[i] = label_image

    time.sleep(0.0001)
    pbar.update(1)
  
  pbar.close()

  print('Saving the Data...')

  np.save(f'{impath}/data/X_train.npy', X_train)
  np.save(f'{impath}/data/y_train.npy', y_train)

  print(f'Data is saved at {impath}/data/')

  return X_train, y_train


def main():
    impath = '/media/aditya/DATA/Test/Satellite Images/'
    if os.path.isdir(f'{impath}/data') == False:
        os.mkdir(f'{impath}/data')
    CreateData()

if __name__ ==  '__main__':
  main()