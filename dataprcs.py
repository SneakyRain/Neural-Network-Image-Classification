import os
import datetime
import time
from tqdm import tqdm
import numpy as np
from cv2 import cv2

def showImage(img, label = '', waitKey = 10**4):
  img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
  print('press any button to close the window')
  
  if waitKey > 0:
    print(f'displaying image for {waitKey/1000} s')
  
  cv2.imshow(label, img)
  cv2.waitKey(waitKey)
  cv2.destroyAllWindows()
  
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

def CreateData(IMG_SIZE):

  if os.path.isdir('raw_data') == False:
        os.mkdir('raw_data')
        os.mkdir('raw_data/Label')
        os.mkdir('raw_data/RGB')
    
  if os.path.isdir('data') == False:
      os.mkdir('data')
      os.mkdir('data/Label')
      os.mkdir('data/RGB')
  
  (IMG_HEIGHT, IMG_WIDTH, CHANNELS) = IMG_SIZE
  numofSamp = int(2**24/(IMG_HEIGHT*IMG_WIDTH))

  impath = 'data'
  file_names = os.listdir(f'{impath}/Label/')

  X_train = np.zeros((IMG_HEIGHT, IMG_WIDTH, CHANNELS))
  y_train = np.zeros((IMG_HEIGHT, IMG_WIDTH))

  pbar = tqdm(total = len(file_names), desc = 'Reading Data', unit = 'files')

  for file, i in zip(file_names, range(len(file_names))):
    #Preparing the X_train
    image_file = file.split('_')
    image_file[-1] = 'RGB.tif'
    image_file = ('_').join(image_file)
    

    if f'{image_file}.npy' in os.listdir('raw_data/RGB'):
      image = np.load(f'raw_data/RGB/{image_file}.npy')
    else:
      image = cv2.imread(f'{impath}/RGB/{image_file}')
      image = cv2.resize(image, (2**12, 2**12))
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      np.save(f'raw_data/RGB/{image_file}.npy', image)

    if f'{file}.npy' in os.listdir('raw_data/Label'):
      label_image = np.load(f'raw_data/Label/{file}.npy')
    else:
      label_image = cv2.imread(f'{impath}/Label/{file}')
      label_image = cv2.resize(label_image, (2**12, 2**12))
      label_image = cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB)
      label_image = CreateLabel(label_image)
      np.save(f'raw_data/Label/{file}.npy', label_image)
    
    if i ==0:
      X_train = image.reshape((numofSamp, IMG_HEIGHT, IMG_WIDTH, CHANNELS))
      y_train = label_image.reshape((numofSamp, IMG_HEIGHT, IMG_WIDTH))
    else:
      image = image.reshape((numofSamp, IMG_HEIGHT, IMG_WIDTH, CHANNELS))
      label_image = label_image.reshape((numofSamp, IMG_HEIGHT, IMG_WIDTH))
      X_train = np.append(X_train, image, axis = 0)
      y_train = np.append(y_train, label_image, axis = 0)
    
    time.sleep(0.0001)
    pbar.update(1)
  
  pbar.close()
  print('saving data...')

  np.save(f'raw_data/X_train.npy', X_train)
  np.save(f'raw_data/y_train.npy', y_train)

  print(f'data is saved at directory raw_data/')
  print('loading data...')

  return X_train, y_train

def main():

  IMG_HEIGHT = 2**7
  IMG_WIDTH = 2**7
  CHANNELS = 3
  IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)

  CreateData(IMG_SIZE)

if __name__ ==  '__main__':
  main()