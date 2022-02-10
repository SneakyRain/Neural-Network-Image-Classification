import os
import datetime
import time
from tqdm import tqdm
import numpy as np
from cv2 import cv2
from sklearn.model_selection import train_test_split
from settings import IMG_HEIGHT, IMG_WIDTH, CHANNELS, IMG_SIZE, CLASSES, numofSamp, batch_size, batches, test_size, RESIZE_HEIGHT, RESIZE_WIDTH

def countBatches(path, file, batches):
  file_count = 0
  for batch in range(batches):
    if f'{file}-batch-{batch}.npy' in os.listdir(path):
      file_count = file_count + 1
    else:
      pass

  return file_count

def readImage(img_path, IMG_SIZE):
  image = cv2.imread(img_path)
  image = cv2.resize(image, (RESIZE_HEIGHT, RESIZE_WIDTH))
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = image.reshape((numofSamp, IMG_HEIGHT, IMG_WIDTH, CHANNELS))
  return image

def CreateLabel(img):
  labeled_img = np.zeros((len(img), len(img[0,]), CLASSES))

  LB = [0, 255, 255] #Light Blue
  DB = [0, 0, 255] #Dark Blue
  G = [0, 255, 0] #Green
  W = [255, 255, 255] #White
  Y = [255, 255, 0] #Yellow
  R = [255, 0, 0] #Red

  pbar = tqdm(total = len(img), desc = 'Labelling Images', unit = 'pixels', unit_scale = len(img[0,]), disable = False, leave = False)

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
        labeled_img[x, y, :] =  [1, 0, 0, 0, 0, 0]
      elif np.prod(building):
        labeled_img[x, y, :] =  [0, 1, 0, 0, 0, 0]
      elif np.prod(trees):
        labeled_img[x, y, :] =  [0, 0, 1, 0, 0, 0]
      elif np.prod(imperv_surf):
        labeled_img[x, y, :] =  [0, 0, 0, 1, 0, 0]
      elif np.prod(car):
        labeled_img[x, y, :] =  [0, 0, 0, 0, 1, 0]
      elif np.prod(clutter):
        labeled_img[x, y, :] =  [0, 0, 0, 0, 0, 1]
    
    time.sleep(0.0001)
    pbar.update(1)
  
  pbar.close()

  return labeled_img

def InvertLabel(labeled_img):

  img = np.zeros((len(labeled_img), len(labeled_img[0,]), 3), dtype = np.uint8)

  LB = [0, 255, 255] #Light Blue
  DB = [0, 0, 255] #Dark Blue
  G = [0, 255, 0] #Green
  W = [255, 255, 255] #White
  Y = [255, 255, 0] #Yellow
  R = [255, 0, 0] #Red

  pbar = tqdm(total = len(img), desc = 'Inverting Labels', unit = 'pixels', unit_scale = len(img[0,]))

  for x in range(len(img)):
    for y in range(len(img[0,])):
      if labeled_img[x, y] == 0:
        img[x, y, :] = LB
      elif labeled_img[x, y] == 1:
        img[x, y, :] = DB 
      elif labeled_img[x, y] == 2:
        img[x, y, :] = G 
      elif labeled_img[x, y] == 3:
        img[x, y, :] = W 
      elif labeled_img[x, y] == 4:
        img[x, y, :] = Y 
      elif labeled_img[x, y] == 5:
        img[x, y, :] = R
      # else:
      #   img[x, y, :] = W
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
      raise Exception('no data to read from')

  impath = 'data'
  file_names = os.listdir(f'{impath}/Label/')

  pbar = tqdm(total = len(file_names)*batches, desc = 'Reading Data', unit = 'batch')

  for file in file_names:
    #Preparing the X_train
    image_file = file.split('_')
    image_file[-1] = 'RGB.tif'
    image_file = ('_').join(image_file)
  
    file_count = countBatches('raw_data/RGB', image_file, batches)
    if file_count < batches:
      image = cv2.imread(f'{impath}/RGB/{image_file}')
      image = cv2.resize(image, (RESIZE_HEIGHT, RESIZE_WIDTH))
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      image = image.reshape((numofSamp, IMG_HEIGHT, IMG_WIDTH, CHANNELS))
      temp = 0      
      for batch in range(batches):
        if f'{image_file}-batch-{batch}.npy' in os.listdir('raw_data/RGB'):
          pass
        else:
          batch_image = image[temp:temp+batch_size, :, :, :]
          np.save(f'raw_data/RGB/{image_file}-batch-{batch}.npy', batch_image)
        temp = temp + 1
    
    file_count = countBatches('raw_data/Label', file, batches)
    
    if file_count < batches:
      label_image = cv2.imread(f'{impath}/Label/{file}')
      label_image = cv2.resize(label_image, (RESIZE_HEIGHT, RESIZE_WIDTH))
      label_image = cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB)
      label_image = CreateLabel(label_image)
      label_image = label_image.reshape((numofSamp, IMG_HEIGHT, IMG_WIDTH, CLASSES))
      temp = 0
      for batch in range(batches):
        if f'{file}-batch-{batch}.npy' in os.listdir('raw_data/Label'):
          pass
        else:
          batch_label = label_image[temp:temp+batch_size, :, :, :]
          np.save(f'raw_data/Label/{file}-batch-{batch}.npy', batch_label)
        temp = temp + 1

    time.sleep(0.0001)
    pbar.update(batches)

  pbar.close()
  print('saving data...')

  print(f'data is saved at directory raw_data/')

def data_generator(file_names):
  # file_names = os.listdir(f'{impath}/raw_data/Label')
  while True:    
    for file in file_names:
      image_file = file.split('_')
      batch_info = image_file[-1].split('-')
      batch_info[0] = 'RGB.tif'
      image_file[-1] = ('-').join(batch_info)
      image_file = ('_').join(image_file)

      img = np.load(f'raw_data/RGB/{image_file}')
      label = np.load(f'raw_data/Label/{file}')
      yield img, label

def data_stream(filepath = 'raw_data/Label'):
  files = os.listdir(filepath)
  train_files, test_files = train_test_split(files, test_size = test_size)
  train_data_generator = data_generator(train_files)
  test_data_generator = data_generator(test_files)
  steps_per_epoch = len(train_files)
  validation_steps = len(test_files)
  return train_data_generator, test_data_generator, steps_per_epoch, validation_steps

def main():
  CreateData(IMG_SIZE)

if __name__ ==  '__main__':
  main()