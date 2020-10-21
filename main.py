import os
import datetime
import time
from tqdm import tqdm
import numpy as np
from cv2 import cv2

from dataprcs import CreateLabel, InvertLabel

def main():

    if os.path.isdir('raw_data') == False:
        os.mkdir('raw_data')
    
    if os.path.isdir('data') == False:
        os.mkdir('data')

    image_file = 'top_potsdam_3_10_RGB.tif'
    label_file = 'top_potsdam_3_10_label.tif'

    if not os.listdir('raw_data/'):
    
        image = cv2.imread(f'data/{image_file}')
        image = cv2.resize(image, (2**12, 2**12))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        np.save(f'raw_data/{image_file}.npy', image)
    
        label_image = cv2.imread(f'data/{label_file}')
        label_image = cv2.resize(label_image, (2**12, 2**12))
        label_image = cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB)
        label_image = CreateLabel(label_image)
        np.save(f'raw_data/{label_file}.npy', label_image)  

if __name__ == '__main__':
    main()