CLASSES = 6
LB = [0, 255, 255] #Light Blue
DB = [0, 0, 255] #Dark Blue
G = [0, 255, 0] #Green
W = [255, 255, 255] #White
Y = [255, 255, 0] #Yellow
R = [255, 0, 0] #Red

IMG_HEIGHT = 2**9
IMG_WIDTH = 2**9
CHANNELS = 3
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)

RESIZE_HEIGHT = 2**9
RESIZE_WIDTH = 2**9

numofSamp = int((RESIZE_HEIGHT*RESIZE_WIDTH)/(IMG_HEIGHT*IMG_WIDTH))

batch_size = 1
batches = int(numofSamp/batch_size)

test_size = 0.3

DISPLAY_HEIGHT = 1000
DISPLAY_WIDTH = 1000