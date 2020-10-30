import os
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, Add
from utils import ignoreWarnings, useDevice
from settings import IMG_SIZE

def Createfcn_8(IMG_SIZE, base=4):
    
    n_classes = 6
    model_name = 'fcn-8'
    b = base

    inputs = tf.keras.layers.Input(IMG_SIZE)
    s = tf.keras.layers.Lambda(lambda x: x/256)(inputs)

    # Block 1
    c1 = Conv2D(2**b, (3, 3), activation = 'elu', padding = 'same', name = 'block1_conv1')(s)
    c1 = Conv2D(2**b, (3, 3), activation = 'elu', padding = 'same', name = 'block1_conv2')(c1)
    p1 = MaxPooling2D((2, 2), strides = (2, 2), name = 'block1_pool')(c1)

    # Block 2
    c2 = Conv2D(2**(b+1), (3, 3), activation = 'elu', padding = 'same', name = 'block2_conv1')(p1)
    c2 = Conv2D(2**(b+1), (3, 3), activation = 'elu', padding = 'same', name = 'block2_conv2')(c2)
    p2 = MaxPooling2D((2, 2), strides = (2, 2), name = 'block2_pool')(c2)

    # Block 3
    c3 = Conv2D(2**(b+2), (3, 3), activation = 'elu', padding = 'same', name = 'block3_conv1')(p2)
    c3 = Conv2D(2**(b+2), (3, 3), activation = 'elu', padding = 'same', name = 'block3_conv2')(c3)
    c3 = Conv2D(2**(b+2), (3, 3), activation = 'elu', padding = 'same', name = 'block3_conv3')(c3)
    p3 = MaxPooling2D((2, 2), strides = (2, 2), name = 'block3_pool')(c3)

    # Block 4
    c4 = Conv2D(2**(b+3), (3, 3), activation = 'elu', padding = 'same', name = 'block4_conv1')(p3)
    c4 = Conv2D(2**(b+3), (3, 3), activation = 'elu', padding = 'same', name = 'block4_conv2')(c4)
    c4 = Conv2D(2**(b+3), (3, 3), activation = 'elu', padding = 'same', name = 'block4_conv3')(c4)
    p4 = MaxPooling2D((2, 2), strides = (2, 2), name = 'block4_pool')(c4)

    # Block 5
    c5 = Conv2D(2**(b+3), (3, 3), activation = 'elu', padding = 'same', name = 'block5_conv1')(p4)
    c5 = Conv2D(2**(b+3), (3, 3), activation = 'elu', padding = 'same', name = 'block5_conv2')(c5)
    c5 = Conv2D(2**(b+3), (3, 3), activation = 'elu', padding = 'same', name = 'block5_conv3')(c5)
    p5 = MaxPooling2D((2, 2), strides = (2, 2), name = 'block5_pool')(c5)

    c6 = Conv2D(2048 , (7, 7) , activation = 'elu' , padding = 'same', name = "block5_conv4")(p5)
    c6 = Dropout(0.5)(c6)
    c6 = Conv2D(2048 , (1, 1) , activation = 'elu' , padding = 'same', name = "block5_conv5")(c6)
    c6 = Dropout(0.5)(c6)

    p4_n = Conv2D(n_classes, (1, 1), activation = 'elu', padding = 'same')(p4)
    u2 = Conv2DTranspose(n_classes, kernel_size=(2, 2), strides = (2, 2), padding = 'same')(c6)
    u2_skip = Add()([p4_n, u2])

    p3_n = Conv2D(n_classes, (1, 1), activation = 'elu', padding = 'same')(p3)
    u4 = Conv2DTranspose(n_classes, kernel_size=(2, 2), strides = (2, 2), padding = 'same')(u2_skip)
    u4_skip = Add()([p3_n, u4])

    outputs = Conv2DTranspose(n_classes, kernel_size = (8, 8), strides = (8, 8), padding = 'same', activation = 'softmax')(u4_skip)

    model = tf.keras.Model(inputs = [inputs], outputs = [outputs], name = model_name)
    model.compile(optimizer = tf.optimizers.Adam(1e-4), loss = 'categorical_crossentropy', metrics = ['acc'])
    # model.summary()

    return model

def main():
  ignoreWarnings()
  useDevice('CPU')
  
  fcn_8 = Createfcn_8(IMG_SIZE)
  fcn_8.summary()

if __name__ == '__main__':
  main()