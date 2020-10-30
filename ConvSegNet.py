import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, ZeroPadding2D, UpSampling2D, Reshape
from tensorflow.keras.activations import softmax
from utils import ignoreWarnings, useDevice
from settings import IMG_SIZE

def convnet_encoder(IMG_SIZE):

	inputs = Input(IMG_SIZE)

	c1 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same', name = 'block1_conv1')(inputs)
	c1 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same', name = 'block1_conv2')(c1)
	p1 = MaxPooling2D((2, 2), strides = (2, 2), name = 'block1_pool')(c1)
	f1 = p1

	c2 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same', name = 'block2_conv1')(p1)
	c2 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same', name = 'block2_conv2')(c2)
	p2 = MaxPooling2D((2, 2), strides = (2, 2), name = 'block2_pool')(c2)
	f2 = p2

	c3 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same', name = 'block3_conv1')(p2)
	c3 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same', name = 'block3_conv2')(c3)
	c3 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same', name = 'block3_conv3')(c3)
	p3 = MaxPooling2D((2, 2), strides = (2, 2), name = 'block3_pool')(c3)
	f3 = p3

	c4 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same', name = 'block4_conv1')(p3)
	c4 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same', name = 'block4_conv2')(c4)
	c4 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same', name = 'block4_conv3')(c4)
	p4 = MaxPooling2D((2, 2), strides = (2, 2), name = 'block4_pool')(c4)
	f4 = p4 

	c5 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same', name = 'block5_conv1')(p4)
	c5 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same', name = 'block5_conv2')(c5)
	c5 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same', name = 'block5_conv3')(c5)
	p5 = MaxPooling2D((2, 2), strides = (2, 2), name = 'block5_pool')(c5)
	f5 = p5

	return inputs , [f1 , f2 , f3 , f4 , f5 ]

def segnet_decoder(feat, n_classes , n_up = 3 ):

	assert n_up >= 2

	f1 = feat
	f1 = ZeroPadding2D(padding = (1,1))(f1)
	f1 = Conv2D(512, (3, 3), padding = 'valid')(f1)
	b1 = BatchNormalization()(f1)


	u2 = UpSampling2D(size = (2,2))(b1)
	f2 = ZeroPadding2D(padding = (1,1))(u2)
	b2 = Conv2D( 256, (3, 3), padding = 'valid')(f2)
	o = BatchNormalization()(b2)

	for _ in range(n_up-1):
		o = UpSampling2D(size = (2,2))(o)
		o = ZeroPadding2D(padding = (1,1))(o)
		o = Conv2D( 128 , (3, 3), padding = 'valid')(o)
		o = BatchNormalization()(o)

	un = UpSampling2D(size = (2,2))(o)
	fn = ZeroPadding2D(padding = (1,1))(un)
	outputs = Conv2D(64 , (3, 3), padding = 'valid')(fn)
	outputs = BatchNormalization()(outputs)
	
	outputs =  Conv2D(n_classes, (3, 3), padding = 'same', activation = 'softmax')(outputs)
	
	return outputs

def CreateSegNet(n_classes, encoder, IMG_SIZE, encoder_level = 3):

	inputs, levels = encoder(IMG_SIZE)
	feat = levels[encoder_level]
	outputs = segnet_decoder(feat, n_classes, n_up = encoder_level)
	# outputs = Reshape((int(IMG_HEIGHT/2)*int(IMG_WIDTH/2), -1))(outputs)
	model = tf.keras.Model(inputs = [inputs], outputs = [outputs], name = 'ConvSegNet')
	model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['acc'])

	return model

def main():
  ignoreWarnings()
  useDevice('CPU')
  
  segnet = CreateSegNet(n_classes = 6, encoder = convnet_encoder, IMG_SIZE = IMG_SIZE, encoder_level = 3)
  segnet.summary()

if __name__ == '__main__':
  main()