# REF [paper] >> "Learning Deconvolution Network for Semantic Segmentation".
# REF [site] >> https://github.com/fabianbormann/Tensorflow-DeconvNet-Segmentation
# REF [site] >> https://github.com/HyeonwooNoh/DeconvNet
# REF [site] >> https://github.com/zizhaozhang/DeconvNet-tensorflow-keras
# REF [site] >> https://github.com/fchollet/keras/blob/master/keras/applications/vgg16.py

from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, ZeroPadding2D

class DeconvNet():
	def __init__(self):
		print('Build DeconvNet ...')

	def get_crop_shape(self, target, refer):
		# Width, the 3rd dimension.
		cw = (target.get_shape()[2] - refer.get_shape()[2]).value
		assert (cw >= 0)
		if cw % 2 != 0:
			cw1, cw2 = int(cw/2), int(cw/2) + 1
		else:
			cw1, cw2 = int(cw/2), int(cw/2)
		# Height, the 2nd dimension.
		ch = (target.get_shape()[1] - refer.get_shape()[1]).value
		assert (ch >= 0)
		if ch % 2 != 0:
			ch1, ch2 = int(ch/2), int(ch/2) + 1
		else:
			ch1, ch2 = int(ch/2), int(ch/2)

		return (ch1, ch2), (cw1, cw2)

	def create_model(self, img_shape=None, backend='tf', tf_input=None):
		if 'tf' == backend:
			inputs = tf_input
			concat_axis = 3
			data_format="channels_last"
		else:
			inputs = Input(shape = img_shape)
			concat_axis = 1
			data_format="channels_first"

		# VGG-16.
		# Block 1.
		x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
		x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
		x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

		# Block 2.
		x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
		x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
		x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

		# Block 3.
		x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
		x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
		x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
		x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

		# Block 4.
		x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
		x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
		x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
		x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

		# Block 5.
		x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
		x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
		x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
		x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

		if 'tf' == backend:
			return conv10
		else:
			model = Model(input=inputs, output=conv10)
			return model

	def train(self):
		raise NotImplementError

	def predict(self):
		raise NotImplementError
