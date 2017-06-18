# REF [paper] >> "U-Net: Convolutional Networks for Biomedical Image Segmentation".
# REF [site] >> https://github.com/zizhaozhang/unet-tensorflow-keras
# REF [file] >> https://github.com/zizhaozhang/unet-tensorflow-keras/blob/master/model.py
# REF [site] >> https://lmb.informatik.uni-freiburg.de/resources/opensource/unet.en.html

from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, ZeroPadding2D

class UNet():
	def __init__(self):
		print('Build U-Net ...')

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

		conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format=data_format, name='conv1_1')(inputs)
		conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format=data_format, name='conv1_2')(conv1)
		pool1 = MaxPooling2D(pool_size=(2, 2), data_format=data_format)(conv1)
		conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format=data_format, name='conv2_1')(pool1)
		conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format=data_format, name='conv2_2')(conv2)
		pool2 = MaxPooling2D(pool_size=(2, 2), data_format=data_format)(conv2)

		conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format=data_format, name='conv3_1')(pool2)
		conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format=data_format, name='conv3_2')(conv3)
		pool3 = MaxPooling2D(pool_size=(2, 2), data_format=data_format)(conv3)

		conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format=data_format, name='conv4_1')(pool3)
		conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format=data_format, name='conv4_2')(conv4)
		pool4 = MaxPooling2D(pool_size=(2, 2), data_format=data_format)(conv4)

		conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', data_format=data_format, name='conv5_1')(pool4)
		conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', data_format=data_format, name='conv5_2')(conv5)

		up_conv5 = UpSampling2D(size=(2, 2), data_format=data_format)(conv5)
		ch, cw = self.get_crop_shape(conv4, up_conv5)
		crop_conv4 = Cropping2D(cropping=(ch,cw), data_format=data_format)(conv4)
		up6 = concatenate([up_conv5, crop_conv4], axis=concat_axis)
		conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format=data_format, name='conv6_1')(up6)
		conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format=data_format, name='conv6_2')(conv6)

		up_conv6 = UpSampling2D(size=(2, 2), data_format=data_format)(conv6)
		ch, cw = self.get_crop_shape(conv3, up_conv6)
		crop_conv3 = Cropping2D(cropping=(ch,cw), data_format=data_format)(conv3)
		up7 = concatenate([up_conv6, crop_conv3], axis=concat_axis)
		conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format=data_format, name='conv7_1')(up7)
		conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format=data_format, name='conv7_2')(conv7)

		up_conv7 = UpSampling2D(size=(2, 2), data_format=data_format)(conv7)
		ch, cw = self.get_crop_shape(conv2, up_conv7)
		crop_conv2 = Cropping2D(cropping=(ch,cw), data_format=data_format)(conv2)
		up8 = concatenate([up_conv7, crop_conv2], axis=concat_axis)
		conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format=data_format, name='conv8_1')(up8)
		conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format=data_format, name='conv8_2')(conv8)

		up_conv8 = UpSampling2D(size=(2, 2), data_format=data_format)(conv8)
		ch, cw = self.get_crop_shape(conv1, up_conv8)
		crop_conv1 = Cropping2D(cropping=(ch,cw), data_format=data_format)(conv1)
		up9 = concatenate([up_conv8, crop_conv1], axis=concat_axis)
		conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format=data_format, name='conv9_1')(up9)
		conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format=data_format, name='conv9_2')(conv9)

		ch, cw = self.get_crop_shape(inputs, conv9)
		conv9 = ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])), data_format=data_format)(conv9)
		conv10 = Conv2D(1, (1, 1), activation='sigmoid', data_format=data_format, name='conv10_1')(conv9)

		if 'tf' == backend:
			return conv10
		else:
			model = Model(input=inputs, output=conv10)
			return model

	def create_original_model(self, img_shape=None, backend='tf', tf_input=None):
		if 'tf' == backend:
			inputs = tf_input
			concat_axis = 3
			data_format="channels_last"
		else:
			inputs = Input(shape = img_shape)
			concat_axis = 1
			data_format="channels_first"

		conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format=data_format, name='conv1_1')(inputs)
		conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format=data_format, name='conv1_2')(conv1)
		pool1 = MaxPooling2D(pool_size=(2, 2), data_format=data_format)(conv1)
		conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format=data_format, name='conv2_1')(pool1)
		conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format=data_format, name='conv2_2')(conv2)
		pool2 = MaxPooling2D(pool_size=(2, 2), data_format=data_format)(conv2)

		conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format=data_format, name='conv3_1')(pool2)
		conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format=data_format, name='conv3_2')(conv3)
		pool3 = MaxPooling2D(pool_size=(2, 2), data_format=data_format)(conv3)

		conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', data_format=data_format, name='conv4_1')(pool3)
		conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', data_format=data_format, name='conv4_2')(conv4)
		pool4 = MaxPooling2D(pool_size=(2, 2), data_format=data_format)(conv4)

		conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same', data_format=data_format, name='conv5_1')(pool4)
		conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same', data_format=data_format, name='conv5_2')(conv5)

		up_conv5 = UpSampling2D(size=(2, 2), data_format=data_format)(conv5)
		ch, cw = self.get_crop_shape(conv4, up_conv5)
		crop_conv4 = Cropping2D(cropping=(ch,cw), data_format=data_format)(conv4)
		up6 = concatenate([up_conv5, crop_conv4], axis=concat_axis)
		conv6 = Conv2D(512, (3, 3), activation='relu', padding='same', data_format=data_format, name='conv6_1')(up6)
		conv6 = Conv2D(512, (3, 3), activation='relu', padding='same', data_format=data_format, name='conv6_2')(conv6)

		up_conv6 = UpSampling2D(size=(2, 2), data_format=data_format)(conv6)
		ch, cw = self.get_crop_shape(conv3, up_conv6)
		crop_conv3 = Cropping2D(cropping=(ch,cw), data_format=data_format)(conv3)
		up7 = concatenate([up_conv6, crop_conv3], axis=concat_axis)
		conv7 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format=data_format, name='conv7_1')(up7)
		conv7 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format=data_format, name='conv7_2')(conv7)

		up_conv7 = UpSampling2D(size=(2, 2), data_format=data_format)(conv7)
		ch, cw = self.get_crop_shape(conv2, up_conv7)
		crop_conv2 = Cropping2D(cropping=(ch,cw), data_format=data_format)(conv2)
		up8 = concatenate([up_conv7, crop_conv2], axis=concat_axis)
		conv8 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format=data_format, name='conv8_1')(up8)
		conv8 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format=data_format, name='conv8_2')(conv8)

		up_conv8 = UpSampling2D(size=(2, 2), data_format=data_format)(conv8)
		ch, cw = self.get_crop_shape(conv1, up_conv8)
		crop_conv1 = Cropping2D(cropping=(ch,cw), data_format=data_format)(conv1)
		up9 = concatenate([up_conv8, crop_conv1], axis=concat_axis)
		conv9 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format=data_format, name='conv9_1')(up9)
		conv9 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format=data_format, name='conv9_2')(conv9)

		ch, cw = self.get_crop_shape(inputs, conv9)
		conv9 = ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])), data_format=data_format)(conv9)
		conv10 = Conv2D(1, (1, 1), activation='sigmoid', data_format=data_format, name='conv10_1')(conv9)

		if 'tf' == backend:
			return conv10
		else:
			model = Model(input=inputs, output=conv10)
			return model

	def train(self):
		raise NotImplementError

	def predict(self):
		raise NotImplementError
