# REF [paper] >> "Learning Deconvolution Network for Semantic Segmentation".
# REF [site] >> https://github.com/HyeonwooNoh/DeconvNet
# REF [site] >> https://github.com/fchollet/keras/blob/master/keras/applications/vgg16.py

from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, Conv2DTranspose, BatchNormalization, Activation, MaxPooling2D, UpSampling2D, Dense, Flatten, Reshape
from neural_network import NeuralNetwork

class DeconvNet(NeuralNetwork):
	# TODO [check] >>
	#	- Batch normalization may differ frome one in the original paper.
	#	- Fully connected layers consists of convolutional layers.
	#	- Deconvolution is not performed by Gaussian.
	#	- Deconvolution layer in FC6-Deconv is replaced by fully connected layer.

	def __init__(self):
		super().__init__()
		print('Build DeconvNet ...')

	def create_model(self, num_classes, backend='tf', input_shape=None, tf_input=None):
		return self.__create_basic_model(num_classes, backend, input_shape, tf_input)
		#return self.__create_model_with_skip_connections(num_classes, backend, input_shape, tf_input)
		#return self.__create_model_without_batch_normalization(num_classes, backend, input_shape, tf_input)

	def __create_basic_model(self, num_classes, backend='tf', input_shape=None, tf_input=None):
		if 'tf' == backend:
			inputs = tf_input
			concat_axis = 3
			data_format = "channels_last"
		else:
			inputs = Input(shape = input_shape)
			concat_axis = 1
			data_format = "channels_first"

		# VGG-16.
		# Conv 1.
		conv1_1 = Conv2D(64, (3, 3), padding='same', data_format=data_format, name='conv1_1')(inputs)
		conv1_1 = BatchNormalization(axis=concat_axis, name='bn1_1')(conv1_1)
		conv1_1 = Activation('relu', name='relu1_1')(conv1_1)
		conv1_2 = Conv2D(64, (3, 3), padding='same', data_format=data_format, name='conv1_2')(conv1_1)
		conv1_2 = BatchNormalization(axis=concat_axis, name='bn1_2')(conv1_2)
		conv1_2 = Activation('relu', name='relu1_2')(conv1_2)
		pool1 = MaxPooling2D((2, 2), strides=(2, 2), data_format=data_format, name='pool1')(conv1_2)

		# Conv 2.
		conv2_1 = Conv2D(128, (3, 3), padding='same', data_format=data_format, name='conv2_1')(pool1)
		conv2_1 = BatchNormalization(axis=concat_axis, name='bn2_1')(conv2_1)
		conv2_1 = Activation('relu', name='relu2_1')(conv2_1)
		conv2_2 = Conv2D(128, (3, 3), padding='same', data_format=data_format, name='conv2_2')(conv2_1)
		conv2_2 = BatchNormalization(axis=concat_axis, name='bn2_2')(conv2_2)
		conv2_2 = Activation('relu', name='relu2_2')(conv2_2)
		pool2 = MaxPooling2D((2, 2), strides=(2, 2), data_format=data_format, name='pool2')(conv2_2)

		# Conv 3.
		conv3_1 = Conv2D(256, (3, 3), padding='same', data_format=data_format, name='conv3_1')(pool2)
		conv3_1 = BatchNormalization(axis=concat_axis, name='bn3_1')(conv3_1)
		conv3_1 = Activation('relu', name='relu3_1')(conv3_1)
		conv3_2 = Conv2D(256, (3, 3), padding='same', data_format=data_format, name='conv3_2')(conv3_1)
		conv3_2 = BatchNormalization(axis=concat_axis, name='bn3_2')(conv3_2)
		conv3_2 = Activation('relu', name='relu3_2')(conv3_2)
		conv3_3 = Conv2D(256, (3, 3), padding='same', data_format=data_format, name='conv3_3')(conv3_2)
		conv3_3 = BatchNormalization(axis=concat_axis, name='bn3_3')(conv3_3)
		conv3_3 = Activation('relu', name='relu3_3')(conv3_3)
		pool3 = MaxPooling2D((2, 2), strides=(2, 2), data_format=data_format, name='pool3')(conv3_3)

		# Conv 4.
		conv4_1 = Conv2D(512, (3, 3), padding='same', data_format=data_format, name='conv4_1')(pool3)
		conv4_1 = BatchNormalization(axis=concat_axis, name='bn4_1')(conv4_1)
		conv4_1 = Activation('relu', name='relu4_1')(conv4_1)
		conv4_2 = Conv2D(512, (3, 3), padding='same', data_format=data_format, name='conv4_2')(conv4_1)
		conv4_2 = BatchNormalization(axis=concat_axis, name='bn4_2')(conv4_2)
		conv4_2 = Activation('relu', name='relu4_2')(conv4_2)
		conv4_3 = Conv2D(512, (3, 3), padding='same', data_format=data_format, name='conv4_3')(conv4_2)
		conv4_3 = BatchNormalization(axis=concat_axis, name='bn4_3')(conv4_3)
		conv4_3 = Activation('relu', name='relu4_3')(conv4_3)
		pool4 = MaxPooling2D((2, 2), strides=(2, 2), data_format=data_format, name='pool4')(conv4_3)

		# Conv 5.
		conv5_1 = Conv2D(512, (3, 3), padding='same', data_format=data_format, name='conv5_1')(pool4)
		conv5_1 = BatchNormalization(axis=concat_axis, name='bn5_1')(conv5_1)
		conv5_1 = Activation('relu', name='relu5_1')(conv5_1)
		conv5_2 = Conv2D(512, (3, 3), padding='same', data_format=data_format, name='conv5_2')(conv5_1)
		conv5_2 = BatchNormalization(axis=concat_axis, name='bn5_2')(conv5_2)
		conv5_2 = Activation('relu', name='relu5_2')(conv5_2)
		conv5_3 = Conv2D(512, (3, 3), padding='same', data_format=data_format, name='conv5_3')(conv5_2)
		conv5_3 = BatchNormalization(axis=concat_axis, name='bn5_3')(conv5_3)
		conv5_3 = Activation('relu', name='relu5_3')(conv5_3)
		pool5 = MaxPooling2D((2, 2), strides=(2, 2), data_format=data_format, name='pool5')(conv5_3)

		pool5_shape = pool5.get_shape().as_list()

		# FC.
		#fc6 = Flatten(name='fc6-flatten')(pool5)
		#fc6 = Dense(4096, name='fc6')(fc6)
		fc6 = Conv2D(4096, (pool5_shape[1], pool5_shape[2]), padding='valid', data_format=data_format, name='fc6')(pool5)
		fc6 = BatchNormalization(axis=concat_axis, name='bn6')(fc6)
		fc6 = Activation('relu', name='relu6')(fc6)

		#fc7 = Dense(4096, name='fc7')(fc6)
		fc7 = Conv2D(4096, (1, 1), data_format=data_format, name='fc7')(fc6)
		fc7 = BatchNormalization(axis=concat_axis, name='bn7')(fc7)
		fc7 = Activation('relu', name='relu7')(fc7)

		# TODO [check] >> Is it correct?
		#fc6_deconv = Conv2DTranspose(512, kernel_size=(pool5_shape[1], pool5_shape[2]), dilation_rate=(pool5_shape[1], pool5_shape[2]), padding='same', data_format=data_format, name='fc6-deconv')(fc7)  # Do not correctly work.
		fc6_deconv = Dense(pool5_shape[1] * pool5_shape[2] * 512, activation='relu', name='fc6-deconv-dense')(fc7)
		fc6_deconv = Reshape((pool5_shape[1], pool5_shape[2], 512), name='fc6-deconv')(fc6_deconv)
		fc6_deconv = BatchNormalization(axis=concat_axis, name='bn6-deconv')(fc6_deconv)
		fc6_deconv = Activation('relu', name='relu6-deconv')(fc6_deconv)

		# Deconv 5.
		unpool5 = UpSampling2D((2, 2), data_format=data_format, name='unpool5')(fc6_deconv)
		deconv5_1 = Conv2DTranspose(512, (3, 3), padding='same', data_format=data_format, name='deconv5_1')(unpool5)
		deconv5_1 = BatchNormalization(axis=concat_axis, name='bn5_1-deconv')(deconv5_1)
		deconv5_1 = Activation('relu', name='relu5_1-deconv')(deconv5_1)
		deconv5_2 = Conv2DTranspose(512, (3, 3), padding='same', data_format=data_format, name='deconv5_2')(deconv5_1)
		deconv5_2 = BatchNormalization(axis=concat_axis, name='bn5_2-deconv')(deconv5_2)
		deconv5_2 = Activation('relu', name='relu5_2-deconv')(deconv5_2)
		deconv5_3 = Conv2DTranspose(512, (3, 3), padding='same', data_format=data_format, name='deconv5_3')(deconv5_2)
		deconv5_3 = BatchNormalization(axis=concat_axis, name='bn5_3-deconv')(deconv5_3)
		deconv5_3 = Activation('relu', name='relu5_3-deconv')(deconv5_3)

		# Deconv 4.
		unpool4 = UpSampling2D((2, 2), data_format=data_format, name='unpool4')(deconv5_3)
		deconv4_1 = Conv2DTranspose(512, (3, 3), padding='same', data_format=data_format, name='deconv4_1')(unpool4)
		deconv4_1 = BatchNormalization(axis=concat_axis, name='bn4_1-deconv')(deconv4_1)
		deconv4_1 = Activation('relu', name='relu4_1-deconv')(deconv4_1)
		deconv4_2 = Conv2DTranspose(512, (3, 3), padding='same', data_format=data_format, name='deconv4_2')(deconv4_1)
		deconv4_2 = BatchNormalization(axis=concat_axis, name='bn4_2-deconv')(deconv4_2)
		deconv4_2 = Activation('relu', name='relu4_2-deconv')(deconv4_2)
		deconv4_3 = Conv2DTranspose(512, (3, 3), padding='same', data_format=data_format, name='deconv4_3')(deconv4_2)
		deconv4_3 = BatchNormalization(axis=concat_axis, name='bn4_3-deconv')(deconv4_3)
		deconv4_3 = Activation('relu', name='relu4_3-deconv')(deconv4_3)

		# Deconv 3.
		unpool3 = UpSampling2D((2, 2), data_format=data_format, name='unpool3')(deconv4_3)
		deconv3_1 = Conv2DTranspose(256, (3, 3), padding='same', data_format=data_format, name='deconv3_1')(unpool3)
		deconv3_1 = BatchNormalization(axis=concat_axis, name='bn3_1-deconv')(deconv3_1)
		deconv3_1 = Activation('relu', name='relu3_1-deconv')(deconv3_1)
		deconv3_2 = Conv2DTranspose(256, (3, 3), padding='same', data_format=data_format, name='deconv3_2')(deconv3_1)
		deconv3_2 = BatchNormalization(axis=concat_axis, name='bn3_2-deconv')(deconv3_2)
		deconv3_2 = Activation('relu', name='relu3_2-deconv')(deconv3_2)
		deconv3_3 = Conv2DTranspose(256, (3, 3), padding='same', data_format=data_format, name='deconv3_3')(deconv3_2)
		deconv3_3 = BatchNormalization(axis=concat_axis, name='bn3_3-deconv')(deconv3_3)
		deconv3_3 = Activation('relu', name='relu3_3-deconv')(deconv3_3)

		# Deconv 2.
		unpool2 = UpSampling2D((2, 2), data_format=data_format, name='unpool2')(deconv3_3)
		deconv2_1 = Conv2DTranspose(128, (3, 3), padding='same', data_format=data_format, name='deconv2_1')(unpool2)
		deconv2_1 = BatchNormalization(axis=concat_axis, name='bn2_1-deconv')(deconv2_1)
		deconv2_1 = Activation('relu', name='relu2_1-deconv')(deconv2_1)
		deconv2_2 = Conv2DTranspose(128, (3, 3), padding='same', data_format=data_format, name='deconv2_2')(deconv2_1)
		deconv2_2 = BatchNormalization(axis=concat_axis, name='bn2_2-deconv')(deconv2_2)
		deconv2_2 = Activation('relu', name='relu2_2-deconv')(deconv2_2)

		# Deconv 1.
		unpool1 = UpSampling2D((2, 2), data_format=data_format, name='unpool1')(deconv2_2)
		deconv1_1 = Conv2DTranspose(64, (3, 3), padding='same', data_format=data_format, name='deconv1_1')(unpool1)
		deconv1_1 = BatchNormalization(axis=concat_axis, name='bn1_1-deconv')(deconv1_1)
		deconv1_1 = Activation('relu', name='relu1_1-deconv')(deconv1_1)
		deconv1_2 = Conv2DTranspose(64, (3, 3), padding='same', data_format=data_format, name='deconv1_2')(deconv1_1)
		deconv1_2 = BatchNormalization(axis=concat_axis, name='bn1_2-deconv')(deconv1_2)
		deconv1_2 = Activation('relu', name='relu1_2-deconv')(deconv1_2)

		# Segmentation score.
		if 2 == num_classes:
			seg_score = Conv2D(1, (1, 1), activation='sigmoid', data_format=data_format, name='seg-score')(deconv1_2)
		elif num_classes > 2:
			# TODO [check] >> is softmax correct?
			seg_score = Conv2D(num_classes, (1, 1), activation='softmax', data_format=data_format, name='seg-score')(deconv1_2)
		else:
			raise ValueError('Invalid number of classes.')

		if 'tf' == backend:
			return seg_score
		else:
			model = Model(input=inputs, output=seg_score)
			return model

	def __create_model_with_skip_connections(self, num_classes, backend='tf', input_shape=None, tf_input=None):
		if 'tf' == backend:
			inputs = tf_input
			concat_axis = 3
			data_format = "channels_last"
		else:
			inputs = Input(shape = input_shape)
			concat_axis = 1
			data_format = "channels_first"

		# VGG-16.
		# Conv 1.
		conv1_1 = Conv2D(64, (3, 3), padding='same', data_format=data_format, name='conv1_1')(inputs)
		conv1_1 = BatchNormalization(axis=concat_axis, name='bn1_1')(conv1_1)
		conv1_1 = Activation('relu', name='relu1_1')(conv1_1)
		conv1_2 = Conv2D(64, (3, 3), padding='same', data_format=data_format, name='conv1_2')(conv1_1)
		conv1_2 = BatchNormalization(axis=concat_axis, name='bn1_2')(conv1_2)
		conv1_2 = Activation('relu', name='relu1_2')(conv1_2)
		pool1 = MaxPooling2D((2, 2), strides=(2, 2), data_format=data_format, name='pool1')(conv1_2)

		# Conv 2.
		conv2_1 = Conv2D(128, (3, 3), padding='same', data_format=data_format, name='conv2_1')(pool1)
		conv2_1 = BatchNormalization(axis=concat_axis, name='bn2_1')(conv2_1)
		conv2_1 = Activation('relu', name='relu2_1')(conv2_1)
		conv2_2 = Conv2D(128, (3, 3), padding='same', data_format=data_format, name='conv2_2')(conv2_1)
		conv2_2 = BatchNormalization(axis=concat_axis, name='bn2_2')(conv2_2)
		conv2_2 = Activation('relu', name='relu2_2')(conv2_2)
		pool2 = MaxPooling2D((2, 2), strides=(2, 2), data_format=data_format, name='pool2')(conv2_2)

		# Conv 3.
		conv3_1 = Conv2D(256, (3, 3), padding='same', data_format=data_format, name='conv3_1')(pool2)
		conv3_1 = BatchNormalization(axis=concat_axis, name='bn3_1')(conv3_1)
		conv3_1 = Activation('relu', name='relu3_1')(conv3_1)
		conv3_2 = Conv2D(256, (3, 3), padding='same', data_format=data_format, name='conv3_2')(conv3_1)
		conv3_2 = BatchNormalization(axis=concat_axis, name='bn3_2')(conv3_2)
		conv3_2 = Activation('relu', name='relu3_2')(conv3_2)
		conv3_3 = Conv2D(256, (3, 3), padding='same', data_format=data_format, name='conv3_3')(conv3_2)
		conv3_3 = BatchNormalization(axis=concat_axis, name='bn3_3')(conv3_3)
		conv3_3 = Activation('relu', name='relu3_3')(conv3_3)
		pool3 = MaxPooling2D((2, 2), strides=(2, 2), data_format=data_format, name='pool3')(conv3_3)

		# Conv 4.
		conv4_1 = Conv2D(512, (3, 3), padding='same', data_format=data_format, name='conv4_1')(pool3)
		conv4_1 = BatchNormalization(axis=concat_axis, name='bn4_1')(conv4_1)
		conv4_1 = Activation('relu', name='relu4_1')(conv4_1)
		conv4_2 = Conv2D(512, (3, 3), padding='same', data_format=data_format, name='conv4_2')(conv4_1)
		conv4_2 = BatchNormalization(axis=concat_axis, name='bn4_2')(conv4_2)
		conv4_2 = Activation('relu', name='relu4_2')(conv4_2)
		conv4_3 = Conv2D(512, (3, 3), padding='same', data_format=data_format, name='conv4_3')(conv4_2)
		conv4_3 = BatchNormalization(axis=concat_axis, name='bn4_3')(conv4_3)
		conv4_3 = Activation('relu', name='relu4_3')(conv4_3)
		pool4 = MaxPooling2D((2, 2), strides=(2, 2), data_format=data_format, name='pool4')(conv4_3)

		# Conv 5.
		conv5_1 = Conv2D(512, (3, 3), padding='same', data_format=data_format, name='conv5_1')(pool4)
		conv5_1 = BatchNormalization(axis=concat_axis, name='bn5_1')(conv5_1)
		conv5_1 = Activation('relu', name='relu5_1')(conv5_1)
		conv5_2 = Conv2D(512, (3, 3), padding='same', data_format=data_format, name='conv5_2')(conv5_1)
		conv5_2 = BatchNormalization(axis=concat_axis, name='bn5_2')(conv5_2)
		conv5_2 = Activation('relu', name='relu5_2')(conv5_2)
		conv5_3 = Conv2D(512, (3, 3), padding='same', data_format=data_format, name='conv5_3')(conv5_2)
		conv5_3 = BatchNormalization(axis=concat_axis, name='bn5_3')(conv5_3)
		conv5_3 = Activation('relu', name='relu5_3')(conv5_3)
		pool5 = MaxPooling2D((2, 2), strides=(2, 2), data_format=data_format, name='pool5')(conv5_3)

		pool5_shape = pool5.get_shape().as_list()

		# FC.
		#fc6 = Flatten(name='fc6-flatten')(pool5)
		#fc6 = Dense(4096, name='fc6')(fc6)
		fc6 = Conv2D(4096, (pool5_shape[1], pool5_shape[2]), padding='valid', data_format=data_format, name='fc6')(pool5)
		fc6 = BatchNormalization(axis=concat_axis, name='bn6')(fc6)
		fc6 = Activation('relu', name='relu6')(fc6)

		#fc7 = Dense(4096, name='fc7')(fc6)
		fc7 = Conv2D(4096, (1, 1), data_format=data_format, name='fc7')(fc6)
		fc7 = BatchNormalization(axis=concat_axis, name='bn7')(fc7)
		fc7 = Activation('relu', name='relu7')(fc7)

		# TODO [check] >> Is it correct?
		#fc6_deconv = Conv2DTranspose(512, kernel_size=(pool5_shape[1], pool5_shape[2]), dilation_rate=(pool5_shape[1], pool5_shape[2]), padding='same', data_format=data_format, name='fc6-deconv')(fc7)  # Do not correctly work.
		fc6_deconv = Dense(pool5_shape[1] * pool5_shape[2] * 512, activation='relu', name='fc6-deconv-dense')(fc7)
		fc6_deconv = Reshape((pool5_shape[1], pool5_shape[2], 512), name='fc6-deconv')(fc6_deconv)
		fc6_deconv = BatchNormalization(axis=concat_axis, name='bn6-deconv')(fc6_deconv)
		fc6_deconv = Activation('relu', name='relu6-deconv')(fc6_deconv)

		# Deconv 5.
		unpool5 = UpSampling2D((2, 2), data_format=data_format, name='unpool5')(fc6_deconv)
		concat5 = concatenate([unpool5, conv5_3], axis=concat_axis, name='concat5')
		deconv5_1 = Conv2DTranspose(512, (3, 3), padding='same', data_format=data_format, name='deconv5_1')(concat5)
		deconv5_1 = BatchNormalization(axis=concat_axis, name='bn5_1-deconv')(deconv5_1)
		deconv5_1 = Activation('relu', name='relu5_1-deconv')(deconv5_1)
		deconv5_2 = Conv2DTranspose(512, (3, 3), padding='same', data_format=data_format, name='deconv5_2')(deconv5_1)
		deconv5_2 = BatchNormalization(axis=concat_axis, name='bn5_2-deconv')(deconv5_2)
		deconv5_2 = Activation('relu', name='relu5_2-deconv')(deconv5_2)
		deconv5_3 = Conv2DTranspose(512, (3, 3), padding='same', data_format=data_format, name='deconv5_3')(deconv5_2)
		deconv5_3 = BatchNormalization(axis=concat_axis, name='bn5_3-deconv')(deconv5_3)
		deconv5_3 = Activation('relu', name='relu5_3-deconv')(deconv5_3)

		# Deconv 4.
		unpool4 = UpSampling2D((2, 2), data_format=data_format, name='unpool4')(deconv5_3)
		concat4 = concatenate([unpool4, conv4_3], axis=concat_axis, name='concat4')
		deconv4_1 = Conv2DTranspose(512, (3, 3), padding='same', data_format=data_format, name='deconv4_1')(concat4)
		deconv4_1 = BatchNormalization(axis=concat_axis, name='bn4_1-deconv')(deconv4_1)
		deconv4_1 = Activation('relu', name='relu4_1-deconv')(deconv4_1)
		deconv4_2 = Conv2DTranspose(512, (3, 3), padding='same', data_format=data_format, name='deconv4_2')(deconv4_1)
		deconv4_2 = BatchNormalization(axis=concat_axis, name='bn4_2-deconv')(deconv4_2)
		deconv4_2 = Activation('relu', name='relu4_2-deconv')(deconv4_2)
		deconv4_3 = Conv2DTranspose(512, (3, 3), padding='same', data_format=data_format, name='deconv4_3')(deconv4_2)
		deconv4_3 = BatchNormalization(axis=concat_axis, name='bn4_3-deconv')(deconv4_3)
		deconv4_3 = Activation('relu', name='relu4_3-deconv')(deconv4_3)

		# Deconv 3.
		unpool3 = UpSampling2D((2, 2), data_format=data_format, name='unpool3')(deconv4_3)
		concat3 = concatenate([unpool3, conv3_3], axis=concat_axis, name='concat3')
		deconv3_1 = Conv2DTranspose(256, (3, 3), padding='same', data_format=data_format, name='deconv3_1')(concat3)
		deconv3_1 = BatchNormalization(axis=concat_axis, name='bn3_1-deconv')(deconv3_1)
		deconv3_1 = Activation('relu', name='relu3_1-deconv')(deconv3_1)
		deconv3_2 = Conv2DTranspose(256, (3, 3), padding='same', data_format=data_format, name='deconv3_2')(deconv3_1)
		deconv3_2 = BatchNormalization(axis=concat_axis, name='bn3_2-deconv')(deconv3_2)
		deconv3_2 = Activation('relu', name='relu3_2-deconv')(deconv3_2)
		deconv3_3 = Conv2DTranspose(256, (3, 3), padding='same', data_format=data_format, name='deconv3_3')(deconv3_2)
		deconv3_3 = BatchNormalization(axis=concat_axis, name='bn3_3-deconv')(deconv3_3)
		deconv3_3 = Activation('relu', name='relu3_3-deconv')(deconv3_3)

		# Deconv 2.
		unpool2 = UpSampling2D((2, 2), data_format=data_format, name='unpool2')(deconv3_3)
		concat2 = concatenate([unpool2, conv2_2], axis=concat_axis, name='concat2')
		deconv2_1 = Conv2DTranspose(128, (3, 3), padding='same', data_format=data_format, name='deconv2_1')(concat2)
		deconv2_1 = BatchNormalization(axis=concat_axis, name='bn2_1-deconv')(deconv2_1)
		deconv2_1 = Activation('relu', name='relu2_1-deconv')(deconv2_1)
		deconv2_2 = Conv2DTranspose(128, (3, 3), padding='same', data_format=data_format, name='deconv2_2')(deconv2_1)
		deconv2_2 = BatchNormalization(axis=concat_axis, name='bn2_2-deconv')(deconv2_2)
		deconv2_2 = Activation('relu', name='relu2_2-deconv')(deconv2_2)

		# Deconv 1.
		unpool1 = UpSampling2D((2, 2), data_format=data_format, name='unpool1')(deconv2_2)
		concat1 = concatenate([unpool1, conv1_2], axis=concat_axis, name='concat1')
		deconv1_1 = Conv2DTranspose(64, (3, 3), padding='same', data_format=data_format, name='deconv1_1')(concat1)
		deconv1_1 = BatchNormalization(axis=concat_axis, name='bn1_1-deconv')(deconv1_1)
		deconv1_1 = Activation('relu', name='relu1_1-deconv')(deconv1_1)
		deconv1_2 = Conv2DTranspose(64, (3, 3), padding='same', data_format=data_format, name='deconv1_2')(deconv1_1)
		deconv1_2 = BatchNormalization(axis=concat_axis, name='bn1_2-deconv')(deconv1_2)
		deconv1_2 = Activation('relu', name='relu1_2-deconv')(deconv1_2)

		# Segmentation score.
		if 2 == num_classes:
			seg_score = Conv2D(1, (1, 1), activation='sigmoid', data_format=data_format, name='seg-score')(deconv1_2)
		elif num_classes > 2:
			# TODO [check] >> is softmax correct?
			seg_score = Conv2D(num_classes, (1, 1), activation='softmax', data_format=data_format, name='seg-score')(deconv1_2)
		else:
			raise ValueError('Invalid number of classes.')

		if 'tf' == backend:
			return seg_score
		else:
			model = Model(input=inputs, output=seg_score)
			return model

	def __create_model_without_batch_normalization(self, num_classes, backend='tf', input_shape=None, tf_input=None):
		if 'tf' == backend:
			inputs = tf_input
			concat_axis = 3
			data_format = "channels_last"
		else:
			inputs = Input(shape = input_shape)
			concat_axis = 1
			data_format = "channels_first"

		# VGG-16.
		# Conv 1.
		conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format=data_format, name='conv1_1')(inputs)
		conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format=data_format, name='conv1_2')(conv1_1)
		pool1 = MaxPooling2D((2, 2), strides=(2, 2), data_format=data_format, name='pool1')(conv1_2)

		# Conv 2.
		conv2_1 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format=data_format, name='conv2_1')(pool1)
		conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format=data_format, name='conv2_2')(conv2_1)
		pool2 = MaxPooling2D((2, 2), strides=(2, 2), data_format=data_format, name='pool2')(conv2_2)

		# Conv 3.
		conv3_1 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format=data_format, name='conv3_1')(pool2)
		conv3_2 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format=data_format, name='conv3_2')(conv3_1)
		conv3_3 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format=data_format, name='conv3_3')(conv3_2)
		pool3 = MaxPooling2D((2, 2), strides=(2, 2), data_format=data_format, name='pool3')(conv3_3)

		# Conv 4.
		conv4_1 = Conv2D(512, (3, 3), activation='relu', padding='same', data_format=data_format, name='conv4_1')(pool3)
		conv4_2 = Conv2D(512, (3, 3), activation='relu', padding='same', data_format=data_format, name='conv4_2')(conv4_1)
		conv4_3 = Conv2D(512, (3, 3), activation='relu', padding='same', data_format=data_format, name='conv4_3')(conv4_2)
		pool4 = MaxPooling2D((2, 2), strides=(2, 2), data_format=data_format, name='pool4')(conv4_3)

		# Conv 5.
		conv5_1 = Conv2D(512, (3, 3), activation='relu', padding='same', data_format=data_format, name='conv5_1')(pool4)
		conv5_2 = Conv2D(512, (3, 3), activation='relu', padding='same', data_format=data_format, name='conv5_2')(conv5_1)
		conv5_3 = Conv2D(512, (3, 3), activation='relu', padding='same', data_format=data_format, name='conv5_3')(conv5_2)
		pool5 = MaxPooling2D((2, 2), strides=(2, 2), data_format=data_format, name='pool5')(conv5_3)

		pool5_shape = pool5.get_shape().as_list()

		# FC.
		#fc6 = Flatten(name='fc6-flatten')(pool5)
		#fc6 = Dense(4096, activation='relu', name='fc6')(fc6)
		fc6 = Conv2D(4096, (pool5_shape[1], pool5_shape[2]), activation='relu', padding='valid', data_format=data_format, name='fc6')(pool5)

		#fc7 = Dense(4096, activation='relu', name='fc7')(fc6)
		fc7 = Conv2D(4096, (1, 1), activation='relu', data_format=data_format, name='fc7')(fc6)

		# TODO [check] >> Is it correct?
		#fc6_deconv = Conv2DTranspose(512, kernel_size=(pool5_shape[1], pool5_shape[2]), dilation_rate=(pool5_shape[1], pool5_shape[2]), activation='relu', padding='same', data_format=data_format, name='fc6-deconv')(fc7)  # Do not correctly work.
		fc6_deconv = Dense(pool5_shape[1] * pool5_shape[2] * 512, activation='relu', name='fc6-deconv-dense')(fc7)
		fc6_deconv = Reshape((pool5_shape[1], pool5_shape[2], 512), name='fc6-deconv')(fc6_deconv)

		# Deconv 5.
		unpool5 = UpSampling2D((2, 2), data_format=data_format, name='unpool5')(fc6_deconv)
		deconv5_1 = Conv2DTranspose(512, (3, 3), activation='relu', padding='same', data_format=data_format, name='deconv5_1')(unpool5)
		deconv5_2 = Conv2DTranspose(512, (3, 3), activation='relu', padding='same', data_format=data_format, name='deconv5_2')(deconv5_1)
		deconv5_3 = Conv2DTranspose(512, (3, 3), activation='relu', padding='same', data_format=data_format, name='deconv5_3')(deconv5_2)

		# Deconv 4.
		unpool4 = UpSampling2D((2, 2), data_format=data_format, name='unpool4')(deconv5_3)
		deconv4_1 = Conv2DTranspose(512, (3, 3), activation='relu', padding='same', data_format=data_format, name='deconv4_1')(unpool4)
		deconv4_2 = Conv2DTranspose(512, (3, 3), activation='relu', padding='same', data_format=data_format, name='deconv4_2')(deconv4_1)
		deconv4_3 = Conv2DTranspose(512, (3, 3), activation='relu', padding='same', data_format=data_format, name='deconv4_3')(deconv4_2)

		# Deconv 3.
		unpool3 = UpSampling2D((2, 2), data_format=data_format, name='unpool3')(deconv4_3)
		deconv3_1 = Conv2DTranspose(256, (3, 3), activation='relu', padding='same', data_format=data_format, name='deconv3_1')(unpool3)
		deconv3_2 = Conv2DTranspose(256, (3, 3), activation='relu', padding='same', data_format=data_format, name='deconv3_2')(deconv3_1)
		deconv3_3 = Conv2DTranspose(256, (3, 3), activation='relu', padding='same', data_format=data_format, name='deconv3_3')(deconv3_2)

		# Deconv 2.
		unpool2 = UpSampling2D((2, 2), data_format=data_format, name='unpool2')(deconv3_3)
		deconv2_1 = Conv2DTranspose(128, (3, 3), activation='relu', padding='same', data_format=data_format, name='deconv2_1')(unpool2)
		deconv2_2 = Conv2DTranspose(128, (3, 3), activation='relu', padding='same', data_format=data_format, name='deconv2_2')(deconv2_1)

		# Deconv 1.
		unpool1 = UpSampling2D((2, 2), data_format=data_format, name='unpool1')(deconv2_2)
		deconv1_1 = Conv2DTranspose(64, (3, 3), activation='relu', padding='same', data_format=data_format, name='deconv1_1')(unpool1)
		deconv1_2 = Conv2DTranspose(64, (3, 3), activation='relu', padding='same', data_format=data_format, name='deconv1_2')(deconv1_1)

		# Segmentation score.
		if 2 == num_classes:
			seg_score = Conv2D(1, (1, 1), activation='sigmoid', data_format=data_format, name='seg-score')(deconv1_2)
		elif num_classes > 2:
			# TODO [check] >> is softmax correct?
			seg_score = Conv2D(num_classes, (1, 1), activation='softmax', data_format=data_format, name='seg-score')(deconv1_2)
		else:
			raise ValueError('Invalid number of classes.')

		if 'tf' == backend:
			return seg_score
		else:
			model = Model(input=inputs, output=seg_score)
			return model

	def train(self):
		raise NotImplementError

	def predict(self):
		raise NotImplementError
