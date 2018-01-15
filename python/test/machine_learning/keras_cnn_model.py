from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
#from keras import optimizers, callbacks
import tensorflow as tf
from dnn_model import DnnBaseModel

#%%------------------------------------------------------------------

class KerasCnnModel(DnnBaseModel):
	def __init__(self, num_classes):
		super(KerasCnnModel, self).__init__(num_classes)

	def __call__(self, input_tensor, is_training=True):
		self.model_output_ = self._create_model_1(input_tensor, self.num_classes_, is_training)
		#self.model_output_ = self._create_model_2(input_tensor, self.num_classes_, is_training)
		return self.model_output_

	def _create_model_1(self, input_tensor, num_classes, is_training=True):
		# REF [site] >> https://keras.io/getting-started/functional-api-guide
		# REF [site] >> https://keras.io/models/model/
		# REF [site] >> https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html
		keep_prob = 0.25 if is_training is True else 1.0

		with tf.variable_scope('keras_cnn_model_1', reuse=None):
			x = Conv2D(32, kernel_size=(5, 5), padding='same', activation='relu')(input_tensor)
			x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

			x = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(x)
			x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

			x = Flatten()(x)

			x = Dense(1024, activation='relu')(x)
			x = Dropout(keep_prob)(x)

			if 2 == num_classes:
				x = Dense(1, activation='sigmoid')(x)
				#x = Dense(1, activation='sigmoid', activity_regularizer=keras.regularizers.activity_l2(0.0001))(x)
			else:
				x = Dense(num_classes, activation='softmax')(x)
				#x = Dense(num_classes, activation='softmax', activity_regularizer=keras.regularizers.activity_l2(0.0001))(x)

			#model = Model(inputs=input_tensor, outputs=x)

			return x

	def _create_model_2(self, input_tensor, num_classes, is_training=True):
		# REF [site] >> https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html
		keep_prob = 0.25 if is_training is True else 1.0

		input_shape = input_tensor.get_shape().as_list()
		input_shape = input_shape[1:]
		#input_tensor = Input(shape=input_shape)

		with tf.variable_scope('keras_cnn_model_2', reuse=None):
			model = Sequential()
			model.add(Conv2D(32, kernel_size=(5, 5), padding='same', activation='relu', input_shape=input_shape))
			model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

			model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
			model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

			model.add(Flatten())

			model.add(Dense(1024, activation='relu'))
			model.add(Dropout(keep_prob))

			if 2 == num_classes:
				model.add(Dense(1, activation='sigmoid'))
				#model.add(Dense(1, activation='sigmoid', activity_regularizer=keras.regularizers.activity_l2(0.0001)))
			else:
				model.add(Dense(num_classes, activation='softmax'))
				#model.add(Dense(num_classes, activation='softmax', activity_regularizer=keras.regularizers.activity_l2(0.0001)))

			return model(input_tensor)
			#return model.output  # Run-time error.
