from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
#from keras import optimizers, callbacks
import tensorflow as tf
from mnist_cnn import MnistCNN

#%%------------------------------------------------------------------

class MnistKerasCNN(MnistCNN):
	def __init__(self, input_shape, output_shape):
		super().__init__(input_shape, output_shape)

	def _create_model(self, input_tensor, is_training_tensor, num_classes):
		return self._create_model_1(input_tensor, num_classes)
		#return self._create_model_2(input_tensor, num_classes)

	def _create_model_1(self, input_tensor, num_classes):
		# REF [site] >> https://keras.io/getting-started/functional-api-guide
		# REF [site] >> https://keras.io/models/model/
		# REF [site] >> https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html

		# Note [info] >> Because is_training_tensor is a TensorFlow tensor, it can not be used as an argument in Keras.
		#	In Keras, K.set_learning_phase(1) or K.set_learning_phase(0) has to be used to set the learning phase, 'train' or 'test' before defining a model.
		#		K.set_learning_phase(1)  # Set the learning phase to 'train'.
		#		K.set_learning_phase(0)  # Set the learning phase to 'test'.
		#drop_rate = 0.75 if True == is_training_tensor else 0.0  # Error: Not working.
		#drop_rate = tf.cond(tf.equal(is_training_tensor, tf.constant(True)), lambda: tf.constant(0.75), lambda: tf.constant(0.0))  # Error: Not working.
		drop_rate = 0.75

		with tf.variable_scope('mnist_keras_cnn_1', reuse=tf.AUTO_REUSE):
			x = Conv2D(32, kernel_size=(5, 5), padding='same', activation='relu')(input_tensor)
			x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

			x = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(x)
			x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

			x = Flatten()(x)

			x = Dense(1024, activation='relu')(x)
			x = Dropout(drop_rate)(x)

			if 2 == num_classes:
				x = Dense(1, activation='sigmoid')(x)
				#x = Dense(1, activation='sigmoid', activity_regularizer=keras.regularizers.activity_l2(0.0001))(x)
			else:
				x = Dense(num_classes, activation='softmax')(x)
				#x = Dense(num_classes, activation='softmax', activity_regularizer=keras.regularizers.activity_l2(0.0001))(x)

			#model = Model(inputs=input_tensor, outputs=x)

			return x

	def _create_model_2(self, input_tensor, num_classes):
		# REF [site] >> https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html

		# Note [info] >> Because is_training_tensor is a TensorFlow tensor, it can not be used as an argument in Keras.
		#	In Keras, K.set_learning_phase(1) or K.set_learning_phase(0) has to be used to set the learning phase, 'train' or 'test' before defining a model. 
		#		K.set_learning_phase(1)  # Set the learning phase to 'train'.
		#		K.set_learning_phase(0)  # Set the learning phase to 'test'.
		#drop_rate = 0.75 if True == is_training_tensor else 0.0  # Error: Not working.
		#drop_rate = tf.cond(tf.equal(is_training_tensor, tf.constant(True)), lambda: tf.constant(0.75), lambda: tf.constant(0.0))  # Error: Not working.
		drop_rate = 0.75

		input_shape = input_tensor.get_shape().as_list()
		input_shape = input_shape[1:]
		#input_tensor = Input(shape=input_shape)

		with tf.variable_scope('mnist_keras_cnn_2', reuse=tf.AUTO_REUSE):
			model = Sequential()
			model.add(Conv2D(32, kernel_size=(5, 5), padding='same', activation='relu', input_shape=input_shape))
			model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

			model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
			model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

			model.add(Flatten())

			model.add(Dense(1024, activation='relu'))
			model.add(Dropout(drop_rate))

			if 2 == num_classes:
				model.add(Dense(1, activation='sigmoid'))
				#model.add(Dense(1, activation='sigmoid', activity_regularizer=keras.regularizers.activity_l2(0.0001)))
			else:
				model.add(Dense(num_classes, activation='softmax'))
				#model.add(Dense(num_classes, activation='softmax', activity_regularizer=keras.regularizers.activity_l2(0.0001)))

			# Display the model summary.
			#model.summary()

			return model(input_tensor)
			#return model.output  # Run-time error.
