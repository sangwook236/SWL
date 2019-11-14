# REF [site] >> https://keras.io/getting-started/functional-api-guide
# REF [site] >> https://keras.io/models/model/
# REF [site] >> https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html

from keras.models import Model, Input
#from keras import optimizers, callbacks
from keras.datasets import mnist
import tensorflow as tf

#--------------------------------------------------------------------

num_classes = 10
input_shape = (28, 28, 1)  # 784 = 28 * 28.
#input_tensor = tf.placeholder(tf.float32, shape=(None,) + input_shape)  # Error.
input_tensor = Input(shape=input_shape)

#--------------------------------------------------------------------
# Create a Tensorflow model.

conv1 = tf.layers.conv2d(input_tensor, 32, 5, activation=tf.nn.relu, name='conv1_1')
conv1 = tf.layers.max_pooling2d(conv1, 2, 2, name='maxpool1_1')

conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu, name='conv2_1')
conv2 = tf.layers.max_pooling2d(conv2, 2, 2, name='maxpool2_1')

fc1 = tf.layers.flatten(conv2, name='flatten1_1')

fc1 = tf.layers.dense(fc1, 1024, activation=tf.nn.relu, name='fc1_1')
fc1 = tf.layers.dropout(fc1, rate=0.25, training=True, name='dropout1_1')

tf_model_output_tensor = tf.layers.dense(fc1, num_classes, activation=tf.nn.softmax, name='fc2_1')
#tf_model_output_tensor = tf.layers.dense(fc1, num_classes, activation=tf.nn.softmax, activity_regularizer=tf.contrib.layers.l2_regularizer(0.0001), name='fc2_1')

#--------------------------------------------------------------------
# Create a Keras model from a Tensorflow model.

# FIXME [fix] >> Not working.

keras_model = Model(inputs=input_tensor, outputs=tf_model_output_tensor)

#--------------------------------------------------------------------

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

keras_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

history = keras_model.fit(X_train, Y_train, batch_size=128, epochs=20)

score, acc = keras_model.evaluate(X_test, Y_test, batch_size=128)
print('Test score: {}, test accuracy: {}', score, acc)
