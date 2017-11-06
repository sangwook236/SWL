# REF [site] >> https://talbaumel.github.io/attention/
# REF [site] >> https://github.com/fchollet/keras/blob/master/examples/lstm_seq2seq.py
# REF [site] >> https://www.tensorflow.org/tutorials/seq2seq
# REF [site] >> https://www.tensorflow.org/tutorials/recurrent

# REF [site] >> https://blog.heuritech.com/2016/01/20/attention-mechanism/
# REF [site] >> https://github.com/philipperemy/keras-attention-mechanism
# REF [paper] >> "Describing Multimedia Content Using Attention-Based Encoder-Decoder Networks", ToM 2015.
# REF [paper] >> "Neural Machine Translation by Jointly Learning to Align and Translate", arXiv 2016.
# REF [paper] >> "Effective Approaches to Attention-based Neural Machine Translation", arXiv 2015.
# REF [paper] >> "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention", ICML 2015.

#%%------------------------------------------------------------------

import os, sys
if 'posix' == os.name:
	swl_python_home_dir_path = '/home/sangwook/work/SWL_github/python'
else:
	swl_python_home_dir_path = 'D:/work/SWL_github/python'

sys.path.append(swl_python_home_dir_path + '/src')

#%%------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Bidirectional
from keras import optimizers
from keras import backend as K

#%%------------------------------------------------------------------
# Generate a toy problem.
# REF [site] >> https://talbaumel.github.io/attention/

from random import choice, randrange

EOS = '<EOS>'  # All strings will end with the End Of String token.
characters = list('abcd')
characters.append(EOS)

int2char = list(characters)
char2int = {c:i for i,c in enumerate(characters)}

VOCAB_SIZE = len(characters)

def sample_model(min_length, max_length):
	random_length = randrange(min_length, max_length)
	# Pick a random length.
	random_char_list = [choice(characters[:-1]) for _ in range(random_length)]
	# Pick random chars.
	random_str = ''.join(random_char_list)
	return random_str, random_str[::-1]  # Return the random string and its reverse.

def add_eos(str):
	str = list(str) + [EOS]
	return [char2int[ch] for ch in str]

# Preprocessing function for character strings.
def preprocess_string(str):
	return add_eos(str)

def create_dataset(dataset, window_size=1):
	dataX, dataY = [], []
	# FIXME [check] >> Which one is correct?
	#for i in range(len(dataset) - window_size - 1):
	for i in range(len(dataset) - window_size):
		dataX.append(dataset[i:(i + window_size)])
		dataY.append(dataset[i + window_size])  # Next character.
	return np.array(dataX), np.array(dataY)

def create_string_dataset(num_data):
	return [sample_model(1, MAX_STRING_LEN) for _ in range(num_data)]

def convert_string_dataset_to_numeric_dataset(dataset):
	data = []
	for input_str, output_str in dataset:
		#_, x = create_dataset(preprocess_string(input_str), window_size=0)
		#_, y = create_dataset(preprocess_string(output_str), window_size=0)
		x = preprocess_string(input_str)
		y = preprocess_string(output_str)
		data.append((x, y))
	return data

#print(sample_model(4, 5))
#print(sample_model(5, 10))

MAX_STRING_LEN = 15
num_train_data = 3000
num_val_data = 100

train_string_set = create_string_dataset(num_train_data)
val_string_set = create_string_dataset(num_val_data)

train_numeric_set = convert_string_dataset_to_numeric_dataset(train_string_set)
val_numeric_set = convert_string_dataset_to_numeric_dataset(val_string_set)

train_input_data = np.empty((num_train_data, MAX_STRING_LEN), dtype=np.int)
train_input_data.fill(char2int[EOS])
train_output_data = np.empty((num_train_data, MAX_STRING_LEN), dtype=np.int)
train_output_data.fill(char2int[EOS])
for i in range(num_train_data):
	train_input_data[i,:len(train_numeric_set[i][0])] = np.array(train_numeric_set[i][0])
	train_output_data[i,:len(train_numeric_set[i][1])] = np.array(train_numeric_set[i][1])
val_input_data = np.empty((num_val_data, MAX_STRING_LEN), dtype=np.int)
val_input_data.fill(char2int[EOS])
val_output_data = np.empty((num_val_data, MAX_STRING_LEN), dtype=np.int)
val_output_data.fill(char2int[EOS])
for i in range(num_val_data):
	val_input_data[i,:len(val_numeric_set[i][0])] = np.array(val_numeric_set[i][0])
	val_output_data[i,:len(val_numeric_set[i][1])] = np.array(val_numeric_set[i][1])

# Reshape input to be (samples, time steps, features) = (num_train_data/num_val_data, MAX_STRING_LEN, VOCAB_SIZE).
train_input_data = keras.utils.to_categorical(train_input_data, VOCAB_SIZE).reshape(train_input_data.shape + (-1,))
train_output_data = keras.utils.to_categorical(train_output_data, VOCAB_SIZE).reshape(train_output_data.shape + (-1,))
val_input_data = keras.utils.to_categorical(val_input_data, VOCAB_SIZE).reshape(val_input_data.shape + (-1,))
val_output_data = keras.utils.to_categorical(val_output_data, VOCAB_SIZE).reshape(val_output_data.shape + (-1,))

#%%------------------------------------------------------------------

def display_history(history):
	# List all data in history.
	print(history.history.keys())

	# Summarize history for accuracy.
	fig = plt.figure()
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	plt.close(fig)
	# Summarize history for loss.
	fig = plt.figure()
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	plt.close(fig)

#%%------------------------------------------------------------------
# Simple RNN.

state_size = 128
dropout_ratio = 0.5
batch_size = 4
num_epochs = 50

# Build a model.
simple_rnn_inputs = Input(shape=(None, VOCAB_SIZE))
#simple_rnn_inputs = Input(shape=(MAX_STRING_LEN, VOCAB_SIZE))
simple_rnn_lstm1 = LSTM(state_size, dropout=dropout_ratio, recurrent_dropout=dropout_ratio, return_sequences=True)
simple_rnn_outputs = simple_rnn_lstm1(simple_rnn_inputs)
simple_rnn_lstm2 = LSTM(state_size, dropout=dropout_ratio, recurrent_dropout=dropout_ratio, return_sequences=True)
simple_rnn_outputs = simple_rnn_lstm2(simple_rnn_outputs)
simple_rnn_dense = Dense(VOCAB_SIZE, activation='softmax')
simple_rnn_outputs = simple_rnn_dense(simple_rnn_outputs)

simple_rnn_model = Model(inputs=simple_rnn_inputs, outputs=simple_rnn_outputs)

# Display the model summary.
#print(simple_rnn_model.summary())

# Train.
optimizer = optimizers.Adam(lr=1.0e-5, decay=1.0e-9, beta_1=0.9, beta_2=0.999, epsilon=1.0e-8)

simple_rnn_model.compile(optimizer=optimizer, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

history = simple_rnn_model.fit(train_input_data, train_output_data,
		batch_size=batch_size, epochs=num_epochs,
		validation_data=(val_input_data, val_output_data))
display_history(history)

# Evaluate.
test_loss, test_accuracy = simple_rnn_model.evaluate(val_input_data, val_output_data, batch_size=batch_size, verbose=1)
print('Test loss = {}, test accuracy = {}'.format(test_loss, test_accuracy))

# Predict.
test_datum = np.empty((1, MAX_STRING_LEN), dtype=np.int)
test_datum.fill(char2int[EOS])
tmp = np.array(preprocess_string('abc'))
test_datum[:,:tmp.shape[0]] = tmp
test_datum = keras.utils.to_categorical(test_datum, VOCAB_SIZE).reshape(test_datum.shape + (-1,))
prediction = simple_rnn_model.predict(test_datum)

#%%------------------------------------------------------------------
# Bidirectional RNN.

state_size = 64
dropout_ratio = 0.5
batch_size = 4
num_epochs = 50

# Build a model.
bi_rnn_inputs = Input(shape=(None, VOCAB_SIZE))
#bi_rnn_inputs = Input(shape=(MAX_STRING_LEN, VOCAB_SIZE))
bi_rnn_lstm1 = Bidirectional(LSTM(state_size, dropout=dropout_ratio, recurrent_dropout=dropout_ratio, return_sequences=True))
bi_rnn_outputs = bi_rnn_lstm1(bi_rnn_inputs)
bi_rnn_lstm2 = Bidirectional(LSTM(state_size, dropout=dropout_ratio, recurrent_dropout=dropout_ratio, return_sequences=True))
bi_rnn_outputs = bi_rnn_lstm2(bi_rnn_outputs)
bi_rnn_dense = Dense(VOCAB_SIZE, activation='softmax')
bi_rnn_outputs = bi_rnn_dense(bi_rnn_outputs)

bi_rnn_model = Model(inputs=bi_rnn_inputs, outputs=bi_rnn_outputs)

# Display the model summary.
#print(bi_rnn_model.summary())

# Train.
optimizer = optimizers.Adam(lr=1.0e-5, decay=1.0e-9, beta_1=0.9, beta_2=0.999, epsilon=1.0e-8)

bi_rnn_model.compile(optimizer=optimizer, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

history = bi_rnn_model.fit(train_input_data, train_output_data,
		batch_size=batch_size, epochs=num_epochs,
		validation_data=(val_input_data, val_output_data))
display_history(history)

# Evaluate.
test_loss, test_accuracy = bi_rnn_model.evaluate(val_input_data, val_output_data, batch_size=batch_size, verbose=1)
print('Test loss = {}, test accuracy = {}'.format(test_loss, test_accuracy))

# Predict.
test_datum = np.empty((1, MAX_STRING_LEN), dtype=np.int)
test_datum.fill(char2int[EOS])
tmp = np.array(preprocess_string('abc'))
test_datum[:,:tmp.shape[0]] = tmp
test_datum = keras.utils.to_categorical(test_datum, VOCAB_SIZE).reshape(test_datum.shape + (-1,))
prediction = bi_rnn_model.predict(test_datum)
