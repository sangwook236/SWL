# REF [site] >> https://blog.heuritech.com/2016/01/20/attention-mechanism/
# REF [site] >> https://talbaumel.github.io/attention/
# REF [site] >> https://github.com/philipperemy/keras-attention-mechanism
# REF [site] >> https://github.com/fchollet/keras/blob/master/examples/lstm_seq2seq.py
# REF [site] >> https://www.tensorflow.org/tutorials/seq2seq
# REF [site] >> https://www.tensorflow.org/tutorials/recurrent
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
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation, Bidirectional
from keras import optimizers, callbacks
from keras import backend as K

#%%------------------------------------------------------------------
# Generate a toy problem.

from random import choice, randrange

EOS = '<EOS>'  # All strings will end with the End Of String token.
characters = list('abcd')
characters.append(EOS)

int2char = list(characters)
char2int = {c:i for i,c in enumerate(characters)}

VOCAB_SIZE = len(characters)

def sample_model(min_length, max_lenth):
	random_length = randrange(min_length, max_lenth)
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

def create_string_dataset(dataset):
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
train_set = [sample_model(1, MAX_STRING_LEN) for _ in range(3000)]
val_set = [sample_model(1, MAX_STRING_LEN) for _ in range(50)]

train_set_numeric = create_string_dataset(train_set)
val_set_numeric = create_string_dataset(val_set)

#%%------------------------------------------------------------------
# Simple RNN.

num_classes = VOCAB_SIZE
state_size = 128
dropout_ratio = 0.5

# Build model.
model = Sequential()
model.add(LSTM(state_size, dropout=dropout_ratio, recurrent_dropout=dropout_ratio, return_sequences=True, input_shape=(1, 1)))
model.add(LSTM(state_size, dropout=dropout_ratio, recurrent_dropout=dropout_ratio))
model.add(Dense(num_classes, activation='softmax'))

# Display the model summary.
#print(model.summary())

optimizer = optimizers.Adam(lr=1.0e-5, decay=1.0e-9, beta_1=0.9, beta_2=0.999, epsilon=1.0e-8)

model.compile(optimizer=optimizer, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

history = model.fit(train_signals[:,:10000,:], train_labels,
		batch_size=batch_size, epochs=num_epochs, initial_epoch=initial_epoch,
		#validation_data=None, validation_split=0.2,
		validation_data=(test_signals[:,:10000,:], test_labels), validation_split=0.0,
		class_weight=None, callbacks=callback_list, shuffle=shuffle, verbose=1)
display_history(history)
