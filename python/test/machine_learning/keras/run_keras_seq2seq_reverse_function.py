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

#--------------------
import os, sys
if 'posix' == os.name:
	swl_python_home_dir_path = '/home/sangwook/work/SWL_github/python'
else:
	swl_python_home_dir_path = 'D:/work/SWL_github/python'

sys.path.append(swl_python_home_dir_path + '/src')

#--------------------
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Bidirectional
from keras import optimizers
from keras import backend as K

#--------------------------------------------------------------------
# Generate a toy problem.
# REF [site] >> https://talbaumel.github.io/attention/

from random import choice, randrange

use_SOS = True

if use_SOS:
	SOS = '<SOS>'  # All strings will start with the Start Of String token.
EOS = '<EOS>'  # All strings will end with the End Of String token.
characters = list('abcd')
if use_SOS:
	characters = [SOS] + characters
characters.append(EOS)

int2char = list(characters)
char2int = {c:i for i,c in enumerate(characters)}

VOCAB_SIZE = len(characters)

def sample_model(min_length, max_length):
	random_length = randrange(min_length, max_length)
	# Pick a random length.
	if use_SOS:
		random_char_list = [choice(characters[1:-1]) for _ in range(random_length)]
	else:
		random_char_list = [choice(characters[:-1]) for _ in range(random_length)]
	# Pick random chars.
	random_str = ''.join(random_char_list)
	return random_str, random_str[::-1]  # Return the random string and its reverse.

def add_eos(str):
	str = list(str) + [EOS]
	return [char2int[ch] for ch in str]

def add_sos_and_eos(str):
	str = [SOS] + list(str) + [EOS]
	return [char2int[ch] for ch in str]

# Preprocessing function for character strings.
def preprocess_string(str):
	return add_sos_and_eos(str) if use_SOS else add_eos(str)

def create_dataset(dataset, window_size=1):
	dataX, dataY = [], []
	# FIXME [check] >> Which one is correct?
	#for i in range(len(dataset) - window_size - 1):
	for i in range(len(dataset) - window_size):
		dataX.append(dataset[i:(i + window_size)])
		dataY.append(dataset[i + window_size])  # Next character.
	return np.array(dataX), np.array(dataY)

def create_string_dataset(num_data, str_len):
	return [sample_model(1, str_len) for _ in range(num_data)]

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
if use_SOS:
	MAX_TOKEN_LEN = MAX_STRING_LEN + 2
else:
	#MAX_TOKEN_LEN = MAX_STRING_LEN
	MAX_TOKEN_LEN = MAX_STRING_LEN + 1
num_train_data = 3000
num_val_data = 100

train_string_set = create_string_dataset(num_train_data, MAX_STRING_LEN)
val_string_set = create_string_dataset(num_val_data, MAX_STRING_LEN)

train_numeric_set = convert_string_dataset_to_numeric_dataset(train_string_set)
val_numeric_set = convert_string_dataset_to_numeric_dataset(val_string_set)

def max_len(dataset):
	num_data = len(dataset)
	ml = 0
	for i in range(num_data):
		if len(dataset[i][0]) > ml:
			ml = len(dataset[i][0])
	return ml

def create_dataset_array(input_output_pairs, str_len):
	num_data = len(input_output_pairs)
	input_data = np.empty((num_data, str_len), dtype=np.int)
	input_data.fill(char2int[EOS])
	output_data = np.empty((num_data, str_len), dtype=np.int)
	output_data.fill(char2int[EOS])
	output_data_ahead_of_one_timestep = np.empty((num_data, str_len), dtype=np.int)
	output_data_ahead_of_one_timestep.fill(char2int[EOS])
	for i in range(num_data):
		inp = input_output_pairs[i][0]
		outp = input_output_pairs[i][1]
		input_data[i,:len(inp)] = np.array(inp)
		outa = np.array(outp)
		output_data[i,:len(outp)] = outa
		output_data_ahead_of_one_timestep[i,:(len(outp) - 1)] = outa[1:]

	return input_data, output_data, output_data_ahead_of_one_timestep

train_input_data, train_output_data, train_output_data_ahead_of_one_timestep = create_dataset_array(train_numeric_set, MAX_TOKEN_LEN)
#val_input_data, _, val_output_data_ahead_of_one_timestep = create_dataset_array(val_numeric_set, MAX_TOKEN_LEN)
val_input_data, val_output_data, val_output_data_ahead_of_one_timestep = create_dataset_array(val_numeric_set, MAX_TOKEN_LEN)

# Reshape input to be (samples, time steps, features) = (num_train_data, MAX_TOKEN_LEN, VOCAB_SIZE).
train_input_data = keras.utils.to_categorical(train_input_data, VOCAB_SIZE).reshape(train_input_data.shape + (-1,))
train_output_data = keras.utils.to_categorical(train_output_data, VOCAB_SIZE).reshape(train_output_data.shape + (-1,))
train_output_data_ahead_of_one_timestep = keras.utils.to_categorical(train_output_data_ahead_of_one_timestep, VOCAB_SIZE).reshape(train_output_data_ahead_of_one_timestep.shape + (-1,))
# Reshape input to be (samples, time steps, features) = (num_val_data, MAX_TOKEN_LEN, VOCAB_SIZE).
val_input_data = keras.utils.to_categorical(val_input_data, VOCAB_SIZE).reshape(val_input_data.shape + (-1,))
val_output_data = keras.utils.to_categorical(val_output_data, VOCAB_SIZE).reshape(val_output_data.shape + (-1,))
val_output_data_ahead_of_one_timestep = keras.utils.to_categorical(val_output_data_ahead_of_one_timestep, VOCAB_SIZE).reshape(val_output_data_ahead_of_one_timestep.shape + (-1,))

#--------------------------------------------------------------------

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

def decode_predicted_sequence(prediction):
	num_tokens = prediction.shape[1]
	predicted_sentence = ''
	for i in range(num_tokens):
		token_index = np.argmax(prediction[0, i, :])
		ch = int2char[token_index]
		predicted_sentence += ch

	return predicted_sentence;

def decode_sequence(encoder_model, decoder_model, input_seq):
	# Encode the input as state vectors.
	states_output = encoder_model.predict(input_seq)

	# Generate empty target sequence of length 1.
	target_seq = np.zeros((1, 1, VOCAB_SIZE))
	# Populate the first character of target sequence with the start character.
	target_seq[0, 0, 0] = 1  # <SOS>.

	# Sampling loop for a batch of sequences (to simplify, here we assume a batch of size 1).
	stop_condition = False
	decoded_sentence = ''
	while not stop_condition:
		output_tokens, h, c = decoder_model.predict([target_seq] + states_output)

		# Sample a token.
		sampled_token_index = np.argmax(output_tokens[0, -1, :])
		sampled_char = int2char[sampled_token_index]
		decoded_sentence += sampled_char

		# Exit condition: either hit max length or find stop character.
		if (sampled_char == EOS or len(decoded_sentence) > MAX_TOKEN_LEN):
			stop_condition = True

		# Update the target sequence (of length 1).
		target_seq = np.zeros((1, 1, VOCAB_SIZE))
		target_seq[0, 0, sampled_token_index] = 1

		# Update states.
		states_output = [h, c]

	return decoded_sentence

#--------------------------------------------------------------------
# Simple RNN.
# REF [site] >> https://talbaumel.github.io/attention/

state_size = 128
dropout_ratio = 0.5
batch_size = 4
num_epochs = 50

# Build a model.
# Input shape = (samples, time-steps, features) = (None, None, VOCAB_SIZE).
simple_rnn_inputs = Input(shape=(None, VOCAB_SIZE))
#simple_rnn_inputs = Input(shape=(MAX_TOKEN_LEN, VOCAB_SIZE))
simple_rnn_lstm1 = LSTM(state_size, dropout=dropout_ratio, recurrent_dropout=dropout_ratio, return_sequences=True)
simple_rnn_outputs = simple_rnn_lstm1(simple_rnn_inputs)
simple_rnn_lstm2 = LSTM(state_size, dropout=dropout_ratio, recurrent_dropout=dropout_ratio, return_sequences=True)
simple_rnn_outputs = simple_rnn_lstm2(simple_rnn_outputs)
simple_rnn_dense = Dense(VOCAB_SIZE, activation='softmax')
# Output shape = (samples, time-steps, features) = (None, None, VOCAB_SIZE).
simple_rnn_outputs = simple_rnn_dense(simple_rnn_outputs)

simple_rnn_model = Model(inputs=simple_rnn_inputs, outputs=simple_rnn_outputs)

# Summarize the model.
#print(simple_rnn_model.summary())

# Train.
optimizer = optimizers.Adam(lr=1.0e-5, decay=1.0e-9, beta_1=0.9, beta_2=0.999, epsilon=1.0e-8)

simple_rnn_model.compile(optimizer=optimizer, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

history = simple_rnn_model.fit(train_input_data, train_output_data,
		batch_size=batch_size, epochs=num_epochs,
		validation_data=(val_input_data, val_output_data))
		#validation_split=0.2)
display_history(history)

# Save the model.
simple_rnn_model.save('keras_seq2seq_reverse_function_simple_rnn.h5')

# Evaluate.
test_loss, test_accuracy = simple_rnn_model.evaluate(val_input_data, val_output_data_ahead_of_one_timestep, batch_size=batch_size, verbose=1)
print('Test loss = {}, test accuracy = {}'.format(test_loss, test_accuracy))

# Predict.
input_seq = 'abc'
test_datum = np.empty((1, MAX_TOKEN_LEN), dtype=np.int)
test_datum.fill(char2int[EOS])
tmp = np.array(preprocess_string(input_seq))
test_datum[:,:tmp.shape[0]] = tmp
test_datum = keras.utils.to_categorical(test_datum, VOCAB_SIZE).reshape(test_datum.shape + (-1,))
prediction = simple_rnn_model.predict(test_datum)

predicted_seq = decode_predicted_sequence(prediction)
print('Predicted sequence of {} = {}'.format(input_seq, predicted_seq))

#--------------------------------------------------------------------
# Bidirectional RNN.
# REF [site] >> https://talbaumel.github.io/attention/

state_size = 128
dropout_ratio = 0.5
batch_size = 4
num_epochs = 50

# Build a model.
# Input shape = (samples, time-steps, features) = (None, None, VOCAB_SIZE).
bi_rnn_inputs = Input(shape=(None, VOCAB_SIZE))
#bi_rnn_inputs = Input(shape=(MAX_TOKEN_LEN, VOCAB_SIZE))
bi_rnn_lstm1 = Bidirectional(LSTM(state_size, dropout=dropout_ratio, recurrent_dropout=dropout_ratio, return_sequences=True))
bi_rnn_outputs = bi_rnn_lstm1(bi_rnn_inputs)
# FIXME [check] >> I don't know why.
# Output shape = (None, state_size * 2).
#bi_rnn_lstm2 = Bidirectional(LSTM(state_size, dropout=dropout_ratio, recurrent_dropout=dropout_ratio))
# Output shape = (None, None, state_size * 2).
bi_rnn_lstm2 = Bidirectional(LSTM(state_size, dropout=dropout_ratio, recurrent_dropout=dropout_ratio, return_sequences=True))
bi_rnn_outputs = bi_rnn_lstm2(bi_rnn_outputs)
bi_rnn_dense = Dense(VOCAB_SIZE, activation='softmax')
bi_rnn_outputs = bi_rnn_dense(bi_rnn_outputs)

bi_rnn_model = Model(inputs=bi_rnn_inputs, outputs=bi_rnn_outputs)

# Summarize the model.
#print(bi_rnn_model.summary())

# Train.
optimizer = optimizers.Adam(lr=1.0e-5, decay=1.0e-9, beta_1=0.9, beta_2=0.999, epsilon=1.0e-8)

bi_rnn_model.compile(optimizer=optimizer, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

history = bi_rnn_model.fit(train_input_data, train_output_data,
		batch_size=batch_size, epochs=num_epochs,
		validation_data=(val_input_data, val_output_data))
		#validation_split=0.2)
display_history(history)

# Save the model.
bi_rnn_model.save('keras_seq2seq_reverse_function_bidirectional_rnn.h5')

# Evaluate.
test_loss, test_accuracy = bi_rnn_model.evaluate(val_input_data, val_output_data_ahead_of_one_timestep, batch_size=batch_size, verbose=1)
print('Test loss = {}, test accuracy = {}'.format(test_loss, test_accuracy))

# Predict.
input_seq = 'abc'
test_datum = np.empty((1, MAX_TOKEN_LEN), dtype=np.int)
test_datum.fill(char2int[EOS])
tmp = np.array(preprocess_string(input_seq))
test_datum[:,:tmp.shape[0]] = tmp
test_datum = keras.utils.to_categorical(test_datum, VOCAB_SIZE).reshape(test_datum.shape + (-1,))
prediction = bi_rnn_model.predict(test_datum)

predicted_seq = decode_predicted_sequence(prediction)
print('Predicted sequence of {} = {}'.format(input_seq, predicted_seq))

#--------------------------------------------------------------------
# Encoder-decoder model.
# REF [site] >> https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
# REF [site] >> https://talbaumel.github.io/attention/

enc_state_size = 128
dec_state_size = 128
dropout_ratio = 0.5
batch_size = 4
num_epochs = 100

#----------
# Build a training model.
# Input shape = (samples, time-steps, features) = (None, None, VOCAB_SIZE).
encdec_enc_inputs = Input(shape=(None, VOCAB_SIZE))
#encdec_enc_inputs = Input(shape=(MAX_TOKEN_LEN, VOCAB_SIZE))
encdec_enc_lstm1 = LSTM(enc_state_size, dropout=dropout_ratio, recurrent_dropout=dropout_ratio, return_sequences=True, return_state=True)
encdec_enc_outputs, encdec_enc_state_h, encdec_enc_state_c = encdec_enc_lstm1(encdec_enc_inputs)
# Discard 'encdec_enc_outputs' and only keep the states.
encdec_enc_states = [encdec_enc_state_h, encdec_enc_state_c]
#encdec_enc_states = keras.layers.concatenate([encdec_enc_state_h, encdec_enc_state_c], axis=-1)

# NOTE [info] >>
#	encdec_enc_outputs.shape = (None, None, enc_state_size).
#	encdec_enc_state_h.shape = (None, enc_state_size).
#	encdec_enc_state_c.shape = (None, enc_state_size).

# Input shape = (samples, time-steps, features).
encdec_dec_inputs = Input(shape=(None, VOCAB_SIZE))
encdec_dec_lstm1 = LSTM(dec_state_size, dropout=dropout_ratio, recurrent_dropout=dropout_ratio, return_sequences=True, return_state=True)
encdec_dec_outputs, _, _ = encdec_dec_lstm1(encdec_dec_inputs, initial_state=encdec_enc_states)
encdec_dec_dense = Dense(VOCAB_SIZE, activation='softmax')
encdec_dec_outputs = encdec_dec_dense(encdec_dec_outputs)

encdec_train_model = Model(inputs=[encdec_enc_inputs, encdec_dec_inputs], outputs=encdec_dec_outputs)

# Summarize the training model.
#print(encdec_train_model.summary())

# Train.
optimizer = optimizers.Adam(lr=1.0e-5, decay=1.0e-9, beta_1=0.9, beta_2=0.999, epsilon=1.0e-8)

encdec_train_model.compile(optimizer=optimizer, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

history = encdec_train_model.fit([train_input_data, train_output_data], train_output_data_ahead_of_one_timestep,
		batch_size=batch_size, epochs=num_epochs,
		validation_data=([val_input_data, val_output_data], val_output_data_ahead_of_one_timestep))
		#validation_split=0.2)
display_history(history)

# Save the training model.
encdec_train_model.save('keras_seq2seq_reverse_function_encdec_model.h5')

#--------------------
# Build inference models.
encdec_inf_encoder_model = Model(inputs=encdec_enc_inputs, outputs=encdec_enc_states)

# Input shape = (samples, time-steps, features).
encdec_state_input_h = Input(shape=(dec_state_size,))
encdec_state_input_c = Input(shape=(dec_state_size,))
encdec_states_inputs = [encdec_state_input_h, encdec_state_input_c]
encdec_dec_outputs, encdec_state_h, encdec_state_c = encdec_dec_lstm1(encdec_dec_inputs, initial_state=encdec_states_inputs)
encdec_dec_states = [encdec_state_h, encdec_state_c]
encdec_dec_outputs = encdec_dec_dense(encdec_dec_outputs)

encdec_inf_decoder_model = Model(inputs=[encdec_dec_inputs] + encdec_states_inputs, outputs=[encdec_dec_outputs] + encdec_dec_states)

# Evaluate.
test_loss, test_accuracy = encdec_inf_decoder_model.evaluate(val_input_data, val_output_data_ahead_of_one_timestep, batch_size=batch_size, verbose=1)
print('Test loss = {}, test accuracy = {}'.format(test_loss, test_accuracy))

# Predict.
input_seq = 'abc'
decoded_seq = decode_sequence(encdec_inf_encoder_model, encdec_inf_decoder_model, preprocess_string(input_seq))
print('Predicted sequence of {} = {}'.format(input_seq, decoded_seq))

#--------------------------------------------------------------------
# Attention model.
# REF [site] >> https://talbaumel.github.io/attention/

enc_state_size = 128
dec_state_size = 128
dropout_ratio = 0.5
batch_size = 4
num_epochs = 50

# Build a model.
# Input shape = (samples, time-steps, features) = (None, None, VOCAB_SIZE).
attn_enc_inputs = Input(shape=(None, VOCAB_SIZE))
#attn_enc_inputs = Input(shape=(MAX_TOKEN_LEN, VOCAB_SIZE))
attn_enc_lstm1 = LSTM(enc_state_size, dropout=dropout_ratio, recurrent_dropout=dropout_ratio, return_sequences=True, return_state=True)
attn_enc_outputs, attn_enc_state_h, attn_enc_state_c = attn_enc_lstm1(attn_enc_inputs)
# Discard 'attn_enc_outputs' and only keep the states.
attn_enc_states = [attn_enc_state_h, attn_enc_state_c]

# Input shape = (samples, time-steps, features).
attn_attn_inputs = Input(shape=(enc_state_size,))
attn_attn_lstm1 = LSTM(dec_state_size, dropout=dropout_ratio, recurrent_dropout=dropout_ratio)
attn_attn_outputs = attn_attn_lstm1(attn_attn_inputs)
attn_attn_dense = Dense(VOCAB_SIZE, activation='softmax')
attn_attn_outputs = attn_attn_dense(attn_attn_outputs)

# Input shape = (samples, time-steps, features).
attn_dec_inputs = Input(shape=(enc_state_size,))
attn_dec_lstm1 = LSTM(dec_state_size, dropout=dropout_ratio, recurrent_dropout=dropout_ratio)
attn_dec_outputs = attn_dec_lstm1(attn_dec_inputs)
attn_dec_dense = Dense(VOCAB_SIZE, activation='softmax')
attn_dec_outputs = attn_dec_dense(attn_dec_outputs)

attn_model = Model(inputs=attn_enc_inputs, outputs=attn_dec_outputs)

# Summarize the model.
#print(attn_model.summary())

# Train.
optimizer = optimizers.Adam(lr=1.0e-5, decay=1.0e-9, beta_1=0.9, beta_2=0.999, epsilon=1.0e-8)

attn_model.compile(optimizer=optimizer, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

history = attn_model.fit(train_input_data, train_output_data_ahead_of_one_timestep,
		batch_size=batch_size, epochs=num_epochs,
		validation_data=(val_input_data, val_output_data_ahead_of_one_timestep))
		#validation_split=0.2)
display_history(history)

# Save the model.
attn_model.save('keras_seq2seq_reverse_function_attention_model.h5')

# Evaluate.
test_loss, test_accuracy = attn_model.evaluate(val_input_data, val_output_data_ahead_of_one_timestep, batch_size=batch_size, verbose=1)
print('Test loss = {}, test accuracy = {}'.format(test_loss, test_accuracy))

# Predict.
