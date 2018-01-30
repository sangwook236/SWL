# REF [site] >> https://talbaumel.github.io/attention/ ==> Neural Attention Mechanism - Sequence To Sequence Attention Models In DyNet.pdf
# REF [site] >> https://github.com/fchollet/keras/blob/master/examples/lstm_seq2seq.py
# REF [site] >> https://www.tensorflow.org/tutorials/seq2seq
# REF [site] >> https://www.tensorflow.org/tutorials/recurrent

# REF [site] >> https://blog.heuritech.com/2016/01/20/attention-mechanism/
# REF [site] >> https://github.com/philipperemy/keras-attention-mechanism

# REF [paper] >> "Describing Multimedia Content Using Attention-Based Encoder-Decoder Networks", ToM 2015.
# REF [paper] >> "Neural Machine Translation by Jointly Learning to Align and Translate", arXiv 2016.
# REF [paper] >> "Effective Approaches to Attention-based Neural Machine Translation", arXiv 2015.
# REF [paper] >> "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention", ICML 2015.

# Path to libcudnn.so.
#export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

#--------------------
import os, sys
if 'posix' == os.name:
	swl_python_home_dir_path = '/home/sangwook/work/SWL_github/python'
	lib_home_dir_path = '/home/sangwook/lib_repo/python'
else:
	swl_python_home_dir_path = 'D:/work/SWL_github/python'
	lib_home_dir_path = 'D:/lib_repo/python'
	#lib_home_dir_path = 'D:/lib_repo/python/rnd'
sys.path.append(swl_python_home_dir_path + '/src')
sys.path.append(lib_home_dir_path + '/tflearn_github')
#sys.path.append('../../../src')

#os.chdir(swl_python_home_dir_path + '/test/machine_learning/tensorflow')

#--------------------
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from random import choice, randrange
from reverse_function_tf_rnn import ReverseFunctionTensorFlowRNN
from reverse_function_tf_encdec import ReverseFunctionTensorFlowEncoderDecoder
#from reverse_function_tf_attention import ReverseFunctionTensorFlowEncoderDecoderWithAttention
from reverse_function_keras_rnn import ReverseFunctionKerasRNN
from reverse_function_rnn_trainer import ReverseFunctionRnnTrainer
from swl.machine_learning.tensorflow.neural_net_evaluator import NeuralNetEvaluator
from swl.machine_learning.tensorflow.neural_net_predictor import NeuralNetPredictor
#from swl.machine_learning.tensorflow.neural_net_trainer import TrainingMode
import keras
import time

#np.random.seed(7)

#%%------------------------------------------------------------------
# Prepare directories.

import datetime

output_dir_prefix = 'reverse_function'
output_dir_suffix = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
#output_dir_suffix = '20180116T212902'

model_dir_path = './result/{}_model_{}'.format(output_dir_prefix, output_dir_suffix)
prediction_dir_path = './result/{}_prediction_{}'.format(output_dir_prefix, output_dir_suffix)
train_summary_dir_path = './log/{}_train_{}'.format(output_dir_prefix, output_dir_suffix)
val_summary_dir_path = './log/{}_val_{}'.format(output_dir_prefix, output_dir_suffix)

#%%------------------------------------------------------------------
# Generate a toy problem.
# REF [site] >> https://talbaumel.github.io/attention/

def sample_model(min_length, max_length):
	random_length = randrange(min_length, max_length)
	# Pick a random length.
	random_char_list = [choice(characters[1:-1]) for _ in range(random_length)]
	# Pick random chars.
	random_str = ''.join(random_char_list)
	return random_str, random_str[::-1]  # Return the random string and its reverse.

# A character string to a numeric datum(numeric list).
def str2datum(str):
	#str = list(str) + [EOS]
	str = [SOS] + list(str) + [EOS]
	return [char2int[ch] for ch in str]

# A numeric datum(numeric list) to a character string.
def datum2str(datum):
	locs = np.where(char2int[EOS] == datum)
	datum = datum[:locs[0][0]]
	#return ''.join([int2char[no] for no in datum[:]])
	return ''.join([int2char[no] for no in datum[1:]])

# Preprocessing function for character strings.
def preprocess_string(str):
	return str2datum(str)

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

def max_len(dataset):
	num_data = len(dataset)
	ml = 0
	for i in range(num_data):
		if len(dataset[i][0]) > ml:
			ml = len(dataset[i][0])
	return ml

# Fixed-size dataset.
def create_array_dataset(input_output_pairs, str_len):
	num_data = len(input_output_pairs)
	input_data = np.full((num_data, str_len), char2int[EOS])
	output_data = np.full((num_data, str_len), char2int[EOS])
	output_data_ahead_of_one_timestep = np.full((num_data, str_len), char2int[EOS])
	for (i, (inp, outp)) in enumerate(input_output_pairs):
		input_data[i,:len(inp)] = np.array(inp)
		outa = np.array(outp)
		output_data[i,:len(outp)] = outa
		output_data_ahead_of_one_timestep[i,:(len(outp) - 1)] = outa[1:]

	return input_data, output_data, output_data_ahead_of_one_timestep

# Variable-size dataset.
def create_list_dataset(input_output_pairs, str_len):
	input_data, output_data, output_data_ahead_of_one_timestep = [], [], []
	for (inp, outp) in input_output_pairs:
		input_data.append(np.array(inp))
		output_data.append(np.array(outp))
		output_data_ahead_of_one_timestep.append(np.array(outp[1:]))

	return input_data, output_data, output_data_ahead_of_one_timestep

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

#%%------------------------------------------------------------------
# Prepare data.
	
SOS = '<SOS>'  # All strings will start with the Start-Of-String token.
EOS = '<EOS>'  # All strings will end with the End-Of-String token.
characters = list('abcd')
characters = [SOS] + characters + [EOS]

int2char = list(characters)
char2int = {c:i for i, c in enumerate(characters)}

VOCAB_SIZE = len(characters)

#print(sample_model(4, 5))
#print(sample_model(5, 10))

MAX_STRING_LEN = 15
#MAX_TOKEN_LEN = MAX_STRING_LEN
#MAX_TOKEN_LEN = MAX_STRING_LEN + 1
MAX_TOKEN_LEN = MAX_STRING_LEN + 2
num_train_data = 3000
num_val_data = 100

train_string_list = create_string_dataset(num_train_data, MAX_STRING_LEN)
val_string_list = create_string_dataset(num_val_data, MAX_STRING_LEN)

train_numeric_list = convert_string_dataset_to_numeric_dataset(train_string_list)
val_numeric_list = convert_string_dataset_to_numeric_dataset(val_string_list)

if True:
	# Uses fixed-size dataset.

	train_data, train_labels, train_labels_ahead_of_one_timestep = create_array_dataset(train_numeric_list, MAX_TOKEN_LEN)
	#val_data, _, val_labels_ahead_of_one_timestep = create_array_dataset(val_numeric_list, MAX_TOKEN_LEN)
	val_data, val_labels, val_labels_ahead_of_one_timestep = create_array_dataset(val_numeric_list, MAX_TOKEN_LEN)

	# Reshape input to be (samples, time-steps, features) = (num_train_data, MAX_TOKEN_LEN, VOCAB_SIZE).
	train_data = keras.utils.to_categorical(train_data, VOCAB_SIZE).reshape(train_data.shape + (-1,))
	train_labels = keras.utils.to_categorical(train_labels, VOCAB_SIZE).reshape(train_labels.shape + (-1,))
	train_labels_ahead_of_one_timestep = keras.utils.to_categorical(train_labels_ahead_of_one_timestep, VOCAB_SIZE).reshape(train_labels_ahead_of_one_timestep.shape + (-1,))
	# Reshape input to be (samples, time-steps, features) = (num_val_data, MAX_TOKEN_LEN, VOCAB_SIZE).
	val_data = keras.utils.to_categorical(val_data, VOCAB_SIZE).reshape(val_data.shape + (-1,))
	val_labels = keras.utils.to_categorical(val_labels, VOCAB_SIZE).reshape(val_labels.shape + (-1,))
	val_labels_ahead_of_one_timestep = keras.utils.to_categorical(val_labels_ahead_of_one_timestep, VOCAB_SIZE).reshape(val_labels_ahead_of_one_timestep.shape + (-1,))
else:
	# Uses variable-size dataset.
	# TensorFlow internally uses np.arary for tf.placeholder. (?)

	train_data, train_labels, train_labels_ahead_of_one_timestep = create_list_dataset(train_numeric_list, MAX_TOKEN_LEN)
	#val_data, _, val_labels_ahead_of_one_timestep = create_list_dataset(val_numeric_list, MAX_TOKEN_LEN)
	val_data, val_labels, val_labels_ahead_of_one_timestep = create_list_dataset(val_numeric_list, MAX_TOKEN_LEN)

	tmp_data, tmp_labels, tmp_labels_ahead = [], [], []
	for (dat, lbl, lbl_ahead) in zip(train_data, train_labels, train_labels_ahead_of_one_timestep):
		tmp_data.append(keras.utils.to_categorical(dat, VOCAB_SIZE).reshape(dat.shape + (-1,)))
		tmp_labels.append(keras.utils.to_categorical(lbl, VOCAB_SIZE).reshape(lbl.shape + (-1,)))
		tmp_labels_ahead.append(keras.utils.to_categorical(lbl_ahead, VOCAB_SIZE).reshape(lbl_ahead.shape + (-1,)))
	train_data, train_labels, train_labels_ahead_of_one_timestep = tmp_data, tmp_labels, tmp_labels_ahead
	tmp_data, tmp_labels, tmp_labels_ahead = [], [], []
	for (dat, lbl, lbl_ahead) in zip(val_data, val_labels, val_labels_ahead_of_one_timestep):
		tmp_data.append(keras.utils.to_categorical(dat, VOCAB_SIZE).reshape(dat.shape + (-1,)))
		tmp_labels.append(keras.utils.to_categorical(lbl, VOCAB_SIZE).reshape(lbl.shape + (-1,)))
		tmp_labels_ahead.append(keras.utils.to_categorical(lbl_ahead, VOCAB_SIZE).reshape(lbl_ahead.shape + (-1,)))
	val_data, val_labels, val_labels_ahead_of_one_timestep = tmp_data, tmp_labels, tmp_labels_ahead
	tmp_data, tmp_labels, tmp_labels_ahead = [], [], []

#%%------------------------------------------------------------------
# Configure tensorflow.

config = tf.ConfigProto()
#config.allow_soft_placement = True
config.log_device_placement = True
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.4  # Only allocate 40% of the total memory of each GPU.

# REF [site] >> https://stackoverflow.com/questions/45093688/how-to-understand-sess-as-default-and-sess-graph-as-default
#graph = tf.Graph()
#session = tf.Session(graph=graph, config=config)
session = tf.Session(config=config)

#%%------------------------------------------------------------------

def train_model(session, rnnModel, batch_size, num_epochs, shuffle, initial_epoch):
	nnTrainer = ReverseFunctionRnnTrainer(rnnModel, initial_epoch)
	session.run(tf.global_variables_initializer())
	with session.as_default() as sess:
		# Save a model every 2 hours and maximum 5 latest models are saved.
		saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

		start_time = time.time()
		history = nnTrainer.train(sess, train_data, train_labels, val_data, val_labels, batch_size, num_epochs, shuffle, saver=saver, model_save_dir_path=model_dir_path, train_summary_dir_path=train_summary_dir_path, val_summary_dir_path=val_summary_dir_path)
		end_time = time.time()

		print('\tTraining time = {}'.format(end_time - start_time))

		# Display results.
		nnTrainer.display_history(history)

def evaluate_model(session, rnnModel, batch_size):
	nnEvaluator = NeuralNetEvaluator()
	with session.as_default() as sess:
		start_time = time.time()
		test_loss, test_acc = nnEvaluator.evaluate(sess, rnnModel, val_data, val_labels, batch_size)
		end_time = time.time()

		print('\tEvaluation time = {}'.format(end_time - start_time))
		print('\tTest loss = {}, test accurary = {}'.format(test_loss, test_acc))

def predict_model(session, rnnModel, batch_size, test_strs):
	nnPredictor = NeuralNetPredictor()
	with session.as_default() as sess:
		# Character strings -> numeric data.
		test_data = np.full((len(test_strs), MAX_TOKEN_LEN), char2int[EOS])
		for (i, str) in enumerate(test_strs):
			tmp = np.array(preprocess_string(str))
			test_data[i,:tmp.shape[0]] = tmp
		test_data.reshape((-1,) + test_data.shape)
		test_data = keras.utils.to_categorical(test_data, VOCAB_SIZE).reshape(test_data.shape + (-1,))

		start_time = time.time()
		predictions = nnPredictor.predict(sess, rnnModel, test_data)
		end_time = time.time()

		# Numeric data -> character strings.
		predictions = np.argmax(predictions, axis=-1)
		predicted_strs = []
		for pred in predictions:
			predicted_strs.append(datum2str(pred))

		print('\tPrediction time = {}'.format(end_time - start_time))
		print('\tTest strings = {}, predicted strings = {}'.format(test_strs, predicted_strs))

is_dynamic = True
if is_dynamic:
	# For dynamic RNNs.
	# TODO [improve] >> Training & validation datasets are still fixed-size (static).
	input_shape = (None, VOCAB_SIZE)
	output_shape = (None, VOCAB_SIZE)
else:
	# For static RNNs.
	input_shape = (MAX_TOKEN_LEN, VOCAB_SIZE)
	output_shape = (MAX_TOKEN_LEN, VOCAB_SIZE)

#%%------------------------------------------------------------------
# Simple RNN.
# REF [site] >> https://talbaumel.github.io/attention/

if False:
	# Build a model.
	is_stacked = True  # Uses multiple layers.
	rnnModel = ReverseFunctionTensorFlowRNN(input_shape, output_shape, is_dynamic=is_dynamic, is_bidirectional=False, is_stacked=is_stacked)
	#from keras import backend as K
	#K.set_learning_phase(1)  # Set the learning phase to 'train'.
	##K.set_learning_phase(0)  # Set the learning phase to 'test'.
	#rnnModel = ReverseFunctionKerasRNN(input_shape, output_shape, is_bidirectional=False, is_stacked=is_stacked)

	#--------------------
	batch_size = 4  # Number of samples per gradient update.
	num_epochs = 20  # Number of times to iterate over training data.

	shuffle = True
	initial_epoch = 0

	train_model(session, rnnModel, batch_size, num_epochs, shuffle, initial_epoch)
	evaluate_model(session, rnnModel, batch_size)
	test_strs = ['abc', 'cba', 'dcb', 'abcd', 'dcba', 'cdacbd', 'bcdaabccdb']
	predict_model(session, rnnModel, batch_size, test_strs)

#%%------------------------------------------------------------------
# Bidirectional RNN.
# REF [site] >> https://talbaumel.github.io/attention/

if False:
	# Build a model.
	is_stacked = True  # Uses multiple layers.
	rnnModel = ReverseFunctionTensorFlowRNN(input_shape, output_shape, is_dynamic=is_dynamic, is_bidirectional=True, is_stacked=is_stacked)
	#from keras import backend as K
	#K.set_learning_phase(1)  # Set the learning phase to 'train'.
	##K.set_learning_phase(0)  # Set the learning phase to 'test'.
	#rnnModel = ReverseFunctionKerasRNN(input_shape, output_shape, is_bidirectional=True, is_stacked=is_stacked)

	#--------------------
	batch_size = 4  # Number of samples per gradient update.
	num_epochs = 120  # Number of times to iterate over training data.

	shuffle = True
	initial_epoch = 0

	train_model(session, rnnModel, batch_size, num_epochs, shuffle, initial_epoch)
	evaluate_model(session, rnnModel, batch_size)
	test_strs = ['abc', 'cba', 'dcb', 'abcd', 'dcba', 'cdacbd', 'bcdaabccdb']
	predict_model(session, rnnModel, batch_size, test_strs)

#%%------------------------------------------------------------------
# Encoder-decoder model.
# REF [site] >> https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
# REF [site] >> https://talbaumel.github.io/attention/

if False:
	is_bidirectional = True
	rnnModel = ReverseFunctionTensorFlowEncoderDecoder(input_shape, output_shape, is_dynamic=is_dynamic, is_bidirectional=is_bidirectional)

	#--------------------
	batch_size = 4  # Number of samples per gradient update.
	num_epochs = 150  # Number of times to iterate over training data.

	shuffle = True
	initial_epoch = 0

	train_model(session, rnnModel, batch_size, num_epochs, shuffle, initial_epoch)
	evaluate_model(session, rnnModel, batch_size)
	test_strs = ['abc', 'cba', 'dcb', 'abcd', 'dcba', 'cdacbd', 'bcdaabccdb']
	predict_model(session, rnnModel, batch_size, test_strs)

#%%------------------------------------------------------------------
# Attention model.
# REF [site] >> https://talbaumel.github.io/attention/

if True:
	is_bidirectional = True
	rnnModel = ReverseFunctionTensorFlowEncoderDecoderWithAttention(input_shape, output_shape, is_dynamic=is_dynamic, is_bidirectional=is_bidirectional)

	#--------------------
	batch_size = 4  # Number of samples per gradient update.
	num_epochs = 50  # Number of times to iterate over training data.

	shuffle = True
	initial_epoch = 0

	train_model(session, rnnModel, batch_size, num_epochs, shuffle, initial_epoch)
	evaluate_model(session, rnnModel, batch_size)
	test_strs = ['abc', 'cba', 'dcb', 'abcd', 'dcba', 'cdacbd', 'bcdaabccdb']
	predict_model(session, rnnModel, batch_size, test_strs)

#%%
enc_state_size = 128
dec_state_size = 128
dropout_rate = 0.5
batch_size = 4
num_epochs = 50

# Build a model.
# Input shape = (samples, time-steps, features) = (None, None, VOCAB_SIZE).
attn_enc_inputs = Input(shape=(None, VOCAB_SIZE))
#attn_enc_inputs = Input(shape=(MAX_TOKEN_LEN, VOCAB_SIZE))
attn_enc_lstm1 = LSTM(enc_state_size, dropout=dropout_rate, recurrent_dropout=dropout_rate, return_sequences=True, return_state=True)
attn_enc_outputs, attn_enc_state_h, attn_enc_state_c = attn_enc_lstm1(attn_enc_inputs)
# Discard 'attn_enc_outputs' and only keep the states.
attn_enc_states = [attn_enc_state_h, attn_enc_state_c]

# Input shape = (samples, time-steps, features).
attn_attn_inputs = Input(shape=(enc_state_size,))
attn_attn_lstm1 = LSTM(dec_state_size, dropout=dropout_rate, recurrent_dropout=dropout_rate)
attn_attn_outputs = attn_attn_lstm1(attn_attn_inputs)
attn_attn_dense = Dense(VOCAB_SIZE, activation='softmax')
attn_attn_outputs = attn_attn_dense(attn_attn_outputs)

# Input shape = (samples, time-steps, features).
attn_dec_inputs = Input(shape=(enc_state_size,))
attn_dec_lstm1 = LSTM(dec_state_size, dropout=dropout_rate, recurrent_dropout=dropout_rate)
attn_dec_outputs = attn_dec_lstm1(attn_dec_inputs)
attn_dec_dense = Dense(VOCAB_SIZE, activation='softmax')
attn_dec_outputs = attn_dec_dense(attn_dec_outputs)

attn_model = Model(inputs=attn_enc_inputs, outputs=attn_dec_outputs)

# Summarize the model.
#print(attn_model.summary())

# Train.
optimizer = optimizers.Adam(lr=1.0e-5, decay=1.0e-9, beta_1=0.9, beta_2=0.999, epsilon=1.0e-8)

attn_model.compile(optimizer=optimizer, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

history = attn_model.fit(train_data, train_labels_ahead_of_one_timestep,
		batch_size=batch_size, epochs=num_epochs,
		validation_data=(val_data, val_labels_ahead_of_one_timestep))
		#validation_split=0.2)
display_history(history)

# Save the model.
attn_model.save('seq2seq_reverse_function_attention_model.h5')

# Evaluate.
test_loss, test_accuracy = attn_model.evaluate(val_data, val_labels_ahead_of_one_timestep, batch_size=batch_size, verbose=1)
print('Test loss = {}, test accuracy = {}'.format(test_loss, test_accuracy))

# Predict.
