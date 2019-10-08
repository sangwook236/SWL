#!/usr/bin/env python

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
#sys.path.append('../../../src')
sys.path.append(os.path.join(swl_python_home_dir_path, 'src'))

#os.chdir(os.path.join(swl_python_home_dir_path, 'test/machine_learning/tensorflow'))

#--------------------
import time, datetime
import numpy as np
import tensorflow as tf
from swl.machine_learning.tensorflow.simple_neural_net_trainer import SimpleGradientClippingNeuralNetTrainer
from swl.machine_learning.tensorflow.neural_net_evaluator import NeuralNetEvaluator
from swl.machine_learning.tensorflow.neural_net_inferrer import NeuralNetInferrer
import swl.util.util as swl_util
import swl.machine_learning.util as swl_ml_util
import swl.machine_learning.tensorflow.util as swl_tf_util
from rda_plant_util import RdaPlantDataset
from simple_seq2seq_encdec import SimpleSeq2SeqEncoderDecoder
from simple_seq2seq_encdec_tf_attention import SimpleSeq2SeqEncoderDecoderWithTfAttention

#--------------------------------------------------------------------

def create_seq2seq_encoder_decoder(encoder_input_shape, decoder_input_shape, decoder_output_shape, dataset, is_time_major):
	# Sequence-to-sequence encoder-decoder model w/o attention.
	return SimpleSeq2SeqEncoderDecoder(encoder_input_shape, decoder_input_shape, decoder_output_shape, dataset.start_token, dataset.end_token, is_time_major=is_time_major)

#--------------------------------------------------------------------

def pad_image(img, target_height, target_width):
	if 2 == img.ndim:
		height, width = img.shape
	elif 3 == img.ndim:
		height, width, _ = img.shape
	else:
		assert 2 == img.ndim or 3 == img.ndim, 'The dimension of an image is not proper.'

	left_margin = (target_width - width) // 2
	right_margin = target_width - width - left_margin
	#top_margin = (target_height - height) // 2
	#bottom_margin = target_height - height - top_margin
	top_margin = target_height - height
	bottom_margin = target_height - height - top_margin
	if 2 == img.ndim:
		return np.pad(img, ((top_margin, bottom_margin), (left_margin, right_margin)), 'edge')
		#return np.pad(img, ((top_margin, bottom_margin), (left_margin, right_margin)), 'constant', constant_values=(0, 0))
	else:
		return np.pad(img, ((top_margin, bottom_margin), (left_margin, right_margin), (0, 0)), 'edge')
		#return np.pad(img, ((top_margin, bottom_margin), (left_margin, right_margin), (0, 0)), 'constant', constant_values=(0, 0))

#--------------------------------------------------------------------

def main():
	#np.random.seed(7)

	#--------------------
	# Parameters.

	does_need_training = True
	does_resume_training = False

	output_dir_prefix = 'reverse_function_seq2seq'
	output_dir_suffix = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
	#output_dir_suffix = '20180222T144236'

	max_gradient_norm = 5
	initial_epoch = 0

	batch_size = 4  # Number of samples per gradient update.
	num_epochs = 70  # Number of times to iterate over training data.
	shuffle = True

	augmenter = None
	is_output_augmented = False

	sess_config = tf.ConfigProto()
	#sess_config.allow_soft_placement = True
	sess_config.log_device_placement = True
	sess_config.gpu_options.allow_growth = True
	#sess_config.gpu_options.per_process_gpu_memory_fraction = 0.4  # Only allocate 40% of the total memory of each GPU.

	#--------------------
	# Prepare directories.

	output_dir_path = os.path.join('.', '{}_{}'.format(output_dir_prefix, output_dir_suffix))
	checkpoint_dir_path = os.path.join(output_dir_path, 'tf_checkpoint')
	inference_dir_path = os.path.join(output_dir_path, 'inference')
	train_summary_dir_path = os.path.join(output_dir_path, 'train_log')
	val_summary_dir_path = os.path.join(output_dir_path, 'val_log')

	swl_util.make_dir(checkpoint_dir_path)
	swl_util.make_dir(inference_dir_path)
	swl_util.make_dir(train_summary_dir_path)
	swl_util.make_dir(val_summary_dir_path)

	#--------------------
	# Prepare data.

	if 'posix' == os.name:
		data_home_dir_path = '/home/sangwook/my_dataset'
	else:
		data_home_dir_path = 'D:/dataset'
	data_dir_path = data_home_dir_path + '/phenotyping/RDA/all_plants_mask'
	plant_mask_list_file_name = '/plant_mask_list.json'

	plant_mask_list, max_size = RdaPlantDataset.load_masks_from_json(data_dir_path, plant_mask_list_file_name)
	#plant: plant_mask_list[*][0]
	#masks: plant_mask_list[*][1][0] ~ plant_mask_list[*][1][n]
	max_len = max(max_size)
	for pm_pair in plant_mask_list:
		pm_pair[0] = pad_image(pm_pair[0], max_len, max_len)
		for (idx, mask) in enumerate(pm_pair[1]):
			#mask = pad_image(mask, max_len, max_len)  # Not correctly working.
			pm_pair[1][idx] = pad_image(mask, max_len, max_len)

	#--------------------
	# Create models, sessions, and graphs.

	# Create graphs.
	if does_need_training:
		train_graph = tf.Graph()
		eval_graph = tf.Graph()
	infer_graph = tf.Graph()

	if does_need_training:
		with train_graph.as_default():
			# Create a model.
			modelForTraining = create_seq2seq_encoder_decoder(encoder_input_shape, decoder_input_shape, decoder_output_shape, dataset, is_time_major)
			modelForTraining.create_training_model()

			# Create a trainer.
			#nnTrainer = SimpleNeuralNetTrainer(modelForTraining, initial_epoch, augmenter, is_output_augmented)
			nnTrainer = SimpleGradientClippingNeuralNetTrainer(modelForTraining, max_gradient_norm, initial_epoch, augmenter, is_output_augmented)

			# Create a saver.
			#	Save a model every 2 hours and maximum 5 latest models are saved.
			train_saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

			initializer = tf.global_variables_initializer()

		with eval_graph.as_default():
			# Create a model.
			modelForEvaluation = create_seq2seq_encoder_decoder(encoder_input_shape, decoder_input_shape, decoder_output_shape, dataset, is_time_major)
			modelForEvaluation.create_evaluation_model()

			# Create an evaluator.
			nnEvaluator = NeuralNetEvaluator(modelForEvaluation)

			# Create a saver.
			eval_saver = tf.train.Saver()

	with infer_graph.as_default():
		# Create a model.
		modelForInference = create_seq2seq_encoder_decoder(encoder_input_shape, decoder_input_shape, decoder_output_shape, dataset, is_time_major)
		modelForInference.create_inference_model()

		# Create an inferrer.
		nnInferrer = NeuralNetInferrer(modelForInference)

		# Create a saver.
		infer_saver = tf.train.Saver()

	# Create sessions.
	if does_need_training:
		train_session = tf.Session(graph=train_graph, config=sess_config)
		eval_session = tf.Session(graph=eval_graph, config=sess_config)
	infer_session = tf.Session(graph=infer_graph, config=sess_config)

	# Initialize.
	if does_need_training:
		train_session.run(initializer)

	#--------------------------------------------------------------------
	# Train and evaluate.

	if does_need_training:
		start_time = time.time()
		with train_session.as_default() as sess:
			with sess.graph.as_default():
				swl_tf_util.train_neural_net_with_decoder_input(sess, nnTrainer, train_encoder_input_seqs, train_decoder_input_seqs, train_decoder_output_seqs, val_encoder_input_seqs, val_decoder_input_seqs, val_decoder_output_seqs, batch_size, num_epochs, shuffle, does_resume_training, train_saver, output_dir_path, checkpoint_dir_path, train_summary_dir_path, val_summary_dir_path)
		print('\tTotal training time = {}'.format(time.time() - start_time))

		start_time = time.time()
		with eval_session.as_default() as sess:
			with sess.graph.as_default():
				swl_tf_util.evaluate_neural_net_with_decoder_input(sess, nnEvaluator, val_encoder_input_seqs, val_decoder_input_seqs, val_decoder_output_seqs, batch_size, eval_saver, checkpoint_dir_path, is_time_major)
		print('\tTotal evaluation time = {}'.format(time.time() - start_time))

	#--------------------------------------------------------------------
	# Infer.
	
	test_strs = ['abc', 'cba', 'dcb', 'abcd', 'dcba', 'cdacbd', 'bcdaabccdb']
	# String data -> numeric data.
	test_data = dataset.to_numeric(test_strs)

	start_time = time.time()
	with infer_session.as_default() as sess:
		with sess.graph.as_default():
			inferences = swl_tf_util.infer_by_neural_net(sess, nnInferrer, test_data, batch_size, infer_saver, checkpoint_dir_path)
	print('\tTotal inference time = {}'.format(time.time() - start_time))

	if inferences is not None:
		# Numeric data -> string data.
		inferred_strs = dataset.to_string(inferences, has_start_token=False)
		print('\tTest strings = {}, inferred strings = {}'.format(test_strs, inferred_strs))
	else:
		print('[SWL] Warning: Invalid inference results.')

	#--------------------
	# Close sessions.

	if does_need_training:
		train_session.close()
		del train_session
		eval_session.close()
		del eval_session
	infer_session.close()
	del infer_session

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
