import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from enum import Enum

#%%------------------------------------------------------------------

class TrainingMode(Enum):
	START_TRAINING = 0  # Start training.
	RESUME_TRAINING = 1  # Resume training.
	USE_SAVED_MODEL = 2  # Use a saved model.

#%%------------------------------------------------------------------

class NeuralNetTrainer(object):
	def __init__(self, neuralNet, initial_epoch=0):
		self._neuralNet = neuralNet
		self._loss, self._accuracy = self._neuralNet.loss, self._neuralNet.accuracy

		self._global_step = tf.Variable(initial_epoch, name='global_step', trainable=False)
		self._train_step = self._get_train_step(self._loss, self._global_step)

	def train(self, session, train_data, train_labels, val_data, val_labels, batch_size, num_epochs, shuffle=True, saver=None, model_save_dir_path=None, train_summary_dir_path=None, val_summary_dir_path=None):
		# Merge all the summaries.
		merged_summary = tf.summary.merge_all()

		# Create writers to write all the summaries out to a directory.
		train_summary_writer = tf.summary.FileWriter(train_summary_dir_path, session.graph) if train_summary_dir_path is not None else None
		val_summary_writer = tf.summary.FileWriter(val_summary_dir_path) if val_summary_dir_path is not None else None

		num_train_examples = 0
		if train_data is not None and train_labels is not None:
			if train_data.shape[0] == train_labels.shape[0]:
				num_train_examples = train_data.shape[0]
			train_steps_per_epoch = ((num_train_examples - 1) // batch_size + 1) if num_train_examples > 0 else 0
		num_val_examples = 0
		if val_data is not None and val_labels is not None:
			if val_data.shape[0] == val_labels.shape[0]:
				num_val_examples = val_data.shape[0]
			val_steps_per_epoch = ((num_val_examples - 1) // batch_size + 1) if num_val_examples > 0 else 0

		history = {
			'acc': [],
			'val_acc': [],
			'loss': [],
			'val_loss': []
		}

		best_val_acc = 0.0
		for epoch in range(1, num_epochs + 1):
			print('Epoch {}/{}'.format(epoch, num_epochs))

			start_time = time.time()

			indices = np.arange(num_train_examples)
			if shuffle:
				np.random.shuffle(indices)

			# Train.
			print('>-', sep='', end='')
			processing_ratio = 0.05
			for step in range(train_steps_per_epoch):
				start = step * batch_size
				end = start + batch_size
				batch_indices = indices[start:end]
				if batch_indices.size > 0:  # If batch_indices is non-empty.
					data_batch, label_batch = train_data[batch_indices,], train_labels[batch_indices,]
					if data_batch.size > 0 and label_batch.size > 0:  # If data_batch and label_batch are non-empty.
						#summary = merged_summary.eval(session=session, feed_dict=self._neuralNet.get_feed_dict(data_batch, label_batch, is_training=True))
						#self._train_step.eval(session=session, feed_dict=self._neuralNet.get_feed_dict(data_batch, label_batch, is_training=True))
						summary, _ = session.run([merged_summary, self._train_step], feed_dict=self._neuralNet.get_feed_dict(data_batch, label_batch, is_training=True))
						if train_summary_writer is not None:
							train_summary_writer.add_summary(summary, epoch)
				if step / train_steps_per_epoch >= processing_ratio:
					print('-', sep='', end='')
					processing_ratio = round(step / train_steps_per_epoch, 2) + 0.05
			print('<')

			# Evaluate training.
			train_loss, train_acc = 0.0, 0.0
			#if False:
			if num_train_examples > 0:
				"""
				batch_indices = indices[0:batch_size]
				if batch_indices.size > 0:  # If batch_indices is non-empty.
					data_batch, label_batch = train_data[batch_indices,], train_labels[batch_indices,]
					if data_batch.size > 0 and label_batch.size > 0:  # If data_batch and label_batch are non-empty.
						#train_loss = self._loss.eval(session=session, feed_dict=self._neuralNet.get_feed_dict(data_batch, label_batch, is_training=False))
						#train_acc = self._accuracy.eval(session=session, feed_dict=self._neuralNet.get_feed_dict(data_batch, label_batch, is_training=False))
						train_loss, train_acc = session.run([self._loss, self._accuracy], feed_dict=self._neuralNet.get_feed_dict(data_batch, label_batch, is_training=False))
				"""
				for step in range(train_steps_per_epoch):
					start = step * batch_size
					end = start + batch_size
					batch_indices = indices[start:end]
					if batch_indices.size > 0:  # If batch_indices is non-empty.
						data_batch, label_batch = train_data[batch_indices,], train_labels[batch_indices,]
						if data_batch.size > 0 and label_batch.size > 0:  # If data_batch and label_batch are non-empty.
							#batch_loss = self._loss.eval(session=session, feed_dict=self._neuralNet.get_feed_dict(data_batch, label_batch, is_training=False))
							#batch_acc = self._accuracy.eval(session=session, feed_dict=self._neuralNet.get_feed_dict(data_batch, label_batch, is_training=False))
							batch_loss, batch_acc = session.run([self._loss, self._accuracy], feed_dict=self._neuralNet.get_feed_dict(data_batch, label_batch, is_training=False))
	
							# TODO [check] >> Is train_loss or train_acc correct?
							train_loss += batch_loss * batch_indices.size
							train_acc += batch_acc * batch_indices.size
				train_loss /= num_train_examples
				train_acc /= num_train_examples

				history['loss'].append(train_loss)
				history['acc'].append(train_acc)

			# Validate.
			val_loss, val_acc = 0.0, 0.0
			#if val_data is not None and val_labels is not None:
			if num_val_examples > 0:
				"""
				data_batch, label_batch = val_data, val_labels
				if data_batch.size > 0 and label_batch.size > 0:  # If data_batch and label_batch are non-empty.
					#summary = merged_summary.eval(session=session, feed_dict=self._neuralNet.get_feed_dict(data_batch, label_batch, is_training=False))
					#val_loss = self._loss.eval(session=session, feed_dict=self._neuralNet.get_feed_dict(data_batch, label_batch, is_training=False))
					#val_acc = self._accuracy.eval(session=session, feed_dict=self._neuralNet.get_feed_dict(data_batch, label_batch, is_training=False))
					summary, val_loss, val_acc = session.run([merged_summary, self._loss, self._accuracy], feed_dict=self._neuralNet.get_feed_dict(data_batch, label_batch, is_training=False))
					if val_summary_writer is not None:
						val_summary_writer.add_summary(summary, epoch)
				"""
				indices = np.arange(num_val_examples)
				if shuffle:
					np.random.shuffle(indices)

				for step in range(val_steps_per_epoch):
					start = step * batch_size
					end = start + batch_size
					batch_indices = indices[start:end]
					if batch_indices.size > 0:  # If batch_indices is non-empty.
						data_batch, label_batch = val_data[batch_indices,], val_labels[batch_indices,]
						if data_batch.size > 0 and label_batch.size > 0:  # If batch_indices and label_batch are non-empty.
							#summary = merged_summary.eval(session=session, feed_dict=self._neuralNet.get_feed_dict(data_batch, label_batch, is_training=False))
							#batch_loss = self._loss.eval(session=session, feed_dict=self._neuralNet.get_feed_dict({data_batch, label_batch, is_training=False))
							#batch_acc = self._accuracy.eval(session=session, feed_dict=self._neuralNet.get_feed_dict(data_batch, label_batch, is_training=False))
							summary, batch_loss, batch_acc = session.run([merged_summary, self._loss, self._accuracy], feed_dict=self._neuralNet.get_feed_dict(data_batch, label_batch, is_training=False))
							if val_summary_writer is not None:
								val_summary_writer.add_summary(summary, epoch)
	
							# TODO [check] >> Is val_loss or val_acc correct?
							val_loss += batch_loss * batch_indices.size
							val_acc += batch_acc * batch_indices.size
				val_loss /= num_val_examples
				val_acc /= num_val_examples

				history['val_loss'].append(val_loss)
				history['val_acc'].append(val_acc)

				# Save a model.
				if saver is not None and model_save_dir_path is not None and val_acc >= best_val_acc:
					saved_model_path = saver.save(session, model_save_dir_path + '/model.ckpt', global_step=self._global_step)
					best_val_acc = val_acc

					print('[SWL] Info: Accurary is improved and the model is saved at {}.'.format(saved_model_path))

			print('\tElapsed time = {}'.format(time.time() - start_time))
			print('\tLoss = {}, accuracy = {}, validation loss = {}, validation accurary = {}'.format(train_loss, train_acc, val_loss, val_acc))

		# Close writers.
		if train_summary_writer is not None:
			train_summary_writer.close()
		if val_summary_writer is not None:
			val_summary_writer.close()

		return history

	def train_seq2seq(self, session, train_encoder_inputs, train_decoder_inputs, train_decoder_outputs, val_encoder_inputs, val_decoder_inputs, val_decoder_outputs, batch_size, num_epochs, shuffle=True, saver=None, model_save_dir_path=None, train_summary_dir_path=None, val_summary_dir_path=None):
		# Merge all the summaries.
		merged_summary = tf.summary.merge_all()

		# Create writers to write all the summaries out to a directory.
		train_summary_writer = tf.summary.FileWriter(train_summary_dir_path, session.graph) if train_summary_dir_path is not None else None
		val_summary_writer = tf.summary.FileWriter(val_summary_dir_path) if val_summary_dir_path is not None else None

		num_train_examples = 0
		if train_encoder_inputs is not None and train_decoder_inputs is not None and train_decoder_outputs is not None:
			if train_encoder_inputs.shape[0] == train_decoder_inputs.shape[0] and train_encoder_inputs.shape[0] == train_decoder_outputs.shape[0]:
				num_train_examples = train_encoder_inputs.shape[0]
			train_steps_per_epoch = ((num_train_examples - 1) // batch_size + 1) if num_train_examples > 0 else 0
		num_val_examples = 0
		if val_encoder_inputs is not None and val_decoder_inputs is not None and val_decoder_outputs is not None:
			if val_encoder_inputs.shape[0] == val_decoder_inputs.shape[0] and val_encoder_inputs.shape[0] == val_decoder_outputs.shape[0]:
				num_val_examples = val_encoder_inputs.shape[0]
			val_steps_per_epoch = ((num_val_examples - 1) // batch_size + 1) if num_val_examples > 0 else 0

		history = {
			'acc': [],
			'val_acc': [],
			'loss': [],
			'val_loss': []
		}

		best_val_acc = 0.0
		for epoch in range(1, num_epochs + 1):
			print('Epoch {}/{}'.format(epoch, num_epochs))

			start_time = time.time()

			indices = np.arange(num_train_examples)
			if shuffle:
				np.random.shuffle(indices)

			# Train.
			print('>-', sep='', end='')
			processing_ratio = 0.05
			for step in range(train_steps_per_epoch):
				start = step * batch_size
				end = start + batch_size
				batch_indices = indices[start:end]
				if batch_indices.size > 0:  # If batch_indices is non-empty.
					enc_input_batch, dec_input_batch, dec_output_batch = train_encoder_inputs[batch_indices,], train_decoder_inputs[batch_indices,], train_decoder_outputs[batch_indices,]
					if enc_input_batch.size > 0 and dec_input_batch.size > 0 and dec_output_batch.size > 0:  # If enc_input_batch, dec_input_batch, and dec_output_batch are non-empty.
						#summary = merged_summary.eval(session=session, feed_dict=self._neuralNet.get_feed_dict(enc_input_batch, dec_input_batch, dec_output_batch, is_training=True))
						#self._train_step.eval(session=session, feed_dict=self._neuralNet.get_feed_dict(enc_input_batch, dec_input_batch, dec_output_batch, is_training=True))
						summary, _ = session.run([merged_summary, self._train_step], feed_dict=self._neuralNet.get_feed_dict(enc_input_batch, dec_input_batch, dec_output_batch, is_training=True))
						if train_summary_writer is not None:
							train_summary_writer.add_summary(summary, epoch)
				if step / train_steps_per_epoch >= processing_ratio:
					print('-', sep='', end='')
					processing_ratio = round(step / train_steps_per_epoch, 2) + 0.05
			print('<')

			# Evaluate training.
			train_loss, train_acc = 0.0, 0.0
			#if False:
			if num_train_examples > 0:
				"""
				batch_indices = indices[0:batch_size]
				if batch_indices.size > 0:  # If batch_indices is non-empty.
					enc_input_batch, dec_input_batch, dec_output_batch = train_encoder_inputs[batch_indices,], train_decoder_inputs[batch_indices,], train_decoder_outputs[batch_indices,]
					if enc_input_batch.size > 0 and dec_input_batch.size > 0 and dec_output_batch.size > 0:  # If enc_input_batch, dec_input_batch, and dec_output_batch are non-empty.
						#train_loss = self._loss.eval(session=session, feed_dict=self._neuralNet.get_feed_dict(enc_input_batch, dec_input_batch, dec_output_batch, is_training=False))
						#train_acc = self._accuracy.eval(session=session, feed_dict=self._neuralNet.get_feed_dict(enc_input_batch, dec_input_batch, dec_output_batch, is_training=False))
						train_loss, train_acc = session.run([self._loss, self._accuracy], feed_dict=self._neuralNet.get_feed_dict(enc_input_batch, dec_input_batch, dec_output_batch, is_training=False))
				"""
				for step in range(train_steps_per_epoch):
					start = step * batch_size
					end = start + batch_size
					batch_indices = indices[start:end]
					if batch_indices.size > 0:  # If batch_indices is non-empty.
						enc_input_batch, dec_input_batch, dec_output_batch = train_encoder_inputs[batch_indices,], train_decoder_inputs[batch_indices,], train_decoder_outputs[batch_indices,]
						if enc_input_batch.size > 0 and dec_input_batch.size > 0 and dec_output_batch.size > 0:  # If enc_input_batch, dec_input_batch, and dec_output_batch are non-empty.
							#batch_loss = self._loss.eval(session=session, feed_dict=self._neuralNet.get_feed_dict(enc_input_batch, dec_input_batch, dec_output_batch, is_training=False))
							#batch_acc = self._accuracy.eval(session=session, feed_dict=self._neuralNet.get_feed_dict(enc_input_batch, dec_input_batch, dec_output_batch, is_training=False))
							batch_loss, batch_acc = session.run([self._loss, self._accuracy], feed_dict=self._neuralNet.get_feed_dict(enc_input_batch, dec_input_batch, dec_output_batch, is_training=False))
	
							# TODO [check] >> Is train_loss or train_acc correct?
							train_loss += batch_loss * batch_indices.size
							train_acc += batch_acc * batch_indices.size
				train_loss /= num_train_examples
				train_acc /= num_train_examples

				history['loss'].append(train_loss)
				history['acc'].append(train_acc)

			# Validate.
			val_loss, val_acc = 0.0, 0.0
			#if val_data is not None and val_labels is not None:
			if num_val_examples > 0:
				"""
				data_batch, label_batch = val_data, val_labels
				enc_input_batch, dec_input_batch, dec_output_batch = val_encoder_inputs[batch_indices,], val_decoder_inputs[batch_indices,], val_decoder_outputs[batch_indices,]
				if enc_input_batch.size > 0 and dec_input_batch.size > 0 and dec_output_batch.size > 0:  # If enc_input_batch, dec_input_batch, and dec_output_batch are non-empty.
					#summary = merged_summary.eval(session=session, feed_dict=self._neuralNet.get_feed_dict(enc_input_batch, dec_input_batch, dec_output_batch, is_training=False))
					#val_loss = self._loss.eval(session=session, feed_dict=self._neuralNet.get_feed_dict(enc_input_batch, dec_input_batch, dec_output_batch, is_training=False))
					#val_acc = self._accuracy.eval(session=session, feed_dict=self._neuralNet.get_feed_dict(enc_input_batch, dec_input_batch, dec_output_batch, is_training=False))
					summary, val_loss, val_acc = session.run([merged_summary, self._loss, self._accuracy], feed_dict=self._neuralNet.get_feed_dict(enc_input_batch, dec_input_batch, dec_output_batch, is_training=False))
					if val_summary_writer is not None:
						val_summary_writer.add_summary(summary, epoch)
				"""
				indices = np.arange(num_val_examples)
				if shuffle:
					np.random.shuffle(indices)

				for step in range(val_steps_per_epoch):
					start = step * batch_size
					end = start + batch_size
					batch_indices = indices[start:end]
					if batch_indices.size > 0:  # If batch_indices is non-empty.
						enc_input_batch, dec_input_batch, dec_output_batch = val_encoder_inputs[batch_indices,], val_decoder_inputs[batch_indices,], val_decoder_outputs[batch_indices,]
						if enc_input_batch.size > 0 and dec_input_batch.size > 0 and dec_output_batch.size > 0:  # If enc_input_batch, dec_input_batch, and dec_output_batch are non-empty.
							#summary = merged_summary.eval(session=session, feed_dict=self._neuralNet.get_feed_dict(enc_input_batch, dec_input_batch, dec_output_batch, is_training=False))
							#batch_loss = self._loss.eval(session=session, feed_dict=self._neuralNet.get_feed_dict(enc_input_batch, dec_input_batch, dec_output_batch, is_training=False))
							#batch_acc = self._accuracy.eval(session=session, feed_dict=self._neuralNet.get_feed_dict(enc_input_batch, dec_input_batch, dec_output_batch, is_training=False))
							summary, batch_loss, batch_acc = session.run([merged_summary, self._loss, self._accuracy], feed_dict=self._neuralNet.get_feed_dict(enc_input_batch, dec_input_batch, dec_output_batch, is_training=False))
							if val_summary_writer is not None:
								val_summary_writer.add_summary(summary, epoch)
	
							# TODO [check] >> Is val_loss or val_acc correct?
							val_loss += batch_loss * batch_indices.size
							val_acc += batch_acc * batch_indices.size
				val_loss /= num_val_examples
				val_acc /= num_val_examples

				history['val_loss'].append(val_loss)
				history['val_acc'].append(val_acc)

				# Save a model.
				if saver is not None and model_save_dir_path is not None and val_acc >= best_val_acc:
					saved_model_path = saver.save(session, model_save_dir_path + '/model.ckpt', global_step=self._global_step)
					best_val_acc = val_acc

					print('[SWL] Info: Accurary is improved and the model is saved at {}.'.format(saved_model_path))

			print('\tElapsed time = {}'.format(time.time() - start_time))
			print('\tLoss = {}, accuracy = {}, validation loss = {}, validation accurary = {}'.format(train_loss, train_acc, val_loss, val_acc))

		# Close writers.
		if train_summary_writer is not None:
			train_summary_writer.close()
		if val_summary_writer is not None:
			val_summary_writer.close()

		return history

	def display_history(self, history):
		# List all data in history.
		#print(history.keys())

		# Summarize history for accuracy.
		fig = plt.figure()
		plt.plot(history['acc'])
		plt.plot(history['val_acc'])
		plt.title('model accuracy')
		plt.xlabel('epochs')
		plt.ylabel('accuracy')
		plt.legend(['train', 'test'], loc='upper left')
		plt.show()
		plt.close(fig)

		# Summarize history for loss.
		plt.figure()
		plt.plot(history['loss'])
		plt.plot(history['val_loss'])
		plt.title('model loss')
		plt.xlabel('epochs')
		plt.ylabel('loss')
		plt.legend(['train', 'test'], loc='upper left')
		plt.show()
		plt.close(fig)

	def _get_train_step(self, loss, global_step=None):
		raise NotImplementedError
