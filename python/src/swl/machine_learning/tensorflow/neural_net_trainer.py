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
		self._train_step = self._train(self._loss, self._global_step)

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
			if True == shuffle:
				np.random.shuffle(indices)

			# Train.
			for step in range(train_steps_per_epoch):
				start = step * batch_size
				end = start + batch_size
				batch_indices = indices[start:end]
				if batch_indices.size > 0:  # If batch_indices is non-empty.
					data_batch, label_batch = train_data[batch_indices,], train_labels[batch_indices,]
					#summary = merged_summary.eval(session=session, feed_dict=self._neuralNet.get_feed_dict(data_batch, label_batch, is_training=True))
					#self._train_step.eval(session=session, feed_dict=self._neuralNet.get_feed_dict(data_batch, label_batch, is_training=True))
					summary, _ = session.run([merged_summary, self._train_step], feed_dict=self._neuralNet.get_feed_dict(data_batch, label_batch, is_training=True))
					if train_summary_writer is not None:
						train_summary_writer.add_summary(summary, epoch)

			# Evaluate training.
			train_loss, train_acc = 0, 0
			#if False:
			if num_train_examples > 0:
				"""
				batch_indices = indices[0:batch_size]
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
			val_loss, val_acc = 0, 0
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
				if True == shuffle:
					np.random.shuffle(indices)

				for step in range(val_steps_per_epoch):
					start = step * batch_size
					end = start + batch_size
					batch_indices = indices[start:end]
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
					model_saved_path = saver.save(session, model_save_dir_path + '/model.ckpt', global_step=self._global_step)
					best_val_acc = val_acc

					print('[SWL] Info: Accurary is improved and the model is saved at {}.'.format(model_saved_path))

			print('\tLoss = {}, accuracy = {}, validation loss = {}, validation accurary = {}, elapsed time = {}'.format(train_loss, train_acc, val_loss, val_acc, time.time() - start_time))

		# Close writers.
		if train_summary_writer is not None:
			train_summary_writer.close()
		if val_summary_writer is not None:
			val_summary_writer.close()

		return history

	def display_history(self, history):
		# List all data in history.
		print(history.keys())

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

	def _train(self, loss, global_step=None):
		with tf.name_scope('learning_rate'):
			#learning_rate = tf.train.exponential_decay(0.001, global_step=global_step, decay_steps=10, decay_rate=0.995, staircase=True)
			learning_rate = 0.001
			tf.summary.scalar('learning_rate', learning_rate)

		with tf.name_scope('train'):
			#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
			optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999)
			train_step = optimizer.minimize(loss, global_step=global_step)
			return train_step
