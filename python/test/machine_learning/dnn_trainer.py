#--------------------
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time

#%%------------------------------------------------------------------

class DnnTrainer(object):
	def __init__(self, dnnModel, input_shape, output_shape):
		self.input_tensor_ph_ = tf.placeholder(tf.float32, shape=(None,) + input_shape)
		self.output_tensor_ph_ = tf.placeholder(tf.float32, shape=(None,) + output_shape)
		self.is_training_ph_ = tf.placeholder(tf.bool)

		self.global_step_ = tf.Variable(0, name='global_step', trainable=False)

		self.model_output_ = dnnModel(self.input_tensor_ph_, self.is_training_ph_)
		self.loss_, self.accuracy_ = self._prepare_evaluation(self.model_output_, self.output_tensor_ph_)
		self.train_step_ = self._prepare_training(self.loss_, self.global_step_)

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
					#summary = merged_summary.eval(session=session, feed_dict={self.input_tensor_ph_: data_batch, self.output_tensor_ph_: label_batch, self.is_training_ph_: True})
					#self.train_step_.eval(session=session, feed_dict={self.input_tensor_ph_: data_batch, self.output_tensor_ph_: label_batch, self.is_training_ph_: True})
					summary, _ = session.run([merged_summary, self.train_step_], feed_dict={self.input_tensor_ph_: data_batch, self.output_tensor_ph_: label_batch, self.is_training_ph_: True})
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
					#train_loss = self.loss_.eval(session=session, feed_dict={self.input_tensor_ph_: data_batch, self.output_tensor_ph_: label_batch, self.is_training_ph_: False})
					#train_acc = self.accuracy_.eval(session=session, feed_dict={self.input_tensor_ph_: data_batch, self.output_tensor_ph_: label_batch, self.is_training_ph_: False})
					train_loss, train_acc = session.run([self.loss_, self.accuracy_], feed_dict={self.input_tensor_ph_: data_batch, self.output_tensor_ph_: label_batch, self.is_training_ph_: False})
				"""
				for step in range(train_steps_per_epoch):
					start = step * batch_size
					end = start + batch_size
					batch_indices = indices[start:end]
					data_batch, label_batch = train_data[batch_indices,], train_labels[batch_indices,]
					if data_batch.size > 0 and label_batch.size > 0:  # If data_batch and label_batch are non-empty.
						#batch_loss = self.loss_.eval(session=session, feed_dict={self.input_tensor_ph_: data_batch, self.output_tensor_ph_: label_batch, self.is_training_ph_: False})
						#batch_acc = self.accuracy_.eval(session=session, feed_dict={self.input_tensor_ph_: data_batch, self.output_tensor_ph_: label_batch, self.is_training_ph_: False})
						batch_loss, batch_acc = session.run([self.loss_, self.accuracy_], feed_dict={self.input_tensor_ph_: data_batch, self.output_tensor_ph_: label_batch, self.is_training_ph_: False})

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
					#summary = merged_summary.eval(session=session, feed_dict={self.input_tensor_ph_: data_batch, self.output_tensor_ph_: label_batch, self.is_training_ph_: False})
					#val_loss = self.loss_.eval(session=session, feed_dict={self.input_tensor_ph_: data_batch, self.output_tensor_ph_: label_batch, self.is_training_ph_: False})
					#val_acc = self.accuracy_.eval(session=session, feed_dict={self.input_tensor_ph_: data_batch, self.output_tensor_ph_: label_batch, self.is_training_ph_: False})
					summary, val_loss, val_acc = session.run([merged_summary, self.loss_, self.accuracy_], feed_dict={self.input_tensor_ph_: data_batch, self.output_tensor_ph_: label_batch, self.is_training_ph_: False})
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
						#summary = merged_summary.eval(session=session, feed_dict={self.input_tensor_ph_: data_batch, self.output_tensor_ph_: label_batch, self.is_training_ph_: False})
						#batch_loss = self.loss_.eval(session=session, feed_dict={self.input_tensor_ph_: data_batch, self.output_tensor_ph_: label_batch, self.is_training_ph_: False})
						#batch_acc = self.accuracy_.eval(session=session, feed_dict={self.input_tensor_ph_: data_batch, self.output_tensor_ph_: label_batch, self.is_training_ph_: False})
						summary, batch_loss, batch_acc = session.run([merged_summary, self.loss_, self.accuracy_], feed_dict={self.input_tensor_ph_: data_batch, self.output_tensor_ph_: label_batch, self.is_training_ph_: False})
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
					model_saved_path = saver.save(session, model_save_dir_path + '/model.ckpt', global_step=self.global_step_)
					best_val_acc = val_acc

					print('[SWL] Info: Improved accurary and saved the model at {}.'.format(model_saved_path))

			print('Loss = {}, accuracy = {}, validation loss = {}, validation accurary = {}, elapsed time = {}'.format(train_loss, train_acc, val_loss, val_acc, time.time() - start_time))

		# Close writers.
		if train_summary_writer is not None:
			train_summary_writer.close()
		if val_summary_writer is not None:
			val_summary_writer.close()

		return history

	def evaluate(self, session, test_images, test_labels, batch_size):
		num_test_examples = test_images.shape[0]

		"""
		#test_loss = self.loss_.eval(session=session, feed_dict={self.input_tensor_ph_: test_data, self.output_tensor_ph_: test_labels, self.is_training_ph_: False})
		#test_acc = self.accuracy_.eval(session=session, feed_dict={self.input_tensor_ph_: test_data, self.output_tensor_ph_: test_labels, self.is_training_ph_: False})
		test_loss, test_acc = session.run([self.loss_, self.accuracy_], feed_dict={self.input_tensor_ph_: test_data, self.output_tensor_ph_: test_labels, self.is_training_ph_: False})
		"""
		test_steps_per_epoch = (num_test_examples - 1) // batch_size + 1

		indices = np.arange(num_test_examples)
		#if True == shuffle:
		#	np.random.shuffle(indices)

		test_loss, test_acc = 0, 0
		for step in range(test_steps_per_epoch):
			start = step * batch_size
			end = start + batch_size
			batch_indices = indices[start:end]
			data_batch, label_batch = test_images[batch_indices,], test_labels[batch_indices,]
			if data_batch.size > 0 and label_batch.size > 0:  # If data_batch and label_batch are non-empty.
				#batch_loss = self.loss_.eval(session=session, feed_dict={self.input_tensor_ph_: data_batch, self.output_tensor_ph_: label_batch, self.is_training_ph_: False})
				#batch_acc = self.accuracy_.eval(session=session, feed_dict={self.input_tensor_ph_: data_batch, self.output_tensor_ph_: label_batch, self.is_training_ph_: False})
				batch_loss, batch_acc = session.run([self.loss_, self.accuracy_], feed_dict={self.input_tensor_ph_: data_batch, self.output_tensor_ph_: label_batch, self.is_training_ph_: False})

				# TODO [check] >> Is test_loss or test_acc correct?
				test_loss += batch_loss * batch_indices.size
				test_acc += batch_acc * batch_indices.size
		test_loss /= num_test_examples
		test_acc /= num_test_examples

		return test_loss, test_acc

	def predict(self, session, test_images, batch_size):
		num_pred_examples = test_images.shape[0]

		"""
		predictions = self.model_output_.eval(session=session, feed_dict={self.input_tensor_ph_: test_images, self.is_training_ph_: False})
		"""
		pred_steps_per_epoch = (num_pred_examples - 1) // batch_size + 1

		indices = np.arange(num_pred_examples)

		predictions = np.array([])
		for step in range(pred_steps_per_epoch):
			start = step * batch_size
			end = start + batch_size
			batch_indices = indices[start:end]
			data_batch = test_images[batch_indices,]
			if data_batch.size > 0:  # If data_batch is non-empty.
				batch_prediction = self.model_output_.eval(session=session, feed_dict={self.input_tensor_ph_: data_batch, self.is_training_ph_: False})
	
				if predictions.size > 0:  # If predictions is non-empty.
					predictions = np.concatenate((predictions, batch_prediction), axis=0)
				else:
					predictions = batch_prediction

		return np.argmax(predictions, 1)

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

	def _loss(self, y, t):
		#cross_entropy = tf.reduce_mean(-tf.reduce_sum(t * tf.log(y), reduction_indices=[1]))
		#cross_entropy = tf.reduce_mean(-tf.reduce_sum(t * tf.log(tf.clip_by_value(y, 1e-10, 1.0)), reduction_indices=[1]))
		#cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=t, logits=y))
		cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=t, logits=y))
		return cross_entropy

	def _accuracy(self, y, t):
		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		return accuracy

	def _train(self, loss, learning_rate, global_step=None):
		#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999)
		train_step = optimizer.minimize(loss, global_step=global_step)
		return train_step

	def _prepare_training(self, loss, global_step):
		with tf.name_scope('learning_rate'):
			learning_rate = tf.train.exponential_decay(0.001, global_step=global_step, decay_steps=10, decay_rate=0.995, staircase=True)
			tf.summary.scalar('learning_rate', learning_rate)
		with tf.name_scope('train'):
			train_step = self._train(loss, learning_rate=0.001, global_step=global_step)
			#train_step = self._train(loss, learning_rate=learning_rate, global_step=global_step)
		return train_step

	def _prepare_evaluation(self, y_pred, y_true):
		with tf.name_scope('loss'):
			loss = self._loss(y_pred, y_true)
			tf.summary.scalar('loss', loss)
		with tf.name_scope('accuracy'):
			accuracy = self._accuracy(y_pred, y_true)
			tf.summary.scalar('accuracy', accuracy)
		return loss, accuracy
