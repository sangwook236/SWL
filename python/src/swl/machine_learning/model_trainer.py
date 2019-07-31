import os, abc, math, time
import numpy as np
import tensorflow as tf
import swl.machine_learning.util as swl_ml_util

#%%------------------------------------------------------------------
# ModelTrainer.

#class ModelTrainer(abc.ABC):
class ModelTrainer(object):
	def __init__(self, model, optimizer, dataGenerator, output_dir_path, model_save_dir_path, train_summary_dir_path, val_summary_dir_path, global_step=None, var_list=None):
		super().__init__()

		self._model = model
		self._optimizer = optimizer
		self._global_step = global_step

		self._dataGenerator = dataGenerator
		self._output_dir_path = output_dir_path
		self._model_save_dir_path = model_save_dir_path
		self._train_summary_dir_path = train_summary_dir_path
		self._val_summary_dir_path = val_summary_dir_path

		# Creates a saver.
		#	Saves a model every 2 hours and maximum 5 latest models are saved.
		self._saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

		self._loss, self._accuracy = self._model.loss, self._model.accuracy
		if self._loss is None:
			raise ValueError('Invalid loss')

		self._train_operation = self._get_train_operation(self._loss, self._global_step, var_list=var_list)

		# Merge all the summaries.
		self._merged_summary = tf.summary.merge_all()

	@property
	def global_step(self):
		return self._global_step

	def train(self, session, batch_size, num_epochs, shuffle=True, is_training_resumed=False):
		if is_training_resumed:
			if self._saver is not None and self._model_save_dir_path is not None:
				print('[SWL] Info: Resume training...')

				# Restore a model.
				# REF [site] >> https://www.tensorflow.org/programmers_guide/saved_model
				# REF [site] >> http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
				ckpt = tf.train.get_checkpoint_state(self._model_save_dir_path)
				ckpt_filepath = ckpt.model_checkpoint_path if ckpt else None
				#ckpt_filepath = tf.train.latest_checkpoint(self._model_save_dir_path)
				if ckpt_filepath:
					#initial_epoch = int(ckpt_filepath.split('-')[1])
					self._saver.restore(session, ckpt_filepath)
				else:
					print('[SWL] Error: Failed to restore a model from {}.'.format(self._model_save_dir_path))
					return
				print('[SWL] Info: Restored a model.')
			else:
				print('[SWL] Error: Invalid model save path, {}.'.format(self._model_save_dir_path))
				return
		else:
			print('[SWL] Info: Start training...')

		start_time = time.time()
		history = self._train(session, batch_size, num_epochs, shuffle)
		print('\tTraining time = {} secs.'.format(time.time() - start_time))

		#--------------------
		# Display results.
		#swl_ml_util.display_train_history(history)
		if self._output_dir_path is not None:
			swl_ml_util.save_train_history(history, self._output_dir_path)
		print('[SWL] Info: End training.')

		"""
		# Save a graph.
		tf.train.write_graph(session.graph_def, self._output_dir_path, 'graph.pb', as_text=False)
		#tf.train.write_graph(session.graph_def, self._output_dir_path, 'graph.pbtxt', as_text=True)

		# Save a serving model.
		builder = tf.saved_model.builder.SavedModelBuilder(self._output_dir_path + '/serving_model')
		builder.add_meta_graph_and_variables(session, [tf.saved_model.tag_constants.SERVING], saver=saver)
		builder.save(as_text=False)
		"""

	def _train(self, session, batch_size, num_epochs, shuffle=True):
		# Create writers to write all the summaries out to a directory.
		train_summary_writer = tf.summary.FileWriter(self._train_summary_dir_path, session.graph) if self._train_summary_dir_path is not None else None
		val_summary_writer = tf.summary.FileWriter(self._val_summary_dir_path) if self._val_summary_dir_path is not None else None

		history = {
			'acc': [],
			'loss': [],
			'val_acc': [],
			'val_loss': []
		}

		best_val_acc = -math.inf
		for epoch in range(1, num_epochs + 1):
			print('Epoch {}/{}'.format(epoch, num_epochs))

			start_time = time.time()

			# Train.
			for batch_data, _ in self._dataGenerator.getTrainBatches(batch_size, shuffle=shuffle):
				#summary = self._merged_summary.eval(session=session, feed_dict=self._model.get_feed_dict(batch_data, is_training=True))
				#self._train_operation.eval(session=session, feed_dict=self._model.get_feed_dict(batch_data, is_training=True))
				summary, _ = session.run([self._merged_summary, self._train_operation], feed_dict=self._model.get_feed_dict(batch_data, is_training=True))
				if train_summary_writer is not None:
					train_summary_writer.add_summary(summary, epoch)

			# Evaluate training.
			train_loss, train_acc = 0.0, 0.0
			if True:
				num_train_examples = 0
				#for batch_data, num_batch_examples in self._dataGenerator.getTrainBatchesForEvaluation(batch_size, shuffle=shuffle):
				for batch_data, num_batch_examples in self._dataGenerator.getTrainBatchesForEvaluation(batch_size, shuffle=False):
					if self._accuracy is None:
						batch_loss = self._loss.eval(session=session, feed_dict=self._model.get_feed_dict(batch_data, is_training=False))
						#batch_acc = self._accuracy.eval(session=session, feed_dict=self._model.get_feed_dict(batch_data, is_training=False))
					else:
						batch_loss, batch_acc = session.run([self._loss, self._accuracy], feed_dict=self._model.get_feed_dict(batch_data, is_training=False))

					# TODO [check] >> Is train_loss or train_acc correct?
					train_loss += batch_loss * num_batch_examples
					train_acc += batch_acc * num_batch_examples
					num_train_examples += num_batch_examples
				if num_train_examples > 0:
					train_loss /= num_train_examples
					train_acc /= num_train_examples

				history['loss'].append(train_loss)
				history['acc'].append(train_acc)

			# Evaluate.
			val_loss, val_acc = 0.0, 0.0
			if self._dataGenerator.hasValidationBatches():
				num_val_examples = 0
				#for batch_data, num_batch_examples in self._dataGenerator.getValidationBatches(batch_size, shuffle=shuffle):
				for batch_data, num_batch_examples in self._dataGenerator.getValidationBatches(batch_size, shuffle=False):
					if self._accuracy is None:
						summary, batch_loss = session.run([self._merged_summary, self._loss], feed_dict=self._model.get_feed_dict(batch_data, is_training=False))
					else:
						#summary = self._merged_summary.eval(session=session, feed_dict=self._model.get_feed_dict(batch_data, is_training=False))
						#batch_loss = self._loss.eval(session=session, feed_dict=self._model.get_feed_dict(batch_data, is_training=False))
						#batch_acc = self._accuracy.eval(session=session, feed_dict=self._model.get_feed_dict(batch_data, is_training=False))
						summary, batch_loss, batch_acc = session.run([self._merged_summary, self._loss, self._accuracy], feed_dict=self._model.get_feed_dict(batch_data, is_training=False))
					if val_summary_writer is not None:
						val_summary_writer.add_summary(summary, epoch)

					# TODO [check] >> Is val_loss or val_acc correct?
					val_loss += batch_loss * num_batch_examples
					val_acc += batch_acc * num_batch_examples
					num_val_examples += num_batch_examples
				if num_val_examples > 0:
					val_loss /= num_val_examples
					val_acc /= num_val_examples

				history['val_loss'].append(val_loss)
				history['val_acc'].append(val_acc)

				# Save a model.
				if self._saver is not None and self._model_save_dir_path is not None and val_acc >= best_val_acc:
					saved_model_path = self._saver.save(session, os.path.join(self._model_save_dir_path, 'model.ckpt'), global_step=self._global_step)
					best_val_acc = val_acc

					print('[SWL] Info: Accuracy is improved and the model is saved at {}.'.format(saved_model_path))

			print('\tElapsed time = {}'.format(time.time() - start_time))
			print('\tTraining:   loss = {}, accuracy = {}'.format(train_loss, train_acc))
			print('\tValidation: loss = {}, accuracy = {}'.format(val_loss, val_acc))

		# Close writers.
		if train_summary_writer is not None:
			train_summary_writer.close()
		if val_summary_writer is not None:
			val_summary_writer.close()

		return history

	def _get_train_operation(self, loss, global_step=None, var_list=None):
		with tf.name_scope('train_op'):
			train_op = self._optimizer.minimize(loss, global_step=global_step, var_list=var_list)
			return train_op

#%%------------------------------------------------------------------

class GradientClippingModelTrainer(ModelTrainer):
	def __init__(self, model, optimizer, dataGenerator, output_dir_path, model_save_dir_path, train_summary_dir_path, val_summary_dir_path, max_gradient_norm, global_step=None, var_list=None):
		self._max_gradient_norm = max_gradient_norm
		super().__init__(model, optimizer, dataGenerator, output_dir_path, model_save_dir_path, train_summary_dir_path, val_summary_dir_path, global_step, var_list)

	def _get_train_operation(self, loss, global_step=None, var_list=None):
		with tf.name_scope('train_op'):
			# Method 1.
			gradients = self._optimizer.compute_gradients(loss, var_list=var_list)
			for i, (g, v) in enumerate(gradients):
				if g is not None:
					gradients[i] = (tf.clip_by_norm(g, self._max_gradient_norm), v)  # Clip gradients.
			train_op = self._optimizer.apply_gradients(gradients, global_step=global_step)
			"""
			# Method 2.
			#	REF [site] >> https://www.tensorflow.org/tutorials/seq2seq
			if var_list is None:
				var_list = tf.trainable_variables()
			gradients = tf.gradients(loss, var_list)
			clipped_gradients, _ = tf.clip_by_global_norm(gradients, self._max_gradient_norm)  # Clip gradients.
			train_op = self._optimizer.apply_gradients(zip(clipped_gradients, params), global_step=global_step)
			"""
			return train_op

#%%------------------------------------------------------------------

class SimpleModelTrainer(ModelTrainer):
	def __init__(self, model, dataGenerator, output_dir_path, model_save_dir_path, train_summary_dir_path, val_summary_dir_path, initial_epoch=0, var_list=None):
		global_step = tf.Variable(initial_epoch, name='global_step', trainable=False)
		with tf.name_scope('learning_rate'):
			learning_rate = 0.001
			#learning_rate = tf.train.exponential_decay(learning_rate=0.001, global_step=global_step, decay_steps=10, decay_rate=0.995, staircase=True)
			tf.summary.scalar('learning_rate', learning_rate)
		with tf.name_scope('optimizer'):
			#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
			optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999)
			#optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=False)

		super().__init__(model, optimizer, dataGenerator, output_dir_path, model_save_dir_path, train_summary_dir_path, val_summary_dir_path, global_step, var_list)

#%%------------------------------------------------------------------

class SimpleGradientClippingModelTrainer(GradientClippingModelTrainer):
	def __init__(self, model, dataGenerator, output_dir_path, model_save_dir_path, train_summary_dir_path, val_summary_dir_path, max_gradient_norm, initial_epoch=0, var_list=None):
		global_step = tf.Variable(initial_epoch, name='global_step', trainable=False)
		with tf.name_scope('learning_rate'):
			learning_rate = 0.001
			#learning_rate = tf.train.exponential_decay(learning_rate=0.001, global_step=global_step, decay_steps=10, decay_rate=0.995, staircase=True)
			tf.summary.scalar('learning_rate', learning_rate)
		with tf.name_scope('optimizer'):
			#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
			optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999)
			#optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=False)

		super().__init__(model, optimizer, dataGenerator, output_dir_path, model_save_dir_path, train_summary_dir_path, val_summary_dir_path, max_gradient_norm, global_step, var_list)
