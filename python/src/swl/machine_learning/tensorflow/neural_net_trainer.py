import sys, time
import numpy as np
import tensorflow as tf
import imgaug as ia
from imgaug import augmenters as iaa

#%%------------------------------------------------------------------

class NeuralNetTrainer(object):
	def __init__(self, model, optimizer, global_step=None, augmenter=None, is_output_augmented=False):
		super().__init__()

		self._model = model
		self._optimizer = optimizer
		self._global_step = global_step
		self._augmenter = augmenter
		self._is_output_augmented = is_output_augmented

		self._loss, self._accuracy = self._model.loss, self._model.accuracy
		if self._loss is None:
			raise ValueError('Invalid loss')

		self._train_operation = self._get_train_operation(self._loss, self._global_step)

		# Merge all the summaries.
		self._merged_summary = tf.summary.merge_all()

	@property
	def global_step(self):
		return self._global_step

	# Supports dense and sparse labels.
	def train_by_batch(self, session, train_data, train_labels, train_summary_writer=None, is_time_major=False, is_sparse_label=False):
		batch_axis = 1 if is_time_major else 0

		num_train_examples = 0
		if train_data is not None and train_labels is not None:
			if is_sparse_label:
				num_train_examples = train_data.shape[batch_axis]
			else:
				if train_data.shape[batch_axis] == train_labels.shape[batch_axis]:
					num_train_examples = train_data.shape[batch_axis]
		#if train_data is None or train_labels is None:
		if num_train_examples <= 0:
			return None, None

		train_loss, train_acc = None, None
		if train_data.size > 0 and (is_sparse_label or train_labels.size > 0):  # If train_data and train_labels are non-empty.
			if self._augmenter is not None:
				# FIXME [fix] >> May not work correctly when using sparse label.
				train_data, train_labels = self._augmenter(train_data, train_labels, self._is_output_augmented)

			#summary = self._merged_summary.eval(session=session, feed_dict=self._model.get_feed_dict(train_data, train_labels, is_training=True))
			#self._train_operation.eval(session=session, feed_dict=self._model.get_feed_dict(train_data, train_labels, is_training=True))
			summary, _ = session.run([self._merged_summary, self._train_operation], feed_dict=self._model.get_feed_dict(train_data, train_labels, is_training=True))
			if train_summary_writer is not None:
				train_summary_writer.add_summary(summary)

			# Evaluate training.
			if self._accuracy is None:
				train_loss = self._loss.eval(session=session, feed_dict=self._model.get_feed_dict(train_data, train_labels, is_training=False))
				#train_acc = self._accuracy.eval(session=session, feed_dict=self._model.get_feed_dict(train_data, train_labels, is_training=False))
			else:
				train_loss, train_acc = session.run([self._loss, self._accuracy], feed_dict=self._model.get_feed_dict(train_data, train_labels, is_training=False))

		return train_acc, train_loss

	# Supports dense and sparse labels.
	# REF [function] >> NeuralNetEvaluator.evaluate_by_batch() in neural_net_evaluator.py
	def evaluate_training_by_batch(self, session, val_data, val_labels, val_summary_writer=None, is_time_major=False, is_sparse_label=False):
		batch_axis = 1 if is_time_major else 0

		num_val_examples = 0
		if val_data is not None and val_labels is not None:
			if is_sparse_label:
				num_val_examples = val_data.shape[batch_axis]
			else:
				if val_data.shape[batch_axis] == val_labels.shape[batch_axis]:
					num_val_examples = val_data.shape[batch_axis]
		#if val_data is None or val_labels is None:
		if num_val_examples <= 0:
			return None, None

		val_loss, val_acc = None, None
		if val_data.size > 0 and (is_sparse_label or val_labels.size > 0):  # If val_data and val_labels are non-empty.
			if self._accuracy is None:
				summary, val_loss = session.run([self._merged_summary, self._loss], feed_dict=self._model.get_feed_dict(val_data, val_labels, is_training=False))
			else:
				#summary = self._merged_summary.eval(session=session, feed_dict=self._model.get_feed_dict(val_data, val_labels, is_training=False))
				#val_loss = self._loss.eval(session=session, feed_dict=self._model.get_feed_dict({val_data, val_labels, is_training=False))
				#val_acc = self._accuracy.eval(session=session, feed_dict=self._model.get_feed_dict(val_data, val_labels, is_training=False))
				summary, val_loss, val_acc = session.run([self._merged_summary, self._loss, self._accuracy], feed_dict=self._model.get_feed_dict(val_data, val_labels, is_training=False))
			if val_summary_writer is not None:
				val_summary_writer.add_summary(summary)

		return val_acc, val_loss

	# Supports a dense label only.
	def train(self, session, train_data, train_labels, val_data, val_labels, batch_size, num_epochs, shuffle=True, saver=None, model_save_dir_path=None, train_summary_dir_path=None, val_summary_dir_path=None, is_time_major=False):
		# TODO [check] >>
		if isinstance(self._augmenter, iaa.Sequential):
			# If imgaug augmenter is used, data are augmented in background augmentation processes. (faster)
			return self._train_by_imgaug(session, train_data, train_labels, val_data, val_labels, batch_size, num_epochs, shuffle, saver, model_save_dir_path, train_summary_dir_path, val_summary_dir_path, is_time_major)
		else:
			return self._train(session, train_data, train_labels, val_data, val_labels, batch_size, num_epochs, shuffle, saver, model_save_dir_path, train_summary_dir_path, val_summary_dir_path, is_time_major)

	def _train(self, session, train_data, train_labels, val_data, val_labels, batch_size, num_epochs, shuffle=True, saver=None, model_save_dir_path=None, train_summary_dir_path=None, val_summary_dir_path=None, is_time_major=False):
		batch_axis = 1 if is_time_major else 0

		# Create writers to write all the summaries out to a directory.
		train_summary_writer = tf.summary.FileWriter(train_summary_dir_path, session.graph) if train_summary_dir_path is not None else None
		val_summary_writer = tf.summary.FileWriter(val_summary_dir_path) if val_summary_dir_path is not None else None

		num_train_examples, train_steps_per_epoch = 0, 0
		if train_data is not None and train_labels is not None:
			if train_data.shape[batch_axis] == train_labels.shape[batch_axis]:
				num_train_examples = train_data.shape[batch_axis]
			train_steps_per_epoch = ((num_train_examples - 1) // batch_size + 1) if num_train_examples > 0 else 0
		#if train_data is None or train_labels is None:
		if num_train_examples <= 0:
			return None

		num_val_examples, val_steps_per_epoch = 0, 0
		if val_data is not None and val_labels is not None:
			if val_data.shape[batch_axis] == val_labels.shape[batch_axis]:
				num_val_examples = val_data.shape[batch_axis]
			val_steps_per_epoch = ((num_val_examples - 1) // batch_size + 1) if num_val_examples > 0 else 0

		history = {
			'acc': [],
			'loss': [],
			'val_acc': [],
			'val_loss': []
		}

		best_val_acc = 0.0
		for epoch in range(1, num_epochs + 1):
			print('Epoch {}/{}'.format(epoch, num_epochs))

			start_time = time.time()

			train_loss, train_acc, val_loss, val_acc = None, None, None, None
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
					# FIXME [fix] >> Does not work correctly in time-major data.
					data_batch, label_batch = train_data[batch_indices], train_labels[batch_indices]
					if data_batch.size > 0 and label_batch.size > 0:  # If data_batch and label_batch are non-empty.
						if self._augmenter is not None:
							data_batch, label_batch = self._augmenter(data_batch, label_batch, self._is_output_augmented)

						#summary = self._merged_summary.eval(session=session, feed_dict=self._model.get_feed_dict(data_batch, label_batch, is_training=True))
						#self._train_operation.eval(session=session, feed_dict=self._model.get_feed_dict(data_batch, label_batch, is_training=True))
						summary, _ = session.run([self._merged_summary, self._train_operation], feed_dict=self._model.get_feed_dict(data_batch, label_batch, is_training=True))
						if train_summary_writer is not None:
							train_summary_writer.add_summary(summary, epoch)
				if step / train_steps_per_epoch >= processing_ratio:
					print('-', sep='', end='')
					processing_ratio = round(step / train_steps_per_epoch, 2) + 0.05
			print('<')

			# Evaluate training.
			train_loss, train_acc = 0.0, 0.0
			if True:
				"""
				batch_indices = indices[0:batch_size]
				if batch_indices.size > 0:  # If batch_indices is non-empty.
					# FIXME [fix] >> Does not work correctly in time-major data.
					data_batch, label_batch = train_data[batch_indices], train_labels[batch_indices]
					if data_batch.size > 0 and label_batch.size > 0:  # If data_batch and label_batch are non-empty.
						if self._accuracy is None:
							train_loss = self._loss.eval(session=session, feed_dict=self._model.get_feed_dict(data_batch, label_batch, is_training=False))
							#train_acc = self._accuracy.eval(session=session, feed_dict=self._model.get_feed_dict(data_batch, label_batch, is_training=False))
						else:
							train_loss, train_acc = session.run([self._loss, self._accuracy], feed_dict=self._model.get_feed_dict(data_batch, label_batch, is_training=False))
				"""
				for step in range(train_steps_per_epoch):
					start = step * batch_size
					end = start + batch_size
					batch_indices = indices[start:end]
					if batch_indices.size > 0:  # If batch_indices is non-empty.
						# FIXME [fix] >> Does not work correctly in time-major data.
						data_batch, label_batch = train_data[batch_indices], train_labels[batch_indices]
						if data_batch.size > 0 and label_batch.size > 0:  # If data_batch and label_batch are non-empty.
							if self._accuracy is None:
								batch_loss = self._loss.eval(session=session, feed_dict=self._model.get_feed_dict(data_batch, label_batch, is_training=False))
								#batch_acc = self._accuracy.eval(session=session, feed_dict=self._model.get_feed_dict(data_batch, label_batch, is_training=False))
							else:
								batch_loss, batch_acc = session.run([self._loss, self._accuracy], feed_dict=self._model.get_feed_dict(data_batch, label_batch, is_training=False))

							# TODO [check] >> Is train_loss or train_acc correct?
							train_loss += batch_loss * batch_indices.size
							train_acc += batch_acc * batch_indices.size
				train_loss /= num_train_examples
				train_acc /= num_train_examples

				history['loss'].append(train_loss)
				history['acc'].append(train_acc)

			# Evaluate.
			val_loss, val_acc = 0.0, 0.0
			#if val_data is not None and val_labels is not None:
			if num_val_examples > 0:
				"""
				data_batch, label_batch = val_data, val_labels
				if data_batch.size > 0 and label_batch.size > 0:  # If data_batch and label_batch are non-empty.
					if self._accuracy is None:
						summary, val_loss = session.run([self._merged_summary, self._loss], feed_dict=self._model.get_feed_dict(data_batch, label_batch, is_training=False))
					else:
						#summary = self._merged_summary.eval(session=session, feed_dict=self._model.get_feed_dict(data_batch, label_batch, is_training=False))
						#val_loss = self._loss.eval(session=session, feed_dict=self._model.get_feed_dict(data_batch, label_batch, is_training=False))
						#val_acc = self._accuracy.eval(session=session, feed_dict=self._model.get_feed_dict(data_batch, label_batch, is_training=False))
						summary, val_loss, val_acc = session.run([self._merged_summary, self._loss, self._accuracy], feed_dict=self._model.get_feed_dict(data_batch, label_batch, is_training=False))
					if val_summary_writer is not None:
						val_summary_writer.add_summary(summary, epoch)
				"""
				indices = np.arange(num_val_examples)
				#if shuffle:
				#	np.random.shuffle(indices)

				for step in range(val_steps_per_epoch):
					start = step * batch_size
					end = start + batch_size
					batch_indices = indices[start:end]
					if batch_indices.size > 0:  # If batch_indices is non-empty.
						# FIXME [fix] >> Does not work correctly in time-major data.
						data_batch, label_batch = val_data[batch_indices], val_labels[batch_indices]
						if data_batch.size > 0 and label_batch.size > 0:  # If data_batch and label_batch are non-empty.
							if self._accuracy is None:
								summary, batch_loss = session.run([self._merged_summary, self._loss], feed_dict=self._model.get_feed_dict(data_batch, label_batch, is_training=False))
							else:
								#summary = self._merged_summary.eval(session=session, feed_dict=self._model.get_feed_dict(data_batch, label_batch, is_training=False))
								#batch_loss = self._loss.eval(session=session, feed_dict=self._model.get_feed_dict({data_batch, label_batch, is_training=False))
								#batch_acc = self._accuracy.eval(session=session, feed_dict=self._model.get_feed_dict(data_batch, label_batch, is_training=False))
								summary, batch_loss, batch_acc = session.run([self._merged_summary, self._loss, self._accuracy], feed_dict=self._model.get_feed_dict(data_batch, label_batch, is_training=False))
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
			print('\tTraining:   loss = {}, accuracy = {}'.format(train_loss, train_acc))
			print('\tValidation: loss = {}, accurary = {}'.format(val_loss, val_acc))

		# Close writers.
		if train_summary_writer is not None:
			train_summary_writer.close()
		if val_summary_writer is not None:
			val_summary_writer.close()

		return history

	@staticmethod
	def _create_batch_generator(train_data, train_labels, is_output_augmented, indices, batch_size, train_steps_per_epoch):
		for step in range(train_steps_per_epoch):
			start = step * batch_size
			end = start + batch_size
			batch_indices = indices[start:end]
			if batch_indices.size > 0:  # If batch_indices is non-empty.
				# FIXME [fix] >> Does not work correctly in time-major data.
				data_batch, label_batch = train_data[batch_indices], train_labels[batch_indices]
				if data_batch.size > 0 and label_batch.size > 0:  # If data_batch and label_batch are non-empty.
					# Add e.g. keypoints=... or bounding_boxes=... here to also augment keypoints / bounding boxes on these images.
					if is_output_augmented:
						yield ia.Batch(images=data_batch, heatmaps=label_batch)
					else:
						yield ia.Batch(images=data_batch, data=label_batch)

	# Supports a dense label only.
	def _train_by_imgaug(self, session, train_data, train_labels, val_data, val_labels, batch_size, num_epochs, shuffle=True, saver=None, model_save_dir_path=None, train_summary_dir_path=None, val_summary_dir_path=None, is_time_major=False):
		batch_axis = 1 if is_time_major else 0

		# Create writers to write all the summaries out to a directory.
		train_summary_writer = tf.summary.FileWriter(train_summary_dir_path, session.graph) if train_summary_dir_path is not None else None
		val_summary_writer = tf.summary.FileWriter(val_summary_dir_path) if val_summary_dir_path is not None else None

		num_train_examples, train_steps_per_epoch = 0, 0
		if train_data is not None and train_labels is not None:
			if train_data.shape[batch_axis] == train_labels.shape[batch_axis]:
				num_train_examples = train_data.shape[batch_axis]
			train_steps_per_epoch = ((num_train_examples - 1) // batch_size + 1) if num_train_examples > 0 else 0
		#if train_data is None or train_labels is None:
		if num_train_examples <= 0:
			return None

		num_val_examples, val_steps_per_epoch = 0, 0
		if val_data is not None and val_labels is not None:
			if val_data.shape[batch_axis] == val_labels.shape[batch_axis]:
				num_val_examples = val_data.shape[batch_axis]
			val_steps_per_epoch = ((num_val_examples - 1) // batch_size + 1) if num_val_examples > 0 else 0

		history = {
			'acc': [],
			'loss': [],
			'val_acc': [],
			'val_loss': []
		}

		with self._augmenter.pool(processes=-1, maxtasksperchild=20, seed=123) as pool:
			best_val_acc = 0.0
			for epoch in range(1, num_epochs + 1):
				print('Epoch {}/{}'.format(epoch, num_epochs))

				start_time = time.time()

				train_loss, train_acc, val_loss, val_acc = None, None, None, None
				indices = np.arange(num_train_examples)
				if shuffle:
					np.random.shuffle(indices)

				# Train.
				batch_gen = NeuralNetTrainer._create_batch_generator(train_data, train_labels, self._is_output_augmented, indices, batch_size, train_steps_per_epoch)
				batch_aug_gen = pool.imap_batches(batch_gen, chunksize=10)
				for batch_aug in batch_aug_gen:
					#summary = self._merged_summary.eval(session=session, feed_dict=self._model.get_feed_dict(batch_aug.images_aug, batch_aug.heatmaps_aug if self._is_output_augmented else batch_aug.data, is_training=True))
					#self._train_operation.eval(session=session, feed_dict=self._model.get_feed_dict(batch_aug.images_aug, batch_aug.heatmaps_aug if self._is_output_augmented else batch_aug.data, is_training=True))
					summary, _ = session.run([self._merged_summary, self._train_operation], feed_dict=self._model.get_feed_dict(batch_aug.images_aug, batch_aug.heatmaps_aug if self._is_output_augmented else batch_aug.data, is_training=True))
					if train_summary_writer is not None:
						train_summary_writer.add_summary(summary, epoch)

				# Evaluate training.
				train_loss, train_acc = 0.0, 0.0
				if True:
					"""
					batch_indices = indices[0:batch_size]
					if batch_indices.size > 0:  # If batch_indices is non-empty.
						# FIXME [fix] >> Does not work correctly in time-major data.
						data_batch, label_batch = train_data[batch_indices], train_labels[batch_indices]
						if data_batch.size > 0 and label_batch.size > 0:  # If data_batch and label_batch are non-empty.
							if self._accuracy is None:
								train_loss = self._loss.eval(session=session, feed_dict=self._model.get_feed_dict(data_batch, label_batch, is_training=False))
								#train_acc = self._accuracy.eval(session=session, feed_dict=self._model.get_feed_dict(data_batch, label_batch, is_training=False))
							else:
								train_loss, train_acc = session.run([self._loss, self._accuracy], feed_dict=self._model.get_feed_dict(data_batch, label_batch, is_training=False))
					"""
					for step in range(train_steps_per_epoch):
						start = step * batch_size
						end = start + batch_size
						batch_indices = indices[start:end]
						if batch_indices.size > 0:  # If batch_indices is non-empty.
							# FIXME [fix] >> Does not work correctly in time-major data.
							data_batch, label_batch = train_data[batch_indices], train_labels[batch_indices]
							if data_batch.size > 0 and label_batch.size > 0:  # If data_batch and label_batch are non-empty.
								if self._accuracy is None:
									batch_loss = self._loss.eval(session=session, feed_dict=self._model.get_feed_dict(data_batch, label_batch, is_training=False))
									#batch_acc = self._accuracy.eval(session=session, feed_dict=self._model.get_feed_dict(data_batch, label_batch, is_training=False))
								else:
									batch_loss, batch_acc = session.run([self._loss, self._accuracy], feed_dict=self._model.get_feed_dict(data_batch, label_batch, is_training=False))

								# TODO [check] >> Is train_loss or train_acc correct?
								train_loss += batch_loss * batch_indices.size
								train_acc += batch_acc * batch_indices.size
					train_loss /= num_train_examples
					train_acc /= num_train_examples

					history['loss'].append(train_loss)
					history['acc'].append(train_acc)

				# Evaluate.
				val_loss, val_acc = 0.0, 0.0
				#if val_data is not None and val_labels is not None:
				if num_val_examples > 0:
					"""
					data_batch, label_batch = val_data, val_labels
					if data_batch.size > 0 and label_batch.size > 0:  # If data_batch and label_batch are non-empty.
						if self._accuracy is None:
							summary, val_loss = session.run([self._merged_summary, self._loss], feed_dict=self._model.get_feed_dict(data_batch, label_batch, is_training=False))
						else:
							#summary = self._merged_summary.eval(session=session, feed_dict=self._model.get_feed_dict(data_batch, label_batch, is_training=False))
							#val_loss = self._loss.eval(session=session, feed_dict=self._model.get_feed_dict(data_batch, label_batch, is_training=False))
							#val_acc = self._accuracy.eval(session=session, feed_dict=self._model.get_feed_dict(data_batch, label_batch, is_training=False))
							summary, val_loss, val_acc = session.run([self._merged_summary, self._loss, self._accuracy], feed_dict=self._model.get_feed_dict(data_batch, label_batch, is_training=False))
						if val_summary_writer is not None:
							val_summary_writer.add_summary(summary, epoch)
					"""
					indices = np.arange(num_val_examples)
					#if shuffle:
					#	np.random.shuffle(indices)

					for step in range(val_steps_per_epoch):
						start = step * batch_size
						end = start + batch_size
						batch_indices = indices[start:end]
						if batch_indices.size > 0:  # If batch_indices is non-empty.
							# FIXME [fix] >> Does not work correctly in time-major data.
							data_batch, label_batch = val_data[batch_indices], val_labels[batch_indices]
							if data_batch.size > 0 and label_batch.size > 0:  # If data_batch and label_batch are non-empty.
								if self._accuracy is None:
									summary, batch_loss = session.run([self._merged_summary, self._loss], feed_dict=self._model.get_feed_dict(data_batch, label_batch, is_training=False))
								else:
									#summary = self._merged_summary.eval(session=session, feed_dict=self._model.get_feed_dict(data_batch, label_batch, is_training=False))
									#batch_loss = self._loss.eval(session=session, feed_dict=self._model.get_feed_dict({data_batch, label_batch, is_training=False))
									#batch_acc = self._accuracy.eval(session=session, feed_dict=self._model.get_feed_dict(data_batch, label_batch, is_training=False))
									summary, batch_loss, batch_acc = session.run([self._merged_summary, self._loss, self._accuracy], feed_dict=self._model.get_feed_dict(data_batch, label_batch, is_training=False))
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
				print('\tTraining:   loss = {}, accuracy = {}'.format(train_loss, train_acc))
				print('\tValidation: loss = {}, accurary = {}'.format(val_loss, val_acc))

			# Close writers.
			if train_summary_writer is not None:
				train_summary_writer.close()
			if val_summary_writer is not None:
				val_summary_writer.close()

			return history

	def train_seq2seq_by_batch(self, session, train_encoder_inputs, train_decoder_inputs, train_decoder_outputs, val_encoder_inputs=None, val_decoder_inputs=None, val_decoder_outputs=None, train_summary_writer=None, val_summary_writer=None, is_time_major=False):
		batch_axis = 1 if is_time_major else 0

		num_train_examples = 0
		if train_encoder_inputs is not None and train_decoder_inputs is not None and train_decoder_outputs is not None:
			if train_encoder_inputs.shape[batch_axis] == train_decoder_inputs.shape[batch_axis] and train_encoder_inputs.shape[batch_axis] == train_decoder_outputs.shape[batch_axis]:
				num_train_examples = train_encoder_inputs.shape[batch_axis]
		#if train_encoder_inputs is None or train_decoder_inputs is None or train_decoder_inputs is None:
		if num_train_examples <= 0:
			return None

		num_val_examples = 0
		if val_encoder_inputs is not None and val_decoder_inputs is not None and val_decoder_outputs is not None:
			if val_encoder_inputs.shape[batch_axis] == val_decoder_inputs.shape[batch_axis] and val_encoder_inputs.shape[batch_axis] == val_decoder_outputs.shape[batch_axis]:
				num_val_examples = val_encoder_inputs.shape[batch_axis]

		train_loss, train_acc, val_loss, val_acc = None, None, None, None
		# Train.
		if train_encoder_inputs.size > 0 and train_decoder_inputs.size > 0 and train_decoder_outputs.size > 0:  # If train_encoder_inputs, train_decoder_inputs, and train_decoder_outputs are non-empty.
			if self._augmenter is not None:
				train_encoder_inputs, train_decoder_inputs, train_decoder_outputs = self._augmenter(train_encoder_inputs, train_decoder_inputs, train_decoder_outputs, self._is_output_augmented)

			#summary = self._merged_summary.eval(session=session, feed_dict=self._model.get_feed_dict(train_encoder_inputs, train_decoder_inputs, train_decoder_outputs, is_training=True))
			#self._train_operation.eval(session=session, feed_dict=self._model.get_feed_dict(train_encoder_inputs, train_decoder_inputs, train_decoder_outputs, is_training=True))
			summary, _ = session.run([self._merged_summary, self._train_operation], feed_dict=self._model.get_feed_dict(train_encoder_inputs, train_decoder_inputs, train_decoder_outputs, is_training=True))
			if train_summary_writer is not None:
				train_summary_writer.add_summary(summary)

		# Evaluate training.
		if True:
			if train_encoder_inputs.size > 0 and train_decoder_inputs.size > 0 and train_decoder_outputs.size > 0:  # If train_encoder_inputs, train_decoder_inputs, and train_decoder_outputs are non-empty.
				if self._accuracy is None:
					train_loss = self._loss.eval(session=session, feed_dict=self._model.get_feed_dict(train_encoder_inputs, train_decoder_inputs, train_decoder_outputs, is_training=False))
					#train_acc = self._accuracy.eval(session=session, feed_dict=self._model.get_feed_dict(train_encoder_inputs, train_decoder_inputs, train_decoder_outputs, is_training=False))
				else:
					train_loss, train_acc = session.run([self._loss, self._accuracy], feed_dict=self._model.get_feed_dict(train_encoder_inputs, train_decoder_inputs, train_decoder_outputs, is_training=False))

		# Evaluate.
		#if val_encoder_inputs is not None and val_decoder_inputs is not None and val_decoder_outputs is not None:
		if num_val_examples > 0:
			if val_encoder_inputs.size > 0 and val_decoder_inputs.size > 0 and val_decoder_outputs.size > 0:  # If val_encoder_inputs, val_decoder_inputs, and val_decoder_outputs are non-empty.
				if self._accuracy is None:
					summary, val_loss = session.run([self._merged_summary, self._loss], feed_dict=self._model.get_feed_dict(val_encoder_inputs, val_decoder_inputs, val_decoder_outputs, is_training=False))
				else:
					#summary = self._merged_summary.eval(session=session, feed_dict=self._model.get_feed_dict(val_encoder_inputs, val_decoder_inputs, val_decoder_outputs, is_training=False))
					#val_loss = self._loss.eval(session=session, feed_dict=self._model.get_feed_dict(val_encoder_inputs, val_decoder_inputs, val_decoder_outputs, is_training=False))
					#val_acc = self._accuracy.eval(session=session, feed_dict=self._model.get_feed_dict(val_encoder_inputs, val_decoder_inputs, val_decoder_outputs, is_training=False))
					summary, val_loss, val_acc = session.run([self._merged_summary, self._loss, self._accuracy], feed_dict=self._model.get_feed_dict(val_encoder_inputs, val_decoder_inputs, val_decoder_outputs, is_training=False))
				if val_summary_writer is not None:
					val_summary_writer.add_summary(summary)

		return train_acc, train_loss, val_acc, val_loss

	def train_seq2seq(self, session, train_encoder_inputs, train_decoder_inputs, train_decoder_outputs, val_encoder_inputs, val_decoder_inputs, val_decoder_outputs, batch_size, num_epochs, shuffle=True, saver=None, model_save_dir_path=None, train_summary_dir_path=None, val_summary_dir_path=None, is_time_major=False):
		batch_axis = 1 if is_time_major else 0

		# Create writers to write all the summaries out to a directory.
		train_summary_writer = tf.summary.FileWriter(train_summary_dir_path, session.graph) if train_summary_dir_path is not None else None
		val_summary_writer = tf.summary.FileWriter(val_summary_dir_path) if val_summary_dir_path is not None else None

		num_train_examples, train_steps_per_epoch = 0, 0
		if train_encoder_inputs is not None and train_decoder_inputs is not None and train_decoder_outputs is not None:
			if train_encoder_inputs.shape[batch_axis] == train_decoder_inputs.shape[batch_axis] and train_encoder_inputs.shape[batch_axis] == train_decoder_outputs.shape[batch_axis]:
				num_train_examples = train_encoder_inputs.shape[batch_axis]
			train_steps_per_epoch = ((num_train_examples - 1) // batch_size + 1) if num_train_examples > 0 else 0
		#if train_encoder_inputs is None or train_decoder_inputs is None or train_decoder_outputs is None:
		if num_train_examples <= 0:
			return None

		num_val_examples, val_steps_per_epoch = 0, 0
		if val_encoder_inputs is not None and val_decoder_inputs is not None and val_decoder_outputs is not None:
			if val_encoder_inputs.shape[batch_axis] == val_decoder_inputs.shape[batch_axis] and val_encoder_inputs.shape[batch_axis] == val_decoder_outputs.shape[batch_axis]:
				num_val_examples = val_encoder_inputs.shape[batch_axis]
			val_steps_per_epoch = ((num_val_examples - 1) // batch_size + 1) if num_val_examples > 0 else 0

		history = {
			'acc': [],
			'loss': [],
			'val_acc': [],
			'val_loss': []
		}

		best_val_acc = 0.0
		for epoch in range(1, num_epochs + 1):
			print('Epoch {}/{}'.format(epoch, num_epochs))

			start_time = time.time()

			train_loss, train_acc, val_loss, val_acc = None, None, None, None
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
					# FIXME [fix] >> Does not work correctly in time-major data.
					enc_input_batch, dec_input_batch, dec_output_batch = train_encoder_inputs[batch_indices], train_decoder_inputs[batch_indices], train_decoder_outputs[batch_indices]
					if enc_input_batch.size > 0 and dec_input_batch.size > 0 and dec_output_batch.size > 0:  # If enc_input_batch, dec_input_batch, and dec_output_batch are non-empty.
						if self._augmenter is not None:
							enc_input_batch, dec_input_batch, dec_output_batch = self._augmenter(enc_input_batch, dec_input_batch, dec_output_batch, self._is_output_augmented)

						#summary = self._merged_summary.eval(session=session, feed_dict=self._model.get_feed_dict(enc_input_batch, dec_input_batch, dec_output_batch, is_training=True))
						#self._train_operation.eval(session=session, feed_dict=self._model.get_feed_dict(enc_input_batch, dec_input_batch, dec_output_batch, is_training=True))
						summary, _ = session.run([self._merged_summary, self._train_operation], feed_dict=self._model.get_feed_dict(enc_input_batch, dec_input_batch, dec_output_batch, is_training=True))
						if train_summary_writer is not None:
							train_summary_writer.add_summary(summary, epoch)
				if step / train_steps_per_epoch >= processing_ratio:
					print('-', sep='', end='')
					processing_ratio = round(step / train_steps_per_epoch, 2) + 0.05
			print('<')

			# Evaluate training.
			train_loss, train_acc = 0.0, 0.0
			if True:
				"""
				batch_indices = indices[0:batch_size]
				if batch_indices.size > 0:  # If batch_indices is non-empty.
					# FIXME [fix] >> Does not work correctly in time-major data.
					enc_input_batch, dec_input_batch, dec_output_batch = train_encoder_inputs[batch_indices], train_decoder_inputs[batch_indices], train_decoder_outputs[batch_indices]
					if enc_input_batch.size > 0 and dec_input_batch.size > 0 and dec_output_batch.size > 0:  # If enc_input_batch, dec_input_batch, and dec_output_batch are non-empty.
						if self._accuracy is None:
							train_loss = self._loss.eval(session=session, feed_dict=self._model.get_feed_dict(enc_input_batch, dec_input_batch, dec_output_batch, is_training=False))
							#train_acc = self._accuracy.eval(session=session, feed_dict=self._model.get_feed_dict(enc_input_batch, dec_input_batch, dec_output_batch, is_training=False))
						else:
							train_loss, train_acc = session.run([self._loss, self._accuracy], feed_dict=self._model.get_feed_dict(enc_input_batch, dec_input_batch, dec_output_batch, is_training=False))
				"""
				for step in range(train_steps_per_epoch):
					start = step * batch_size
					end = start + batch_size
					batch_indices = indices[start:end]
					if batch_indices.size > 0:  # If batch_indices is non-empty.
						# FIXME [fix] >> Does not work correctly in time-major data.
						enc_input_batch, dec_input_batch, dec_output_batch = train_encoder_inputs[batch_indices], train_decoder_inputs[batch_indices], train_decoder_outputs[batch_indices]
						if enc_input_batch.size > 0 and dec_input_batch.size > 0 and dec_output_batch.size > 0:  # If enc_input_batch, dec_input_batch, and dec_output_batch are non-empty.
							if self._accuracy is None:
								batch_loss = self._loss.eval(session=session, feed_dict=self._model.get_feed_dict(enc_input_batch, dec_input_batch, dec_output_batch, is_training=False))
								#batch_acc = self._accuracy.eval(session=session, feed_dict=self._model.get_feed_dict(enc_input_batch, dec_input_batch, dec_output_batch, is_training=False))
							else:
								batch_loss, batch_acc = session.run([self._loss, self._accuracy], feed_dict=self._model.get_feed_dict(enc_input_batch, dec_input_batch, dec_output_batch, is_training=False))

							# TODO [check] >> Is train_loss or train_acc correct?
							train_loss += batch_loss * batch_indices.size
							train_acc += batch_acc * batch_indices.size
				train_loss /= num_train_examples
				train_acc /= num_train_examples

				history['loss'].append(train_loss)
				history['acc'].append(train_acc)

			# Evaluate.
			val_loss, val_acc = 0.0, 0.0
			#if val_data is not None and val_labels is not None:
			if num_val_examples > 0:
				"""
				# FIXME [fix] >> Does not work correctly in time-major data.
				enc_input_batch, dec_input_batch, dec_output_batch = val_encoder_inputs[batch_indices], val_decoder_inputs[batch_indices], val_decoder_outputs[batch_indices]
				if enc_input_batch.size > 0 and dec_input_batch.size > 0 and dec_output_batch.size > 0:  # If enc_input_batch, dec_input_batch, and dec_output_batch are non-empty.
					if self._accuracy is None:
						summary, val_loss = session.run([self._merged_summary, self._loss], feed_dict=self._model.get_feed_dict(enc_input_batch, dec_input_batch, dec_output_batch, is_training=False))
					else:
						#summary = self._merged_summary.eval(session=session, feed_dict=self._model.get_feed_dict(enc_input_batch, dec_input_batch, dec_output_batch, is_training=False))
						#val_loss = self._loss.eval(session=session, feed_dict=self._model.get_feed_dict(enc_input_batch, dec_input_batch, dec_output_batch, is_training=False))
						#val_acc = self._accuracy.eval(session=session, feed_dict=self._model.get_feed_dict(enc_input_batch, dec_input_batch, dec_output_batch, is_training=False))
						summary, val_loss, val_acc = session.run([self._merged_summary, self._loss, self._accuracy], feed_dict=self._model.get_feed_dict(enc_input_batch, dec_input_batch, dec_output_batch, is_training=False))
					if val_summary_writer is not None:
						val_summary_writer.add_summary(summary, epoch)
				"""
				indices = np.arange(num_val_examples)
				#if shuffle:
				#	np.random.shuffle(indices)

				for step in range(val_steps_per_epoch):
					start = step * batch_size
					end = start + batch_size
					batch_indices = indices[start:end]
					if batch_indices.size > 0:  # If batch_indices is non-empty.
						# FIXME [fix] >> Does not work correctly in time-major data.
						enc_input_batch, dec_input_batch, dec_output_batch = val_encoder_inputs[batch_indices], val_decoder_inputs[batch_indices], val_decoder_outputs[batch_indices]
						if enc_input_batch.size > 0 and dec_input_batch.size > 0 and dec_output_batch.size > 0:  # If enc_input_batch, dec_input_batch, and dec_output_batch are non-empty.
							if self._accuracy is None:
								summary, batch_loss = session.run([self._merged_summary, self._loss], feed_dict=self._model.get_feed_dict(enc_input_batch, dec_input_batch, dec_output_batch, is_training=False))
							else:
								#summary = self._merged_summary.eval(session=session, feed_dict=self._model.get_feed_dict(enc_input_batch, dec_input_batch, dec_output_batch, is_training=False))
								#batch_loss = self._loss.eval(session=session, feed_dict=self._model.get_feed_dict(enc_input_batch, dec_input_batch, dec_output_batch, is_training=False))
								#batch_acc = self._accuracy.eval(session=session, feed_dict=self._model.get_feed_dict(enc_input_batch, dec_input_batch, dec_output_batch, is_training=False))
								summary, batch_loss, batch_acc = session.run([self._merged_summary, self._loss, self._accuracy], feed_dict=self._model.get_feed_dict(enc_input_batch, dec_input_batch, dec_output_batch, is_training=False))
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
			print('\tTraining:   loss = {}, accuracy = {}'.format(train_loss, train_acc))
			print('\tValidation: loss = {}, accurary = {}'.format(val_loss, val_acc))

		# Close writers.
		if train_summary_writer is not None:
			train_summary_writer.close()
		if val_summary_writer is not None:
			val_summary_writer.close()

		return history

	def train_unsupervisedly_by_batch(self, session, train_data, val_data=None, train_summary_writer=None, val_summary_writer=None, is_time_major=False):
		batch_axis = 1 if is_time_major else 0

		num_train_examples = 0
		if train_data is not None:
			num_train_examples = train_data.shape[batch_axis]
		#if train_data is None:
		if num_train_examples <= 0:
			return None, None

		num_val_examples = 0
		if val_data is not None:
			num_val_examples = val_data.shape[batch_axis]

		train_loss, val_loss = None, None
		# Train.
		if train_data.size > 0:  # If train_data is non-empty.
			if self._augmenter is not None:
				train_data, _ = self._augmenter(train_data, None, False)

			#summary = self._merged_summary.eval(session=session, feed_dict=self._model.get_feed_dict(train_data, is_training=True))
			#self._train_operation.eval(session=session, feed_dict=self._model.get_feed_dict(train_data, is_training=True))
			summary, _ = session.run([self._merged_summary, self._train_operation], feed_dict=self._model.get_feed_dict(train_data, is_training=True))
			if train_summary_writer is not None:
				train_summary_writer.add_summary(summary)

		# Evaluate training.
		if True:
			if train_data.size > 0:  # If train_data is non-empty.
				train_loss = self._loss.eval(session=session, feed_dict=self._model.get_feed_dict(train_data, is_training=False))

		# Evaluate.
		#if val_data is not None:
		if num_val_examples > 0:
			if val_data.size > 0:  # If val_data is non-empty.
				#summary = self._merged_summary.eval(session=session, feed_dict=self._model.get_feed_dict(val_data, is_training=False))
				#val_loss = self._loss.eval(session=session, feed_dict=self._model.get_feed_dict({val_data, is_training=False))
				summary, val_loss = session.run([self._merged_summary, self._loss], feed_dict=self._model.get_feed_dict(val_data, is_training=False))
				if val_summary_writer is not None:
					val_summary_writer.add_summary(summary)

		return train_loss, val_loss

	def train_unsupervisedly(self, session, train_data, val_data, batch_size, num_epochs, shuffle=True, saver=None, model_save_dir_path=None, train_summary_dir_path=None, val_summary_dir_path=None, is_time_major=False):
		batch_axis = 1 if is_time_major else 0

		# Create writers to write all the summaries out to a directory.
		train_summary_writer = tf.summary.FileWriter(train_summary_dir_path, session.graph) if train_summary_dir_path is not None else None
		val_summary_writer = tf.summary.FileWriter(val_summary_dir_path) if val_summary_dir_path is not None else None

		num_train_examples, train_steps_per_epoch = 0, 0
		if train_data is not None:
			num_train_examples = train_data.shape[0]
			train_steps_per_epoch = ((num_train_examples - 1) // batch_size + 1) if num_train_examples > 0 else 0
		#if train_data is None:
		if num_train_examples <= 0:
			return None

		num_val_examples, val_steps_per_epoch = 0, 0
		if val_data is not None:
			num_val_examples = val_data.shape[0]
			val_steps_per_epoch = ((num_val_examples - 1) // batch_size + 1) if num_val_examples > 0 else 0

		history = {
			'acc': None,
			'loss': [],
			'val_acc': None,
			'val_loss': []
		}

		best_val_loss = sys.float_info.max
		for epoch in range(1, num_epochs + 1):
			print('Epoch {}/{}'.format(epoch, num_epochs))

			start_time = time.time()

			train_loss, val_loss = 0.0, 0.0
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
					# FIXME [fix] >> Does not work correctly in time-major data.
					data_batch = train_data[batch_indices]
					if data_batch.size > 0:  # If data_batch is non-empty.
						if self._augmenter is not None:
							data_batch, _ = self._augmenter(data_batch, None, False)

						#summary = self._merged_summary.eval(session=session, feed_dict=self._model.get_feed_dict(data_batch, is_training=True))
						#self._train_operation.eval(session=session, feed_dict=self._model.get_feed_dict(data_batch, is_training=True))
						summary, _ = session.run([self._merged_summary, self._train_operation], feed_dict=self._model.get_feed_dict(data_batch, is_training=True))
						if train_summary_writer is not None:
							train_summary_writer.add_summary(summary, epoch)
				if step / train_steps_per_epoch >= processing_ratio:
					print('-', sep='', end='')
					processing_ratio = round(step / train_steps_per_epoch, 2) + 0.05
			print('<')

			# Evaluate training.
			if True:
				"""
				batch_indices = indices[0:batch_size]
				if batch_indices.size > 0:  # If batch_indices is non-empty.
					# FIXME [fix] >> Does not work correctly in time-major data.
					data_batch = train_data[batch_indices]
					if data_batch.size > 0:  # If data_batch is non-empty.
						train_loss = self._loss.eval(session=session, feed_dict=self._model.get_feed_dict(data_batch, is_training=False))
				"""
				for step in range(train_steps_per_epoch):
					start = step * batch_size
					end = start + batch_size
					batch_indices = indices[start:end]
					if batch_indices.size > 0:  # If batch_indices is non-empty.
						# FIXME [fix] >> Does not work correctly in time-major data.
						data_batch = train_data[batch_indices]
						if data_batch.size > 0:  # If data_batch is non-empty.
							batch_loss = self._loss.eval(session=session, feed_dict=self._model.get_feed_dict(data_batch, is_training=False))

							# TODO [check] >> Is train_loss correct?
							train_loss += batch_loss * batch_indices.size
				train_loss /= num_train_examples

				history['loss'].append(train_loss)

			# Evaluate.
			#if val_data is not None:
			if num_val_examples > 0:
				"""
				data_batch = val_data
				if data_batch.size > 0:  # If data_batch is non-empty.
					#summary = self._merged_summary.eval(session=session, feed_dict=self._model.get_feed_dict(data_batch, is_training=False))
					#val_loss = self._loss.eval(session=session, feed_dict=self._model.get_feed_dict(data_batch, is_training=False))
					summary, val_loss = session.run([self._merged_summary, self._loss], feed_dict=self._model.get_feed_dict(data_batch, is_training=False))
					if val_summary_writer is not None:
						val_summary_writer.add_summary(summary, epoch)
				"""
				indices = np.arange(num_val_examples)
				#if shuffle:
				#	np.random.shuffle(indices)

				for step in range(val_steps_per_epoch):
					start = step * batch_size
					end = start + batch_size
					batch_indices = indices[start:end]
					if batch_indices.size > 0:  # If batch_indices is non-empty.
						# FIXME [fix] >> Does not work correctly in time-major data.
						data_batch = val_data[batch_indices]
						if data_batch.size > 0:  # If data_batch is non-empty.
							#summary = self._merged_summary.eval(session=session, feed_dict=self._model.get_feed_dict(data_batch, is_training=False))
							#batch_loss = self._loss.eval(session=session, feed_dict=self._model.get_feed_dict({data_batch, is_training=False))
							summary, batch_loss = session.run([self._merged_summary, self._loss], feed_dict=self._model.get_feed_dict(data_batch, is_training=False))
							if val_summary_writer is not None:
								val_summary_writer.add_summary(summary, epoch)

							# TODO [check] >> Is val_loss correct?
							val_loss += batch_loss * batch_indices.size
				val_loss /= num_val_examples

				history['val_loss'].append(val_loss)

				# Save a model.
				if saver is not None and model_save_dir_path is not None and val_loss <= best_val_loss:
					saved_model_path = saver.save(session, model_save_dir_path + '/model.ckpt', global_step=self._global_step)
					best_val_loss = val_loss

					print('[SWL] Info: Loss is improved and the model is saved at {}.'.format(saved_model_path))

			print('\tElapsed time = {}'.format(time.time() - start_time))
			print('\tTraining:   loss = {}'.format(train_loss))
			print('\tValidation: loss = {}'.format(val_loss))

		# Close writers.
		if train_summary_writer is not None:
			train_summary_writer.close()
		if val_summary_writer is not None:
			val_summary_writer.close()

		return history

	def _get_train_operation(self, loss, global_step=None):
		with tf.name_scope('train_op'):
			train_op = self._optimizer.minimize(loss, global_step=global_step)
			return train_op

#%%------------------------------------------------------------------

class GradientClippingNeuralNetTrainer(NeuralNetTrainer):
	def __init__(self, neuralNet, optimizer, max_gradient_norm, global_step=None, augmenter=None, is_output_augmented=False):
		self._max_gradient_norm = max_gradient_norm
		super().__init__(neuralNet, optimizer, global_step, augmenter, is_output_augmented)

	def _get_train_operation(self, loss, global_step=None):
		with tf.name_scope('train_op'):
			# Method 1.
			gradients = self._optimizer.compute_gradients(loss)
			for i, (g, v) in enumerate(gradients):
				if g is not None:
					gradients[i] = (tf.clip_by_norm(g, self._max_gradient_norm), v)  # Clip gradients.
			train_op = self._optimizer.apply_gradients(gradients, global_step=global_step)
			"""
			# Method 2.
			#	REF [site] >> https://www.tensorflow.org/tutorials/seq2seq
			params = tf.trainable_variables()
			gradients = tf.gradients(loss, params)
			clipped_gradients, _ = tf.clip_by_global_norm(gradients, self._max_gradient_norm)  # Clip gradients.
			train_op = self._optimizer.apply_gradients(zip(clipped_gradients, params), global_step=global_step)
			"""
			return train_op
