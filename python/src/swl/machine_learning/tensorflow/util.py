import time
import numpy as np
import tensorflow as tf
import swl.machine_learning.util as swl_ml_util

#%%------------------------------------------------------------------

# Supports lists of dense or sparse outputs.
def train_neural_net_by_batch_generator(session, nnTrainer, trainBatchGenerator, valBatchGenerator, num_epochs, does_resume_training, saver, output_dir_path, checkpoint_dir_path, train_summary_dir_path, val_summary_dir_path, is_time_major, is_sparse_output):
	if does_resume_training:
		print('[SWL] Info: Resume training...')

		# Load a model.
		# REF [site] >> https://www.tensorflow.org/programmers_guide/saved_model
		# REF [site] >> http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir_path)
		saver.restore(session, ckpt.model_checkpoint_path)
		#saver.restore(session, tf.train.latest_checkpoint(checkpoint_dir_path))
		print('[SWL] Info: Restored a model.')
	else:
		print('[SWL] Info: Start training...')

	# Create writers to write all the summaries out to a directory.
	train_summary_writer = tf.summary.FileWriter(train_summary_dir_path, session.graph) if train_summary_dir_path is not None else None
	val_summary_writer = tf.summary.FileWriter(val_summary_dir_path) if val_summary_dir_path is not None else None

	history = {
		'acc': [],
		'loss': [],
		'val_acc': [],
		'val_loss': []
	}

	batch_axis = 1 if is_time_major else 0

	best_val_acc = 0.0
	for epoch in range(1, num_epochs + 1):
		print('Epoch {}/{}'.format(epoch, num_epochs))

		start_time = time.time()

		#--------------------
		print('>-', sep='', end='')
		step = 0
		train_loss, train_acc, num_train_examples = 0.0, 0.0, 0
		batches = trainBatchGenerator.generateBatches()  # Generates and augments batches.
		for batch_data, num_batch_examples in batches:
			batch_acc, batch_loss = nnTrainer.train_by_batch(session, batch_data[0], batch_data[1], train_summary_writer, is_time_major, is_sparse_output)

			# TODO [check] >> Are these calculations correct?
			train_acc += batch_acc * num_batch_examples
			train_loss += batch_loss * num_batch_examples
			num_train_examples += num_batch_examples

			step += 1
			if 0 == step % 10:
				print('-', sep='', end='')
		print('<')

		train_acc /= num_train_examples
		train_loss /= num_train_examples

		#--------------------
		val_loss, val_acc, num_val_examples = 0.0, 0.0, 0
		batches = valBatchGenerator.generateBatches()  # Generates and augments batches.
		for batch_data, num_batch_examples in batches:
			batch_acc, batch_loss = nnTrainer.evaluate_training_by_batch(session, batch_data[0], batch_data[1], val_summary_writer, is_time_major, is_sparse_output)

			# TODO [check] >> Are these calculations correct?
			val_acc += batch_acc * num_batch_examples
			val_loss += batch_loss * num_batch_examples
			num_val_examples += num_batch_examples

		val_acc /= num_val_examples
		val_loss /= num_val_examples

		print('\tTraining time = {}'.format(time.time() - start_time))
		print('\tTraining:   loss = {}, accuracy = {}'.format(train_loss, train_acc))
		print('\tValidation: loss = {}, accurary = {}'.format(val_loss, val_acc))

		history['acc'].append(train_acc)
		history['loss'].append(train_loss)
		history['val_acc'].append(val_acc)
		history['val_loss'].append(val_loss)

		# Save a model.
		if saver is not None and checkpoint_dir_path is not None and val_acc >= best_val_acc:
			saved_model_path = saver.save(session, checkpoint_dir_path + '/model.ckpt', global_step=nnTrainer.global_step)
			best_val_acc = val_acc
			print('[SWL] Info: Accurary is improved and the model is saved at {}.'.format(saved_model_path))

	# Close writers.
	if train_summary_writer is not None:
		train_summary_writer.close()
	if val_summary_writer is not None:
		val_summary_writer.close()

	#--------------------
	# Display results.
	#swl_ml_util.display_train_history(history)
	if output_dir_path is not None:
		swl_ml_util.save_train_history(history, output_dir_path)
	print('[SWL] Info: End training...')

	"""
	# Save a graph.
	tf.train.write_graph(session.graph_def, output_dir_path, 'graph.pb', as_text=False)
	#tf.train.write_graph(session.graph_def, output_dir_path, 'graph.pbtxt', as_text=True)

	# Save a serving model.
	builder = tf.saved_model.builder.SavedModelBuilder(output_dir_path + '/serving_model')
	builder.add_meta_graph_and_variables(session, [tf.saved_model.tag_constants.SERVING], saver=saver)
	builder.save(as_text=False)
	"""

# Supports lists of dense or sparse outputs.
def train_neural_net_by_file_batch_loader(session, nnTrainer, trainFileBatchLoader, valFileBatchLoader, trainDirMgr, valDirMgr, num_epochs, does_resume_training, saver, output_dir_path, checkpoint_dir_path, train_summary_dir_path, val_summary_dir_path, is_time_major, is_sparse_output):
	if does_resume_training:
		print('[SWL] Info: Resume training...')

		# Load a model.
		# REF [site] >> https://www.tensorflow.org/programmers_guide/saved_model
		# REF [site] >> http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir_path)
		saver.restore(session, ckpt.model_checkpoint_path)
		#saver.restore(session, tf.train.latest_checkpoint(checkpoint_dir_path))
		print('[SWL] Info: Restored a model.')
	else:
		print('[SWL] Info: Start training...')

	# Create writers to write all the summaries out to a directory.
	train_summary_writer = tf.summary.FileWriter(train_summary_dir_path, session.graph) if train_summary_dir_path is not None else None
	val_summary_writer = tf.summary.FileWriter(val_summary_dir_path) if val_summary_dir_path is not None else None

	history = {
		'acc': [],
		'loss': [],
		'val_acc': [],
		'val_loss': []
	}

	batch_axis = 1 if is_time_major else 0

	best_val_acc = 0.0
	for epoch in range(1, num_epochs + 1):
		print('Epoch {}/{}'.format(epoch, num_epochs))

		print('\tWaiting for a train batch directory...')
		while True:
			train_dir_path = trainDirMgr.requestDirectory()
			if train_dir_path is not None:
				break
			else:
				time.sleep(0.1)
		print('\tGot a train batch directory: {}.'.format(train_dir_path))

		print('\tWaiting for a validation batch directory...')
		while True:
			val_dir_path = valDirMgr.requestDirectory()
			if val_dir_path is not None:
				break
			else:
				time.sleep(0.1)
		print('\tGot a validation batch directory: {}.'.format(val_dir_path))

		start_time = time.time()

		#--------------------
		print('>-', sep='', end='')
		step = 0
		train_loss, train_acc, num_train_examples = 0.0, 0.0, 0
		batches = trainFileBatchLoader.loadBatches(train_dir_path)  # Loads batches.
		for batch_data, num_batch_examples in batches:
			batch_acc, batch_loss = nnTrainer.train_by_batch(session, batch_data[0], batch_data[1], train_summary_writer, is_time_major, is_sparse_output)

			# TODO [check] >> Are these calculations correct?
			train_acc += batch_acc * num_batch_examples
			train_loss += batch_loss * num_batch_examples
			num_train_examples += num_batch_examples

			step += 1
			if 0 == step % 10:
				print('-', sep='', end='')
		print('<')

		trainDirMgr.returnDirectory(train_dir_path)				

		train_acc /= num_train_examples
		train_loss /= num_train_examples

		#--------------------
		val_loss, val_acc, num_val_examples = 0.0, 0.0, 0
		batches = valFileBatchLoader.loadBatches(val_dir_path)  # Loads batches.
		for batch_data, num_batch_examples in batches:
			batch_acc, batch_loss = nnTrainer.evaluate_training_by_batch(session, batch_data[0], batch_data[1], val_summary_writer, is_time_major, is_sparse_output)

			# TODO [check] >> Are these calculations correct?
			val_acc += batch_acc * num_batch_examples
			val_loss += batch_loss * num_batch_examples
			num_val_examples += num_batch_examples

		valDirMgr.returnDirectory(val_dir_path)				

		val_acc /= num_val_examples
		val_loss /= num_val_examples

		print('\tTraining time = {}'.format(time.time() - start_time))
		print('\tTraining:   loss = {}, accuracy = {}'.format(train_loss, train_acc))
		print('\tValidation: loss = {}, accurary = {}'.format(val_loss, val_acc))

		history['acc'].append(train_acc)
		history['loss'].append(train_loss)
		history['val_acc'].append(val_acc)
		history['val_loss'].append(val_loss)

		# Save a model.
		if saver is not None and checkpoint_dir_path is not None and val_acc >= best_val_acc:
			saved_model_path = saver.save(session, checkpoint_dir_path + '/model.ckpt', global_step=nnTrainer.global_step)
			best_val_acc = val_acc
			print('[SWL] Info: Accurary is improved and the model is saved at {}.'.format(saved_model_path))

	# Close writers.
	if train_summary_writer is not None:
		train_summary_writer.close()
	if val_summary_writer is not None:
		val_summary_writer.close()

	#--------------------
	# Display results.
	#swl_ml_util.display_train_history(history)
	if output_dir_path is not None:
		swl_ml_util.save_train_history(history, output_dir_path)
	print('[SWL] Info: End training...')

	"""
	# Save a graph.
	tf.train.write_graph(session.graph_def, output_dir_path, 'graph.pb', as_text=False)
	#tf.train.write_graph(session.graph_def, output_dir_path, 'graph.pbtxt', as_text=True)

	# Save a serving model.
	builder = tf.saved_model.builder.SavedModelBuilder(output_dir_path + '/serving_model')
	builder.add_meta_graph_and_variables(session, [tf.saved_model.tag_constants.SERVING], saver=saver)
	builder.save(as_text=False)
	"""

# NOTE [info] >> Use train_neural_net_by_batch_generator().
# Supports lists of dense or sparse outputs.
def train_neural_net_by_batch_manager(session, nnTrainer, trainBatchMgr, valBatchMgr, num_epochs, does_resume_training, saver, output_dir_path, checkpoint_dir_path, train_summary_dir_path, val_summary_dir_path, is_time_major, is_sparse_output):
	if does_resume_training:
		print('[SWL] Info: Resume training...')

		# Load a model.
		# REF [site] >> https://www.tensorflow.org/programmers_guide/saved_model
		# REF [site] >> http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir_path)
		saver.restore(session, ckpt.model_checkpoint_path)
		#saver.restore(session, tf.train.latest_checkpoint(checkpoint_dir_path))
		print('[SWL] Info: Restored a model.')
	else:
		print('[SWL] Info: Start training...')

	# Create writers to write all the summaries out to a directory.
	train_summary_writer = tf.summary.FileWriter(train_summary_dir_path, session.graph) if train_summary_dir_path is not None else None
	val_summary_writer = tf.summary.FileWriter(val_summary_dir_path) if val_summary_dir_path is not None else None

	history = {
		'acc': [],
		'loss': [],
		'val_acc': [],
		'val_loss': []
	}

	batch_axis = 1 if is_time_major else 0

	best_val_acc = 0.0
	for epoch in range(1, num_epochs + 1):
		print('Epoch {}/{}'.format(epoch, num_epochs))

		start_time = time.time()

		#--------------------
		print('>-', sep='', end='')
		step = 0
		train_loss, train_acc, num_train_examples = 0.0, 0.0, 0
		batches = trainBatchMgr.getBatches()  # Generates and augments batches.
		for train_inputs, train_outputs in batches:
			batch_acc, batch_loss = nnTrainer.train_by_batch(session, train_inputs, train_outputs, train_summary_writer, is_time_major, is_sparse_output)

			# TODO [check] >> Are these calculations correct?
			batch_size = train_inputs.shape[batch_axis]
			train_acc += batch_acc * batch_size
			train_loss += batch_loss * batch_size
			num_train_examples += batch_size

			step += 1
			if 0 == step % 10:
				print('-', sep='', end='')
		print('<')

		train_acc /= num_train_examples
		train_loss /= num_train_examples

		#--------------------
		val_loss, val_acc, num_val_examples = 0.0, 0.0, 0
		batches = valBatchMgr.getBatches()  # Generates and augments batches.
		for val_inputs, val_outputs in batches:
			batch_acc, batch_loss = nnTrainer.evaluate_training_by_batch(session, val_inputs, val_outputs, val_summary_writer, is_time_major, is_sparse_output)

			# TODO [check] >> Are these calculations correct?
			batch_size = val_inputs.shape[batch_axis]
			val_acc += batch_acc * batch_size
			val_loss += batch_loss * batch_size
			num_val_examples += batch_size

		val_acc /= num_val_examples
		val_loss /= num_val_examples

		print('\tTraining time = {}'.format(time.time() - start_time))
		print('\tTraining:   loss = {}, accuracy = {}'.format(train_loss, train_acc))
		print('\tValidation: loss = {}, accurary = {}'.format(val_loss, val_acc))

		history['acc'].append(train_acc)
		history['loss'].append(train_loss)
		history['val_acc'].append(val_acc)
		history['val_loss'].append(val_loss)

		# Save a model.
		if saver is not None and checkpoint_dir_path is not None and val_acc >= best_val_acc:
			saved_model_path = saver.save(session, checkpoint_dir_path + '/model.ckpt', global_step=nnTrainer.global_step)
			best_val_acc = val_acc
			print('[SWL] Info: Accurary is improved and the model is saved at {}.'.format(saved_model_path))

	# Close writers.
	if train_summary_writer is not None:
		train_summary_writer.close()
	if val_summary_writer is not None:
		val_summary_writer.close()

	#--------------------
	# Display results.
	#swl_ml_util.display_train_history(history)
	if output_dir_path is not None:
		swl_ml_util.save_train_history(history, output_dir_path)
	print('[SWL] Info: End training...')

	"""
	# Save a graph.
	tf.train.write_graph(session.graph_def, output_dir_path, 'graph.pb', as_text=False)
	#tf.train.write_graph(session.graph_def, output_dir_path, 'graph.pbtxt', as_text=True)

	# Save a serving model.
	builder = tf.saved_model.builder.SavedModelBuilder(output_dir_path + '/serving_model')
	builder.add_meta_graph_and_variables(session, [tf.saved_model.tag_constants.SERVING], saver=saver)
	builder.save(as_text=False)
	"""

# NOTE [info] >> Use train_neural_net_by_file_batch_loader().
# Supports lists of dense or sparse outputs.
def train_neural_net_by_file_batch_manager(session, nnTrainer, trainFileBatchMgr, valFileBatchMgr, trainDirMgr, valDirMgr, num_epochs, does_resume_training, saver, output_dir_path, checkpoint_dir_path, train_summary_dir_path, val_summary_dir_path, is_time_major, is_sparse_output):
	if does_resume_training:
		print('[SWL] Info: Resume training...')

		# Load a model.
		# REF [site] >> https://www.tensorflow.org/programmers_guide/saved_model
		# REF [site] >> http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir_path)
		saver.restore(session, ckpt.model_checkpoint_path)
		#saver.restore(session, tf.train.latest_checkpoint(checkpoint_dir_path))
		print('[SWL] Info: Restored a model.')
	else:
		print('[SWL] Info: Start training...')

	# Create writers to write all the summaries out to a directory.
	train_summary_writer = tf.summary.FileWriter(train_summary_dir_path, session.graph) if train_summary_dir_path is not None else None
	val_summary_writer = tf.summary.FileWriter(val_summary_dir_path) if val_summary_dir_path is not None else None

	history = {
		'acc': [],
		'loss': [],
		'val_acc': [],
		'val_loss': []
	}

	batch_axis = 1 if is_time_major else 0

	best_val_acc = 0.0
	for epoch in range(1, num_epochs + 1):
		print('Epoch {}/{}'.format(epoch, num_epochs))

		print('\tWaiting for a train batch directory...')
		while True:
			train_dir_path = trainDirMgr.requestAvailableDirectory()
			if train_dir_path is not None:
				break
			else:
				time.sleep(0.1)
		print('\tGot a train batch directory: {}.'.format(train_dir_path))

		print('\tWaiting for a validation batch directory...')
		while True:
			val_dir_path = valDirMgr.requestAvailableDirectory()
			if val_dir_path is not None:
				break
			else:
				time.sleep(0.1)
		print('\tGot a validation batch directory: {}.'.format(val_dir_path))

		start_time = time.time()

		#--------------------
		trainFileBatchMgr.putBatches(train_dir_path)  # Generates, augments, and saves batches.

		print('>-', sep='', end='')
		step = 0
		train_loss, train_acc, num_train_examples = 0.0, 0.0, 0
		batches = trainFileBatchMgr.getBatches(train_dir_path)  # Loads batches.
		for train_inputs, train_outputs in batches:
			batch_acc, batch_loss = nnTrainer.train_by_batch(session, train_inputs, train_outputs, train_summary_writer, is_time_major, is_sparse_output)

			# TODO [check] >> Are these calculations correct?
			batch_size = train_inputs.shape[batch_axis]
			train_acc += batch_acc * batch_size
			train_loss += batch_loss * batch_size
			num_train_examples += batch_size

			step += 1
			if 0 == step % 10:
				print('-', sep='', end='')
		print('<')

		trainDirMgr.returnDirectory(train_dir_path)				

		train_acc /= num_train_examples
		train_loss /= num_train_examples

		#--------------------
		valFileBatchMgr.putBatches(val_dir_path)  # Generates, augments, and saves batches.

		val_loss, val_acc, num_val_examples = 0.0, 0.0, 0
		batches = valFileBatchMgr.getBatches(val_dir_path)  # Loads batches.
		for val_inputs, val_outputs in batches:
			batch_acc, batch_loss = nnTrainer.evaluate_training_by_batch(session, val_inputs, val_outputs, val_summary_writer, is_time_major, is_sparse_output)

			# TODO [check] >> Are these calculations correct?
			batch_size = val_inputs.shape[batch_axis]
			val_acc += batch_acc * batch_size
			val_loss += batch_loss * batch_size
			num_val_examples += batch_size

		valDirMgr.returnDirectory(val_dir_path)				

		val_acc /= num_val_examples
		val_loss /= num_val_examples

		print('\tTraining time = {}'.format(time.time() - start_time))
		print('\tTraining:   loss = {}, accuracy = {}'.format(train_loss, train_acc))
		print('\tValidation: loss = {}, accurary = {}'.format(val_loss, val_acc))

		history['acc'].append(train_acc)
		history['loss'].append(train_loss)
		history['val_acc'].append(val_acc)
		history['val_loss'].append(val_loss)

		# Save a model.
		if saver is not None and checkpoint_dir_path is not None and val_acc >= best_val_acc:
			saved_model_path = saver.save(session, checkpoint_dir_path + '/model.ckpt', global_step=nnTrainer.global_step)
			best_val_acc = val_acc
			print('[SWL] Info: Accurary is improved and the model is saved at {}.'.format(saved_model_path))

	# Close writers.
	if train_summary_writer is not None:
		train_summary_writer.close()
	if val_summary_writer is not None:
		val_summary_writer.close()

	#--------------------
	# Display results.
	#swl_ml_util.display_train_history(history)
	if output_dir_path is not None:
		swl_ml_util.save_train_history(history, output_dir_path)
	print('[SWL] Info: End training...')

	"""
	# Save a graph.
	tf.train.write_graph(session.graph_def, output_dir_path, 'graph.pb', as_text=False)
	#tf.train.write_graph(session.graph_def, output_dir_path, 'graph.pbtxt', as_text=True)

	# Save a serving model.
	builder = tf.saved_model.builder.SavedModelBuilder(output_dir_path + '/serving_model')
	builder.add_meta_graph_and_variables(session, [tf.saved_model.tag_constants.SERVING], saver=saver)
	builder.save(as_text=False)
	"""

# Supports lists of dense or sparse outputs.
def train_neural_net_by_batch_list(session, nnTrainer, train_inputs_list, train_outputs_list, val_inputs_list, val_outputs_list, num_epochs, shuffle, does_resume_training, saver, output_dir_path, checkpoint_dir_path, train_summary_dir_path, val_summary_dir_path, is_time_major, is_sparse_output):
	num_train_batches, num_val_batches = len(train_inputs_list), len(val_inputs_list)
	if len(train_outputs_list) != num_train_batches or len(val_outputs_list) != num_val_batches:
		raise ValueError('Invalid parameter length')

	if does_resume_training:
		print('[SWL] Info: Resume training...')

		# Load a model.
		# REF [site] >> https://www.tensorflow.org/programmers_guide/saved_model
		# REF [site] >> http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir_path)
		saver.restore(session, ckpt.model_checkpoint_path)
		#saver.restore(session, tf.train.latest_checkpoint(checkpoint_dir_path))
		print('[SWL] Info: Restored a model.')
	else:
		print('[SWL] Info: Start training...')

	# Create writers to write all the summaries out to a directory.
	train_summary_writer = tf.summary.FileWriter(train_summary_dir_path, session.graph) if train_summary_dir_path is not None else None
	val_summary_writer = tf.summary.FileWriter(val_summary_dir_path) if val_summary_dir_path is not None else None

	history = {
		'acc': [],
		'loss': [],
		'val_acc': [],
		'val_loss': []
	}

	batch_axis = 1 if is_time_major else 0

	best_val_acc = 0.0
	for epoch in range(1, num_epochs + 1):
		print('Epoch {}/{}'.format(epoch, num_epochs))

		start_time = time.time()

		#--------------------
		indices = np.arange(num_train_batches)
		if shuffle:
			np.random.shuffle(indices)

		print('>-', sep='', end='')
		processing_ratio = 0.05
		train_loss, train_acc, num_train_examples = 0.0, 0.0, 0
		for step in indices:
			train_inputs, train_outputs = train_inputs_list[step], train_outputs_list[step]
			batch_acc, batch_loss = nnTrainer.train_by_batch(session, train_inputs, train_outputs, train_summary_writer, is_time_major, is_sparse_output)

			# TODO [check] >> Are these calculations correct?
			batch_size = train_inputs.shape[batch_axis]
			train_acc += batch_acc * batch_size
			train_loss += batch_loss * batch_size
			num_train_examples += batch_size

			if step / num_train_batches >= processing_ratio:
				print('-', sep='', end='')
				processing_ratio = round(step / num_train_batches, 2) + 0.05
		print('<')

		train_acc /= num_train_examples
		train_loss /= num_train_examples

		#--------------------
		indices = np.arange(num_val_batches)
		#if shuffle:
		#	np.random.shuffle(indices)

		val_loss, val_acc, num_val_examples = 0.0, 0.0, 0
		for step in indices:
			val_inputs, val_outputs = val_inputs_list[step], val_outputs_list[step]
			batch_acc, batch_loss = nnTrainer.evaluate_training_by_batch(session, val_inputs, val_outputs, val_summary_writer, is_time_major, is_sparse_output)

			# TODO [check] >> Are these calculations correct?
			batch_size = val_inputs.shape[batch_axis]
			val_acc += batch_acc * batch_size
			val_loss += batch_loss * batch_size
			num_val_examples += batch_size

		val_acc /= num_val_examples
		val_loss /= num_val_examples

		print('\tTraining time = {}'.format(time.time() - start_time))
		print('\tTraining:   loss = {}, accuracy = {}'.format(train_loss, train_acc))
		print('\tValidation: loss = {}, accurary = {}'.format(val_loss, val_acc))

		history['acc'].append(train_acc)
		history['loss'].append(train_loss)
		history['val_acc'].append(val_acc)
		history['val_loss'].append(val_loss)

		# Save a model.
		if saver is not None and checkpoint_dir_path is not None and val_acc >= best_val_acc:
			saved_model_path = saver.save(session, checkpoint_dir_path + '/model.ckpt', global_step=nnTrainer.global_step)
			best_val_acc = val_acc
			print('[SWL] Info: Accurary is improved and the model is saved at {}.'.format(saved_model_path))

	# Close writers.
	if train_summary_writer is not None:
		train_summary_writer.close()
	if val_summary_writer is not None:
		val_summary_writer.close()

	#--------------------
	# Display results.
	#swl_ml_util.display_train_history(history)
	if output_dir_path is not None:
		swl_ml_util.save_train_history(history, output_dir_path)
	print('[SWL] Info: End training...')

	"""
	# Save a graph.
	tf.train.write_graph(session.graph_def, output_dir_path, 'graph.pb', as_text=False)
	#tf.train.write_graph(session.graph_def, output_dir_path, 'graph.pbtxt', as_text=True)

	# Save a serving model.
	builder = tf.saved_model.builder.SavedModelBuilder(output_dir_path + '/serving_model')
	builder.add_meta_graph_and_variables(session, [tf.saved_model.tag_constants.SERVING], saver=saver)
	builder.save(as_text=False)
	"""

# Supports a dense output only.
def train_neural_net_after_generating_batch_list(session, nnTrainer, train_inputs, train_outputs, val_inputs, val_outputs, batch_size, num_epochs, shuffle, does_resume_training, saver, output_dir_path, checkpoint_dir_path, train_summary_dir_path, val_summary_dir_path, is_time_major):
	batch_axis = 1 if is_time_major else 0

	num_train_examples, num_train_steps = 0, 0
	if train_inputs is not None and train_outputs is not None:
		if train_inputs.shape[batch_axis] == train_outputs.shape[batch_axis]:
			num_train_examples = train_inputs.shape[batch_axis]
		num_train_steps = ((num_train_examples - 1) // batch_size + 1) if num_train_examples > 0 else 0
	num_val_examples, num_val_steps = 0, 0
	if val_inputs is not None and val_outputs is not None:
		if val_inputs.shape[batch_axis] == val_outputs.shape[batch_axis]:
			num_val_examples = val_inputs.shape[batch_axis]
		num_val_steps = ((num_val_examples - 1) // batch_size + 1) if num_val_examples > 0 else 0

	#--------------------
	indices = np.arange(num_train_examples)
	if shuffle:
		np.random.shuffle(indices)

	train_inputs_list, train_outputs_list = list(), list()
	for step in range(num_train_steps):
		start = step * batch_size
		end = start + batch_size
		batch_indices = indices[start:end]
		if batch_indices.size > 0:  # If batch_indices is non-empty.
			# FIXME [fix] >> Does not work correctly in time-major data.
			train_inputs_list.append(train_inputs[batch_indices])
			train_outputs_list.append(train_outputs[batch_indices])

	#--------------------
	indices = np.arange(num_val_examples)
	#if shuffle:
	#	np.random.shuffle(indices)

	val_inputs_list, val_outputs_list = list(), list()
	for step in range(num_val_steps):
		start = step * batch_size
		end = start + batch_size
		batch_indices = indices[start:end]
		if batch_indices.size > 0:  # If batch_indices is non-empty.
			# FIXME [fix] >> Does not work correctly in time-major data.
			val_inputs_list.append(val_inputs[batch_indices])
			val_outputs_list.append(val_outputs[batch_indices])

	train_neural_net_by_batch_list(session, nnTrainer, train_inputs_list, train_outputs_list, val_inputs_list, val_outputs_list, num_epochs, shuffle, does_resume_training, saver, output_dir_path, checkpoint_dir_path, train_summary_dir_path, val_summary_dir_path, is_time_major, False)

# Supports a dense output only.
def train_neural_net(session, nnTrainer, train_inputs, train_outputs, val_inputs, val_outputs, batch_size, num_epochs, shuffle, does_resume_training, saver, output_dir_path, checkpoint_dir_path, train_summary_dir_path, val_summary_dir_path):
	if does_resume_training:
		print('[SWL] Info: Resume training...')

		# Load a model.
		# REF [site] >> https://www.tensorflow.org/programmers_guide/saved_model
		# REF [site] >> http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir_path)
		saver.restore(session, ckpt.model_checkpoint_path)
		#saver.restore(session, tf.train.latest_checkpoint(checkpoint_dir_path))
		print('[SWL] Info: Restored a model.')
	else:
		print('[SWL] Info: Start training...')

	start_time = time.time()
	history = nnTrainer.train(session, train_inputs, train_outputs, val_inputs, val_outputs, batch_size, num_epochs, shuffle, saver=saver, model_save_dir_path=checkpoint_dir_path, train_summary_dir_path=train_summary_dir_path, val_summary_dir_path=val_summary_dir_path)
	print('\tTraining time = {}'.format(time.time() - start_time))

	#--------------------
	# Display results.
	#swl_ml_util.display_train_history(history)
	if output_dir_path is not None:
		swl_ml_util.save_train_history(history, output_dir_path)
	print('[SWL] Info: End training...')

	"""
	# Save a graph.
	tf.train.write_graph(session.graph_def, output_dir_path, 'graph.pb', as_text=False)
	#tf.train.write_graph(session.graph_def, output_dir_path, 'graph.pbtxt', as_text=True)

	# Save a serving model.
	builder = tf.saved_model.builder.SavedModelBuilder(output_dir_path + '/serving_model')
	builder.add_meta_graph_and_variables(session, [tf.saved_model.tag_constants.SERVING], saver=saver)
	builder.save(as_text=False)
	"""

def train_neural_net_with_decoder_input(session, nnTrainer, train_encoder_input_seqs, train_decoder_input_seqs, train_decoder_output_seqs, val_encoder_input_seqs, val_decoder_input_seqs, val_decoder_output_seqs, batch_size, num_epochs, shuffle, does_resume_training, saver, output_dir_path, checkpoint_dir_path, train_summary_dir_path, val_summary_dir_path):
	if does_resume_training:
		print('[SWL] Info: Resume training...')

		# Load a model.
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir_path)
		saver.restore(session, ckpt.model_checkpoint_path)
		#saver.restore(session, tf.train.latest_checkpoint(checkpoint_dir_path))
		print('[SWL] Info: Restored a model.')
	else:
		print('[SWL] Info: Start training...')

	start_time = time.time()
	history = nnTrainer.train_seq2seq(session, train_encoder_input_seqs, train_decoder_input_seqs, train_decoder_output_seqs, val_encoder_input_seqs, val_decoder_input_seqs, val_decoder_output_seqs, batch_size, num_epochs, shuffle, saver=saver, model_save_dir_path=checkpoint_dir_path, train_summary_dir_path=train_summary_dir_path, val_summary_dir_path=val_summary_dir_path)
	print('\tTraining time = {}'.format(time.time() - start_time))

	#--------------------
	# Display results.
	#swl_ml_util.display_train_history(history)
	if output_dir_path is not None:
		swl_ml_util.save_train_history(history, output_dir_path)
	print('[SWL] Info: End training...')

	"""
	# Save a graph.
	tf.train.write_graph(session.graph_def, output_dir_path, 'graph.pb', as_text=False)
	#tf.train.write_graph(session.graph_def, output_dir_path, 'graph.pbtxt', as_text=True)

	# Save a serving model.
	builder = tf.saved_model.builder.SavedModelBuilder(output_dir_path + '/serving_model')
	builder.add_meta_graph_and_variables(session, [tf.saved_model.tag_constants.SERVING], saver=saver)
	builder.save(as_text=False)
	"""

def train_neural_net_unsupervisedly(session, nnTrainer, train_inputs, val_inputs, batch_size, num_epochs, shuffle, does_resume_training, saver, output_dir_path, checkpoint_dir_path, train_summary_dir_path):
	if does_resume_training:
		print('[SWL] Info: Resume training...')

		# Load a model.
		# REF [site] >> https://www.tensorflow.org/programmers_guide/saved_model
		# REF [site] >> http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir_path)
		saver.restore(session, ckpt.model_checkpoint_path)
		#saver.restore(session, tf.train.latest_checkpoint(checkpoint_dir_path))
		print('[SWL] Info: Restored a model.')
	else:
		print('[SWL] Info: Start training...')

	start_time = time.time()
	history = nnTrainer.train_unsupervisedly(session, train_inputs, val_inputs, batch_size, num_epochs, shuffle, saver=saver, model_save_dir_path=checkpoint_dir_path, train_summary_dir_path=train_summary_dir_path)
	print('\tTraining time = {}'.format(time.time() - start_time))

	#--------------------
	# Display results.
	#swl_ml_util.display_train_history(history)
	if output_dir_path is not None:
		swl_ml_util.save_train_history(history, output_dir_path)
	print('[SWL] Info: End training...')

	"""
	# Save a graph.
	tf.train.write_graph(session.graph_def, output_dir_path, 'mnist_draw_graph.pb', as_text=False)
	#tf.train.write_graph(session.graph_def, output_dir_path, 'mnist_draw_graph.pbtxt', as_text=True)

	# Save a serving model.
	builder = tf.saved_model.builder.SavedModelBuilder(output_dir_path + '/serving_model')
	builder.add_meta_graph_and_variables(session, [tf.saved_model.tag_constants.SERVING], saver=saver)
	builder.save(as_text=False)
	"""

#%%------------------------------------------------------------------

# Supports lists of dense or sparse outputs.
def evaluate_neural_net_by_batch_generator(session, nnEvaluator, valBatchGenerator, saver=None, checkpoint_dir_path=None, is_time_major=False, is_sparse_output=False):
	batch_axis = 1 if is_time_major else 0

	if saver is not None and checkpoint_dir_path is not None:
		# Load a model.
		# REF [site] >> https://www.tensorflow.org/programmers_guide/saved_model
		# REF [site] >> http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir_path)
		saver.restore(session, ckpt.model_checkpoint_path)
		#saver.restore(session, tf.train.latest_checkpoint(checkpoint_dir_path))
		print('[SWL] Info: Loaded a model.')

	print('[SWL] Info: Start evaluation...')
	start_time = time.time()

	val_loss, val_acc, num_val_examples = 0.0, 0.0, 0
	batches = valBatchGenerator.generateBatches()  # Generates and augments batches.
	for batch_data, num_batch_examples in batches:
		batch_acc, batch_loss = nnEvaluator.evaluate_by_batch(session, batch_data[0], batch_data[1], is_time_major, is_sparse_output)

		# TODO [check] >> Are these calculations correct?
		val_acc += batch_acc * num_batch_examples
		val_loss += batch_loss * num_batch_examples
		num_val_examples += num_batch_examples

	val_acc /= num_val_examples
	val_loss /= num_val_examples

	print('\tEvaluation time = {}'.format(time.time() - start_time))
	print('\tValidation: loss = {}, accurary = {}'.format(val_loss, val_acc))
	print('[SWL] Info: End evaluation...')

# Supports lists of dense or sparse outputs.
def evaluate_neural_net_by_file_batch_loader(session, nnEvaluator, valFileBatchLoader, valDirMgr, saver=None, checkpoint_dir_path=None, is_time_major=False, is_sparse_output=False):
	batch_axis = 1 if is_time_major else 0

	if saver is not None and checkpoint_dir_path is not None:
		# Load a model.
		# REF [site] >> https://www.tensorflow.org/programmers_guide/saved_model
		# REF [site] >> http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir_path)
		saver.restore(session, ckpt.model_checkpoint_path)
		#saver.restore(session, tf.train.latest_checkpoint(checkpoint_dir_path))
		print('[SWL] Info: Loaded a model.')

	print('\tWaiting for a validation batch directory...')
	while True:
		val_dir_path = valDirMgr.requestDirectory()
		if val_dir_path is not None:
			break
		else:
			time.sleep(0.1)
	print('\tGot a validation batch directory: {}.'.format(val_dir_path))

	print('[SWL] Info: Start evaluation...')
	start_time = time.time()

	val_loss, val_acc, num_val_examples = 0.0, 0.0, 0
	batches = valFileBatchLoader.loadBatches(val_dir_path)  # Loads batches.
	for batch_data, num_batch_examples in batches:
		batch_acc, batch_loss = nnEvaluator.evaluate_by_batch(session, batch_data[0], batch_data[1], is_time_major, is_sparse_output)

		# TODO [check] >> Are these calculations correct?
		val_acc += batch_acc * num_batch_examples
		val_loss += batch_loss * num_batch_examples
		num_val_examples += num_batch_examples

	valDirMgr.returnDirectory(val_dir_path)				

	val_acc /= num_val_examples
	val_loss /= num_val_examples

	print('\tEvaluation time = {}'.format(time.time() - start_time))
	print('\tValidation: loss = {}, accurary = {}'.format(val_loss, val_acc))
	print('[SWL] Info: End evaluation...')

# NOTE [info] >> Use evaluate_neural_net_by_batch_generator().
# Supports lists of dense or sparse outputs.
def evaluate_neural_net_by_batch_manager(session, nnEvaluator, valBatchMgr, saver=None, checkpoint_dir_path=None, is_time_major=False, is_sparse_output=False):
	batch_axis = 1 if is_time_major else 0

	if saver is not None and checkpoint_dir_path is not None:
		# Load a model.
		# REF [site] >> https://www.tensorflow.org/programmers_guide/saved_model
		# REF [site] >> http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir_path)
		saver.restore(session, ckpt.model_checkpoint_path)
		#saver.restore(session, tf.train.latest_checkpoint(checkpoint_dir_path))
		print('[SWL] Info: Loaded a model.')

	print('[SWL] Info: Start evaluation...')
	start_time = time.time()

	val_loss, val_acc, num_val_examples = 0.0, 0.0, 0
	batches = valBatchMgr.getBatches()  # Generates and augments batches.
	for val_inputs, val_outputs in batches:
		batch_acc, batch_loss = nnEvaluator.evaluate_by_batch(session, val_inputs, val_outputs, is_time_major, is_sparse_output)

		# TODO [check] >> Are these calculations correct?
		batch_size = val_inputs.shape[batch_axis]
		val_acc += batch_acc * batch_size
		val_loss += batch_loss * batch_size
		num_val_examples += batch_size

	val_acc /= num_val_examples
	val_loss /= num_val_examples

	print('\tEvaluation time = {}'.format(time.time() - start_time))
	print('\tValidation: loss = {}, accurary = {}'.format(val_loss, val_acc))
	print('[SWL] Info: End evaluation...')

# NOTE [info] >> evaluate_neural_net_by_file_batch_loader().
# Supports lists of dense or sparse outputs.
def evaluate_neural_net_by_file_batch_manager(session, nnEvaluator, valFileBatchMgr, dirMgr, saver=None, checkpoint_dir_path=None, is_time_major=False, is_sparse_output=False):
	batch_axis = 1 if is_time_major else 0

	if saver is not None and checkpoint_dir_path is not None:
		# Load a model.
		# REF [site] >> https://www.tensorflow.org/programmers_guide/saved_model
		# REF [site] >> http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir_path)
		saver.restore(session, ckpt.model_checkpoint_path)
		#saver.restore(session, tf.train.latest_checkpoint(checkpoint_dir_path))
		print('[SWL] Info: Loaded a model.')

	print('\tWaiting for a validation batch directory...')
	dir_path = dirMgr.requestAvailableDirectory()
	if dir_path is None:
		print('[SWL] Error: No available directory.')
		return
	print('\tGot a validation batch directory: {}.'.format(dir_path))

	print('[SWL] Info: Start evaluation...')
	start_time = time.time()

	valFileBatchMgr.putBatches(dir_path)  # Generates, augments, and saves batches.

	val_loss, val_acc, num_val_examples = 0.0, 0.0, 0
	batches = valFileBatchMgr.getBatches(dir_path)  # Loads batches.
	for val_inputs, val_outputs in batches:
		batch_acc, batch_loss = nnEvaluator.evaluate_by_batch(session, val_inputs, val_outputs, is_time_major, is_sparse_output)

		# TODO [check] >> Are these calculations correct?
		batch_size = val_inputs.shape[batch_axis]
		val_acc += batch_acc * batch_size
		val_loss += batch_loss * batch_size
		num_val_examples += batch_size

	dirMgr.returnDirectory(dir_path)				

	val_acc /= num_val_examples
	val_loss /= num_val_examples

	print('\tEvaluation time = {}'.format(time.time() - start_time))
	print('\tValidation: loss = {}, accurary = {}'.format(val_loss, val_acc))
	print('[SWL] Info: End evaluation...')

# Supports lists of dense or sparse outputs.
def evaluate_neural_net_by_batch_list(session, nnEvaluator, val_inputs_list, val_outputs_list, saver=None, checkpoint_dir_path=None, is_time_major=False, is_sparse_output=False):
	num_val_batches = len(val_inputs_list)
	if len(val_outputs_list) != num_val_batches:
		raise ValueError('Invalid parameter length')

	batch_axis = 1 if is_time_major else 0

	if saver is not None and checkpoint_dir_path is not None:
		# Load a model.
		# REF [site] >> https://www.tensorflow.org/programmers_guide/saved_model
		# REF [site] >> http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir_path)
		saver.restore(session, ckpt.model_checkpoint_path)
		#saver.restore(session, tf.train.latest_checkpoint(checkpoint_dir_path))
		print('[SWL] Info: Loaded a model.')

	print('[SWL] Info: Start evaluation...')
	start_time = time.time()
	indices = np.arange(num_val_batches)
	#if shuffle:
	#	np.random.shuffle(indices)

	val_loss, val_acc, num_val_examples = 0.0, 0.0, 0
	for step in indices:
		batch_acc, batch_loss = nnEvaluator.evaluate_by_batch(session, val_inputs_list[step], val_outputs_list[step], is_time_major, is_sparse_output)

		# TODO [check] >> Are these calculations correct?
		batch_size = val_inputs_list[step].shape[batch_axis]
		val_acc += batch_acc * batch_size
		val_loss += batch_loss * batch_size
		num_val_examples += batch_size

	val_acc /= num_val_examples
	val_loss /= num_val_examples
	print('\tEvaluation time = {}'.format(time.time() - start_time))
	print('\tValidation: loss = {}, accurary = {}'.format(val_loss, val_acc))
	print('[SWL] Info: End evaluation...')

# Supports dense or sparse outputs.
# But when outputs are sparse, all dataset is processed at once.
def evaluate_neural_net(session, nnEvaluator, val_inputs, val_outputs, batch_size, saver=None, checkpoint_dir_path=None, is_time_major=False, is_sparse_output=False):
	batch_axis = 1 if is_time_major else 0

	num_val_examples = 0
	if val_inputs is not None and val_outputs is not None:
		if is_sparse_output:
			num_val_examples = val_inputs.shape[batch_axis]
		else:
			if val_inputs.shape[batch_axis] == val_outputs.shape[batch_axis]:
				num_val_examples = val_inputs.shape[batch_axis]

	if num_val_examples > 0:
		if saver is not None and checkpoint_dir_path is not None:
			# Load a model.
			# REF [site] >> https://www.tensorflow.org/programmers_guide/saved_model
			# REF [site] >> http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
			ckpt = tf.train.get_checkpoint_state(checkpoint_dir_path)
			saver.restore(session, ckpt.model_checkpoint_path)
			#saver.restore(session, tf.train.latest_checkpoint(checkpoint_dir_path))
			print('[SWL] Info: Loaded a model.')

		print('[SWL] Info: Start evaluation...')
		start_time = time.time()
		#val_loss, val_acc = nnEvaluator.evaluate(session, val_inputs, val_outputs, batch_size)
		val_loss, val_acc = nnEvaluator.evaluate(session, val_inputs, val_outputs, num_val_examples if is_sparse_output else batch_size, is_time_major, is_sparse_output)
		print('\tEvaluation time = {}'.format(time.time() - start_time))
		print('\tValidation: loss = {}, accurary = {}'.format(val_loss, val_acc))
		print('[SWL] Info: End evaluation...')
	else:
		print('[SWL] Error: The number of validation inputs is not equal to that of validation outputs.')

def evaluate_neural_net_with_decoder_input(session, nnEvaluator, val_encoder_input_seqs, val_decoder_input_seqs, val_decoder_output_seqs, batch_size, saver=None, checkpoint_dir_path=None, is_time_major=False, is_sparse_output=False):
	batch_axis = 1 if is_time_major else 0

	num_val_examples = 0
	if val_encoder_input_seqs is not None and val_decoder_input_seqs is not None and val_decoder_output_seqs is not None:
		if is_sparse_output:
			num_val_examples = val_encoder_input_seqs.shape[batch_axis]
		else:
			if val_encoder_input_seqs.shape[batch_axis] == val_decoder_input_seqs.shape[batch_axis] and val_encoder_input_seqs.shape[batch_axis] == val_decoder_output_seqs.shape[batch_axis]:
				num_val_examples = val_encoder_input_seqs.shape[batch_axis]

	if num_val_examples > 0:
		if saver is not None and checkpoint_dir_path is not None:
			# Load a model.
			ckpt = tf.train.get_checkpoint_state(checkpoint_dir_path)
			saver.restore(session, ckpt.model_checkpoint_path)
			#saver.restore(session, tf.train.latest_checkpoint(checkpoint_dir_path))
			print('[SWL] Info: Loaded a model.')

		print('[SWL] Info: Start evaluation...')
		start_time = time.time()
		#val_loss, val_acc = nnEvaluator.evaluate_seq2seq(session, val_encoder_input_seqs, val_decoder_input_seqs, val_decoder_output_seqs, batch_size)
		val_loss, val_acc = nnEvaluator.evaluate_seq2seq(session, val_encoder_input_seqs, val_decoder_input_seqs, val_decoder_output_seqs, num_val_examples if is_sparse_output else batch_size, is_time_major, is_sparse_output)
		print('\tEvaluation time = {}'.format(time.time() - start_time))
		print('\tValidation: loss = {}, accurary = {}'.format(val_loss, val_acc))
		print('[SWL] Info: End evaluation...')
	else:
		print('[SWL] Error: The numbers of validation inputs and outputs are not equal.')

#%%------------------------------------------------------------------

# Supports lists of dense or sparse outputs.
def infer_by_neural_net_and_batch_generator(session, nnInferrer, testBatchGenerator, saver=None, checkpoint_dir_path=None, is_time_major=False):
	batch_axis = 1 if is_time_major else 0

	if saver is not None and checkpoint_dir_path is not None:
		# Load a model.
		# REF [site] >> https://www.tensorflow.org/programmers_guide/saved_model
		# REF [site] >> http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir_path)
		saver.restore(session, ckpt.model_checkpoint_path)
		#saver.restore(session, tf.train.latest_checkpoint(checkpoint_dir_path))
		print('[SWL] Info: Loaded a model.')

	print('[SWL] Info: Start inferring...')
	start_time = time.time()

	inf_outputs_list = list()
	batches = testBatchGenerator.generateBatches()  # Generates and augments batches.
	for batch_data, _ in batches:
		batch_outputs = nnInferrer.infer_by_batch(session, batch_data[0], is_time_major)
		inf_outputs_list.append(batch_outputs)
	print('\tInference time = {}'.format(time.time() - start_time))
	print('[SWL] Info: End inferring...')

	return inf_outputs_list

# Supports lists of dense or sparse outputs.
def infer_by_neural_net_and_file_batch_loader(session, nnInferrer, testFileBatchLoader, testDirMgr, saver=None, checkpoint_dir_path=None, is_time_major=False):
	batch_axis = 1 if is_time_major else 0

	if saver is not None and checkpoint_dir_path is not None:
		# Load a model.
		# REF [site] >> https://www.tensorflow.org/programmers_guide/saved_model
		# REF [site] >> http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir_path)
		saver.restore(session, ckpt.model_checkpoint_path)
		#saver.restore(session, tf.train.latest_checkpoint(checkpoint_dir_path))
		print('[SWL] Info: Loaded a model.')

	print('\tWaiting for an inference batch directory...')
	while True:
		inf_dir_path = testDirMgr.requestDirectory()
		if inf_dir_path is not None:
			break
		else:
			time.sleep(0.1)
	print('\tGot an inference batch directory: {}.'.format(inf_dir_path))

	print('[SWL] Info: Start inferring...')
	start_time = time.time()

	inf_outputs_list = list()
	batches = testFileBatchLoader.loadBatches(inf_dir_path)  # Loads batches.
	for batch_data, _ in batches:
		batch_outputs = nnInferrer.infer_by_batch(session, batch_data[0], is_time_major)
		inf_outputs_list.append(batch_outputs)

	testDirMgr.returnDirectory(inf_dir_path)				

	print('\tInference time = {}'.format(time.time() - start_time))
	print('[SWL] Info: End inferring...')

	return inf_outputs_list

# NOTE [info] >> Use infer_by_neural_net_and_batch_generator().
# Supports lists of dense or sparse outputs.
def infer_by_neural_net_and_batch_manager(session, nnInferrer, testBatchMgr, saver=None, checkpoint_dir_path=None, is_time_major=False):
	batch_axis = 1 if is_time_major else 0

	if saver is not None and checkpoint_dir_path is not None:
		# Load a model.
		# REF [site] >> https://www.tensorflow.org/programmers_guide/saved_model
		# REF [site] >> http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir_path)
		saver.restore(session, ckpt.model_checkpoint_path)
		#saver.restore(session, tf.train.latest_checkpoint(checkpoint_dir_path))
		print('[SWL] Info: Loaded a model.')

	print('[SWL] Info: Start inferring...')
	start_time = time.time()

	inf_outputs_list = list()
	batches = testBatchMgr.getBatches()  # Generates and augments batches.
	for test_inputs, _ in batches:
		batch_outputs = nnInferrer.infer_by_batch(session, test_inputs, is_time_major)
		inf_outputs_list.append(batch_outputs)
	print('\tInference time = {}'.format(time.time() - start_time))
	print('[SWL] Info: End inferring...')

	return inf_outputs_list

# NOTE [info] >> Use infer_by_neural_net_and_file_batch_loader().
# Supports lists of dense or sparse outputs.
def infer_by_neural_net_and_file_batch_manager(session, nnInferrer, testFileBatchMgr, dirMgr, saver=None, checkpoint_dir_path=None, is_time_major=False):
	batch_axis = 1 if is_time_major else 0

	if saver is not None and checkpoint_dir_path is not None:
		# Load a model.
		# REF [site] >> https://www.tensorflow.org/programmers_guide/saved_model
		# REF [site] >> http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir_path)
		saver.restore(session, ckpt.model_checkpoint_path)
		#saver.restore(session, tf.train.latest_checkpoint(checkpoint_dir_path))
		print('[SWL] Info: Loaded a model.')

	print('\tWaiting for an inference batch directory...')
	dir_path = dirMgr.requestAvailableDirectory()
	if dir_path is None:
		print('[SWL] Error: No available directory.')
		return
	print('\tGot an inference batch directory: {}.'.format(dir_path))

	print('[SWL] Info: Start inferring...')
	start_time = time.time()

	testFileBatchMgr.putBatches(dir_path)  # Generates, augments, and saves batches.

	inf_outputs_list = list()
	batches = testFileBatchMgr.getBatches(dir_path)  # Loads batches.
	for test_inputs, _ in batches:
		batch_outputs = nnInferrer.infer_by_batch(session, test_inputs, is_time_major)
		inf_outputs_list.append(batch_outputs)

	dirMgr.returnDirectory(dir_path)				

	print('\tInference time = {}'.format(time.time() - start_time))
	print('[SWL] Info: End inferring...')

	return inf_outputs_list

# Supports lists of dense or sparse outputs.
def infer_from_batch_list_by_neural_net(session, nnInferrer, test_inputs_list, saver=None, checkpoint_dir_path=None, is_time_major=False):
	num_inf_batches = len(test_inputs_list)

	batch_axis = 1 if is_time_major else 0

	if saver is not None and checkpoint_dir_path is not None:
		# Load a model.
		# REF [site] >> https://www.tensorflow.org/programmers_guide/saved_model
		# REF [site] >> http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir_path)
		saver.restore(session, ckpt.model_checkpoint_path)
		#saver.restore(session, tf.train.latest_checkpoint(checkpoint_dir_path))
		print('[SWL] Info: Loaded a model.')

	print('[SWL] Info: Start inferring...')
	start_time = time.time()
	indices = np.arange(num_inf_batches)
	#if shuffle:
	#	np.random.shuffle(indices)

	inf_outputs_list = list()
	for step in indices:
		batch_outputs = nnInferrer.infer_by_batch(session, test_inputs_list[step], is_time_major)
		inf_outputs_list.append(batch_outputs)
	print('\tInference time = {}'.format(time.time() - start_time))
	print('[SWL] Info: End inferring...')

	return inf_outputs_list

# Supports dense or sparse outputs.
# But when outputs are sparse, all dataset is processed at once.
def infer_by_neural_net(session, nnInferrer, test_inputs, batch_size, saver=None, checkpoint_dir_path=None, is_time_major=False, is_sparse_output=False):
	batch_axis = 1 if is_time_major else 0

	num_inf_examples = 0
	if test_inputs is not None:
		num_inf_examples = test_inputs.shape[batch_axis]

	if num_inf_examples > 0:
		if saver is not None and checkpoint_dir_path is not None:
			# Load a model.
			# REF [site] >> https://www.tensorflow.org/programmers_guide/saved_model
			# REF [site] >> http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
			ckpt = tf.train.get_checkpoint_state(checkpoint_dir_path)
			saver.restore(session, ckpt.model_checkpoint_path)
			#saver.restore(session, tf.train.latest_checkpoint(checkpoint_dir_path))
			print('[SWL] Info: Loaded a model.')

		print('[SWL] Info: Start inferring...')
		start_time = time.time()
		#inferences = nnInferrer.infer(session, test_inputs, batch_size)
		inferences = nnInferrer.infer(session, test_inputs, num_inf_examples if is_sparse_output else batch_size, is_time_major)
		print('\tInference time = {}'.format(time.time() - start_time))
		print('[SWL] Info: End inferring...')

		return inferences
	else:
		print('[SWL] Error: Invalid test inputs.')
		return None

def infer_by_neural_net_with_decoder_input(session, nnInferrer, dataset, test_inputs, batch_size, saver=None, checkpoint_dir_path=None, is_time_major=False, is_sparse_output=False):
	return infer_by_neural_net(session, nnInferrer, test_inputs, batch_size, saver, checkpoint_dir_path, is_time_major, is_sparse_output)
