import abc, time
import tensorflow as tf

#%%------------------------------------------------------------------
# ModelEvaluator.

#class ModelEvaluator(abc.ABC):
class ModelEvaluator(object):
	def __init__(self, model, dataGenerator, model_save_dir_path):
		super().__init__()

		self._model = model
		self._dataGenerator = dataGenerator
		self._model_save_dir_path = model_save_dir_path

		# Creates a saver.
		self._saver = tf.train.Saver()

	def evaluate(self, session, batch_size=None, shuffle=False):
		if not self._dataGenerator.hasValidationData():
			print('[SWL] Error: No validation data.')
			return

		if self._saver is not None and self._model_save_dir_path is not None:
			# Load a model.
			# REF [site] >> https://www.tensorflow.org/programmers_guide/saved_model
			# REF [site] >> http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
			ckpt = tf.train.get_checkpoint_state(self._model_save_dir_path)
			self._saver.restore(session, ckpt.model_checkpoint_path)
			#self._saver.restore(session, tf.train.latest_checkpoint(self._model_save_dir_path))
			print('[SWL] Info: Loaded a model.')

		print('[SWL] Info: Start evaluation...')
		start_time = time.time()
		val_loss, val_acc = self._evaluate(session, batch_size, shuffle)
		print('\tEvaluation time = {}'.format(time.time() - start_time))
		print('\tValidation: loss = {}, accurary = {}'.format(val_loss, val_acc))
		print('[SWL] Info: End evaluation...')

	def _evaluate(self, session, batch_size=None, shuffle=False):
		loss, accuracy = self._model.loss, self._model.accuracy

		if batch_size is None:
			val_data, num_val_examples = self._dataGenerator.getValidationData()
			#val_loss = loss.eval(session=session, feed_dict=self._model.get_feed_dict(val_data, is_training=False))
			#val_acc = accuracy.eval(session=session, feed_dict=self._model.get_feed_dict(val_data, is_training=False))
			val_loss, val_acc = session.run([loss, accuracy], feed_dict=self._model.get_feed_dict(val_data, is_training=False))
		else:
			if batch_size <= 0:
				raise ValueError('Invalid batch size: {}'.format(batch_size))

			val_loss, val_acc = 0.0, 0.0
			num_val_examples = 0
			for batch_data, num_batch_examples in self._dataGenerator.getValidationBatches(batch_size, shuffle):
				#batch_loss = loss.eval(session=session, feed_dict=self._model.get_feed_dict(batch_data, is_training=False))
				#batch_acc = accuracy.eval(session=session, feed_dict=self._model.get_feed_dict(batch_data, is_training=False))
				batch_loss, batch_acc = session.run([loss, accuracy], feed_dict=self._model.get_feed_dict(batch_data, is_training=False))

				# TODO [check] >> Is val_loss or val_acc correct?
				val_loss += batch_loss * num_batch_examples
				val_acc += batch_acc * num_batch_examples
				num_val_examples += num_batch_examples
			val_loss /= num_val_examples
			val_acc /= num_val_examples

		return val_loss, val_acc
