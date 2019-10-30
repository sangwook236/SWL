import abc, time
import tensorflow as tf

#--------------------------------------------------------------------
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
		if not self._dataGenerator.hasValidationBatches():
			print('[SWL] Error: No validation data.')
			return

		if self._saver is not None and self._model_save_dir_path is not None:
			# Load a model.
			# REF [site] >> https://www.tensorflow.org/programmers_guide/saved_model
			# REF [site] >> http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
			ckpt = tf.train.get_checkpoint_state(self._model_save_dir_path)
			ckpt_filepath = ckpt.model_checkpoint_path if ckpt else None
			#ckpt_filepath = tf.train.latest_checkpoint(self._model_save_dir_path)
			if ckpt_filepath:
				self._saver.restore(session, ckpt_filepath)
			else:
				print('[SWL] Error: Failed to load a model from {}.'.format(self._model_save_dir_path))
				return
			print('[SWL] Info: Loaded a model.')
		else:
			print('[SWL] Error: Invalid model save path, {}.'.format(self._model_save_dir_path))
			return

		print('[SWL] Info: Start evaluating...')
		start_time = time.time()
		val_loss, val_acc = self._evaluate(session, batch_size, shuffle)
		print('\tEvaluation time = {} secs.'.format(time.time() - start_time))
		print('\tValidation: loss = {}, accuracy = {}'.format(val_loss, val_acc))
		print('[SWL] Info: End evaluating.')

	def _evaluate(self, session, batch_size=None, shuffle=False):
		loss, accuracy = self._model.loss, self._model.accuracy

		val_loss, val_acc = 0.0, 0.0
		num_val_examples = 0
		for batch_data, num_batch_examples in self._dataGenerator.getValidationBatches(batch_size, shuffle):
			#batch_loss = loss.eval(session=session, feed_dict=self._model.get_feed_dict(batch_data, num_batch_examples, is_training=False))
			#batch_acc = accuracy.eval(session=session, feed_dict=self._model.get_feed_dict(batch_data, num_batch_examples, is_training=False))
			batch_loss, batch_acc = session.run([loss, accuracy], feed_dict=self._model.get_feed_dict(batch_data, num_batch_examples, is_training=False))

			# TODO [check] >> Is val_loss or val_acc correct?
			val_loss += batch_loss * num_batch_examples
			val_acc += batch_acc * num_batch_examples
			num_val_examples += num_batch_examples
		if num_val_examples > 0:
			val_loss /= num_val_examples
			val_acc /= num_val_examples

		return val_loss, val_acc
