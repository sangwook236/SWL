import numpy as np

#--------------------------------------------------------------------

class NeuralNetInferrer(object):
	def __init__(self, model):
		self._model = model

	def infer_by_batch(self, session, test_data, is_time_major=False):
		batch_axis = 1 if is_time_major else 0

		num_inf_examples = 0
		if test_data is not None:
			num_inf_examples = test_data.shape[batch_axis]
		#if test_data is None:
		if num_inf_examples <= 0:
			return None

		#if test_data is not None:
		if num_inf_examples > 0:
			if test_data.size > 0:  # If test_data is non-empty.
				#inferences = self._model.model_output.eval(session=session, feed_dict=self._model.get_feed_dict(test_data, is_training=False))
				inferences = session.run(self._model.model_output, feed_dict=self._model.get_feed_dict(test_data, is_training=False))  # Can support a model output of list type.

		return inferences

	def infer(self, session, test_data, batch_size=None, is_time_major=False):
		batch_axis = 1 if is_time_major else 0

		num_inf_examples = 0
		if test_data is not None:
			num_inf_examples = test_data.shape[batch_axis]
		#if test_data is None:
		if num_inf_examples <= 0:
			return None

		if batch_size is None or num_inf_examples <= batch_size:
			#inferences = self._model.model_output.eval(session=session, feed_dict=self._model.get_feed_dict(test_data, is_training=False))
			inferences = session.run(self._model.model_output, feed_dict=self._model.get_feed_dict(test_data, is_training=False))  # Can support a model output of list type.
		else:
			inf_steps_per_epoch = (num_inf_examples - 1) // batch_size + 1

			indices = np.arange(num_inf_examples)
			#if shuffle:
			#	np.random.shuffle(indices)

			inferences = None
			for step in range(inf_steps_per_epoch):
				start = step * batch_size
				end = start + batch_size
				batch_indices = indices[start:end]
				if batch_indices.size > 0:  # If batch_indices is non-empty.
					data_batch = test_data[batch_indices]
					if data_batch.size > 0:  # If data_batch is non-empty.
						#batch_inference = self._model.model_output.eval(session=session, feed_dict=self._model.get_feed_dict(data_batch, is_training=False))
						batch_inference = session.run(self._model.model_output, feed_dict=self._model.get_feed_dict(data_batch, is_training=False))  # Can support a model output of list type.

						inferences = batch_inference if inferences is None else np.concatenate((inferences, batch_inference), axis=0)

		return inferences

	def infer_seq2seq(self, session, test_encoder_inputs, batch_size=None, is_time_major=False):
		return self.infer(session, test_encoder_inputs, batch_size, is_time_major)

	def infer_unsupervisedly(self, session, test_data, batch_size=None, is_time_major=False):
		return self.infer(session, test_data, batch_size, is_time_major)
