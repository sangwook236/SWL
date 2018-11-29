import numpy as np

#%%------------------------------------------------------------------

class NeuralNetInferrer(object):
	def __init__(self, neuralNet):
		self._neuralNet = neuralNet

	def infer(self, session, test_data, batch_size=None, is_time_major=False):
		batch_dim = 1 if is_time_major else 0
		num_inf_examples = test_data.shape[batch_dim]

		if batch_size is None or num_inf_examples <= batch_size:
			inferences = self._neuralNet.model_output.eval(session=session, feed_dict=self._neuralNet.get_feed_dict(test_data, is_training=False))
		else:
			inf_steps_per_epoch = (num_inf_examples - 1) // batch_size + 1

			indices = np.arange(num_inf_examples)

			inferences = np.array([])
			for step in range(inf_steps_per_epoch):
				start = step * batch_size
				end = start + batch_size
				batch_indices = indices[start:end]
				if batch_indices.size > 0:  # If batch_indices is non-empty.
					data_batch = test_data[batch_indices]
					if data_batch.size > 0:  # If data_batch is non-empty.
						batch_inference = self._neuralNet.model_output.eval(session=session, feed_dict=self._neuralNet.get_feed_dict(data_batch, is_training=False))

						if inferences.size > 0:  # If inferences is non-empty.
							inferences = np.concatenate((inferences, batch_inference), axis=0)
						else:
							inferences = batch_inference

		return inferences

	def infer_seq2seq(self, session, test_encoder_inputs, batch_size=None, is_time_major=False):
		return self.infer(session, test_encoder_inputs, batch_size, is_time_major)

	def infer_unsupervisedly(self, session, test_data, batch_size=None, is_time_major=False):
		return self.infer(session, test_data, batch_size, is_time_major)
