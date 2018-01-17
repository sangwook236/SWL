import numpy as np

#%%------------------------------------------------------------------

class NeuralNetPredictor(object):
	def predict(self, session, neuralNet, test_data, batch_size):
		self._model_output = neuralNet.model_output

		num_pred_examples = test_data.shape[0]

		"""
		predictions = self._model_output.eval(session=session, feed_dict=neuralNet.fill_feed_dict(test_data, is_training=False))
		"""
		pred_steps_per_epoch = (num_pred_examples - 1) // batch_size + 1

		indices = np.arange(num_pred_examples)

		predictions = np.array([])
		for step in range(pred_steps_per_epoch):
			start = step * batch_size
			end = start + batch_size
			batch_indices = indices[start:end]
			data_batch = test_data[batch_indices,]
			if data_batch.size > 0:  # If data_batch is non-empty.
				batch_prediction = self._model_output.eval(session=session, feed_dict=neuralNet.fill_feed_dict(data_batch, is_training=False))
	
				if predictions.size > 0:  # If predictions is non-empty.
					predictions = np.concatenate((predictions, batch_prediction), axis=0)
				else:
					predictions = batch_prediction

		return predictions
