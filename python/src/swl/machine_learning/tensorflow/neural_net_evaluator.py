import numpy as np

#%%------------------------------------------------------------------

class NeuralNetEvaluator(object):
	def evaluate(self, session, neuralNet, test_data, test_labels, batch_size=None):
		self._loss, self._accuracy = neuralNet.loss, neuralNet.accuracy

		num_test_examples = test_data.shape[0]

		if batch_size is None or num_test_examples <= batch_size:
			#test_loss = self._loss.eval(session=session, feed_dict=neuralNet.get_feed_dict(test_data, test_labels, is_training=False))
			#test_acc = self._accuracy.eval(session=session, feed_dict=neuralNet.get_feed_dict(test_data, test_labels, is_training=False))
			test_loss, test_acc = session.run([self._loss, self._accuracy], feed_dict=neuralNet.get_feed_dict(test_data, test_labels, is_training=False))
		else:
			test_steps_per_epoch = (num_test_examples - 1) // batch_size + 1

			indices = np.arange(num_test_examples)
			#if shuffle:
			#	np.random.shuffle(indices)

			test_loss, test_acc = 0, 0
			for step in range(test_steps_per_epoch):
				start = step * batch_size
				end = start + batch_size
				batch_indices = indices[start:end]
				data_batch, label_batch = test_data[batch_indices,], test_labels[batch_indices,]
				if data_batch.size > 0 and label_batch.size > 0:  # If data_batch and label_batch are non-empty.
					#batch_loss = self._loss.eval(session=session, feed_dict=neuralNet.get_feed_dict(data_batch, label_batch, is_training=False))
					#batch_acc = self._accuracy.eval(session=session, feed_dict=neuralNet.get_feed_dict(data_batch, label_batch, is_training=False))
					batch_loss, batch_acc = session.run([self._loss, self._accuracy], feed_dict=neuralNet.get_feed_dict(data_batch, label_batch, is_training=False))

					# TODO [check] >> Is test_loss or test_acc correct?
					test_loss += batch_loss * batch_indices.size
					test_acc += batch_acc * batch_indices.size
			test_loss /= num_test_examples
			test_acc /= num_test_examples

		return test_loss, test_acc
