# REF [file] >> https://github.com/zizhaozhang/unet-tensorflow-keras/blob/master/model.py

class KerasNeuralNet(object):
	def __init__(self):
		pass

	def create_model(self, num_classes, backend='tf', input_shape=None, tf_input=None):
		raise NotImplementedError

	def train(self):
		raise NotImplementedError

	def predict(self):
		raise NotImplementedError
