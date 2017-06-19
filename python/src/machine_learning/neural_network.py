# REF [file] >> https://github.com/zizhaozhang/unet-tensorflow-keras/blob/master/model.py

class NeuralNetwork():
	def __init__(self):
		pass

	def get_crop_shape(self, target, refer):
		# Width, the 3rd dimension.
		cw = (target.get_shape()[2] - refer.get_shape()[2]).value
		assert (cw >= 0)
		if cw % 2 != 0:
			cw1, cw2 = int(cw/2), int(cw/2) + 1
		else:
			cw1, cw2 = int(cw/2), int(cw/2)
		# Height, the 2nd dimension.
		ch = (target.get_shape()[1] - refer.get_shape()[1]).value
		assert (ch >= 0)
		if ch % 2 != 0:
			ch1, ch2 = int(ch/2), int(ch/2) + 1
		else:
			ch1, ch2 = int(ch/2), int(ch/2)

		return (ch1, ch2), (cw1, cw2)

	def create_model(self, num_classes, backend='tf', input_shape=None, tf_input=None):
		raise NotImplementError

	def train(self):
		raise NotImplementError

	def predict(self):
		raise NotImplementError
