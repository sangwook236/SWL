import tensorflow as tf

#%%------------------------------------------------------------------

class DnnBaseModel(object):
	def __init__(self, num_classes):
		self.num_classes_ = num_classes
		self.model_output_ = None
