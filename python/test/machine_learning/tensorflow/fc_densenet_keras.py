import densenet_fc as dc
import tensorflow as tf
from simple_neural_net import SimpleNeuralNet

#%%------------------------------------------------------------------

class FcDenseNetUsingKeras(SimpleNeuralNet):
	def __init__(self, input_shape, output_shape):
		super().__init__(input_shape, output_shape)

	def _create_single_model(self, input_tensor, input_shape, output_shape, is_training):
		num_classes = output_shape[-1]
		with tf.variable_scope('fc_densenet_using_keras'):
			fc_densenet_model = dc.DenseNetFCN(input_shape[1:], nb_dense_block=5, growth_rate=16, nb_layers_per_block=4, upsampling_type='upsampling', classes=num_classes)

			# Display the model summary.
			#fc_densenet_model.summary()

			return fc_densenet_model(input_tensor)
