import densenet_fc as dc
import tensorflow as tf
from simple_neural_net import SimpleNeuralNet

#%%------------------------------------------------------------------

class PlantFcDenseNet(SimpleNeuralNet):
	def __init__(self, input_shape, output_shape):
		super().__init__(input_shape, output_shape)

	def _create_model(self, input_tensor, is_training_tensor, input_shape, output_shape):
		num_classes = output_shape[-1]
		with tf.name_scope('plant_fc_densenet'):
			fc_densenet_model = dc.DenseNetFCN(input_shape[1:], nb_dense_block=5, growth_rate=16, nb_layers_per_block=4, upsampling_type='upsampling', classes=num_classes)

			# Display the model summary.
			#fc_densenet_model.summary()

			return fc_densenet_model(input_tensor)
