import tensorflow as tf

# REF [site] >> https://github.com/titu1994/keras-non-local-nets
def non_local_block(inputs, intermediate_dim=None, compression=2, mode='embedded', add_residual=True, data_format='NHWC'):
	"""
	Adds a Non-Local block for self attention to the input tensor.
	Input tensor can be or rank 3 (temporal), 4 (spatial) or 5 (spatio-temporal).

	Arguments:
		inputs: input tensor
		intermediate_dim: The dimension of the intermediate representation. Can be
			'None' or a positive integer greater than 0. If 'None', computes the
			intermediate dimension as half of the input channel dimension.
		compression: None or positive integer. Compresses the intermediate
			representation during the dot products to reduce memory consumption.
			Default is set to 2, which states halve the time/space/spatio-time
			dimension for the intermediate step. Set to 1 to prevent computation
			compression. None or 1 causes no reduction.
		mode: Mode of operation. Can be one of 'embedded', 'gaussian', 'dot' or
			'concatenate'.
		add_residual: Boolean value to decide if the residual connection should be
			added or not. Default is True for ResNets, and False for Self Attention.
		data_format: Specify the data format of the input and output data. 'NHWC' or 'NCHW'.

	Returns:
		a tensor of same shape as input
	"""

	def _convND(inputs, rank, channels, kernel_initializer=None):
		assert rank in [3, 4, 5], 'Rank of input must be 3, 4 or 5'

		if 3 == rank:
			x = tf.layers.conv1d(inputs, channels, 1, padding='same', use_bias=False, kernel_initializer=kernel_initializer)
		elif 4 == rank:
			x = tf.layers.conv2d(inputs, channels, (1, 1), padding='same', use_bias=False, kernel_initializer=kernel_initializer)
		else:
			x = tf.layers.conv3d(inputs, channels, (1, 1, 1), padding='same', use_bias=False, kernel_initializer=kernel_initializer)
		return x

	#kernel_initializer = None
	kernel_initializer = tf.initializers.he_normal()

	channel_dim = 1 if 'NCHW' == data_format else -1
	input_shape = inputs.shape

	if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
		raise ValueError("'mode' must be one of 'gaussian', 'embedded', 'dot' or 'concatenate'")

	if compression is None:
		compression = 1

	dim1, dim2, dim3 = None, None, None

	# Check rank and calculate the input shape.
	if 3 == len(input_shape):  # Temporal / time series data.
		rank = 3
		batchsize, dim1, channels = input_shape
	elif 4 == len(input_shape):  # Spatial / image data.
		rank = 4
		if 1 == channel_dim:
			batchsize, channels, dim1, dim2 = input_shape
		else:
			batchsize, dim1, dim2, channels = input_shape
	elif 5 == len(input_shape):  # Spatio-temporal / Video or Voxel data.
		rank = 5
		if 1 == channel_dim:
			batchsize, channels, dim1, dim2, dim3 = input_shape
		else:
			batchsize, dim1, dim2, dim3, channels = input_shape
	else:
		raise ValueError('Input dimension has to be either 3 (temporal), 4 (spatial) or 5 (spatio-temporal)')

	# Verify correct intermediate dimension specified.
	if intermediate_dim is None:
		intermediate_dim = channels // 2

		if intermediate_dim < 1:
			intermediate_dim = 1
	else:
		intermediate_dim = int(intermediate_dim)

		if intermediate_dim < 1:
			raise ValueError("'intermediate_dim' must be either 'None' or positive integer greater than 1.")

	if 'gaussian' == mode:  # Gaussian instantiation.
		x1 = tf.reshape(inputs, (-1, channels))  # x_i.
		x2 = tf.reshape(inputs, (-1, channels))  # x_j.
		f = tf.keras.layers.dot((x1, x2), axes=2)
		f = tf.nn.softmax(f)
	elif 'dot' == mode:  # Dot instantiation.
		# theta path.
		theta = _convND(inputs, rank, intermediate_dim, kernel_initializer)
		theta = tf.reshape(theta, (-1, intermediate_dim))

		# phi path.
		phi = _convND(inputs, rank, intermediate_dim, kernel_initializer)
		phi = tf.reshape(phi, (-1, intermediate_dim))

		f = tf.keras.layers.dot((theta, phi), axes=2)

		# Scale the values to make it size invariant.
		f = f / float(f.shape[-1])
	elif 'concatenate' == mode:  # Concatenation instantiation
		raise NotImplementedError('Concatenate model has not been implemented yet')

		# FIXME [check] >> Not yet tested.

		# theta path.
		theta = _convND(inputs, rank, intermediate_dim, kernel_initializer)
		theta = tf.reshape(theta, (-1, intermediate_dim))

		# phi path.
		phi = _convND(inputs, rank, intermediate_dim, kernel_initializer)
		phi = tf.reshape(phi, (-1, intermediate_dim))
		
		theta_phi = tf.keras.layers.concatenate((theta, phi), axis=-1)

		w_f = tf.get_variable('w_f', shape, initializer=kernel_initializer)
		f = tf.keras.layers.dot((w_f, theta_phi), axes=2)

		# Scale the values to make it size invariant.
		f = f / float(f.shape[-1])
	else:  # Embedded Gaussian instantiation.
		# theta path.
		theta = _convND(inputs, rank, intermediate_dim, kernel_initializer)
		theta = tf.reshape(theta, (-1, intermediate_dim))

		# phi path.
		phi = _convND(inputs, rank, intermediate_dim, kernel_initializer)
		phi = tf.reshape(phi, (-1, intermediate_dim))

		if compression > 1:
			# Shielded computation.
			phi = tf.layers.max_pooling1d(phi, compression)

		f = tf.keras.layers.dot((theta, phi), axes=2)
		f = tf.nn.softmax(f)

	# g path.
	g = _convND(inputs, rank, intermediate_dim, kernel_initializer)
	g = tf.reshape(g, (-1, intermediate_dim))

	if compression > 1 and 'embedded' == mode:
		# Shielded computation.
		g = tf.layers.max_pooling1d(g, compression)

	# Compute output path.
	y = tf.keras.layers.dot((f, g), axes=(2, 1))

	# Reshape to input tensor format.
	if 3 == rank:
		y = tf.reshape((dim1, intermediate_dim))(y)
	elif 4 == rank:
		if -1 == channel_dim:
			y = tf.reshape(y, (dim1, dim2, intermediate_dim))
		else:
			y = tf.reshape(y, (intermediate_dim, dim1, dim2))
	else:
		if -1 == channel_dim:
			y = tf.reshape(y, (dim1, dim2, dim3, intermediate_dim))
		else:
			y = tf.reshape(y, (intermediate_dim, dim1, dim2, dim3))

	# Project filters.
	y = _convND(y, rank, channels, kernel_initializer)

	# Residual connection.
	if add_residual:
		y = tf.keras.layers.add((inputs, y))

	return y
