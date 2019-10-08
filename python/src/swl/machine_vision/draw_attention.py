import math
import tensorflow as tf

#--------------------------------------------------------------------

class DrawAttentionBase(object):
	@staticmethod
	def filter(img, width, height, Fx, Fy, gamma, patch_width, patch_height):
		Fxt = tf.transpose(Fx, perm=[0, 2, 1])
		img = tf.reshape(img, [-1, height, width])
		glimpse = tf.matmul(Fy, tf.matmul(img, Fxt))
		glimpse = tf.reshape(glimpse, [-1, patch_width * patch_height])
		return glimpse * tf.reshape(gamma, [-1, 1])

	@staticmethod
	def linear_transform(x, output_dim):
		"""
		Affine transformation W * x + b.
		Assumes x.shape = (batch_size, num_features).
		"""
		W = tf.get_variable('W', [x.get_shape()[1], output_dim]) 
		b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.0))
		return tf.matmul(x, W) + b

#--------------------------------------------------------------------

# REF [paper] >> "DRAW: A Recurrent Neural Network For Image Generation", arXiv 2015
#	REF [site] >> https://github.com/ericjang/draw
class DrawAttention(DrawAttentionBase):
	@staticmethod
	def getWriteAttention(ctx, batch_size, width, height, patch_size, reuse=tf.AUTO_REUSE, eps=1.0e-8):
		Fx, Fy, gamma = DrawAttention.getAttentionParameters(ctx, width, height, patch_size, 'draw_write_attention', reuse, eps)

		with tf.variable_scope('draw_writing_patch', reuse=reuse):
			w = DrawAttention.linear_transform(ctx, patch_size * patch_size)  # batch_size * (patch_size * patch_size).
		w = tf.reshape(w, [batch_size, patch_size, patch_size])
		Fyt = tf.transpose(Fy, perm=[0, 2, 1])
		wr = tf.matmul(Fyt, tf.matmul(w, Fx))
		wr = tf.reshape(wr, [batch_size, height * width])
		#gamma = tf.tile(gamma, [1, height * width])
		return wr * tf.reshape(1.0 / gamma, [-1, 1])

	@staticmethod
	def getReadAttention(x, ctx, width, height, patch_size, reuse=tf.AUTO_REUSE, eps=1.0e-8):
		Fx, Fy, gamma = DrawAttention.getAttentionParameters(ctx, width, height, patch_size, 'draw_read_attention', reuse, eps)

		return DrawAttention.filter(x, width, height, Fx, Fy, gamma, patch_size, patch_size)  # batch_size * (patch_size * patch_size).

	@staticmethod
	def getAttentionParameters(ctx, width, height, patch_size, scope, reuse, eps=1.0e-8):
		with tf.variable_scope(scope, reuse=reuse):
			params = DrawAttention.linear_transform(ctx, 5)  # 5 parameters.

		# Grid center (gx, gy), stride (delta), isotropic variance (sigma^2), scalar intensity (gamma).
		#gx_tilde, gy_tilde, log_sigma2, log_delta, log_gamma = tf.split(1, 5, params)
		gx_tilde, gy_tilde, log_sigma2, log_delta, log_gamma = tf.split(params, 5, 1)
		gx = (width + 1) / 2 * (gx_tilde + 1)
		gy = (height + 1) / 2 * (gy_tilde + 1)
		sigma2 = tf.exp(log_sigma2)
		delta = (max(width, height) - 1) / (patch_size - 1) * tf.exp(log_delta)  # batch_size * patch_size.

		# Attention parameters: Fx, Fy, gamma.
		return DrawAttention._filterbank(width, height, gx, gy, sigma2, delta, patch_size, eps) + (tf.exp(log_gamma),)

	@staticmethod
	def _filterbank(width, height, gx, gy, sigma2, delta, patch_size, eps=1.0e-8):
		grid_i = tf.reshape(tf.cast(tf.range(patch_size), tf.float32), [1, -1])
		mu_x = gx + (grid_i - patch_size / 2 - 0.5) * delta  # Eqn 19.
		mu_y = gy + (grid_i - patch_size / 2 - 0.5) * delta  # Eqn 20.
		a = tf.reshape(tf.cast(tf.range(width), tf.float32), [1, 1, -1])
		b = tf.reshape(tf.cast(tf.range(height), tf.float32), [1, 1, -1])
		mu_x = tf.reshape(mu_x, [-1, patch_size, 1])
		mu_y = tf.reshape(mu_y, [-1, patch_size, 1])
		sigma2 = tf.reshape(sigma2, [-1, 1, 1])
		Fx = tf.exp(-tf.square(a - mu_x) / (2 * sigma2))
		Fy = tf.exp(-tf.square(b - mu_y) / (2 * sigma2))  # batch_size * patch_size * height.
		# Normalize, sum over width and height dims.
		Fx = Fx / tf.maximum(tf.reduce_sum(Fx, 2, keepdims=True), eps)
		Fy = Fy / tf.maximum(tf.reduce_sum(Fy, 2, keepdims=True), eps)
		return Fx, Fy

#--------------------------------------------------------------------

# REF [paper] >> "End-to-End Instance Segmentation with Recurrent Attention", arXiv 2017
#	REF [site] >> https://github.com/renmengye/rec-attend-public
class DrawRectangularAttention(DrawAttentionBase):
	@staticmethod
	def getWriteAttention(ctx, batch_size, width, height, patch_width, patch_height, reuse=tf.AUTO_REUSE, eps=1.0e-8):
		Fx, Fy, gamma = DrawRectangularAttention.getAttentionParameters(ctx, width, height, patch_width, patch_height, 'draw_write_attention', reuse, eps)

		with tf.variable_scope('draw_writing_patch', reuse=reuse):
			w = DrawRectangularAttention.linear_transform(ctx, patch_width * patch_height)  # batch_size * (patch_width * patch_height).
		w = tf.reshape(w, [batch_size, patch_height, patch_width])
		Fyt = tf.transpose(Fy, perm=[0, 2, 1])
		wr = tf.matmul(Fyt, tf.matmul(w, Fx))
		wr = tf.reshape(wr, [batch_size, height * width])
		#gamma = tf.tile(gamma, [1, height * width])
		return wr * tf.reshape(1.0 / gamma, [-1, 1])

	@staticmethod
	def getReadAttention(x, ctx, width, height, patch_width, patch_height, reuse=tf.AUTO_REUSE, eps=1.0e-8):
		Fx, Fy, gamma = DrawRectangularAttention.getAttentionParameters(ctx, width, height, patch_width, patch_height, 'draw_read_attention', reuse, eps)

		return DrawRectangularAttention.filter(x, width, height, Fx, Fy, gamma, patch_width, patch_height)  # batch_size * (patch_width * patch_height).

	@staticmethod
	def getAttentionParameters(ctx, width, height, patch_width, patch_height, scope, reuse, eps=1.0e-8):
		with tf.variable_scope(scope, reuse=reuse):
			params = DrawRectangularAttention.linear_transform(ctx, 7)  # 7 parameters.

		# Grid center (gx, gy), stride (deltax, deltay), anisotropic variance (sigmax^2, sigmay^2), scalar intensity (gamma).
		#gx_tilde, gy_tilde, log_sigmax2, log_sigmay2, log_deltax, log_deltay, log_gamma = tf.split(1, 7, params)
		gx_tilde, gy_tilde, log_sigmax2, log_sigmay2, log_deltax, log_deltay, log_gamma = tf.split(params, 7, 1)
		gx = (gx_tilde + 1) * width / 2
		gy = (gy_tilde + 1) * height / 2
		sigmax2 = tf.exp(log_sigmax2)
		sigmay2 = tf.exp(log_sigmay2)
		deltax = tf.exp(log_deltax) * width  # batch_size * patch_width.
		deltay = tf.exp(log_deltay) * height  # batch_size * patch_height.

		# Attention parameters: Fx, Fy, gamma.
		return DrawRectangularAttention._filterbank(width, height, gx, gy, sigmax2, sigmay2, deltax, deltay, patch_width, patch_height, eps) + (tf.exp(log_gamma),)

	@staticmethod
	def _filterbank(width, height, gx, gy, sigmax2, sigmay2, deltax, deltay, patch_width, patch_height, eps=1.0e-8):
		grid_ix = tf.reshape(tf.cast(tf.range(patch_width), tf.float32), [1, -1])
		grid_iy = tf.reshape(tf.cast(tf.range(patch_height), tf.float32), [1, -1])
		mu_x = gx + (deltax + 1) * (grid_ix - patch_width / 2 + 0.5) / patch_width
		mu_y = gy + (deltay + 1) * (grid_iy - patch_height / 2 + 0.5) / patch_height
		a = tf.reshape(tf.cast(tf.range(width), tf.float32), [1, 1, -1])
		b = tf.reshape(tf.cast(tf.range(height), tf.float32), [1, 1, -1])
		mu_x = tf.reshape(mu_x, [-1, patch_width, 1])
		mu_y = tf.reshape(mu_y, [-1, patch_height, 1])
		sigmax2 = tf.reshape(sigmax2, [-1, 1, 1])
		sigmay2 = tf.reshape(sigmay2, [-1, 1, 1])
		Fx = tf.exp(-tf.square(a - mu_x) / (2 * sigmax2)) / (math.sqrt(2 * math.pi) * tf.sqrt(sigmax2))  # batch_size * patch_width * width.
		Fy = tf.exp(-tf.square(b - mu_y) / (2 * sigmay2)) / (math.sqrt(2 * math.pi) * tf.sqrt(sigmay2))  # batch_size * patch_height * height.
		# Normalize, sum over width and height dims.
		Fx = Fx / tf.maximum(Fx, eps)
		Fy = Fy / tf.maximum(Fy, eps)
		return Fx, Fy
