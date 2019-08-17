#!/usr/bin/env python

import sys
sys.path.append('../../src')

import numpy as np
import matplotlib.pyplot as plt
import keras
import swl.machine_learning.util as swl_ml_util

def to_one_hot_encoding_test():
	y = np.array([ 1, 3, 5, 2, 4, 0 ])

	np.set_printoptions(threshold=np.nan)

	print('[1-1] swl_ml_util.to_one_hot_encoding ->', swl_ml_util.to_one_hot_encoding(y), sep='\n')
	print('[1-2] keras.utils.to_categorical ->', keras.utils.to_categorical(y), sep='\n')

	print('[1-3] swl_ml_util.to_one_hot_encoding ->', swl_ml_util.to_one_hot_encoding(y, 10), sep='\n')
	print('[1-4] keras.utils.to_categorical ->', keras.utils.to_categorical(y, 10), sep='\n')

	#--------------------
	Y = np.reshape(y, [2, 3, 1])
	print('The shape of Y = {}'.format(Y.shape))

	YY = swl_ml_util.to_one_hot_encoding(Y)
	print('[2-1] swl_ml_util.to_one_hot_encoding -> {}'.format(YY.shape))
	print(YY)

	YY = keras.utils.to_categorical(Y)
	print('[2-2] keras.utils.to_categorical -> {}'.format(YY.shape))
	print(YY)

	print(keras.utils.to_categorical(Y).reshape(Y.shape[:-1] + (-1,)))  # Not correctly working.
	print('[2-3] keras.utils.to_categorical -> {}'.format(YY.shape))
	print(YY)

	YY = swl_ml_util.to_one_hot_encoding(Y, 10)
	print('[2-4] swl_ml_util.to_one_hot_encoding -> {}'.format(YY.shape))
	print(YY)

	YY = keras.utils.to_categorical(Y, 10)
	print('[2-5] keras.utils.to_categorical -> {}'.format(YY.shape))
	print(YY)

	YY = keras.utils.to_categorical(Y, 10).reshape(Y.shape[:-1] + (-1,))  # Not correctly working.
	print('[2-6] keras.utils.to_categorical -> {}'.format(YY.shape))
	print(YY)

	#--------------------
	Y = np.reshape(y, [2, 3])
	print('The shape of Y = {}'.format(Y.shape))

	YY = swl_ml_util.to_one_hot_encoding(Y)
	print('[3-1] swl_ml_util.to_one_hot_encoding -> {}'.format(YY.shape))
	print(YY)

	YY = keras.utils.to_categorical(Y)
	print('[3-2] keras.utils.to_categorical -> {}'.format(YY.shape))
	print(YY)

	print(keras.utils.to_categorical(Y).reshape(Y.shape[:-1] + (-1,)))  # Not correctly working.
	print('[3-3] keras.utils.to_categorical -> {}'.format(YY.shape))
	print(YY)

	YY = swl_ml_util.to_one_hot_encoding(Y, 10)
	print('[3-4] swl_ml_util.to_one_hot_encoding -> {}'.format(YY.shape))
	print(YY)

	YY = keras.utils.to_categorical(Y, 10)
	print('[3-5] keras.utils.to_categorical -> {}'.format(YY.shape))
	print(YY)

	YY = keras.utils.to_categorical(Y, 10).reshape(Y.shape[:-1] + (-1,))  # Not correctly working.
	print('[3-6] keras.utils.to_categorical -> {}'.format(YY.shape))
	print(YY)

def time_based_learning_rate_test():
	initial_learning_rate = 0.1
	decay_rate = 0.001

	epochs = range(10000)
	learning_rates = []
	for epoch in epochs:
		learning_rates.append(swl_ml_util.time_based_learning_rate(epoch, initial_learning_rate, decay_rate))

	plt.figure()
	plt.plot(epochs, learning_rates)

def drop_based_learning_rate_test():
	initial_learning_rate = 0.001
	drop_rate = 0.5
	epoch_drop = 10000.0

	epochs = range(100000)
	learning_rates = []
	for epoch in epochs:
		learning_rates.append(swl_ml_util.drop_based_learning_rate(epoch, initial_learning_rate, drop_rate, epoch_drop))

	plt.figure()
	plt.plot(epochs, learning_rates)

def sequences_to_sparse_test():
	sequences = [
		[2, 0],
		[2],
		[1, 2],
		[2, 0, 1, 2]
	]
	
	sparse_indices, sparse_values, dense_shape = swl_ml_util.sequences_to_sparse(sequences, dtype=np.int32)
	print('Sparse indices =\n', sparse_indices)
	print('Sparse values =\n', sparse_values)
	print('Dense shape =\n', dense_shape)

def sparse_to_sequences_test():
	sparse_indices = np.array([[0, 0], [0, 1], [1, 0], [2, 0], [2, 1], [3, 0], [3, 1], [3, 2], [3, 3]], dtype=np.int32)
	sparse_values = np.array([2, 0, 2, 1, 2, 2, 0, 1, 2], dtype=np.int32)
	dense_shape = np.array([4, 4], dtype=np.int32)

	sequences = swl_ml_util.sparse_to_sequences(sparse_indices, sparse_values, dense_shape, dtype=np.int32)
	print('Sequenses =\n', sequences)

def sequences_to_dense_test():
	sequences = [
		[2, 0],
		[2],
		[1, 2],
		[2, 0, 1, 2]
	]
	
	dense = swl_ml_util.sequences_to_dense(sequences, default_value=-1, dtype=np.int32)
	print('Dense tensor =\n', dense)

def dense_to_sequences_test():
	default_value = -1
	dense = np.array([
		[2, 0, default_value, default_value],
		[2, default_value, default_value, default_value],
		[1, 2, default_value, default_value],
		[2, 0, 1, 2]
	])

	sequences = swl_ml_util.dense_to_sequences(dense, default_value=default_value, dtype=np.int32)
	print('Sequenses =\n', sequences)

def dense_to_sparse_test():
	default_value = -1
	dense = np.array([
		[2, 0, default_value, default_value],
		[2, default_value, default_value, default_value],
		[1, 2, default_value, default_value],
		[2, 0, 1, 2]
	])

	sparse_indices, sparse_values, dense_shape = swl_ml_util.dense_to_sparse(dense, default_value=default_value, dtype=np.int32)
	print('Sparse indices =\n', sparse_indices)
	print('Sparse values =\n', sparse_values)
	print('Dense shape =\n', dense_shape)

def sparse_to_dense_test():
	sparse_indices = np.array([[0, 0], [0, 1], [1, 0], [2, 0], [2, 1], [3, 0], [3, 1], [3, 2], [3, 3]], dtype=np.int32)
	sparse_values = np.array([2, 0, 2, 1, 2, 2, 0, 1, 2], dtype=np.int32)
	dense_shape = np.array([4, 4], dtype=np.int32)

	dense = swl_ml_util.sparse_to_dense(sparse_indices, sparse_values, dense_shape, default_value=-1, dtype=np.int32)
	print('Dense tensor =\n', dense)

def main():
	#to_one_hot_encoding_test()

	#time_based_learning_rate_test()
	#drop_based_learning_rate_test()

	sequences_to_sparse_test()
	sparse_to_sequences_test()
	sequences_to_dense_test()
	dense_to_sequences_test()
	dense_to_sparse_test()
	sparse_to_dense_test()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
