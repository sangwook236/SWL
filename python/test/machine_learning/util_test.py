#!/usr/bin/env python

import sys
sys.path.append('../../src')

import numpy as np
import matplotlib.pyplot as plt
import keras
import swl

def to_one_hot_encoding_test():
	y = np.array([ 1, 3, 5, 2, 4, 0 ])

	np.set_printoptions(threshold=np.nan)

	print('[1-1] swl.machine_learning.util.to_one_hot_encoding ->', swl.machine_learning.util.to_one_hot_encoding(y), sep='\n')
	print('[1-2] keras.utils.to_categorical ->', keras.utils.to_categorical(y), sep='\n')

	print('[1-3] swl.machine_learning.util.to_one_hot_encoding ->', swl.machine_learning.util.to_one_hot_encoding(y, 10), sep='\n')
	print('[1-4] keras.utils.to_categorical ->', keras.utils.to_categorical(y, 10), sep='\n')

	#--------------------
	Y = np.reshape(y, [2, 3, 1])
	print('The shape of Y = {}'.format(Y.shape))

	YY = swl.machine_learning.util.to_one_hot_encoding(Y)
	print('[2-1] swl.machine_learning.util.to_one_hot_encoding -> {}'.format(YY.shape))
	print(YY)

	YY = keras.utils.to_categorical(Y)
	print('[2-2] keras.utils.to_categorical -> {}'.format(YY.shape))
	print(YY)

	print(keras.utils.to_categorical(Y).reshape(Y.shape[:-1] + (-1,)))  # Not correctly working.
	print('[2-3] keras.utils.to_categorical -> {}'.format(YY.shape))
	print(YY)

	YY = swl.machine_learning.util.to_one_hot_encoding(Y, 10)
	print('[2-4] swl.machine_learning.util.to_one_hot_encoding -> {}'.format(YY.shape))
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

	YY = swl.machine_learning.util.to_one_hot_encoding(Y)
	print('[3-1] swl.machine_learning.util.to_one_hot_encoding -> {}'.format(YY.shape))
	print(YY)

	YY = keras.utils.to_categorical(Y)
	print('[3-2] keras.utils.to_categorical -> {}'.format(YY.shape))
	print(YY)

	print(keras.utils.to_categorical(Y).reshape(Y.shape[:-1] + (-1,)))  # Not correctly working.
	print('[3-3] keras.utils.to_categorical -> {}'.format(YY.shape))
	print(YY)

	YY = swl.machine_learning.util.to_one_hot_encoding(Y, 10)
	print('[3-4] swl.machine_learning.util.to_one_hot_encoding -> {}'.format(YY.shape))
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
		learning_rates.append(swl.machine_learning.util.time_based_learning_rate(epoch, initial_learning_rate, decay_rate))

	plt.figure()
	plt.plot(epochs, learning_rates)

def drop_based_learning_rate_test():
	initial_learning_rate = 0.001
	drop_rate = 0.5
	epoch_drop = 10000.0

	epochs = range(100000)
	learning_rates = []
	for epoch in epochs:
		learning_rates.append(swl.machine_learning.util.drop_based_learning_rate(epoch, initial_learning_rate, drop_rate, epoch_drop))

	plt.figure()
	plt.plot(epochs, learning_rates)

def main():
	to_one_hot_encoding_test()

	time_based_learning_rate_test()
	drop_based_learning_rate_test()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
