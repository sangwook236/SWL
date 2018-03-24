# REF [site] >> https://github.com/zizhaozhang/unet-tensorflow-keras
# REF [file] >> https://github.com/zizhaozhang/unet-tensorflow-keras/blob/master/utils.py

import tensorflow as tf

def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = tf.layers.flatten(y_true)
    y_pred_f = tf.layers.flatten(y_pred)
    intersection = tf.layers.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.layers.sum(y_true_f) + tf.layers.sum(y_pred_f) + smooth)

def dice_coeff_loss(y_true, y_pred):
    return -dice_coeff(y_true, y_pred)
