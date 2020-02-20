from keras import backend as K

# REF [site] >> https://github.com/zizhaozhang/unet-tensorflow-keras
# REF [file] >> https://github.com/zizhaozhang/unet-tensorflow-keras/blob/master/utils.py
def dice_coeff(y_true, y_pred):
    smooth = 1.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coeff_loss(y_true, y_pred):
    return -dice_coeff(y_true, y_pred)
