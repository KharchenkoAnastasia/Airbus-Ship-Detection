from tensorflow.keras.losses import binary_crossentropy
import tensorflow.keras.backend as K
import tensorflow as tf

def dice_coeff(target, pred):
  # target__==target
  smooth = 1.0
  intersection = K.sum(target * pred, axis=[1,2,3])
  union = K.sum(target, axis=[1,2,3]) + K.sum(pred, axis=[1,2,3])
  return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
  # return ((2*((pred*target).sum()))+smooth) / (pred.sum()+target.sum()+smooth)

def loss(target, pred):
  bce = binary_crossentropy(target, pred)
  dice_loss = 1-dice_coeff(target, pred)
  return bce-tf.math.log(1-dice_loss)
