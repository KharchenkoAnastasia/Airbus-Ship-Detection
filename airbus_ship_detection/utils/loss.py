import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.losses import binary_crossentropy


def dice_coeff(target: tf.Tensor, pred: tf.Tensor) -> tf.Tensor:
  """
  Compute the Dice coefficient, which measures the overlap between two samples.

  Args:
      target (tf.Tensor): Ground truth tensor.
      pred (tf.Tensor): Predicted tensor.

  Returns:
      tf.Tensor: Dice coefficient, a measure of similarity between target and prediction.
  """
  smooth = 1.0
  intersection = K.sum(target * pred, axis=[1,2,3])
  union = K.sum(target, axis=[1,2,3]) + K.sum(pred, axis=[1,2,3])
  return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)


def loss(target: tf.Tensor, pred: tf.Tensor) -> tf.Tensor:
  """
  Compute the custom loss function combining binary crossentropy and Dice loss.

  Args:
      target (tf.Tensor): Ground truth tensor.
      pred (tf.Tensor): Predicted tensor.

  Returns:
      tf.Tensor: Custom loss value.
  """
  bce = binary_crossentropy(target, pred)
  dice_loss = 1-dice_coeff(target, pred)
  return bce-tf.math.log(1-dice_loss)