# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 10:42:23 2022

@author: kharc
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from skimage.io import imread
from sklearn.model_selection import train_test_split
from tensorflow import keras
import matplotlib.pyplot as plt
from matplotlib import pyplot
from keras.models import Model
from keras.layers import Input, Concatenate, Activation
from keras.layers import  UpSampling2D, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import binary_crossentropy
import tensorflow.keras.backend as K
import tensorflow as tf

TRAIN_V2='../Airbus Ship Detection/train_v2/'
SEGMENTATION ='../Airbus Ship Detection/train_ship_segmentations_v2.csv'

print(os.listdir("../Airbus Ship Detection"))

train_v2 = os.listdir(TRAIN_V2)
print("Len train_v2:",len(train_v2))


train_csv = pd.read_csv(SEGMENTATION)
train_csv['withShip'] = ~train_csv['EncodedPixels'].isnull()
print(train_csv.head())




# =============================================================================
def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction


# =============================================================================
# Example data
ImageId='0a2c6480e.jpg'
img = imread(TRAIN_V2 + ImageId)
img_masks = train_csv.loc[train_csv['ImageId'] == ImageId, 'EncodedPixels'].tolist()
all_masks = np.zeros((768, 768))
for mask in img_masks:
    all_masks += rle_decode(mask)

fig, axarr = plt.subplots(1, 3, figsize=(15, 40))
axarr[0].axis('off')
axarr[1].axis('off')
axarr[2].axis('off')
axarr[0].imshow(img)
axarr[1].imshow(all_masks)
axarr[2].imshow(img)
axarr[2].imshow(all_masks, alpha=0.4)
plt.tight_layout(h_pad=0.1, w_pad=0.1)
plt.show()

# =============================================================================
# Image Ship Count Distribution
figdf = train_csv.fillna(0).groupby('ImageId').sum()['withShip'].value_counts()
plt.bar(figdf.index, figdf.values)
plt.xlabel('Ship Count', fontsize=14)
plt.ylabel('Images counts', fontsize=14)
plt.title('Image Ship Count Distribution', fontsize=18)
plt.show()


# ============================================================================
# Balance the data
DROP_NO_SHIP_FRACTION = 0.8
bal_train_csv=train_csv.set_index('ImageId').drop(
    train_csv.loc[
        train_csv.isna().any(axis=1),
        'ImageId'
    ]).reset_index()

bal_train_csv=bal_train_csv.sample( frac = DROP_NO_SHIP_FRACTION , random_state=1)


# =============================================================================
#Balanced Dataset w/o Ship
figdf=bal_train_csv.groupby('ImageId').count()['withShip'].value_counts()
plt.bar(figdf.index, figdf.values)
plt.xlabel('Ship Count', fontsize=14)
plt.ylabel('Images counts', fontsize=14)
plt.title('Balanced Dataset w/o Ship', fontsize=18)
plt.show()



# =============================================================================
# Split train and validate data
b_train_csv, b_valid_csv = train_test_split(bal_train_csv, test_size = 0.3)
# ----------------------------------------------------------------------------------
# Keras data generator
def masks_as_image(in_mask_list):
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((768, 768), dtype = np.int16)
    #if isinstance(in_mask_list, list):
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)
    return np.expand_dims(all_masks, -1)
# Hyper parameters
IMG_SCALING = (1, 1)

def keras_generator(gen_df, batch_size=4):
    all_batches = list(gen_df.groupby('ImageId'))
    out_rgb = []
    out_mask = []
    while True:
        np.random.shuffle(all_batches)
        for c_img_id, c_masks in all_batches:
            rgb_path = os.path.join('train_v2', c_img_id)
            c_img = imread(rgb_path)
            c_mask = masks_as_image(c_masks['EncodedPixels'].values)
            if IMG_SCALING is not None:
                c_img = c_img[::IMG_SCALING[0], ::IMG_SCALING[1]]
                c_mask = c_mask[::IMG_SCALING[0], ::IMG_SCALING[1]]
            out_rgb += [c_img]
            out_mask += [c_mask]
            if len(out_rgb)>=batch_size:
                yield np.stack(out_rgb, 0)/255.0, np.stack(out_mask, 0).astype(np.float)
                
                out_rgb, out_mask=[], []
                
# Testing the Generator
train_gen = keras_generator(bal_train_csv,6)
train_x, train_y = next(train_gen)
print('x', train_x.shape, train_x.min(), train_x.max())
print('y', train_y.shape, train_y.min(), train_y.max())

# =============================================================================
# 
# =============================================================================
# Design the Model

inp = Input(shape=(768, 768, 3))

# first block
conv_1_1 = Conv2D(32, (3, 3), padding='same')(inp)
conv_1_1 = Activation('relu')(conv_1_1)

conv_1_2 = Conv2D(32, (3, 3), padding='same')(conv_1_1)
conv_1_2 = Activation('relu')(conv_1_2)

pool_1 = MaxPooling2D(2)(conv_1_2)


# second block
conv_2_1 = Conv2D(64, (3, 3), padding='same')(pool_1)
conv_2_1 = Activation('relu')(conv_2_1)

conv_2_2 = Conv2D(64, (3, 3), padding='same')(conv_2_1)
conv_2_2 = Activation('relu')(conv_2_2)

pool_2 = MaxPooling2D(2)(conv_2_2)


# third block
conv_3_1 = Conv2D(128, (3, 3), padding='same')(pool_2)
conv_3_1 = Activation('relu')(conv_3_1)

conv_3_2 = Conv2D(128, (3, 3), padding='same')(conv_3_1)
conv_3_2 = Activation('relu')(conv_3_2)

pool_3 = MaxPooling2D(2)(conv_3_2)


# fourth block
conv_4_1 = Conv2D(256, (3, 3), padding='same')(pool_3)
conv_4_1 = Activation('relu')(conv_4_1)

conv_4_2 = Conv2D(256, (3, 3), padding='same')(conv_4_1)
conv_4_2 = Activation('relu')(conv_4_2)

pool_4 = MaxPooling2D(2)(conv_4_2)


# fifth block
conv_5_1 = Conv2D(512, (3, 3), padding='same')(pool_4)
conv_5_1 = Activation('relu')(conv_5_1)

conv_5_2 = Conv2D(512, (3, 3), padding='same')(conv_5_1)
conv_5_2 = Activation('relu')(conv_5_2)

pool_5 = MaxPooling2D(2)(conv_5_2)


# first decoding block
up_1 = UpSampling2D(2, interpolation='bilinear')(pool_5)
conc_1 = Concatenate()([conv_5_2, up_1])

conv_up_1_1 = Conv2D(512, (3, 3), padding='same')(conc_1)
conv_up_1_1 = Activation('relu')(conv_up_1_1)

conv_up_1_2 = Conv2D(512, (3, 3), padding='same')(conv_up_1_1)
conv_up_1_2 = Activation('relu')(conv_up_1_2)

# second decoding block
up_2 = UpSampling2D(2, interpolation='bilinear')(conv_up_1_2)
conc_2 = Concatenate()([conv_4_2, up_2])

conv_up_2_1 = Conv2D(256, (3, 3), padding='same')(conc_2)
conv_up_2_1 = Activation('relu')(conv_up_2_1)

conv_up_2_2 = Conv2D(256, (3, 3), padding='same')(conv_up_2_1)
conv_up_2_2 = Activation('relu')(conv_up_2_2)


# third decodinc block
up_3 = UpSampling2D(2, interpolation='bilinear')(conv_up_2_2)
conc_3 = Concatenate()([conv_3_2, up_3])

conv_up_3_1 = Conv2D(128, (3, 3), padding='same')(conc_3)
conv_up_3_1 = Activation('relu')(conv_up_3_1)

conv_up_3_2 = Conv2D(128, (3, 3), padding='same')(conv_up_3_1)
conv_up_3_2 = Activation('relu')(conv_up_3_2)


# fourth decoding block
up_4 = UpSampling2D(2, interpolation='bilinear')(conv_up_3_2)
conc_4 = Concatenate()([conv_2_2, up_4])

conv_up_4_1 = Conv2D(64, (3, 3), padding='same')(conc_4)
conv_up_4_1 = Activation('relu')(conv_up_4_1)

conv_up_4_2 = Conv2D(64, (3, 3), padding='same')(conv_up_4_1)
conv_up_4_2 = Activation('relu')(conv_up_4_2)


# fifth decoding block
up_5 = UpSampling2D(2, interpolation='bilinear')(conv_up_4_2)
conc_5 = Concatenate()([conv_1_2, up_5])
conv_up_5_1 = Conv2D(32, (3, 3), padding='same')(conc_5)
conv_up_5_1 = Activation('relu')(conv_up_5_1)

conv_up_5_2 = Conv2D(1, (3, 3), padding='same')(conv_up_5_1)
result = Activation('sigmoid')(conv_up_5_2)


unet_model = Model(inputs=inp, outputs=result)

unet_model.summary()



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


adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False) 
unet_model.compile(optimizer='adam', loss=loss, metrics=[ 'binary_accuracy'])
mc = ModelCheckpoint('best_model.h5', monitor='val_dice_coef', mode='max', verbose=1, save_best_only=True)

loss_history = unet_model.fit_generator(keras_generator(b_train_csv),
                                        steps_per_epoch=20, 
                                        epochs=10, 
                                        validation_data=keras_generator(b_valid_csv),
                                        validation_steps=5)

#=============================================================================
# Plotting Results

fig = plt.figure()
pyplot.plot(loss_history.history['loss'], label='train')
pyplot.plot(loss_history.history['val_loss'], label='test')
pyplot.title('loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend()
pyplot.show()

fig = plt.figure()
pyplot.plot(loss_history.history['binary_accuracy'], label='train')
pyplot.plot(loss_history.history['val_binary_accuracy'], label='test')
pyplot.title('binary_accuracy')
pyplot.ylabel('binary_accuracy')
pyplot.xlabel('epoch')
pyplot.legend()
pyplot.show()

# unet_model.load_weights('best_model.h5')
unet_model.save_weights('unet_model_w.h5')
unet_model.save('unet_model.h5')


