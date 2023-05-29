import numpy as np
import pandas as pd 
import os
from skimage.io import imread
from sklearn.model_selection import train_test_split
from tensorflow import keras
import matplotlib.pyplot as plt
from matplotlib import pyplot
from keras.models import Model
from keras.layers import Input, Concatenate, Activation
from keras.layers import  UpSampling2D, Conv2D, MaxPooling2D
from tensorflow.keras.losses import binary_crossentropy
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras import layers, models
from loss import loss


TRAIN_V2='D:/Airbus Ship Detection/train_v2/'
SEGMENTATION ='D:/Airbus Ship Detection/train_ship_segmentations_v2.csv'

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

def prepare_dataset():
    
    train_v2 = os.listdir(TRAIN_V2)
    print("Len train_v2:",len(train_v2))
    train_csv = pd.read_csv(SEGMENTATION)
    train_csv['withShip'] = ~train_csv['EncodedPixels'].isnull()
    print(train_csv.head())
   
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

    # Image Ship Count Distribution
    figdf = train_csv.fillna(0).groupby('ImageId').sum()['withShip'].value_counts()
    plt.bar(figdf.index, figdf.values)
    plt.xlabel('Ship Count', fontsize=14)
    plt.ylabel('Images counts', fontsize=14)
    plt.title('Image Ship Count Distribution', fontsize=18)
    plt.show()


    # Balance the data
    DROP_NO_SHIP_FRACTION = 0.8
    bal_train_csv=train_csv.set_index('ImageId').drop(
        train_csv.loc[
            train_csv.isna().any(axis=1),
            'ImageId'
        ]).reset_index()
    
    bal_train_csv=bal_train_csv.sample( frac = DROP_NO_SHIP_FRACTION , random_state=1)
    

    #Balanced Dataset w/o Ship
    figdf=bal_train_csv.groupby('ImageId').count()['withShip'].value_counts()
    plt.bar(figdf.index, figdf.values)
    plt.xlabel('Ship Count', fontsize=14)
    plt.ylabel('Images counts', fontsize=14)
    plt.title('Balanced Dataset w/o Ship', fontsize=18)
    plt.show()
    

    # Split train and validate data
    b_train_csv, b_valid_csv = train_test_split(bal_train_csv, test_size = 0.3)

    return b_train_csv, b_valid_csv 

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
IMG_SCALING = (3,3)

def keras_generator(gen_df, batch_size=4):
    all_batches = list(gen_df.groupby('ImageId'))
    out_rgb = []
    out_mask = []
    while True:
        np.random.shuffle(all_batches)
        for c_img_id, c_masks in all_batches:
            rgb_path = os.path.join(TRAIN_V2, c_img_id)
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


def build_model():
    


    # Design the Model

    
    
    img = layers.Input((256, 256, 3))

    c1 = layers.Conv2D(8, (3, 3), activation='relu', padding='same') (img)
    c1 = layers.Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
    p1 = layers.MaxPooling2D((2, 2)) (c1)

    c2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
    c2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
    p2 = layers.MaxPooling2D((2, 2)) (c2)

    c3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
    c3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
    p3 = layers.MaxPooling2D((2, 2)) (c3)

    c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
    c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
    p4 = layers.MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same') (p4)
    c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same') (c5)

    u6 = layers.UpSampling2D((2, 2)) (c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
    c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same') (c6)

    u7 = layers.UpSampling2D((2, 2)) (c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
    c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same') (c7)

    u8 = layers.UpSampling2D((2, 2)) (c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
    c8 = layers.Conv2D(16, (3, 3), activation='relu', padding='same') (c8)

    u9 = layers.UpSampling2D((2, 2)) (c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
    c9 = layers.Conv2D(8, (3, 3), activation='relu', padding='same') (c9)

    o = layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same') (c9)

    unet_model = models.Model(inputs=[img], outputs=[o])
        
    unet_model.summary()
    return unet_model




def train_model(unet_model, train_csv, valid_csv, steps_per_epoch, epochs,validation_steps): 
    # Compile the model
    adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    # adam = tf.keras.optimizers.legacy.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False) 
    unet_model.compile(optimizer='adam', loss=loss, metrics=[ 'binary_accuracy'])
    #mc = ModelCheckpoint('best_model.h5', monitor='val_dice_coef', mode='max', verbose=1, save_best_only=True)
    
    # Train the model
    history = unet_model.fit_generator(keras_generator(train_csv),
                                            steps_per_epoch=steps_per_epoch, 
                                            epochs=epochs, 
                                            validation_data=keras_generator(valid_csv),
                                            validation_steps=validation_steps)
   
    return history


def evaluate_model(history):
    # Plotting Results  
    fig = plt.figure()
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.title('loss')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend()
    pyplot.show()
    
    fig = plt.figure()
    pyplot.plot(history.history['binary_accuracy'], label='train')
    pyplot.plot(history.history['val_binary_accuracy'], label='test')
    pyplot.title('binary_accuracy')
    pyplot.ylabel('binary_accuracy')
    pyplot.xlabel('epoch')
    pyplot.legend()
    pyplot.show()

def save_model(model):
    # Save the model
    model.save_weights('unet_model_w.h5')
    model.save('unet_model.h5')

def main():
    # Set hyperparameter
    steps_per_epoch=100
    epochs = 10
    validation_steps=50

    # Prepare the dataset
    train_csv, valid_csv  = prepare_dataset()

    #Build the model

    model = build_model()
    
    

    #Train the model
    history=train_model(model, train_csv, valid_csv,  steps_per_epoch, epochs,validation_steps)

    # Evaluate the model
    evaluate_model(history)

    #Save the model
    save_model(model)



if __name__ == '__main__':
    main()
