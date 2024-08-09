import os
import numpy as np
import pandas as pd
from skimage.io import imread
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

def ship_count_distribution(train_csv):
    """
    Create a bar plot showing the distribution of ship counts in the dataset.

    Args:
        train_csv (DataFrame): DataFrame containing ship information.

    Returns:
        None: Displays the plot.
    """
    plt.figure()
    ship_count = train_csv['EncodedPixels'].notnull().astype(int).groupby(train_csv['ImageId']).sum().value_counts()
    plt.bar(ship_count.index, ship_count.values)
    plt.xlabel('Ship Count', fontsize=14)
    plt.ylabel('Image Counts', fontsize=14)
    plt.title('Image Ship Count Distribution', fontsize=18)
    plt.show()

def rle_decode(mask_rle, shape=(768, 768)):
    """
        Decode a run-length encoded mask and return it as a numpy array.

        Args:
            mask_rle (str): Run-length encoded mask in the format (start length).
            shape (tuple): Desired shape (height, width) of the returned array.

        Returns:
            numpy.ndarray: Decoded mask where 1 represents the mask and 0 represents the background.

        """
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    mask_rle = mask_rle.split()
    mask_starts = [int(mask_rle[i]) - 1 for i in range(0, len(mask_rle), 2)]
    mask_lengths = [int(mask_rle[i + 1]) for i in range(0, len(mask_rle), 2)]
    for start, length in zip(mask_starts, mask_lengths):
        mask[start: start + length] = 1

    mask = mask.reshape(shape).T
    return mask


def show_example_data(ImageId, train_csv,image_dataset):
    """"
        Display an example of an image and its corresponding masks.
    """
    img = imread(os.path.join(image_dataset,ImageId))
    img_masks = train_csv.loc[train_csv['ImageId'] == ImageId, 'EncodedPixels'].tolist()
    all_masks = np.zeros((768, 768))
    all_masks = np.bitwise_or.reduce([rle_decode(mask) for mask in img_masks], axis=0)

    fig, axarr = plt.subplots(1, 3, figsize=(15, 40))
    for ax in axarr:
        ax.axis('off')
    axarr[0].imshow(img)
    axarr[1].imshow(all_masks)
    axarr[2].imshow(img)
    axarr[2].imshow(all_masks, alpha=0.4)
    plt.tight_layout(h_pad=0.1, w_pad=0.1)
    plt.show()

def balance_data(train_csv, drop_no_ship_fraction=0.8):
    """
        Balance the training data by removing images without ships and sampling a fraction of the remaining data.

        Args:
            train_csv (DataFrame): DataFrame containing the training data.
            drop_no_ship_fraction (float): Fraction of images without ships to drop. Default is 0.8.

        Returns:
            DataFrame: Balanced training data DataFrame, containing a subset of images with ships.

    """
    # Delete images without ships
    bal_train_csv = train_csv.set_index('ImageId').drop(
        train_csv.loc[
            train_csv.isna().any(axis=1),
            'ImageId'
        ]).reset_index()

    # Fraction of images
    bal_train_csv = bal_train_csv.sample(frac=drop_no_ship_fraction, random_state=1)

    return bal_train_csv

def masks_as_image(in_mask_list):
    """
    Take the individual ship masks and create a single mask array for all ships
    """
    all_masks = np.zeros((768, 768), dtype=np.int16)
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks |= rle_decode(mask)
    return np.expand_dims(all_masks, -1)


def keras_generator(gen_df,  image_dataset, batch_size=8):
    """
   Generate batches of RGB images and corresponding masks from the input DataFrame.

   Args:
       gen_df (DataFrame): DataFrame containing image and mask information.
       batch_size (int): Number of samples per batch. Default is BATCH_SIZE.
       image_dataset: directory where images are stored.

   Yields:
       tuple: A tuple containing the batch of RGB images and corresponding masks.
              The RGB images are normalized to the range [0, 1], and masks are not normalized.
   """
    IMG_SCALING = (3, 3)
    all_batches = list(gen_df.groupby('ImageId'))
    out_rgb = []
    out_mask = []
    while True:
        np.random.shuffle(all_batches)
        for c_img_id, c_masks in all_batches:
            rgb_path = os.path.join(image_dataset, c_img_id)
            c_img = imread(rgb_path)
            c_mask = masks_as_image(c_masks['EncodedPixels'].values)
            if IMG_SCALING is not None:
                c_img = c_img[::IMG_SCALING[0], ::IMG_SCALING[1]]
                c_mask = c_mask[::IMG_SCALING[0], ::IMG_SCALING[1]]
            out_rgb.append(c_img)
            out_mask.append(c_mask)
            if len(out_rgb) >= batch_size:
                yield np.stack(out_rgb, 0) / 255.0, np.stack(out_mask, 0).astype(float)
                out_rgb, out_mask = [], []


def augment_images(images):
    """
    Generate augmented images and masks using image data augmentation.

    Args:
        images (iterable): Iterable containing pairs of original images and masks.

    Yields:
        tuple: A tuple containing the augmented images and masks.

    """
    # Define the augmentation settings
    aug_args = dict(
        rotation_range=45,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.01,
        zoom_range=[0.9, 1.25],
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='reflect',
        brightness_range=[0.5, 1.5]
    )

    # Create an image data generator for augmentation
    image_datagen = ImageDataGenerator(**aug_args)

    # Create a mask data generator for augmentation
    aug_args_mask = aug_args.copy()
    del aug_args_mask['brightness_range']
    mask_datagen = ImageDataGenerator(**aug_args_mask)

    # Generate augmented images and masks
    for image, mask in images:
        augmented_images = image_datagen.flow(255 * image, batch_size=image.shape[0], shuffle=True)
        augmented_masks = mask_datagen.flow(mask, batch_size=image.shape[0], shuffle=True)

        yield next(augmented_images) / 255.0, next(augmented_masks)
