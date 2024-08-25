from pathlib import Path
from typing import Generator, Iterable, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from skimage.io import imread
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from airbus_ship_detection.data_exploration import masks_as_image


def balance_data(train_csv: pd.DataFrame, fraction: float = 0.8) -> pd.DataFrame:
    """
    Balance the training data by removing images without ships and sampling a fraction of the remaining data.

    Args:
        train_csv (DataFrame): DataFrame containing the training data.
        fraction (float): Fraction of images without ships to drop. Default is 0.8.

    Returns:
        pd.DataFrame: Balanced training data DataFrame, containing a subset of images with ships.

    """
    # Delete images without ships
    bal_train_csv = (
        train_csv.set_index("ImageId")
        .drop(train_csv.loc[train_csv.isna().any(axis=1), "ImageId"])
        .reset_index()
    )

    # Fraction of images
    bal_train_csv = bal_train_csv.sample(frac=fraction, random_state=1)

    return bal_train_csv


def keras_generator(
    gen_df: pd.DataFrame, image_dataset: Union[str, Path], batch_size: int = 8
) -> Generator[Tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]], None, None]:
    """
    Generate batches of RGB images and corresponding masks from the input DataFrame.

    Args:
        gen_df (pd.DataFrame): DataFrame containing image and mask information.
        image_dataset (Union[str, Path]): Directory where images are stored.
        batch_size (int): Number of samples per batch. Default is 8.

    Yields:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the batch of RGB images and corresponding masks.
              The RGB images are normalized to the range [0, 1], and masks are not normalized.
    """
    IMG_SCALING = (3, 3)
    all_batches = list(gen_df.groupby("ImageId"))
    out_rgb = []
    out_mask = []

    image_dataset_path = Path(image_dataset)

    while True:
        np.random.shuffle(all_batches)
        for c_img_id, c_masks in all_batches:
            rgb_path = image_dataset_path / c_img_id
            c_img: npt.NDArray[np.uint8] = imread(rgb_path)  # type: ignore
            c_mask = masks_as_image(c_masks["EncodedPixels"].values)

            if IMG_SCALING is not None:
                c_img = c_img[:: IMG_SCALING[0], :: IMG_SCALING[1]]
                c_mask = c_mask[:: IMG_SCALING[0], :: IMG_SCALING[1]]

            out_rgb.append(c_img)
            out_mask.append(c_mask)

            if len(out_rgb) >= batch_size:
                yield np.stack(out_rgb, 0).astype(np.uint8), np.stack(out_mask, 0).astype(np.uint8)
                out_rgb, out_mask = [], []


def augment_images(
    images: Iterable[Tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]],
) -> Generator[Tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]], None, None]:
    """
    Generate augmented images and masks using image data augmentation.

    Args:
        images (Iterable[Tuple[np.ndarray, np.ndarray]]): Iterable containing pairs of original images and masks.

    Yields:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the augmented images and masks.
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
        fill_mode="reflect",
        brightness_range=[0.5, 1.5],
    )

    # Create an image data generator for augmentation
    image_datagen = ImageDataGenerator(**aug_args)

    # Create a mask data generator for augmentation
    aug_args_mask = aug_args.copy()
    del aug_args_mask["brightness_range"]
    mask_datagen = ImageDataGenerator(**aug_args_mask)

    # Generate augmented images and masks
    for image, mask in images:
        augmented_images = image_datagen.flow(255 * image, batch_size=image.shape[0], shuffle=True)
        augmented_masks = mask_datagen.flow(mask, batch_size=image.shape[0], shuffle=True)

        yield next(augmented_images) / 255.0, next(augmented_masks)
