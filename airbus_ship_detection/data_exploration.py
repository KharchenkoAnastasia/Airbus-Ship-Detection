from pathlib import Path
from typing import List, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image


def ship_count_distribution(train_csv: pd.DataFrame) -> None:
    """
    Create a bar plot showing the distribution of ship counts in the dataset.

    Args:
        train_csv (pd.DataFrame): DataFrame containing ship information.

    Returns:
        None: Displays the plot.
    """
    plt.figure()
    ship_count = (
        train_csv["EncodedPixels"]
        .notnull()
        .astype(int)
        .groupby(train_csv["ImageId"])
        .sum()
        .value_counts()
    )
    plt.bar(ship_count.index, ship_count.values)
    plt.xlabel("Ship Count", fontsize=14)
    plt.ylabel("Image Counts", fontsize=14)
    plt.title("Image Ship Count Distribution", fontsize=18)
    plt.show()


def rle_decode(
    mask_rle: str, shape: tuple[int, int] = (768, 768)
) -> npt.NDArray[np.uint8]:
    """
    Decode a run-length encoded mask and return it as a numpy array.

    Args:
        mask_rle (str): Run-length encoded mask in the format (start length).
        shape (tuple[int, int]): Desired shape (height, width) of the returned array.

    Returns:
        numpy.ndarray: Decoded mask where 1 represents the mask and 0 represents the background.
    """
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    # mask = np.zeros(shape[0] * shape[1])
    mask_rle_split = mask_rle.split()
    mask_starts = [int(mask_rle_split[i]) - 1 for i in range(0, len(mask_rle_split), 2)]
    mask_lengths = [
        int(mask_rle_split[i + 1]) for i in range(0, len(mask_rle_split), 2)
    ]
    for start, length in zip(mask_starts, mask_lengths):
        mask[start : start + length] = 1

    mask = mask.reshape(shape).T
    return mask


def show_example_data(
    ImageId: str, train_csv: pd.DataFrame, image_dataset: Union[str, Path]
) -> None:
    """
    Display an example of an image and its corresponding masks.

    Args:
        ImageId (str): Identifier for the image.
        train_csv (pd.DataFrame): DataFrame containing image information and masks.
        image_dataset (str): Path to the dataset directory.

    Returns:
        None: Displays the image and mask plot.
    """
    img = np.array(Image.open(Path(image_dataset) / ImageId), dtype=np.uint8)
    img_masks = train_csv.loc[train_csv["ImageId"] == ImageId, "EncodedPixels"].tolist()
    all_masks = masks_as_image(img_masks)

    fig, axarr = plt.subplots(1, 3, figsize=(15, 40))

    for ax in axarr:
        ax.axis("off")
    axarr[0].imshow(img)
    axarr[1].imshow(all_masks)
    axarr[2].imshow(img)
    axarr[2].imshow(all_masks, alpha=0.4)

    plt.tight_layout(h_pad=0.1, w_pad=0.1)
    plt.axis("off")
    plt.show()


def masks_as_image(in_mask_list: List[str]) -> npt.NDArray[np.uint8]:
    """
    Take the individual ship masks and create a single mask array for all ships
    """
    all_masks = np.zeros((768, 768), dtype=np.uint8)
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks |= rle_decode(mask)
    return np.expand_dims(all_masks, axis=-1)
