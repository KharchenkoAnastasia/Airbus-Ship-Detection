from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from skimage.io import imread


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
    img = imread(Path(image_dataset) / ImageId)
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


def masks_as_image(in_mask_list):
    """
    Take the individual ship masks and create a single mask array for all ships
    """
    all_masks = np.zeros((768, 768), dtype=np.int16)
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks |= rle_decode(mask)
    return np.expand_dims(all_masks, -1)
