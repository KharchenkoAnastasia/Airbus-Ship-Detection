import numpy as np
import pandas as pd
import pytest

from airbus_ship_detection.data_processing import balance_data


@pytest.fixture
def sample_train_csv():
    # Create a sample DataFrame
    data = {
        "ImageId": ["img1", "img2", "img3", "img4", "img5"],
        "EncodedPixels": [np.nan, "1 2 3 4", "5 6 7 8", np.nan, "9 10 11 12"],
    }
    return pd.DataFrame(data)


def test_balance_data(sample_train_csv):
    # Test with default fraction
    balanced_df = balance_data(sample_train_csv)

    # Check that images without ships were removed
    assert not balanced_df["ImageId"].isin(["img1", "img4"]).any()


@pytest.mark.parametrize(
    "fraction, expected_length",
    [
        (1.0, 3),  # 100% of the data, should retain all 3 images
        (0.5, 2),  # 50% of the data, 0.5*3=1.5 should round to 2 images
        (0.2, 1),  # 20% of the data, 0.2*3=0.6 should round to 1 image
        (0.1, 0),  # 10% of the data, 0.1*3=0.3 should round to 0
    ],
)
def test_balance_data_fraction(sample_train_csv, fraction, expected_length):
    # Balance the data with the given fraction
    balanced_df = balance_data(sample_train_csv, fraction=fraction)

    # Check that the correct number of images remain
    assert len(balanced_df) == expected_length


@pytest.fixture
def sample_images():
    # Create dummy images and masks
    images = np.random.randint(0, 256, (4, 256, 256, 3), dtype=np.uint8)
    masks = np.random.randint(0, 2, (4, 256, 256, 1), dtype=np.uint8)
    yield (np.stack(images, 0), np.stack(masks, 0))


def test_augment_images(sample_images):
    # Call the augment_images function with the sample images
    # augmented_images_gen = augment_images(sample_images)
    pass


if __name__ == "__main__":
    pytest.main()
