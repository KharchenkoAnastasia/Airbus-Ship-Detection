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
    assert "img1" not in balanced_df["ImageId"].values
    assert "img4" not in balanced_df["ImageId"].values

    # Check that the fraction of remaining images is as expected
    expected_length = int(0.8 * 3)  # 3 images have ships, 80% of that is 2.4, which rounds to 2
    assert len(balanced_df) == expected_length


def test_balance_data_fraction(sample_train_csv):
    # Test with a different fraction
    balanced_df = balance_data(sample_train_csv, fraction=0.5)

    # Check that the correct number of images remain
    expected_length = round(0.5 * 3)  # 3 images have ships, 50% of that is 1.5, which rounds to 2
    assert len(balanced_df) == expected_length


def test_balance_data_no_removal(sample_train_csv):
    # Test with fraction=1.0 to ensure all images with ships are retained
    balanced_df = balance_data(sample_train_csv, fraction=1.0)

    # Check that all images with ships are still present
    assert len(balanced_df) == 3
    assert set(balanced_df["ImageId"].values) == {"img2", "img3", "img5"}


if __name__ == "__main__":
    pytest.main()
