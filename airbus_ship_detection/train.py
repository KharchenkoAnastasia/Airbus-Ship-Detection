from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from airbus_ship_detection.data_exploration import (
    ship_count_distribution,
    show_example_data,
)
from airbus_ship_detection.data_processing import balance_data
from airbus_ship_detection.model import UNet

ROOT_DIR = Path(__file__).parent.parent
TRAIN_V2 = ROOT_DIR / "data" / "train_v2"
SEGMENTATION = ROOT_DIR / "data" / "train_ship_segmentations_v2.csv"


def main() -> None:
    # Load and inspect the data
    train_v2 = list(TRAIN_V2.iterdir())
    print("Len train_v2:", len(train_v2))
    train_csv = pd.read_csv(SEGMENTATION)
    print(train_csv.head())

    # Example data
    ImageId = "0a1a7f395.jpg"
    show_example_data(ImageId, train_csv, TRAIN_V2)

    # Image Ship Count Distribution
    ship_count_distribution(train_csv)

    # Balance the data
    train_csv = balance_data(train_csv)

    # Balanced Dataset w/o Ship
    ship_count_distribution(train_csv)

    # Split train and validate data
    train_csv, valid_csv = train_test_split(train_csv, test_size=0.3)

    # Set epochs
    epochs = 1

    # Create and train the UNet model
    unet_model = UNet()
    unet_model.compile_model()

    unet_model.fit_model(train_csv, TRAIN_V2, valid_csv, TRAIN_V2, epochs)
    unet_model.plot_training_history()
    unet_model.save_model()


if __name__ == "__main__":
    main()
