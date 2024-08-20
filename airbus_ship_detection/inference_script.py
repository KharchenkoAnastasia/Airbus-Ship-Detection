import sys
from pathlib import Path
from typing import Union

import numpy as np
import tensorflow as tf
from PIL import Image

from airbus_ship_detection.utils.loss import loss

# Load the saved model
model_path = Path("../model") / "unet_model.h5"
unet_model = tf.keras.models.load_model(model_path, custom_objects={"loss": loss})


def main(directory: Union[str, Path]) -> None:
    print(f"Processing directory: {directory}")
    directory = Path(directory)

    # Create the "image_mask" folder if it doesn't exist
    mask_folder = directory / "image_mask"
    mask_folder.mkdir(parents=True, exist_ok=True)

    # Find all image files in the directory
    image_files = [f for f in directory.iterdir() if f.suffix.lower() in {".png", ".jpg", ".jpeg"}]

    # Perform segmentation and save the results
    for image_file in image_files:
        image = Image.open(image_file).resize((256, 256))
        image_array = np.expand_dims(image, 0) / 255.0
        segmentation = unet_model.predict(image_array)

        # Save the mask segmentation image
        mask_image = (segmentation[0, :, :, 0] * 255).astype(np.uint8)
        save_path = mask_folder / f"{image_file.stem}_mask.png"
        pil_image = Image.fromarray(mask_image)
        pil_image.save(save_path)
        print(f"Saved mask segmentation for {image_file.name}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inference_script.py <directory_path>")
        sys.exit(1)

    # directory_path = r"{}".format(sys.argv[1])
    directory_path = sys.argv[1]
    main(directory_path)
