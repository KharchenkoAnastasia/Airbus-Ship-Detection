from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import tensorflow as tf
from PIL import Image
from utils.loss import dice_coeff, loss


def mask_to_rle(mask: npt.NDArray[np.uint8]) -> str:
    """
    Convert a binary mask to Run-Length Encoding (RLE) format.

    Args:
        mask (np.ndarray): Binary mask of shape (height, width) with values 0 or 1.

    Returns:
        str: RLE string (e.g., "1 3 10 5") or empty string if no ships detected.
    """
    pixels = mask.flatten(order="F")  # Flatten in column-major order
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs) if np.any(mask) else ""


def load_and_preprocess_image(image_path: Path) -> npt.NDArray[np.uint8]:
    """
    Load and preprocess an image for model prediction.

    Args:
        image_path (Path): Path to the image file.

    Returns:
        np.ndarray: Downscaled image array of shape (256, 256, 3).

    Raises:
        ValueError: If the image does not have the expected shape (768, 768, 3).
    """
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    if img_array.shape != (768, 768, 3):
        raise ValueError(f"Image {image_path.name} has unexpected shape {img_array.shape}")
    # Downscale from (768, 768, 3) to (256, 256, 3)
    img_downscaled = img_array[::3, ::3, :]
    return img_downscaled.astype(np.uint8)


def predict_mask(
    model: tf.keras.Model,
    image: npt.NDArray[np.uint8],
    threshold: float = 0.5,
) -> npt.NDArray[np.uint8]:
    """
    Predict a segmentation mask using the pre-trained model.

    Args:
        model (tf.keras.Model): Loaded U-Net model.
        image (np.ndarray): Preprocessed image of shape (256, 256, 3).
        threshold (float): Threshold for binarizing the prediction. Default is 0.5.

    Returns:
        np.ndarray: Binary mask of shape (256, 256).
    """
    # pred = model.predict(image[None, ...])[0, ..., 0]
    # Normalize image to [0, 1]
    image_normalized = image.astype(np.float32) / 255.0

    # Add batch dimension → (1, 256, 256, 3)
    input_tensor = np.expand_dims(image_normalized, axis=0)

    # Model prediction
    prediction = model.predict(input_tensor)

    # Explicitly cast for mypy type safety
    prediction = cast(npt.NDArray[np.float32], prediction)

    # Remove batch dimension → (256, 256, 1)
    pred = np.squeeze(prediction, axis=0)

    return (pred > threshold).astype(np.uint8)


def upsample_mask(
    mask: npt.NDArray[np.uint8],
    target_size: tuple[int, int] = (768, 768),
) -> npt.NDArray[np.uint8]:
    """
    Upsample a binary mask to the target size.

    Args:
        mask (np.ndarray): Binary mask of shape (256, 256).
        target_size (tuple[int, int]): Desired output size, default (768, 768).

    Returns:
        np.ndarray: Upsampled mask of shape target_size.
    """
    # Add channel dimension: (256, 256) -> (256, 256, 1)
    mask_3d = mask[..., None]

    # Perform nearest-neighbor upsampling
    upsampled = tf.image.resize(mask_3d, target_size, method="nearest")

    # Convert tensor -> uint8 numpy array
    result = tf.cast(upsampled, tf.uint8).numpy()[..., 0]

    # Explicit cast for mypy type checking
    return cast(npt.NDArray[np.uint8], result)


def generate_submission() -> None:
    """
    Generate a submission CSV file for ship detection using a pre-trained U-Net model.
    """
    # Define paths
    root_dir = Path(__file__).resolve().parent.parent
    test_dir = root_dir / "data" / "test_images"
    model_path = root_dir / "model" / "unet_model.h5"

    # Load the pre-trained model
    model = tf.keras.models.load_model(
        model_path, custom_objects={"loss": loss, "dice_coeff": dice_coeff}
    )

    # Get test image paths
    test_images = list(test_dir.iterdir())
    results: list[dict[str, Any]] = []

    # Process each image
    for image_path in test_images:
        try:
            # Load and preprocess
            img_downscaled = load_and_preprocess_image(image_path)

            # Predict mask
            binary_mask = predict_mask(model, img_downscaled)

            # Upsample mask
            upsampled_mask = upsample_mask(binary_mask)

            # Convert to RLE
            rle = mask_to_rle(upsampled_mask)

            # Store result
            results.append({"ImageId": image_path.name, "EncodedPixels": rle})
        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")

    # Save results to CSV
    submission_df = pd.DataFrame(results, columns=["ImageId", "EncodedPixels"])
    submission_path = test_dir / "submission.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")


if __name__ == "__main__":
    generate_submission()
