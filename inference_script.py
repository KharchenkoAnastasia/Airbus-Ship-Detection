import os
import numpy as np
#mport cv2
from PIL import Image
import tensorflow as tf
import sys
from utils.loss import loss

# Load the saved model
model_path = os.path.join('model', 'unet_model.h5')
unet_model = tf.keras.models.load_model(model_path,custom_objects={'loss': loss})

def main(directory):   
        # Create the "image_mask" folder if it doesn't exist
        mask_folder = os.path.join(directory, "image_mask")
        os.makedirs(mask_folder, exist_ok=True)
        
        # Find all image files in the directory
        image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        
        # Perform segmentation and save the results      
        for image_file in image_files:
            image_path  = os.path.join(directory, image_file)
            image = Image.open(image_path).resize((256, 256))
            image_array = np.expand_dims(image, 0) / 255.0
            segmentation = unet_model.predict(image_array)
                
            # Save the mask segmentation image
            mask_image = (segmentation[0, :, :, 0] * 255).astype(np.uint8)
            save_path = os.path.join(mask_folder, f'{os.path.splitext(image_file)[0]}_mask.png')
            pil_image = Image.fromarray(np.uint8(mask_image))
            pil_image.save(save_path)
            print(f'Saved mask segmentation for {image_file}')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python inference_script.py <directory_path>")
        sys.exit(1)

    directory_path = sys.argv[1]
    main(directory_path)

