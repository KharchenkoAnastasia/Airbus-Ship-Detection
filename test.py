import os
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from keras.losses import binary_crossentropy
import keras.backend as K
import sys
import loss 
from loss import loss



# Load the saved model
unet_model = tf.keras.models.load_model('unet_model.h5',custom_objects={'loss': loss})

def main(directory):
    
        # Create the "image_mask" folder if it doesn't exist
        mask_folder = os.path.join(directory, "image_mask")
        os.makedirs(mask_folder, exist_ok=True)
        # Find all image files in the directory
        image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        
        # Perform print the results       
        for image_file in image_files:
            c_path = os.path.join(directory, image_file)
            c_img = Image.open(c_path).resize((256, 256))
            first_img = np.expand_dims(c_img, 0)/255.0
            first_seg = unet_model.predict(first_img)
            
        
            # Save the mask segmentation image
            mask_image = (first_seg[0, :, :, 0] * 255).astype(np.uint8)
            save_path = os.path.join(mask_folder, f'{os.path.splitext(image_file)[0]}_mask.png')
            cv2.imwrite(save_path, mask_image)
            print(f'Saved mask segmentation for {image_file}')




if __name__ == '__main__':
    # main("D:/Airbus Ship Detection/Airbus Ship Detection/test_images")
    if len(sys.argv) != 2:
        print("Usage: python test.py <directory_path>")
        sys.exit(1)
        

    directory_path = sys.argv[1]
    main(directory_path)
