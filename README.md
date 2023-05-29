# Airbus-Ship-Detection
### **Description**


These notebooks detail my solution to Kaggle's Airbus Ship Detection challenge.

The goal of the competition is to analyze satellite images of container ships and produce segmentation masks of the ships.

- **unet_model.h5** - binary file that stores model
- **train.py** - python file that prepare, train and save model
- **loss.py** -python file that implements the loss function used to train the segmentation model
- **test.py** - python file that takes one CLI argument that is path to directory with image samples. The output is that the segmented masks are saved as separate image files in the "image_mask" folder
- **requarements.txt** - file  that prepares the environment
- **train.ipynb,test.ipynb** - created to Colaboratory



### **Prerequisites**
- Python 3.x
- TensorFlow
- Keras
- Scikit-learn
- NumPy
- Matplotlib
- Pandas
- PIL (Python Imaging Library)



### **Installation**

#### Install the required dependencies:


```bash
pip install -r requirements.txt 
```


### **test.py**

The inference_script.py takes a directory path as a command-line argument and performs inference on all images in that directory.The output is that the segmented masks are saved as separate image files in the "image_mask" folder in the user-specified directory.

1. Open a terminal or command prompt on your computer.
2. Navigate to the directory where the "inference_script.py" file. It is necessary that the files test.py, loss.py, unet_model.h5 were in the same directory
3. Replace <directory_path> in the command with the actual path to the directory containing your image samples. Make sure to provide the full path or relative path depending on your file system. 

```bash
python test.py <directory_path>
```

### **train.py**
The train.py prepares, trains, and saves the neural network model. It is responsible for designing the architecture and training the model using the dataset.


To run the training script, use the following command:

```bash
python train.py
```


Ensure that the training dataset is properly configured and accessible within the script. You may need to adjust the script parameters, such as the number of epochs or batch size, to suit your specific requirements.

#### **Dataset**



- train_ship_segmentations_v2.csv: This file contains the RLE (Run Length Encoded is a way to encode image pixels in a more summerized way, especially when images have a black or white background) masks of ships in each image. If there are no ships, the EncodedPixel column is blank.
- train_v2: v2 contains the combined Train and Test images of the original dataset.
- test_v2: A folder with test images, size 768x768 px.
- sample_submission_v2csv: a file containing all the ImageId for the predictions of ships on those images.





#### **Data Preparation**


``` python
         ImageId                                      EncodedPixels  withShip
0  00003e153.jpg                                                NaN     False
1  0001124c7.jpg                                                NaN     False
2  000155de5.jpg  264661 17 265429 33 266197 33 266965 33 267733...      True
3  000194a2d.jpg  360486 1 361252 4 362019 5 362785 8 363552 10 ...      True
4  000194a2d.jpg  51834 9 52602 9 53370 9 54138 9 54906 9 55674 ...      True
```




The sample_submission_v2.csv file contained an “ImageId” column and an “EncodedPixels” column where the “ImageId” contains image file names and the “EncodedPixels” column contain RLE encoded masks (me target model outputs). The decoded masks contain 0’s for the pixel positions in the corresponding image that are not part of a ship and contain 1’s for pixel positions that are part of a ship. Also, each row either represents no ship or 1 ship, so there can be multiple rows with the same image ID. For the image IDs that contain no ships, the encoded pixels value is n/a. 

RLE decoding: The segmentation masks for the training images were encoded by RLE (run-length encoding) for the purpose of reducing file size. To feed the masks into segmentation models, we need to decode the masks. I used the decoding def rle_decode(mask_rle, shape=(768, 768)) function.
Example data


![Example data](https://user-images.githubusercontent.com/47922202/185091218-07f6bfea-4ba6-488c-a913-6590ab79e433.jpg)



I plotted the histogram for the image ship counts and noticed that most images contain no ship.

![image](https://github.com/KharchenkoAnastasia/Airbus-Ship-Detection/assets/47922202/59cf07ec-ff6c-4ccc-8670-0140d02df9a7)

Only images with ships were taken and 80% images

```python
    # Balance the data
    DROP_NO_SHIP_FRACTION = 0.8
    bal_train_csv=train_csv.set_index('ImageId').drop(
        train_csv.loc[
            train_csv.isna().any(axis=1),
            'ImageId'
        ]).reset_index()
    
    bal_train_csv=bal_train_csv.sample( frac = DROP_NO_SHIP_FRACTION , random_state=1)
```

Below is an image of the histogram for the down sampled distribution.

![image](https://github.com/KharchenkoAnastasia/Airbus-Ship-Detection/assets/47922202/8d9e5f10-6ac1-4b2a-aafc-066d3096846d)



#### **Generate data for model**


I did an 70/30% split of data for training and validation.
Keras data generator. A data generator is used to load and process images during training. The dataset is too large to be loaded and porcessed onced, by using a data generator only a small portion of the imagies is loaded at a time.

```python
# Keras data generator
def masks_as_image(in_mask_list):
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((768, 768), dtype = np.int16)
    #if isinstance(in_mask_list, list):
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)
    return np.expand_dims(all_masks, -1)

# Hyper parameters
IMG_SCALING = (3,3)

def keras_generator(gen_df, batch_size=4):
    all_batches = list(gen_df.groupby('ImageId'))
    out_rgb = []
    out_mask = []
    while True:
        np.random.shuffle(all_batches)
        for c_img_id, c_masks in all_batches:
            rgb_path = os.path.join(TRAIN_V2, c_img_id)
            c_img = imread(rgb_path)
            c_mask = masks_as_image(c_masks['EncodedPixels'].values)
            if IMG_SCALING is not None:
                c_img = c_img[::IMG_SCALING[0], ::IMG_SCALING[1]]
                c_mask = c_mask[::IMG_SCALING[0], ::IMG_SCALING[1]]
            out_rgb += [c_img]
            out_mask += [c_mask]
            if len(out_rgb)>=batch_size:
                yield np.stack(out_rgb, 0)/255.0, np.stack(out_mask, 0).astype(np.float)
                
                out_rgb, out_mask=[], []

```


#### **Design the Model**

 
The U-Net model is a popular architecture commonly used for image segmentation tasks, including ship detection. It is named after its U-shaped architecture.
The model consists of a contracting path (encoder) and an expanding path (decoder). The contracting path captures the context and reduces the spatial dimensions of the input image, while the expanding path recovers the spatial information and generates the segmentation mask.
   



#### **Train the Neural Network**

```python
    # Compile the model
    adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    unet_model.compile(optimizer='adam', loss=loss, metrics=[ 'binary_accuracy'])
    # Train the model
    history = unet_model.fit_generator(keras_generator(train_csv),
                                            steps_per_epoch=steps_per_epoch, 
                                            epochs=epochs, 
                                            validation_data=keras_generator(valid_csv),
                                            validation_steps=validation_steps)
```




#### **Evaluate the model**

![image](https://github.com/KharchenkoAnastasia/Airbus-Ship-Detection/assets/47922202/dc96c09c-4be7-48e2-96fb-c569ec9dba6d)

![image](https://github.com/KharchenkoAnastasia/Airbus-Ship-Detection/assets/47922202/25504a6e-35f7-4c7a-b6e4-295425c4b06c)



#### **Visualize predictions

![image](https://user-images.githubusercontent.com/47922202/185146257-acd22268-652e-4df3-beb0-72a96e5cb2ba.png)




