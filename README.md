# Airbus-Ship-Detection
### **Description**


These notebooks detail my solution to Kaggle's Airbus Ship Detection challenge.

The goal of the competition is to analyze satellite images of container ships and produce segmentation masks of the ships.

- **unet_model.h5** - binary file that stores model
- **train.py** - python file that prepare, train and save model
- **test.py** - python file that takes one CLI argument that is path to directory with image samples. The output is that the segmented masks are saved as separate image files in the "image_mask" folder
- **requarements.txt** - file  that prepares the environment
- **train.ipynb,test.ipynb** - created to Colaboratory



### **Prerequisites**
- Python 3.x
- TensorFlow
- Keras
- scikit-learn
- NumPy
- -matplotlib
- pandas
- PIL (Python Imaging Library)



### **Installation**

#### Install the required dependencies:


```bash
pip install -r requirements.txt 
```


### **test.py**

The inference_script.py takes a directory path as a command-line argument and performs inference on all images in that directory.The output is that the segmented masks are saved as separate image files in the "image_mask" folder.

1. Open a terminal or command prompt on your computer.
2. Navigate to the directory where the "inference_script.py" file
3. Replace <directory_path> in the command with the actual path to the directory containing your image samples. Make sure to provide the full path or relative path depending on your file system. 

```bash
python inference_script.py <directory_path>
```

### **train.py**
The handwritten_recog.py prepares, trains, and saves the neural network model. It is responsible for designing the architecture and training the model using the dataset.


To run the training script, use the following command:

```bash
python handwritten_recog.py
```


Ensure that the training dataset is properly configured and accessible within the script. You may need to adjust the script parameters, such as the number of epochs or batch size, to suit your specific requirements.

#### **Dataset**



- train_ship_segmentations_v2.csv: This file contains the RLE (Run Length Encoded is a way to encode image pixels in a more summerized way, especially when images have a black or white background) masks of ships in each image. If there are no ships, the EncodedPixel column is blank.
- train_v2: v2 contains the combined Train and Test images of the original dataset.
- test_v2: A folder with test images, size 768x768 px.
- sample_submission_v2csv: a file containing all the ImageId for the predictions of ships on those images.



----------------------------------------------------------------------------------------------------------------------------------------------------------------------

1. Data Preparation


![sample_submission_v2 jpg](https://user-images.githubusercontent.com/47922202/185092819-fba413bc-f65c-4fc5-9177-94537e539034.png)




The sample_submission_v2.csv file contained an “ImageId” column and an “EncodedPixels” column where the “ImageId” contains image file names and the “EncodedPixels” column contain RLE encoded masks (me target model outputs). The decoded masks contain 0’s for the pixel positions in the corresponding image that are not part of a ship and contain 1’s for pixel positions that are part of a ship. Also, each row either represents no ship or 1 ship, so there can be multiple rows with the same image ID. For the image IDs that contain no ships, the encoded pixels value is n/a. 

RLE decoding: The segmentation masks for the training images were encoded by RLE (run-length encoding) for the purpose of reducing file size. To feed the masks into segmentation models, we need to decode the masks. I used the decoding def rle_decode(mask_rle, shape=(768, 768)) function.
Example data


![Example data](https://user-images.githubusercontent.com/47922202/185091218-07f6bfea-4ba6-488c-a913-6590ab79e433.jpg)



I plotted the histogram for the image ship counts and noticed that most images contain no ship.

![the image ship counts ](https://user-images.githubusercontent.com/47922202/185091790-fdd19bd0-44d2-4297-94f1-bd6ec697c480.jpg)

Only images with ships were taken and 80% images


![balace data](https://user-images.githubusercontent.com/47922202/185091750-961e8563-0f33-40e0-84a7-f657319c0350.jpg)

Below is an image of the histogram for the down sampled distribution.



----------------------------------------------------------------------------------------------------------------------------------------------------------------------


2. Generate data for model


I did an 70/30% split of data for training and validation.
Keras data generator. A data generator is used to load and process images during training. The dataset is too large to be loaded and porcessed onced, by using a data generator only a small portion of the imagies is loaded at a time.


![generator](https://user-images.githubusercontent.com/47922202/185092242-957ee84a-b360-4b46-b2bd-09a113117a05.jpg)



![test_generator](https://user-images.githubusercontent.com/47922202/185092350-ae30a360-fc1c-43f2-a62a-13a900dd68e8.jpg)

----------------------------------------------------------------------------------------------------------------------------------------------------------------------


3. Design the Model

----------------------------------------------------------------------------------------------------------------------------------------------------------------------

4. Train the Neural Network

    Optimizer: Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    Loss: dice 

    teps_per_epoch=20

    epochs=10

    validation_steps=5


----------------------------------------------------------------------------------------------------------------------------------------------------------------------

5. Plotting Results


![image](https://user-images.githubusercontent.com/47922202/185146091-d65acd83-79f1-4dff-806c-a7638c2dee04.png)

![image](https://user-images.githubusercontent.com/47922202/185146191-e58aea6e-7a4a-4db8-8886-9e658a540d45.png)

----------------------------------------------------------------------------------------------------------------------------------------------------------------------

6. Visualize predictions

![predict](https://user-images.githubusercontent.com/47922202/185101620-fb25e941-7b56-4bbd-aad6-d9fc4d5e85ea.jpg)


![image](https://user-images.githubusercontent.com/47922202/185146257-acd22268-652e-4df3-beb0-72a96e5cb2ba.png)




