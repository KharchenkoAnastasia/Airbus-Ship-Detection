import os
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Concatenate, UpSampling2D, Conv2D, MaxPooling2D
from utils.loss import loss
from airbus_ship_detection.data_processing import augment_images,keras_generator
import matplotlib.pyplot as plt


class UNet(Model):
    def __init__(self):
        super(UNet, self).__init__()
        self.build_model()

    def build_model(self):
        """
        Build the U-Net model architecture.
        """
        img = Input((256, 256, 3))

        c1 = Conv2D(8, (3, 3), activation='relu', padding='same')(img)
        c1 = Conv2D(8, (3, 3), activation='relu', padding='same')(c1)
        p1 = MaxPooling2D((2, 2))(c1)

        c2 = Conv2D(16, (3, 3), activation='relu', padding='same')(p1)
        c2 = Conv2D(16, (3, 3), activation='relu', padding='same')(c2)
        p2 = MaxPooling2D((2, 2))(c2)

        c3 = Conv2D(32, (3, 3), activation='relu', padding='same')(p2)
        c3 = Conv2D(32, (3, 3), activation='relu', padding='same')(c3)
        p3 = MaxPooling2D((2, 2))(c3)

        c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(p3)
        c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(c4)
        p4 = MaxPooling2D(pool_size=(2, 2))(c4)

        c5 = Conv2D(128, (3, 3), activation='relu', padding='same')(p4)
        c5 = Conv2D(128, (3, 3), activation='relu', padding='same')(c5)

        u6 = UpSampling2D((2, 2))(c5)
        u6 = Concatenate()([u6, c4])
        c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(u6)
        c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(c6)

        u7 = UpSampling2D((2, 2))(c6)
        u7 = Concatenate()([u7, c3])
        c7 = Conv2D(32, (3, 3), activation='relu', padding='same')(u7)
        c7 = Conv2D(32, (3, 3), activation='relu', padding='same')(c7)

        u8 = UpSampling2D((2, 2))(c7)
        u8 = Concatenate()([u8, c2])
        c8 = Conv2D(16, (3, 3), activation='relu', padding='same')(u8)
        c8 = Conv2D(16, (3, 3), activation='relu', padding='same')(c8)

        u9 = UpSampling2D((2, 2))(c8)
        u9 = Concatenate()([u9, c1])
        c9 = Conv2D(8, (3, 3), activation='relu', padding='same')(u9)
        c9 = Conv2D(8, (3, 3), activation='relu', padding='same')(c9)

        o = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(c9)

        self.unet_model = Model(inputs=[img], outputs=[o])
        self.unet_model.summary()

    def compile_model(self):
        """
        Compile the U-Net model.
        """
        self.adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0,
                                             amsgrad=False)
        self.unet_model.compile(optimizer='adam', loss=loss, metrics=['binary_accuracy'])

    def fit_model(self, train_csv, train_images, valid_csv, valid_images, epochs):
        """
        Fit the U-Net model to the training data.
        """
        self.batch_size = 4
        self.steps_per_epoch = 100
        self.validation_steps = 50
        self.history = self.unet_model.fit(augment_images(keras_generator(train_csv, train_images, self.batch_size)),
                                           steps_per_epoch=self.steps_per_epoch,
                                           epochs=epochs,
                                           validation_data=keras_generator(valid_csv,valid_images),
                                           validation_steps=self.validation_steps)

    def evaluate_model(self):
        """
        Evaluate and plot the training and validation results.
        """
        plt.figure()
        plt.plot(self.history.history['loss'], label='train')
        plt.plot(self.history.history['val_loss'], label='test')
        plt.title('loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(self.history.history['binary_accuracy'], label='train')
        plt.plot(self.history.history['val_binary_accuracy'], label='test')
        plt.title('binary_accuracy')
        plt.ylabel('binary_accuracy')
        plt.xlabel('epoch')
        plt.legend()
        plt.show()

    def save_model(self):
        # Save the model
        model_folder = os.path.join(os.path.dirname(__file__), '..', 'model')

        # Create the directory if it doesn't exist
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        model_path = os.path.join(model_folder, 'unet_model.h5')
        self.unet_model.save(model_path)