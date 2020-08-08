import numpy as np
import os
import pandas as pd
import tensorflow as tf
import skimage.transform as sktransform
import random
import matplotlib.image as mpimg
import shutil
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPool2D, Dropout, Flatten
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.optimizers import Adam
from sklearn import model_selection

path = '/opt/carnd_p3/data'

#Reference code from https://github.com/navoshta/behavioral-cloning/blob/master/data.py
# Cameras we will use
cameras = ['left', 'center', 'right']
cameras_steering_correction = [.25, 0., -.25]

def preprocess(image, top_offset=.375, bottom_offset=.125):
    """
    Applies preprocessing pipeline to an image: crops `top_offset` and `bottom_offset`
    portions of image, resizes to 32x128 px and scales pixel values to [0, 1].
    """
    top = int(top_offset * image.shape[0])
    bottom = int(bottom_offset * image.shape[0])
    image = sktransform.resize(image[top:-bottom, :], (32, 128, 3))
    return image

def generate_samples(data, root_path, augment=True):
    """
    Keras generator yielding batches of training/validation data.
    Applies data augmentation pipeline if `augment` is True.
    """
    while True:
        # Generate random batch of indices
        indices = np.random.permutation(data.count()[0])
        batch_size = 128
        for batch in range(0, len(indices), batch_size):
            batch_indices = indices[batch:(batch + batch_size)]
            # Output arrays
            x = np.empty([0, 32, 128, 3], dtype=np.float32)
            y = np.empty([0], dtype=np.float32)
            # Read in and preprocess a batch of images
            for i in batch_indices:
                # Randomly select camera
                camera = np.random.randint(len(cameras)) if augment else 1
                # Read frame image and work out steering angle
                image = mpimg.imread(os.path.join(root_path, data[cameras[camera]].values[i].strip()))
                angle = data.steering.values[i] + cameras_steering_correction[camera]
                if augment:
                    # Add random shadow as a vertical slice of image
                    h, w = image.shape[0], image.shape[1]
                    [x1, x2] = np.random.choice(w, 2, replace=False)
                    k = h / (x2 - x1)
                    b = - k * x1
                    for i in range(h):
                        c = int((i - b) / k)
                        image[i, :c, :] = (image[i, :c, :] * .5).astype(np.int32)
                # Randomly shift up and down while preprocessing
                v_delta = .05 if augment else 0
                image = preprocess(
                    image,
                    top_offset=random.uniform(.375 - v_delta, .375 + v_delta),
                    bottom_offset=random.uniform(.125 - v_delta, .125 + v_delta)
                )
                # Append to batch
                x = np.append(x, [image], axis=0)
                y = np.append(y, [angle])
            # Randomly flip half of images in the batch
            flip_indices = random.sample(range(x.shape[0]), int(x.shape[0] / 2))
            x[flip_indices] = x[flip_indices, :, ::-1, :]
            y[flip_indices] = -y[flip_indices]
            yield (x, y)

if __name__ == '__main__':
    
    df = pd.read_csv(os.path.join(path, 'driving_log.csv'))
    train, valid = model_selection.train_test_split(df, test_size=0.2)
    
    print("Data acquired!")

    # Model architecture from model.png
    model = Sequential()
    model.add(Conv2D(16, 3, 3, input_shape=(32, 128, 3), activation='relu'))
    model.add(MaxPool2D(strides=2))
    model.add(Conv2D(32, 3, 3, activation='relu'))
    model.add(MaxPool2D(strides=2))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(MaxPool2D(strides=2))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1))
    model.build()
    model.compile(optimizer=Adam(lr=6e-04), loss='mean_squared_error')
    
    print("Model built!")

    history = model.fit_generator(
        generate_samples(train, path),
        samples_per_epoch=train.shape[0],
        nb_epoch=32,
        validation_data=generate_samples(valid, path,augment = False),
        nb_val_samples=valid.shape[0]
    )

    model.save("model.h5")
    
    with open(os.path.join('./', 'model.json'), 'w') as file:
        file.write(model.to_json())
        
    print("Finished!")

    backend.clear_session()