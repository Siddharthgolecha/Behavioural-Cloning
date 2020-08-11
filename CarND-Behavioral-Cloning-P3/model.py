import numpy as np
import os
import pandas as pd
import tensorflow as tf
import skimage.transform as sktransform
import random
import matplotlib.image as mpimg
import shutil
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPool2D, Dropout, Flatten, Lambda
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn import model_selection
from datetime import datetime

path = '/opt/carnd_p3/data'

cameras = ['left', 'center', 'right']
cameras_steering_correction = [0.25, 0.0, -0.25]

def preprocess(image, top_offset=.375, bottom_offset=.125):
    """
    Applies preprocessing pipeline to an image: crops `top_offset` and `bottom_offset`
    portions of image, resizes to 66x200 px and scales pixel values to [0, 1].
    """
    top = int(top_offset * image.shape[0])
    bottom = int(bottom_offset * image.shape[0])
    image = sktransform.resize(image[top:-bottom, :], (66, 200, 3))
    return image

def flip( x, y):
    flip_indices = random.sample(range(x.shape[0]), int(x.shape[0] / 2))
    x[flip_indices] = x[flip_indices, :, ::-1, :]
    y[flip_indices] = -y[flip_indices]
    return (x,y)

def make_dataset( dataset, train = True):
    n = int(len(dataset)/4)
    x = []
    y = []
    for i in range(len(dataset)):
        val = dataset.steering.values[i]
        j = np.random.randint(len(cameras)) if train else 1
        #if i%n==0:
            #print(val)
            #print(cameras[j])
        x.append(preprocess(mpimg.imread(os.path.join(path, dataset[cameras[j]].values[i].strip()))))
        y.append( val + cameras_steering_correction[j])
    x = np.array(x)#/127
    y = np.array(y)#/127
    return flip(x, y)
    #return (x, y)

if __name__ == '__main__':

    try:
    
        log = open("log.txt" , "a")
        log.write(str(datetime.now().strftime('%c')))

        test_size = 0.3

        df = pd.read_csv(os.path.join(path, 'driving_log.csv'))
        train, valid = model_selection.train_test_split(df, test_size=test_size)
        train_x , train_y = make_dataset(train)
        #print(len(train_y))
        #print(sum(train_y > 0))
        print("Training Data acquired!")

        valid_x , valid_y = make_dataset(valid , False)
        print("Validation Data acquired!")

        print("Data acquired!")

        # Model architecture from NVIDIA Research Paper
        model = Sequential()
        model.add(Lambda(lambda x: x / 64 - 0.5, input_shape=(66, 200, 3)))
        model.add(Conv2D(24, 5, strides = (2,2), input_shape=(66, 200, 3), activation='relu'))
        model.add(Conv2D(36, 5, strides = (2,2), input_shape=(31, 98, 24), activation='relu'))
        model.add(Conv2D(48, 5, strides = (2,2), input_shape=(14, 47, 36), activation='relu'))
        model.add(Conv2D(64, 3, input_shape=(5, 22, 48), activation='relu'))
        model.add(Conv2D(64, 3, input_shape=(3, 20, 64), activation='relu'))
        model.add(Flatten())
        model.add(Dense(1164, activation='relu'))
        #model.add(Dropout(0.5))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1))
        model.build()
        model.summary(print_fn = lambda x : log.write(x+'\n'))
        model.compile(optimizer=Adam(lr=1e-04), loss= "mse")

        print("Model built!")
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=0.2, min_lr=1e-6)
        datagen = ImageDataGenerator(
                    rotation_range = 20,
                    width_shift_range = 0.2,\
                    height_shift_range = 0.2,\
                    brightness_range = (0.7,0.9)
                    )

        datagen.fit(train_x)

        history = model.fit_generator(
            datagen.flow(train_x,train_y, batch_size = 32) ,
            epochs=12,
            validation_data= (valid_x, valid_y),
            callbacks=[reduce_lr]
        )
    
        log.write(str(history.history))
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['training loss', 'validation loss'], loc='upper left')
        plt.savefig('model_loss.png')
       
        model.save("model.h5")

        with open(os.path.join('./', 'model.json'), 'w') as file:
            file.write(model.to_json())

        log.write("\n=================================================================================\n\n")
        log.close()

        print("Finished!")
    
    except Exception as e:
        log.write(e)