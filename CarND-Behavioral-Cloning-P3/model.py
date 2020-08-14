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
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from keras.losses import logcosh
from sklearn import model_selection
from datetime import datetime
from sklearn.utils import shuffle
from tqdm import tqdm

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
    #print(image[top:-bottom, :].shape)
    image = sktransform.resize(image[top:-bottom, :], (66, 200, 3))
    return image

def flip( x, y):
    '''
    Flips random half of the dataset images and values to distribute the data
    equally on both the sides.
    '''
    flip_indices = random.sample(range(x.shape[0]), int(x.shape[0] / 2))
    x[flip_indices] = x[flip_indices, :, ::-1, :]
    y[flip_indices] = -y[flip_indices]
    return (x,y)

def make_dataset( dataset, train = True, cases = None):
    '''
    Augments the data and make the dataset: Adds the augmented values and images of
    left and right steering( Since the given data is extremely biased). Returns a 
    dataset.
    '''
    n = int(len(dataset)/4)
    x = []
    y = []
    if train:
        for i in tqdm(range(len(dataset)), desc= "Loading"):
            val = dataset.steering.values[i]
            j = np.random.randint(len(cameras))
            img = mpimg.imread(os.path.join(path, dataset[cameras[j]].values[i].strip()))
            if val!=0:
                indices = np.random.randint(17)
                for k in range(indices):
                    shift = np.random.uniform(-0.07,0.07)
                    #print(shift)
                    x.append(preprocess(img,top_offset= .375 + shift, bottom_offset = .125 + shift))
                    y.append((val + round(np.random.normal(shift,0.0003),5)+ cameras_steering_correction[j])*10)
            x.append(preprocess(img))
            y.append(( val + cameras_steering_correction[j])*10)
        x = np.array(x)#/127
        y = np.array(y)#/127
        x, y = flip(x,y)
    else:
        if cases == None:
            x = np.array([ preprocess(mpimg.imread(os.path.join(path, \
                                                                dataset[cameras[1]].values[i].strip())))\
                          for i in tqdm(range(len(dataset)), desc= "Loading")])
            y = np.array([dataset.steering.values[i]*10 for i in range(len(dataset))])
        elif isinstance(cases,int):
            indices = random.sample(range(len(dataset)), cases)
            x = np.array([ preprocess(mpimg.imread(os.path.join(path, \
                                                                dataset[cameras[1]].values[i].strip())))\
                          for i in indices])
            y = np.array([dataset.steering.values[i] for i in indices])
        else:
            raise ValueError("Invalid type for cases!")
    return shuffle(x, y)
                 
    
if __name__ == '__main__':
    
    log = open("log.txt" , "a")

    try:
        log.write(str(datetime.now().strftime('%c')))

        test_size = 0.3

        df = pd.read_csv(os.path.join(path, 'driving_log.csv'))
        train, valid = model_selection.train_test_split(df, test_size=test_size)
        train_x , train_y = make_dataset(train)
        print(len(train_y))
        #print(sum(train_y > 0))
        plt.figure();
        plt.hist(train_y/10,bins=101,alpha=0.5)
        plt.title('Data Distribution after augmentation')
        plt.ylabel('Frequency')
        plt.xlabel('Steering Angle')
        plt.savefig('Data_Distribution.png')
        plt.close()
        print("Training Data acquired!")
        
        valid_x , valid_y = make_dataset(valid , False)
        print("Validation Data acquired!")

        # Model architecture from NVIDIA Research Paper
        model = Sequential()
        #model.add(Lambda(lambda x: x / 64 - 0.5, input_shape=(66, 200, 3)))
        model.add(Conv2D(24, 5, strides = (2,2), input_shape=(66, 200, 3), activation='relu'))
        model.add(Conv2D(36, 5, strides = (2,2), input_shape=(31, 98, 24), activation='relu'))
        model.add(Conv2D(48, 5, strides = (2,2), input_shape=(14, 47, 36), activation='relu'))
        model.add(Conv2D(64, 3, input_shape=(5, 22, 48), activation='relu'))
        model.add(Conv2D(64, 3, input_shape=(3, 20, 64), activation='relu'))
        model.add(Flatten())
        model.add(Dense(1164, activation='relu'))
        model.add(Dropout(0.5)) #Added dropout layer to avoid overfitting
        model.add(Dense(100, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(0.5)) #Added dropout layer to avoid overfitting
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1))
        model.build()
        model.compile(optimizer=Adam(lr=2e-04), loss= "mse")
        model.summary(print_fn = lambda x : log.write(x+'\n'))

        print("Model built!")
        datagen = ImageDataGenerator(
                    brightness_range = (0.7,0.9)
                    ) #Further data augmentation

        
        datagen.fit(train_x)

        history = model.fit_generator(
            datagen.flow(train_x,train_y, batch_size = 64) ,
            epochs=64,
            validation_data= (valid_x, valid_y)
        )
    
        log.write(str(history.history))
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['training loss', 'validation loss'], loc='upper left')
        plt.savefig('model_loss.png')
        plt.close()

        model.save("model.h5")

        with open(os.path.join('./', 'model.json'), 'w') as file:
            file.write(model.to_json())

        log.write("\n=================================================================================\n\n")

        print("Finished!")
    
    except Exception as e:
        log.write(str(e))
        
    finally: 
        log.close()