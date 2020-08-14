import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from keras.models import model_from_json
from keras.optimizers import Adam
from keras.losses import logcosh
import tensorflow as tf
from model import preprocess, make_dataset
import random

path = '/opt/carnd_p3/data'

if __name__ == '__main__':
    df = pd.read_csv(os.path.join(path, 'driving_log.csv'))
    x , y = make_dataset(df, False, 10)
    with open("model.json", 'r') as jfile:
        model = model_from_json(jfile.read())

    model.compile(optimizer=Adam(lr=2e-04), loss= "mse")
    weights_file = "model.h5"
    model.load_weights(weights_file)
    
    pred = [float(model.predict(img[None,:,:,:], batch_size=1))/4 for img in x]
    
    plt.plot(pred)
    plt.plot(y)
    plt.title('Prediction vs Actual')
    plt.ylabel('Values')
    plt.xlabel('Number of tests')
    plt.legend(['Prediction', 'Actual'], loc='upper left')
    plt.savefig('prediction.png')
    