import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from keras.models import model_from_json
from keras.optimizers import Adam
import tensorflow as tf
from model import preprocess, make_dataset
import random

path = '/opt/carnd_p3/data'

if __name__ == '__main__':
    df = pd.read_csv(os.path.join(path, 'driving_log.csv'))
    x , y = make_dataset(df, False)
    with open("model.json", 'r') as jfile:
        model = model_from_json(jfile.read())

    model.compile(optimizer=Adam(lr=1e-04), loss= "mean_squared_error")
    weights_file = "model.h5"
    model.load_weights(weights_file)
    
    #metrics = model.evaluate(x, y)
    #print(metrics)
    
    #print(df.describe())
    
    pred = []
    actual = []
    
    test_i = random.sample(range(x.shape[0]), 10)
    for i in test_i:
        img = x[i][None, : , : , :]
        #print(img.shape)
        pred.append(float(model.predict(img, batch_size=1)))
        actual.append(y[i])
    
    plt.plot(pred)
    plt.plot(actual)
    plt.title('Prediction vs Actual')
    plt.ylabel('Values')
    plt.xlabel('Number of tests')
    plt.legend(['Prediction', 'Actual'], loc='upper left')
    plt.savefig('prediction.png')
    