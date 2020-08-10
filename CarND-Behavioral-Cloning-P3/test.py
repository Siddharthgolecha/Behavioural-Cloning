import matplotlib.image as mpimg
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
    x , y = make_dataset(df)
    with open("model.json", 'r') as jfile:
        model = model_from_json(jfile.read())

    model.compile(optimizer=Adam(lr=1e-04), loss= "mean_squared_error")
    weights_file = "model.h5"
    model.load_weights(weights_file)
    
    #metrics = model.evaluate(x, y)
    #print(metrics)
    
    #print(df.describe())
    
    test_i = random.sample(range(x.shape[0]), 10)
    for i in test_i:
        img = x[i][None, : , : , :]
        #print(img.shape)
        print("Prediction: ",str(float(model.predict(img, batch_size=1))))
        print("Actual: ",str(y[i]))
    