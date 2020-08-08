# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---


### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
| File                         | Description                                                                        |
| ---------------------------- | ---------------------------------------------------------------------------------- |
| `model.py`                   | Contains the model architecture, image preprocesing techniques and runs the training pipeline.                      |
| `model.json`                 | JSON file containing model architecture.             |
| `model.h5`                   | containing a trained convolution neural network                               |
| `drive.py`                   |for driving the car in autonomous mode. |
| `writeup_report.md`          |          summarizing the results |

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.json
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

While researching, I found this model for Behavorial Cloning which I found very easy to implement.
![Model](./model.png)

My final model consisted of the following layers:

| Layer         		| Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x128x3 RGB image   							|
|  Convolution 3x3     	| 3x3 stride, valid padding, outputs 30x126x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 15x63x16				|
|  Convolution 3x3     	| 3x3 stride, valid padding, outputs 13x61x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 6x30x32 				|
|  Convolution 3x3     	| 3x3 stride, valid padding, outputs 4x28x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 2x14x64 				|
|  Flatten     	| Inputs 2x14x64 , outputs 500 	|
| RELU					|												|
| Droupout | 0.5 |
|  Flatten     	| Inputs 500 , outputs 100 	|
| RELU					|												|
| Droupout | 0.5 |
|  Flatten     	| Inputs 100 , outputs 20 	|
| RELU					|												|
|  Flatten     	| Inputs 20 , outputs 1 	|

```
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
```


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 95). 
The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 81). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.
Also the model used Adam Optimizer to optimize with the learning rate of `6e-04` and mean squared error was taken into account.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 81).

#### 4. Appropriate training data

I did not created my custom dataset. I used the one provided by the udacity. Also for data augmentation and preprocessing, I didn't knew much and the results for my initial augmentation, did not yeild good results. The validation loss was about 0.3. So I took the reference/code from another project.
![Reference](https://github.com/navoshta/behavioral-cloning/blob/master/data.py)
I hope, this does not count as plagarism. Also, I came to know about this model from this project only.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to implement Behavorial Cloning and which is simple and fast as well. 

My first step was to use a convolution neural network model similar to the one provided in 1[research paper](https://arxiv.org/pdf/1604.07316.pdf) by NVIDIA, but for me it was a bit hard. So I was searching the internet for another behavorial model and I found this ![one](https://github.com/navoshta/behavioral-cloning). For me, it was simple to implement and that's why I decided that.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my model had a low mean squared error on the training set as well as on the validation set. This implied that the model was performing well and I didn't had to introduce much changes. But still, I introduced two dropout layers to avoid any overfitting at all. 

The final step was to run the simulator to see how well the car was driving around track one. The car is not that much stable that it should be, however it drove well.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 86-101) consisted of a convolution neural network with the following layers and layer sizes.

| Layer         		| Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x128x3 RGB image   							|
|  Convolution 3x3     	| 3x3 stride, valid padding, outputs 30x126x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 15x63x16				|
|  Convolution 3x3     	| 3x3 stride, valid padding, outputs 13x61x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 6x30x32 				|
|  Convolution 3x3     	| 3x3 stride, valid padding, outputs 4x28x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 2x14x64 				|
|  Flatten     	| Inputs 2x14x64 , outputs 500 	|
| RELU					|												|
| Droupout | 0.5 |
|  Flatten     	| Inputs 500 , outputs 100 	|
| RELU					|												|
| Droupout | 0.5 |
|  Flatten     	| Inputs 100 , outputs 20 	|
| RELU					|												|
|  Flatten     	| Inputs 20 , outputs 1 	|

```
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
```

![Model](./model.png)

#### Helpful Links

https://github.com/navoshta/behavioral-cloning
https://stackoverflow.com/questions/53212672/read-only-mode-in-keras
https://stackoverflow.com/questions/42339876/error-unicodedecodeerror-utf-8-codec-cant-decode-byte-0xff-in-position-0-in
https://stackoverflow.com/questions/17569679/python-attributeerror-io-textiowrapper-object-has-no-attribute-split