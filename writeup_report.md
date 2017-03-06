#**Behavioral Cloning** 
**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 and 5x5 filter sizes and depths in 24, 36, 48 and 64 (model.py lines 59-73) .

This model is based on Nvidia model from classroom. The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer . I also introduce dropout layer to reduce overfitting. 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 70). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 75).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and try driving smoothly using mouse.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

My first step was to use a convolution neural network model similar to the traffic sign classifier. I thought this model might be appropriate because the road condition is simple so the lenet5 can easily cope with it.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model to nvidia model that introduced in the classroom. this model is used by nvidia to drive a real autodrived car. Then I add dropout layer in last two fully connected layer and the model works even better.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 58-73) consisted of a convolution neural network with the following layers and layer sizes
1:convolutional layer: filter size: 5*5 depth 24 subsample filter size: 2x2
activation:relu
2:convolutional layer: filter size: 5*5 depth 36 subsample filter size: 2x2
activation:relu
3:convolutional layer: filter size: 5*5 depth 48 subsample filter size: 2x2
activation:relu
4:convolutional layer: filter size: 3*3 depth 64
activation:relu
5:convolutional layer: filter size: 3*3 depth 64
activation:relu
6:FC layer: 100
7:FC layer: 50
dropout 20%
8:FC layer:10
dropout 10%
9:output node

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first drive the car smoothly using my mouse to navigate and keep the car in the center of the road as possible as I can.

then I recorded few laps on track using center lane driving with keyboard arrow keys to introduce some big angle input.

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recovery from the edge of the road.

finally I drive the car smoothly using mouse just like the first step.

To augment the data set, I also flipped images and angles because the driving is always in left turn. After the collection process, I had 16000 number of data points. I then preprocessed this data by crop it and using keras lambda layer to normalized the data.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 100 and after that the accuracy is not improved at all. I used an adam optimizer so that manually training the learning rate wasn't necessary.
