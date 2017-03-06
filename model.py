import csv
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import sklearn

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)


from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)#split the train and validation samples

def generator(samples, batch_size=256):
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		sklearn.utils.shuffle(samples)
		for offset in range(0,num_samples, batch_size):
			batch_samples=samples[offset:offset+batch_size]
			
			images = []
			measurements = []
			for line in batch_samples:
				for i in range(3):#3 images, left, middle and right
					source_path = line[i]
					filename = source_path.split('/')[-1]
					current_path = './data/IMG/' + filename
					image = cv2.imread(current_path)
					image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)					
					measurement=float(line[3])

					images.append(image)
					measurements.append(measurement)
					images.append(cv2.flip(image,1))#data augmentation
					measurements.append(measurement*-1.0)

			X_train=np.array(images)
			y_train=np.array(measurements)
			
			yield sklearn.utils.shuffle(X_train, y_train)


from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Activation, Dropout
from keras.layers import Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D

#using generator to train much more data(I have a 32 Gb RAM which is quite enough for 10 laps driving data so this is not that necessary in this case)
train_generator = generator(train_samples)
validation_generator = generator(validation_samples)

#nvidia model(from classroom with dropout layers added)
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))#regularization and normalization
model.add(Cropping2D(cropping=((70,25), (0,0))))#crop the sky and ground of the images to make training faster
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))#using relu to introduce nonlinearity
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dropout(0.2))#dropout layer to reduce overfitting
model.add(Dense(10))
model.add(Dropout(0.1))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')#using adam optimizer to fit a regression model
model.fit_generator(train_generator, samples_per_epoch=
            6*len(train_samples), nb_epoch=100, validation_data=validation_generator, 
            nb_val_samples=6*len(validation_samples))
model.save('model.h5')
