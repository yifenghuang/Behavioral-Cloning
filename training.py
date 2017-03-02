import os
import csv
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import sklearn

lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines[0:7000]:
	for i in range(3):
		source_path = line[i]
		filename = source_path.split('/')[-1]
		current_path = './data/IMG/' + filename
		image = cv2.imread(current_path)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		images.append(image)
		measurement=float(line[3])
		measurements.append(measurement)

y_train=np.array(measurements)

augmented_images, augmented_measurements = [], []
for image,measurement in zip(images, measurements):
	augmented_images.append(image)
	augmented_measurements.append(measurement)
	augmented_images.append(cv2.flip(image,1))
	augmented_measurements.append(measurement*-1.0)

X_train=np.array(augmented_images)
y_train=np.array(augmented_measurements)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Activation, Dropout
from keras.layers import Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D

#nvidia model
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.9))
model.add(Dense(50))
model.add(Dropout(0.9))
model.add(Dense(10))
model.add(Dropout(0.9))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, batch_size=512, validation_split=0.2, shuffle=True, nb_epoch=100)
model.save('model.h5')
exit()
