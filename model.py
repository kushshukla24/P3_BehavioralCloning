import pandas as pd
import numpy as np
import cv2
import os.path
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Conv2D, Cropping2D, MaxPooling2D
from keras import regularizers
from keras.optimizers import Adam

# this value is empirical and taken after reading blog posts
STEERING_ANGLE_CORRECTION = 0.25

def loadData():
	'''Returns pandas dataframe containing the file location for Images from center, left and right
	camera, steering angle, throttle, break and speed from driving_log.csv file.
	Returned data frame has columns = ["Center Image", "Left Image", "Right Image", "Steering Angle", "Throttle", "Break", "Speed"]
	driving_csv.log file should be in the run directory
	'''
	if not os.path.exists("driving_log.csv"):
		print("driving_log.csv file missing. Can't load the data!")
		return None
	
	df = pd.read_csv("driving_log.csv")
	df.columns = ["Center Image", "Left Image", "Right Image", "Steering Angle", "Throttle", "Break", "Speed"]
	
	# these columns are not required, so better to drop these column and move ahead with less data
	df.drop(columns = ["Throttle", "Break", "Speed"], inplace=True)
	
	# lambda function to clear the path names with absolute path and return only the image file name
	cleanPath = lambda x: "IMG/"+os.path.basename(x)
	
	df["Center Image"] = df["Center Image"].apply(cleanPath)
	df["Left Image"] = df["Left Image"].apply(cleanPath)
	df["Right Image"] = df["Right Image"].apply(cleanPath)
	
	# creating a Series of all images captured and steering angle with correction
	images = df["Center Image"].append(df["Left Image"]).append(df["Right Image"])
	angles = df["Steering Angle"].append(df["Steering Angle"]+STEERING_ANGLE_CORRECTION).append(df["Steering Angle"]-STEERING_ANGLE_CORRECTION)
	
	df = pd.concat([images, angles], axis=1)
	df.columns = ["Images", "Steering Angle"]
	return df

def createTrainValidSet(df, validation_proportion = 0.2):
	'''
	Returns a shuffles list for training and validation
	based on the provided dataframe and the validation proportion
	'''
	df = shuffle(df, random_state = 0)
	X_train, X_valid, y_train, y_valid = train_test_split(df["Images"].values, df["Steering Angle"].values, test_size=validation_proportion)
	return X_train, X_valid, y_train, y_valid

def resizeImage(image, final_size=(66,200)):
	'''
	Returns the resizedImage, given an image and final required size of the image.
	Default values for the final size are choosen specifically for the current images 
	generated from Udacity Simulator
	'''
	return cv2.resize(image, final_size[::-1])

def cropImage(image, top=60, bottom=30, left=0, right=0):
	'''
	Returns the cropped image, given an image and required prunning from the sides of the image.
	Default values for the prunning from sides are choosen specifically for the current images 
	generated from Udacity Simulator
	'''
	h,w,c = image.shape
	return image[top:h-bottom][left:w-right]

def generator(X, y, batch_size=32):
	'''
	Returns batch size (X,Y) pairs conatining the preprocessed image
	This is required to save memory. As the data is huge, it is intelligent approach to 
	load and preprocess the data on the go with the generator
	'''
	m = len(y)
	while True:
		for offset in range(0, m, batch_size):
			images = []
			angles = []
			for X_batch, y_batch in zip(X[offset:offset+batch_size], y[offset:offset+batch_size]):
				image = resizeImage(cropImage(cv2.imread(X_batch)))
				image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
				images.append(image)
				angles.append(y_batch)
			
			yield np.asarray(images), np.asarray(angles)

def Lenet5():
	'''
	Returns the Lenet-5 model
	used for training on images and steering angle
	'''
	model = Sequential()
	model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=images[0].shape))
	model.add(Lambda(lambda x: (x/255.)-0.5))
	model.add(Conv2D(6,kernel_size=(5,5), strides=1, padding='valid', activation="relu"))
	model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
	model.add(Conv2D(16,kernel_size=(5,5), strides=1, padding='valid', activation="relu"))
	model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
	model.add(Conv2D(64,kernel_size=(5,5), strides=1, padding='valid', activation="relu"))
	model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
	model.add(Flatten())
	model.add(Dropout(0.5))
	model.add(Dense(120, activation="relu"))
	model.add(Dropout(0.5))
	model.add(Dense(84, activation="relu"))
	model.add(Dropout(0.5))
	model.add(Dense(1))
	return model
	
def ConvNet():
	'''
	Returns the model (from NVIDIA paper)
    http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
	used for training on images and steering angle
	'''
	model = Sequential()
	model.add(Lambda(lambda x: (x/255.)-0.5, input_shape=(66,200,3)))
	model.add(Conv2D(24,kernel_size=(5,5), strides=2, padding='valid', activation="relu", kernel_regularizer=regularizers.l2(0.001)))
	model.add(Conv2D(36,kernel_size=(5,5), strides=2, padding='valid', activation="relu", kernel_regularizer=regularizers.l2(0.001)))
	model.add(Conv2D(48,kernel_size=(3,3), strides=1, padding='valid', activation="relu", kernel_regularizer=regularizers.l2(0.001)))
	model.add(Conv2D(64,kernel_size=(3,3), strides=1, padding='valid', activation="relu", kernel_regularizer=regularizers.l2(0.001)))
	model.add(Conv2D(64,kernel_size=(3,3), strides=1, padding='valid', activation="relu", kernel_regularizer=regularizers.l2(0.001)))
	model.add(Flatten())
	model.add(Dropout(0.5))
	model.add(Dense(100, activation="relu"))
	model.add(Dropout(0.5))
	model.add(Dense(50, activation="relu"))
	model.add(Dropout(0.5))
	model.add(Dense(10, activation="relu"))
	model.add(Dropout(0.5))
	model.add(Dense(1))
	return model

def drawSteeringAngleDistribution(data, bins=15):
	'''
	Plots the histogram based on the provided data (expected Steering angle data for this project)
	'''
	plt.hist(data, bins=15, align='left', color='green', alpha=0.8, edgecolor='black')
	plt.title("Distribution of Steering angle in collected data")
	plt.xlabel("Steering Angles")
	plt.ylabel("Bin Frequency")
	plt.show()
	pass


def visualizeLossOverEpochs(history_object):
	'''
	Plots the Line Plot based on the training and history
	object from keras fit_generator method
	'''
	plt.plot(history_object.history['loss'])
	plt.plot(history_object.history['val_loss'])
	plt.title('model mean squared error loss')
	plt.ylabel('mean squared error loss')
	plt.xlabel('epoch')
	plt.legend(['training set', 'validation set'], loc='upper right')
	plt.show()
	pass


# Program Execution Begins Here
df = loadData()
print("Collected Data:" + str(len(df)))

drawSteeringAngleDistribution(df['Steering Angle'])

X_train, X_valid, y_train, y_valid = createTrainValidSet(df)

train_generator = generator(X_train, y_train, batch_size=64)
validation_generator = generator(X_valid, y_valid, batch_size=64)

model = ConvNet()
# The learning rate is intentionally taken smaller than default value (0.001), to converge slowly and adhere to small tweeks in steering angle
model.compile(loss='mse', optimizer=Adam(lr=1e-4))

history_object = model.fit_generator(train_generator, steps_per_epoch=356, validation_data = validation_generator, validation_steps=87, nb_epoch=10, verbose=1)
model.save('model.h5')

visualizeLossOverEpochs(history_object)
