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

def loadData():
	'''Returns pandas dataframe containing the file location for Images from center, left and right
	camera, steering angle, throttle, break and speed from driving_log.csv file.
	Returned data frame has columns = ["Center Image", "Left Image", "Right Image", "Steering Angle", "Throttle", "Break", "Speed"]
	driving_csv.log file should be in the run directory
	'''
	if not os.path.exists("driving_log.csv"):
		print("driving_log.csv file missing. Can't load the data!")
		return None

	cleanPath = lambda x: "IMG/"+os.path.basename(x)
	
	df = pd.read_csv("driving_log.csv")
	df.columns = ["Center Image", "Left Image", "Right Image", "Steering Angle", "Throttle", "Break", "Speed"]
	df.drop(columns = ["Throttle", "Break", "Speed"], inplace=True)
	df["Center Image"] = df["Center Image"].apply(cleanPath)
	df["Left Image"] = df["Left Image"].apply(cleanPath)
	df["Right Image"] = df["Right Image"].apply(cleanPath)
	
	images = df["Center Image"].append(df["Left Image"]).append(df["Right Image"])
	angles = df["Steering Angle"].append(df["Steering Angle"]+0.25).append(df["Steering Angle"]-0.25)
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
	Retuns the resizedImage
	'''
	return cv2.resize(image, final_size[::-1])

def cropImage(image, top=60, bottom=30, left=0, right=0):
	'''
	Returns the cropped image
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
			measurements = []
			for X_batch, y_batch in zip(X[offset:offset+batch_size], y[offset:offset+batch_size]):
				image = resizeImage(cropImage(cv2.imread(X_batch)))
				image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
				images.append(image)
				measurements.append(y_batch)
			
			yield np.asarray(images), np.asarray(measurements)

def ConvNet():
	'''
	Returns the model (from NVIDIA paper)
	used for training
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

df = loadData()
print("Collected Data:" + str(len(df)))
X_train, X_valid, y_train, y_valid = createTrainValidSet(df)

train_generator = generator(X_train, y_train, batch_size=64)
validation_generator = generator(X_valid, y_valid, batch_size=64)
model = ConvNet()
model.compile(loss='mse', optimizer=Adam(lr=1e-4))

history_object = model.fit_generator(train_generator, steps_per_epoch=356, validation_data = validation_generator, validation_steps=87, nb_epoch=5, verbose=1)
model.save('model.h5')

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

#print(X_train[0])
#img = cv2.imread(X_train[0])
#print(y_train[0])
#cv2.imwrite("writeup_images/original.jpg", img)
#img = cropImage(img)
#cv2.imwrite("writeup_images/cropped.jpg", img)
#img = resizeImage(img)
#cv2.imwrite("writeup_images/resized.jpg", img)
#plt.show()


# # #model.fit(images, measurments, validation_split=0.2, shuffle=True, nb_epoch=10)

# print(df.describe())
# print(df.dtypes)
#df.loc[:, "Steering Angle"].plot()
# plt.hist(df['Steering Angle'], bins=15, align='left')
# plt.show()
#(images, measurments) = read_data_and_create_working_data(df)
# measurments = []
# images = []
# leftSteeringCorrection = 0.25
# rightSteeringCorrection = -0.25
# for centerImagePath, leftImagePath, rightImagePath, steerAngle in zip(df["Center Image"], df["Left Image"], df["Right Image"], df["Steering Angle"]):
	# image = cv2.imread(centerImagePath)
	# images.append(image)
	# measurments.append(steerAngle)
	# #image = cv2.imread(leftImagePath)
	# #images.append(image)
	# #measurments.append(steerAngle + leftSteeringCorrection)
	# #image = cv2.imread(rightImagePath)
	# #images.append(image)
	# #measurments.append(steerAngle + rightSteeringCorrection)
	# #images.append(np.fliplr(image))
	# #measurments.append(-steerAngle)

# images = np.asarray(images)
# measurments = np.asarray(measurments)

# imageSize = images[0].shape
# #images = images.reshape((len(images), imageSize[0], imageSize[1], imageSize[2]))
# print("Images are of shape: " + str(imageSize))
# print("Augmented data set examples: " + str(measurments.shape[0]))
# print(images.shape)

# from sklearn.utils import shuffle
# def generator(X, y, batch_size=32):
    # m = len(y)
    # while 1: # Loop forever so the generator never terminates
        # for offset in range(0, m, batch_size):
            # yield shuffle(X[offset:offset+batch_size], y[offset:offset+batch_size])

# # compile and train the model using the generator function
# from sklearn.model_selection import train_test_split
# X_train, X_valid, y_train, y_valid = train_test_split(images, measurments, test_size=0.2)

# train_generator = generator(X_train, y_train, batch_size=64)
# validation_generator = generator(X_valid, y_valid, batch_size=64)

# 

# def Lenet5():
	# model = Sequential()
	# model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=images[0].shape))
	# model.add(Lambda(lambda x: (x/255.)-0.5))
	# model.add(Conv2D(6,kernel_size=(5,5), strides=1, padding='valid', activation="relu"))
	# model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
	# model.add(Conv2D(16,kernel_size=(5,5), strides=1, padding='valid', activation="relu"))
	# model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
	# model.add(Conv2D(64,kernel_size=(5,5), strides=1, padding='valid', activation="relu"))
	# model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
	# model.add(Flatten())
	# model.add(Dropout(0.5))
	# model.add(Dense(120, activation="relu"))
	# model.add(Dropout(0.5))
	# model.add(Dense(84, activation="relu"))
	# model.add(Dropout(0.5))
	# model.add(Dense(1))
	# return model


