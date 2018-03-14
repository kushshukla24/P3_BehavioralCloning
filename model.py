import pandas as pd
import numpy as np
import cv2

df = pd.read_csv("driving_log.csv")
df.columns = ["Center Image", "Left Image", "Right Image", "Steering Angle", "Throttle", "Break", "Speed"]
df["Center Image"] = df["Center Image"].str.replace("\\", "/")
df["Left Image"] = df["Left Image"].str.replace("\\", "/")
df["Right Image"] = df["Right Image"].str.replace("\\", "/")
df["Center Image"] = df["Center Image"].str.replace("D:/Work/GitHubRepos/Udacity/Self-Driving_Cars/windows_sim", ".")
df["Left Image"] = df["Left Image"].str.replace("D:/Work/GitHubRepos/Udacity/Self-Driving_Cars/windows_sim", ".")
df["Right Image"] = df["Right Image"].str.replace("D:/Work/GitHubRepos/Udacity/Self-Driving_Cars/windows_sim", ".")
df["Center Image"] = df["Center Image"].str.replace("D:/Work/GitHubRepos/Udacity/Self-Driving_Cars/trial", ".")
df["Left Image"] = df["Left Image"].str.replace("D:/Work/GitHubRepos/Udacity/Self-Driving_Cars/trial", ".")
df["Right Image"] = df["Right Image"].str.replace("D:/Work/GitHubRepos/Udacity/Self-Driving_Cars/trial", ".")

print("Collected Data:" + str(len(df)))
measurments = []
images = []
for image_path, steer_angle in zip(df["Center Image"], df["Steering Angle"]):
	image = cv2.imread(image_path)
	images.append(image)
	measurments.append(steer_angle)
	#images.append(np.fliplr(image))
	#measurments.append(-steer_angle)

images = np.asarray(images)
measurments = np.asarray(measurments)

print("Images are of shape: " + str(images[0].shape))
print("Augmented data set examples: " + str(measurments.shape[0]))

from sklearn.utils import shuffle
def generator(X, y, batch_size=32):
    m = len(y)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, m, batch_size):
            yield shuffle(X[offset:offset+batch_size], y[offset:offset+batch_size])

# compile and train the model using the generator function
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(images, measurments, test_size=0.2)

train_generator = generator(X_train, y_train, batch_size=64)
validation_generator = generator(X_valid, y_valid, batch_size=64)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Conv2D, Cropping2D, MaxPooling2D

def Lenet5():
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
	model = Sequential()
	model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=images[0].shape))
	model.add(Lambda(lambda x: (x/255.)-0.5))
	model.add(Conv2D(3,kernel_size=(5,5), strides=2, padding='valid', activation="relu"))
	#model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
	model.add(Conv2D(24,kernel_size=(5,5), strides=2, padding='valid', activation="relu"))
	#model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
	model.add(Conv2D(36,kernel_size=(5,5), strides=2, padding='valid', activation="relu"))
	#model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
	model.add(Conv2D(48,kernel_size=(3,3), strides=1, padding='valid', activation="relu"))
	#model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
	model.add(Conv2D(64,kernel_size=(3,3), strides=1, padding='valid', activation="relu"))
	#model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
	model.add(Flatten())
	model.add(Dropout(0.5))
	model.add(Dense(1164, activation="relu"))
	model.add(Dropout(0.5))
	model.add(Dense(100, activation="relu"))
	model.add(Dropout(0.5))
	model.add(Dense(50, activation="relu"))
	model.add(Dropout(0.5))
	model.add(Dense(10, activation="relu"))
	model.add(Dropout(0.5))
	model.add(Dense(1))
	return model

model = Lenet5()
model.compile(loss='mse', optimizer='adam')

history_object = model.fit_generator(train_generator, steps_per_epoch=190, validation_data = validation_generator, validation_steps=48, nb_epoch=5, verbose=1)
# #model.fit(images, measurments, validation_split=0.2, shuffle=True, nb_epoch=10)
model.save('model.h5')

# ### plot the training and validation loss for each epoch
import matplotlib.pyplot as plt
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()