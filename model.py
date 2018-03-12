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

def create_data_from_image(image_path):
	return cv2.imread(image_path)

measurments = df["Steering Angle"].as_matrix()

images = []
for image_path in df["Center Image"]:
	images.append(create_data_from_image(image_path))

images = np.asarray(images)
print(images.shape)
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda

model = Sequential()
model.add(Lambda(lambda x: (x/255.)-0.5, input_shape=images[0].shape))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(images, measurments, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')