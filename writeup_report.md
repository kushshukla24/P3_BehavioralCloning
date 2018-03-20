# **Behavioral Cloning** 

---

**Behavioral Cloning System**

Self-driving cars faces numerous situations when drove on the road. Its essential that the car is trained on a real scenario with good driving behaviour. This will prepare the self-driving car to act correctly, independent of any instructions from a driver. 

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/center.jpg "CenterImage"
[image2]: ./writeup_images/center_track2.jpg "CenterImageTrack2"
[image3]: ./writeup_images/collectedDataSteeringAngleDistribution.png "Initial Histogram"
[image4]: ./writeup_images/collectedIncludingLRDataSteeringAngleDistribution.png "Final Histogram"
[image5]: ./writeup_images/VisualizingLossAcrossEpochs.png "Loss Over Epochs"
[image6]: ./writeup_images/preprocessing.png "Preprocessed Image"
[image7]: ./writeup_images/nVidia_model.png "ModelVisualization"

---
### Concepts Used
* Deep Neural Network
* Convolutional Neural Network
* Overfitting / Underfitting of Neural Network Model
* Regularization
* Hyperparameter Tuning

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* ***model.py*** containing the script to create and train the model
* ***drive.py*** for driving the car in autonomous mode
* ***model.h5*** containing a trained convolution neural network 
* ***writeup_report.md*** summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The ***model.py*** file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model consists of total 9 layers, distributed as follows:
* a normalization layer, 
* 5 convolution layer, and,
* 3 fully connected layer. 

The model is defined using Keras framework within **ConvNet** method of ***model.py (line 79-100)***. 

The normalization layer utilizes hard-coded normalizer to accelerate processing via GPU.

The first 3 convolutional layers are strided (2x2) with kernel size of 5x5, the remaining 2 convolutional layers are non-strided with kernel size of 3x3. All the convolutional layers weights are introduced with valid padding, penalized with L2 regularization to avoid overfitting, and followed by ReLU activation for introducing non-linearity in the model.

The last 3 fully connected layers also utilize ReLU for non-linearity. Between the fully connected layers, dropout layers are inserted to enhance the robustness of the model over test set. 

#### 2. Attempts to reduce overfitting in the model

The model contains L2 regularization as kernel regularizer for convolutional layers as well as 4 dropout layers between fully connected layers in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer with (small then default) learning rate of 0.0001, to converge slowly and capture small tweeks in steering angle.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. 

I used a combination of center, left and right camera images with 1 lap in forward direction on track 1, one lap in backward direction on track 1 and 1 lap in forward direction on track 2.
The steering angles were corrected with a correction factor of 0.25 to make up for images taken from left and right camera.

I also cropped the images from top and bottom to deduct the features not required for training, hence less data and faster computation.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to work in small increments of progress.

My first step was to use convolution neural network model similar to the LeNet-5 architecture. I thought this model will be appropriate because it contains convolution layers along with fully connected layers which will help to train images. This model worked well for traffic sign classification and here we had a similar input however the output is continuous as opposed to discrete classification. Therefore, I kept "Mean Squared Error" as loss function to update and develop single continuous response from training. The model was able to learn and can drive to some extent autonomously on simulator. 

I later learned about nVidia model published (http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) specifically for self-driving cars and tried to apply the same. With this new model in place I can see some improvements in validation set loss and also my car can drive to more distance on the track 1 in autonomous mode of simulator.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set (80% training and 20% validation). I found that my new model had a low mean squared error on the training set (~0.246) but a high mean squared error (~0.462) on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model to introduce L2 Regularizations in the convolutional layers and Dropout layers between the fully connected layers. This resulted my validation loss going below the training set. 

![alt text][image5]

I suspected this behavior and found below reason to convince myself :
> One possibility: If you are using dropout regularization layer in your network, it is reasonable that the validation error is smaller than training error. Because usually dropout is activated when training but deactivated when evaluating on the validation set. You get a more smooth (usually means better) function in the latter case.

https://stats.stackexchange.com/questions/187335/validation-error-less-than-training-error

But still the error is ~10%, and driving autonomously the car is felling out of the track and unable to recover. To improve the driving behavior in these cases, I utilized left and right camera images with steering correction as well as included data from track 2, to import more variety of training to the model.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 115-137) consisted of a convolution neural network with the following layers:

```sh
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lambda_1 (Lambda)            (None, 66, 200, 3)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 31, 98, 24)        1824
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 14, 47, 36)        21636
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 12, 45, 48)        15600
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 10, 43, 64)        27712
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 8, 41, 64)         36928
_________________________________________________________________
flatten_1 (Flatten)          (None, 20992)             0
_________________________________________________________________
dropout_1 (Dropout)          (None, 20992)             0
_________________________________________________________________
dense_1 (Dense)              (None, 100)               2099300
_________________________________________________________________
dropout_2 (Dropout)          (None, 100)               0
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050
_________________________________________________________________
dropout_3 (Dropout)          (None, 50)                0
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510
_________________________________________________________________
dropout_4 (Dropout)          (None, 10)                0
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11
=================================================================
Total params: 2,208,571
Trainable params: 2,208,571
Non-trainable params: 0
_________________________________________________________________
```


Here is a visualization of the architecture from nVidia paper (http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) as this project follows the same with additional L2 regularization and dropout layers:

![alt text][image7]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded around 1 laps on track 1 clockwise and 1 lap on track 1 counter-clockwise using center lane driving. The track 1 consists of mostly left turns and only a single right turn. Therefore for avoiding bias in training data I recorded data in reverse direction from track 1. Here is an example image of center lane driving:

![alt text][image1]

I then recorded 1 lap on track 2 in clockwise direction, as track 2 consists of lot many sharp turns in both left and right side, this will help to train the model for large steerting angle on either side.

![alt text][image2]

With this data, I checked for the distribution of training data steering angle through histogram (15 bins):

![alt text][image3]

Clearly, the data contains mostly steering angle ~0, as most of the time the car is moving in the center on straight road. 

So, to balance and add more examples of left and right turn through steering. I included left and right images (which were unused previously) and added an correction of +0.25 and -0.25 to steering angle. The values 0.25 is choosen empirically from reading blog posts. This augmented my training data set (3X) and provided me new examples with left and right turn. Now the distribution became:

![alt text][image4]

An alternative way could be to record data through simulator with instances car recovering from the left side and right sides of the road back to center. But the usage of left and right images with correction, this is not required as enough data is present to train the network for smooth driving over turns. 

After the collection process, I had 28,308 data points. I then preprocessed this data by cropping each image from top and bottom, as the top portion of the image mostly consisted of the scenary and the bottom the hood of the car. Additionally, I resized the image to 66x200 and converted YUV color scale as suggested in nVidia model. (http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
I randomly shuffled the data set and put 20% of the data into a validation set

![alt text][image6]

This preprocessing helped me reduce the pixel content in the image. But still given the volume of aggregate pixel information embedded in the data, it was not possible to load data in memory at once. So, I wrote a generator method and utilized keras fit_generator to load data only in batches as per batch size, shuffle and train.


I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I trained my model for 10 epochs with Adam optimizer taking a learning rate of 0.0001