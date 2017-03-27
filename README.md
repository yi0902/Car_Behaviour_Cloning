# README

This program aimes to solve the problem givin in the Udacity Self-driving Car - Behavioral Cloning Project. The program will continually output a steering angle for an input image and feed it into Udacity's autonomous drive simulator to make the car turn in order to go through successfully a track.

## Solution Design

To solve the problem, I first tried to generate training data from simulator's training mode. While the control of car's steering with keyboard tended to be difficult and the generated steerings were not so smooth which could introduce too volatility when training model. Thus I decided to directly use Udacity's sample data.

### Data Treatement

As most steerings in the sample data are zero, in order to not train model to predict zero, as well as to increase number of observations of recovery cases, I used following strategy:

- **Up-sampling center images with big steerings:**
	- for center images with a steering bigger than the average level, load repeatly **5 times** the image into the input data
	- for center images with an extremely big steering (`abs(steer)> 0.5`), load repeatly **30 times** the image into the input data
	- for other center images, load just **1 time** into input data

- **Make usage of left and right cameras' images:**
	- 	for images with steering not equal to 0:
		-  load its left camera image with an offset of **+0.2** to its original steering 
		-  load its right camera image with an offset of **-0.2** to its original steering

Thus I got about **31K** images to train and validate my model. I then normalized image data to be between [`-1, 1`]. In order to fit all these data into memory for training, as well as to reduce the training time, I resized the normalized image data to be 50% of their original size (`80x160` instead of `160x320`).

### Model Architecture
For the model part, I decided to start with CommaAI's architecture which made the car get out of the track quickly. I changed quickly to NVIDIA's architecture, trained from scratch and observed the car succeeded in passing over the bridge. So I decided to continue with NVIDIA's architecture on adapting my input data treatement strategy (introducing more recovery data) and adding dropout layers into the model to control overfitting.

The network was finally composed of : 

- the input layer of depth 3
- 3 convolutional layers with relu activation
- followed by a dropout layer which drops 20% of neurons, then a relu activation
- then 2 convolutional layers with relu activation
- followed by a dropout layer which drops 50% of neurons, then a relu activation
- 1 flatten layer
- 1 fully connected layer of 1164 neurons
- followed by a dropout layer which drops 50% of neurons, then a relu activation
- 3 fully connected layers of respectively 100, 50, 10 neurons
- the output layer of 1 neuron 

I used Adam as the optimizer and the learning rate was chosed after several tries among (`0.01, 0.0001, 0.00001`) where `0.00001` made the car go better through the track. 

### Train & Prediction

The model was trained with **20 epochs** and batch size as 128 which took almost 1h on CPU on a Macbook Pro 2015.

The `drive.py` script was adapted to be able to preprocess test image data and send a lower throttle when steering angle is larger than a threshold. 

## Summary

The data treatement strategy and model architecture were defined and improuved through a test & learn method which was really a time consuming process even I started from an existing network architecture. Meanwhile, the CarND forum was really helpful for inspiration and working out this solution. 

For further improvement of the model, I would try the [live trainer](https://github.com/thomasantony/sdc-live-trainer) that could let people manual correct car's direction in autonomous mode and take that as input for training.

