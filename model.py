import pandas as pd
import numpy as np
import cv2

from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.models import Sequential, model_from_json
from keras.layers import Conv2D, MaxPooling2D, Input, AveragePooling2D, Flatten, Dense, Dropout, ELU, Activation
from keras.optimizers import SGD, Adam
from keras.regularizers import l2


ch, row, col = 3, 66, 200
ext_steer = 0.5
rep_size = 5
ext_size = 30
offset = 0.2


def read_data(file_name):

	# load driving log data
	df = pd.read_csv('data/%s' % file_name, sep=',')
	# load left and right cameras images when steering is not zero
	df_side = df[df.steering != 0]

	input_images = []
	labels_list = []

	avg_steer = np.mean(df.steering.tolist())

	# load center images
	images_list = list(df.center)
	for i in range(0, len(images_list)):
		
		img = np.asarray(Image.open('data/%s' % images_list[i]))
		steer = df.steering[i]

		# upsize images with extreme steering 
		if abs(steer) > ext_steer:
			for j in range(0, ext_size):
				input_images.append(img)
				labels_list.append(steer)

		# upsize images with steering bigger than averagge level
		elif abs(steer) > avg_steer:
			for j in range(0, rep_size):
				input_images.append(img)
				labels_list.append(steer)

		# for zero steering or small steering, just load one time the image
		else:
			input_images.append(img)
			labels_list.append(steer)
    
	# load side images 
	side_images_list = list(df_side.left) + list(df_side.right)
	for img_name in side_images_list:
		img_name = img_name[1:]
		img = np.asarray(Image.open('data/%s' % img_name))
		input_images.append(img)

	# load side labels
	steering_left_image = [x + offset for x in df_side.steering]
	steering_right_image = [x - offset for x in df_side.steering]
	labels_list = labels_list + steering_left_image + steering_right_image
    
	# convert to array format
	input_images = np.asarray(input_images)
	output_labels = np.asarray(labels_list)

	print("Data loading done!")
	print(input_images.shape)
	print(output_labels.shape)

	return input_images, output_labels


# Define preprocess function for images
def preprocess_input(image_array):
	
	# normalization
	image_array = image_array.astype('float32')
	image_array = image_array / 128 - 0.5

	# resize images
	images_resized = []
	for img in image_array:
		img_resized = cv2.resize(img, (col, row))
		images_resized.append(img_resized)

	images_resized = np.asarray(images_resized)

	print("Data processing done!")
	return images_resized


# Define the model
def creat_model_nvidia():
    
    model = Sequential()

    model.add(Conv2D(24, 5, 5, input_shape=(row, col, ch), subsample=(2, 2), border_mode="valid", activation='relu'))
    model.add(Conv2D(36, 5, 5, subsample=(2, 2), border_mode="valid", activation='relu'))
    model.add(Conv2D(48, 5, 5, subsample=(2, 2), border_mode="valid"))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    
    model.add(Conv2D(64, 3, 3, subsample=(1, 1), border_mode="valid", activation='relu'))
    model.add(Conv2D(64, 3, 3, subsample=(1, 1), border_mode="valid"))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
        
    model.add(Flatten())
    
    model.add(Dense(1164))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    
    model.add(Dense(100))
    model.add(Dense(50))    
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(optimizer=Adam(lr=0.00001), loss="mse")

    return model


if __name__ == '__main__':

 	# load image and label data
 	input_images, output_labels = read_data('driving_log.csv')

 	# preprocess image data
 	input_images_resized = preprocess_input(input_images)

 	# split train & test data
 	X_train, X_test, y_train, y_test = train_test_split(input_images_resized, output_labels, test_size=0.2, random_state=0)

 	# create model
 	model = creat_model_nvidia()
 	model.summary()

 	# train the model
 	model.fit(X_train, y_train, batch_size=128, nb_epoch=20, verbose=1, shuffle=True, validation_data=(X_test, y_test))

 	# save model and weights
 	model_json = model.to_json()
 	with open("model.json", "w") as json_file:
 		json_file.write(model_json)

 	model.save_weights("model.h5")
 	print("Model and weights saved!")