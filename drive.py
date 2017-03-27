import argparse
import base64
import json

import numpy as np
import cv2
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.optimizers import Adam

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf


sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

def preprocess(img):
	# Normalization of image
	image_array = np.asarray(img).astype('float32')
	image_array = image_array/128 - 0.5
	# Resize of image
	row, col = 66, 200
	return cv2.resize(image_array, (col, row))

@sio.on('telemetry')
def telemetry(sid, data):
    
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    
    # The current throttle of the car
    throttle = data["throttle"]
    
    # The current speed of the car
    speed = data["speed"]
    
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
 
 	# Preprocess input image 
    image_array = preprocess(image)
    transformed_image_array = image_array[None, :, :, :]
    
    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    steering_angle = float(model.predict(transformed_image_array, batch_size=1))
    
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    #throttle = 0.2
    throttle_max = 1.0
    throttle_min = -1.0
    steering_threshold = 3/25

    # Targets for speed controller
    nominal_set_speed = 20
    steering_set_speed = 5

    # Proportional gain
    K = 0.35

    # Slow down for turns
    if abs(steering_angle) > steering_threshold:
        set_speed = steering_set_speed
    else:
        set_speed = nominal_set_speed

    throttle = (set_speed - float(speed)) * K
    throttle = min(throttle_max, throttle)
    throttle = max(throttle_min, throttle)

    print(steering_angle, throttle)
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        model = model_from_json(jfile.read())

    adam = Adam(lr=0.0001)
    model.compile(adam, "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)