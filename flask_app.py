# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
import io

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None

def load_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global model
    model = load_model('EVA model/fer2013_mini_XCEPTION.119-0.65.hdf5')