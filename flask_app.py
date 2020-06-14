# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from keras.models import load_model
from PIL import Image
import numpy as np
import flask
import io


# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None

