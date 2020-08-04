from flask import Flask, jsonify, make_response, request, abort, redirect, send_file
import logging
import cv2
from cv2 import CascadeClassifier
from cv2 import COLOR_BGR2GRAY
import keras
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

app = Flask(__name__)


def get_model():
    global model
    model = load_model('EVA model/_mini_XCEPTION.102-0.66.hdf5')
    print(" * Model loaded!")



print(" * Loading Keras model...")
get_model()

@app.route('/')
def index():
    return redirect("https://google.com", code=302)

@app.route('/classifyImage', methods=['POST'])
def upload():
        image = request.files['image'].read()
        face_cascade = cv2.CascadeClassifier('detection_models/haarcascade_frontalface_default.xml')
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            crop = img[y:y+h, x:x+w]
        roi = cv2.resize(crop, (64, 64))
        roi = roi.astype("float") / 255.0
        roi = image.img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        preds = model.predict(roi)
        emotion_probability = np.max(preds)
        EMOTIONS = ["angry","disgust","scared", "happy", "sad", "surprised","neutral"]
        label = EMOTIONS[np.argmax(preds)]
        
        return jsonify(label)
  

if __name__ == '__main__':
    app.run(debug=True)
