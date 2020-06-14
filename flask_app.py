# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from PIL import Image
import numpy as np
import imutils
import cv2
import flask
from flask import jsonify
from flask import request
import io
import sys
 
# initialize our Flask application and the Keras model
app = flask.Flask(__name__)

def load_model():
    global emotion_model_path
    # parameters for loading data and images
    emotion_model_path = 'EVA model/fer2013_mini_XCEPTION.119-0.65.hdf5'
    #img_path = 'img/2020-06-1413:57:52.jpg'
    emotion_classifier = load_model(emotion_model_path, compile=False)
    

def prepare_image(img_path):

    global detection_model_path
    detection_model_path = 'detection_models/haarcascade_frontalface_default.xml'
    face_detection = cv2.CascadeClassifier(detection_model_path)
    EMOTIONS = ["angry","disgust","scared", "happy", "sad", "surprised","neutral"]
    #reading the frame
    orig_frame = cv2.imread(img_path)
    frame = cv2.imread(img_path,0)
    faces = face_detection.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    
    if len(faces) > 0:
    faces = sorted(faces, reverse=True,key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
    (fX, fY, fW, fH) = faces
    roi = frame[fY:fY + fH, fX:fX + fW]
    roi = cv2.resize(roi, (48, 48))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)
    
    return roi


#### Loading model one time in the memory so we don't have to reload it every time
#print(" * loading model...")
#load_model()


@app.route("/predict", methods=["POST"])
def predict():
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image = prepare_image(image)

            # classify the input image and then initialize the list
            # of predictions to return to the client
            preds = emotion_classifier.predict(image)
            emotion_probability = np.max(preds)
            label = EMOTIONS[preds.argmax()]

            response = {
                'prediction':{
                    label : emotion_probability
                }
            }

            return jsonify(response)        

   

    

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_model()
    app.run()


