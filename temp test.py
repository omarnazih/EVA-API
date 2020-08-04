import cv2
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

#emotion detection model path
emotion_model_path = 'EVA model/_mini_XCEPTION.102-0.66.hdf5'


emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry","disgust","scared", "happy", "sad", "surprised","neutral"]
# Load the cascade
face_cascade = cv2.CascadeClassifier('detection_models/haarcascade_frontalface_default.xml')
# Read the input image
img = cv2.imread('img/Screenshot from 2020-07-25 15.14.29.jpeg')
# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
# Draw rectangle around the faces
for (x, y, w, h) in faces:
    crop = img[y:y+h, x:x+w]
# Display the output
cv2.imshow('img', crop)
cv2.waitKey()


crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
roi = cv2.resize(crop, (64, 64))
roi = roi.astype("float") / 255.0
roi = image.img_to_array(roi)
roi = np.expand_dims(roi, axis=0)
preds = emotion_classifier.predict(roi)[0]
emotion_probability = np.max(preds)
label = EMOTIONS[preds.argmax()]


print(label)  