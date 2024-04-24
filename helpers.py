import os
import cv2
import base64
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf  # noqa: F401, E402
from keras.models import load_model  # noqa: E402


model = load_model("utils/model.h5")

SIZE = 48
OBJECTS = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")

face_cascade = cv2.CascadeClassifier("utils/face.xml")


def b64_to_image(b64_string):
    """
    Convert base64 string to image
    """
    sbuf = base64.decodebytes(b64_string.encode())
    pimg = np.frombuffer(sbuf, dtype=np.uint8)
    image = cv2.imdecode(pimg, cv2.IMREAD_UNCHANGED)
    if isinstance(image.shape, tuple):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)        
        return image
    print("Invalid image")
    return None


def detect_single_face(image):
    """
    Detects a single face in an image and returns the face image
    """

    faces = face_cascade.detectMultiScale(image, 1.3, 5, minSize=(SIZE, SIZE))
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face_image = image[y : y + h, x : x + w]
        return face_image
    print("No face detected")
    return None


def preprocessing(face_image):
    """
    Preprocesses the face image
    """

    x = cv2.resize(face_image, (SIZE, SIZE))
    x = x.reshape(x.shape + (1,))
    x = np.array(x, dtype="float32")
    x = np.expand_dims(x, axis=0)
    if x.max() > 1:
        x = x / 255.0
        return x
    print("Invalid image, unable to preprocess")
    return None


def predict(b64_string):
    """
    Load the model and predict emotions from the image
    """

    image = b64_to_image(b64_string)
    if image is None:
        return None
    face_image = detect_single_face(image)
    if face_image is None:
        return None
    preprocessed = preprocessing(face_image)
    if preprocessed is None:
        return None
    predictions = model.predict(preprocessed)
    if isinstance(predictions, np.ndarray):
        predictions = OBJECTS[np.argmax(predictions)]
        return predictions
    print("Unable to predict")
    return None
