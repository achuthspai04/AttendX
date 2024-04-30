import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import cv2
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
import datetime

# Initialize Firebase Admin SDK
cred = credentials.Certificate("D:/AttendX/attend-x-3cb5f-firebase-adminsdk-r7yme-899df73190.json"
)  # Replace with your service account key file path
firebase_admin.initialize_app(cred)
db = firestore.client()

# Define the threshold for confidence scores
confidence_threshold = 0.5  # Adjust this threshold as needed

# Load the pre-trained model
model = tf.keras.models.load_model('C:/Users/achut/OneDrive/Documents/AttendX.data/model.h5')

# Open the webcam
cap = cv2.VideoCapture(0)

# Load the haarcascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define class labels
class_labels = ['vishnu', 'jopaul', 'unknown']

# Define a function to add data to Firestore
def add_data_to_firestore(name):
    doc_ref = db.collection(u'attendances').document()
    doc_ref.set({
        u'name': name,
        u'time': datetime.datetime.now()
    })
    print("Data added to Firestore:", name)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (224, 224))
        face_img = np.expand_dims(face_img, axis=0)
        
        # Ensure the image is float type before division
        face_img = face_img.astype('float32')
        face_img /= 255.0  # Normalize the image
        
        # Predict the face
        prediction = model.predict(face_img)
        predicted_class = np.argmax(prediction)
        confidence = prediction[0][predicted_class]
        
        # Determine the label based on the predicted class and confidence score
        if confidence >= confidence_threshold:
            label = class_labels[predicted_class]
        else:
            label = 'unknown'
        
        # Display the result on the frame
        cv2.putText(frame, f'{label} ({confidence:.2f})', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # If the 'c' key is pressed, add data to Firestore
        if cv2.waitKey(1) & 0xFF == ord('c'):
            add_data_to_firestore(label)
    
    # Display the frame
    cv2.imshow('Face Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
