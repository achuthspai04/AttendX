import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import cv2
import numpy as np
import sqlite3
import datetime

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

# Define roll numbers
roll_numbers = {'vishnu': 'SCM22CA031', 'jopaul': 'SCM22CA019'}

# Initialize SQLite3 database connection
conn = sqlite3.connect('attendance.db')
cursor = conn.cursor()

# Create a table if it doesn't exist
cursor.execute('''CREATE TABLE IF NOT EXISTS attendances
                (name TEXT, roll_no TEXT, time TIMESTAMP)''')

# Define a function to add data to SQLite3
def add_data_to_sqlite(name, roll_no):
    current_time = datetime.datetime.now()
    cursor.execute("INSERT INTO attendances (name, roll_no, time) VALUES (?, ?, ?)", (name, roll_no, current_time))
    conn.commit()
    print("Data added to SQLite3:", name, roll_no)

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
            roll_no = roll_numbers.get(label, '')  # Get roll number corresponding to the label
        else:
            label = 'unknown'
            roll_no = ''
        
        # Display the result on the frame
        cv2.putText(frame, f'{label} ({confidence:.2f})', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        cv2.putText(frame, f'Roll No: {roll_no}', (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # If the 'c' key is pressed, add data to SQLite3
        if cv2.waitKey(1) & 0xFF == ord('c'):
            add_data_to_sqlite(label, roll_no)
    
    # Display the frame
    cv2.imshow('Face Recognition', frame)
    
    # If the 'q' key is pressed, break out of the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
