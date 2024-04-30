import cv2

# Path to OpenCV data directory
opencv_data_dir = cv2.data.haarcascades

# Check if XML file exists
xml_file_path = opencv_data_dir + 'haarcascade_frontalface_default.xml'
if cv2.data.haarcascades:
    print("XML file exists:", xml_file_path)
else:
    print("XML file not found.")










import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import cv2
import numpy as np

# Print TensorFlow version
print(tf.__version__)

# Define directories
train_dir = 'C:/Users/achut/OneDrive/Documents/AttendX.data/train'
valid_dir = 'C:/Users/achut/OneDrive/Documents/AttendX.data/valid'
test_dir = 'C:/Users/achut/OneDrive/Documents/AttendX.data/test'

# Define batch size and image size
batch_size = 32
image_size = (224, 224)

# Create image data generators
train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Flow images from directories
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Define the model architecture
model = Sequential([
    Flatten(input_shape=(224, 224, 3)),  # Flatten layer to flatten the input
    Dense(128, activation='relu'),       # Dense layer with 128 units and ReLU activation
    Dense(3, activation='softmax')       # Dense output layer with 3 units and softmax activation
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Display model summary
model.summary()

# Train the model
model.fit(train_generator, validation_data=valid_generator, epochs=10)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(test_generator)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Save the model
model.save('C:/Users/achut/OneDrive/Documents/AttendX.data/model.h5')
print("Model saved successfully.")
