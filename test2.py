import numpy as np
import cv2
import tensorflow as tf

def predict_emotion(image_path, model):
    # Update emotion labels to match the three-emotion model
    emotion_labels = ["angry", "sad", "happy"]
    
    # Load and preprocess the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48)) / 255.0  # Normalize and resize
    img = img.reshape(1, 48, 48, 1)  # Reshape for model input

    # Make prediction
    prediction = model.predict(img)
    emotion = emotion_labels[np.argmax(prediction)]
    
    return emotion

# Load the model trained on three emotions
model = tf.keras.models.load_model("three_emotion_model.h5")

# Predict
image_path = "/home/sid/model_train/data/testimg.jpeg"  # Change to your image path
predicted_emotion = predict_emotion(image_path, model)
print(f"Predicted Emotion: {predicted_emotion}")