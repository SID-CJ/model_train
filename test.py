import numpy as np
import cv2
import tensorflow as tf




def predict_emotion(image_path, model):
    emotion_labels = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48)) / 255.0
    img = img.reshape(1, 48, 48, 1)

    prediction = model.predict(img)
    emotion = emotion_labels[np.argmax(prediction)]
    
    return emotion

model = tf.keras.models.load_model("improved_emotion_model.h5")

# Predict
image_path = "/home/sid/dev/train_model/train/data/MainImage.jpg"  # Change to your image path
predicted_emotion = predict_emotion(image_path, model)
print(f"Predicted Emotion: {predicted_emotion}")