import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping

# Define dataset path
dataset_path = "data/archive/train"  # Update this to your dataset location

# Define emotion labels (adjust based on dataset)
emotion_labels = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]
label_map = {label: idx for idx, label in enumerate(emotion_labels)}

# Load images and labels
X, y = [], []

for emotion in os.listdir(dataset_path):  # Iterate over emotion folders
    emotion_folder = os.path.join(dataset_path, emotion)
    
    if os.path.isdir(emotion_folder):
        for img_name in os.listdir(emotion_folder):
            img_path = os.path.join(emotion_folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
            img = cv2.resize(img, (48, 48))  # Resize to 48x48
            X.append(img)
            y.append(label_map[emotion])  # Convert label to integer

# Convert to NumPy arrays
X = np.array(X) / 255.0  # Normalize pixel values
y = to_categorical(y, num_classes=len(emotion_labels))  # One-hot encoding

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape for CNN
X_train = X_train.reshape(-1, 48, 48, 1)
X_test = X_test.reshape(-1, 48, 48, 1)

print("✅ Dataset Loaded Successfully!")

# 🔹 DATA AUGMENTATION
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest"
)
datagen.fit(X_train)

# 🔹 CLASS WEIGHT HANDLING
class_weights = compute_class_weight('balanced', classes=np.unique(y.argmax(axis=1)), y=y.argmax(axis=1))
class_weights = dict(enumerate(class_weights))

# 🔹 CNN MODEL
model = Sequential([
    Conv2D(64, (3,3), activation='relu', input_shape=(48,48,1)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(256, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(512, (3,3), activation='relu'),
    BatchNormalization(),
    GlobalAveragePooling2D(),  # Reduces parameters and improves generalization
    
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(emotion_labels), activation='softmax')
])

# 🔹 COMPILE MODEL
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# 🔹 EARLY STOPPING CALLBACK
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 🔹 TRAIN MODEL
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    epochs=50,  # Increase epochs
    validation_data=(X_test, y_test),
    class_weight=class_weights,  # Balance class distribution
    callbacks=[early_stopping]
)

# 🔹 SAVE MODEL
model.save("improved_emotion_model.h5")
print("✅ Model Training Completed and Saved!")

# 🔹 EVALUATE MODEL
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"🎯 Test Accuracy: {test_acc:.4f}")
