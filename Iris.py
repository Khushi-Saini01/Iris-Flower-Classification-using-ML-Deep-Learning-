# ===========================================
# Iris Flower Classification using ML & DL
# Author: Khushi Saini
# ===========================================

# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# TensorFlow / Keras imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# ===========================================
# 1. Load Dataset
# ===========================================
df = pd.read_csv("iris_org.csv")

# Quick overview
print(df.info())
print(df['Species'].value_counts())

# Visualize relationships
sns.pairplot(df, hue='Species')
plt.show()

# ===========================================
# 2. Preprocessing
# ===========================================
X = df.drop(columns=['Species','Id'])
y = df['Species']

# Encode target labels
encoder = LabelEncoder()
y_int = encoder.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y_int, test_size=0.2, random_state=42, stratify=y_int
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===========================================
# 3. Machine Learning Model - Perceptron
# ===========================================
per = Perceptron(max_iter=1000, random_state=42)
per.fit(X_train_scaled, y_train)

# Predictions & Evaluation
y_pred_percep = per.predict(X_test_scaled)
print("Perceptron Accuracy:", accuracy_score(y_test, y_pred_percep))
print(classification_report(y_test, y_pred_percep))

# ===========================================
# 4. Neural Network - Deep Learning
# ===========================================

# Convert labels to one-hot encoding
y_train_categorical = to_categorical(y_train, num_classes=3)
y_test_categorical = to_categorical(y_test, num_classes=3)

# Build model
model = Sequential([
    Dense(16, input_dim=4, activation='relu'),
    Dense(8, activation='relu'),
    Dense(3, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(
    X_train_scaled, y_train_categorical,
    epochs=100, batch_size=8, validation_split=0.2, verbose=1
)

# Evaluate on test data
loss, acc = model.evaluate(X_test_scaled, y_test_categorical, verbose=1)
print("Neural Network Accuracy:", acc*100)

# ===========================================
# 5. Plot Training Curves
# ===========================================
plt.figure(figsize=(12,5))

# Accuracy
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
