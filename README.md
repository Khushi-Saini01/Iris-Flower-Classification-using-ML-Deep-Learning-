# Iris-Flower-Classification-using-ML-Deep-Learning-
Iris Flower Classification using ML &amp; DL: Built an end-to-end classification system predicting Iris species using Perceptron and Neural Networks. Performed data preprocessing, feature scaling, label encoding, and EDA. Evaluated models with accuracy, confusion matrix, and classification report, demonstrating ML vs DL comparison.
# Iris Flower Classification Using Machine Learning & Deep Learning

## Project Overview
This project focuses on classifying iris flowers into three species: *Iris-setosa*, *Iris-versicolor*, and *Iris-virginica* using both classical Machine Learning and Deep Learning approaches. The project demonstrates an **end-to-end workflow** from data preprocessing to model evaluation.

## Dataset
- **Source:** Iris Dataset (CSV format)
- **Features:** Sepal length, Sepal width, Petal length, Petal width
- **Target:** Species (3 classes)

## Key Steps
1. **Data Loading & Exploration**
   - Load dataset using Pandas
   - Understand data using `df.info()`, `value_counts()`
   - Visualize relationships with `sns.pairplot()`

2. **Data Preprocessing**
   - Separate features (X) and target (y)
   - Encode categorical target labels into integers
   - Scale features using StandardScaler

3. **Machine Learning Model**
   - Trained a **Perceptron** classifier as baseline
   - Evaluated using accuracy, confusion matrix, and classification report

4. **Deep Learning Model**
   - Built a **Sequential Neural Network** with:
     - Two hidden layers (16 & 8 neurons, ReLU activation)
     - Output layer with 3 neurons (softmax)
   - Converted labels to **one-hot encoding**
   - Compiled model with **Adam optimizer** and **categorical crossentropy** loss
   - Trained with **epochs=100**, **batch_size=8**, and **validation split=0.2**

5. **Model Evaluation**
   - Evaluated on unseen test data
   - Achieved high accuracy and visualized training curves

## Tools & Technologies
- Python
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn (Perceptron, preprocessing, metrics)
- TensorFlow, Keras (ANN)

## Usage
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
