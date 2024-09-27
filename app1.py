#pytorch with 410065006 parameters and correct o/p


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Check if GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset
def load_data(data_path):
    return pd.read_csv(data_path)

# PyTorch Neural Network Model
class EmotionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(EmotionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 20000)   # 20,000 neurons
        self.fc2 = nn.Linear(20000, 10000)         # 10,000 neurons
        self.fc3 = nn.Linear(10000, 5000)          # 5,000 neurons
        self.fc4 = nn.Linear(5000, output_size)    # Output layer
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Function to calculate total parameters in the PyTorch model
def calculate_total_params(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params

# Vectorization and model training for Logistic Regression
def train_logistic_regression(X_train, y_train):
    vectorizer = TfidfVectorizer(max_features=8000)
    X_train_vec = vectorizer.fit_transform(X_train)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)
    return vectorizer, model

# Function to load PyTorch model
def load_pytorch_model(input_size, output_size):
    model = EmotionModel(input_size=input_size, output_size=output_size)
    model.to(device)  # Move the model to the GPU if available
    return model

# Prediction function for PyTorch model
def predict_emotion_pytorch(text, model, vectorizer, label_mapping):
    text_vec = vectorizer.transform([text]).toarray()
    text_tensor = torch.tensor(text_vec, dtype=torch.float32).to(device)  # Move to GPU if available
    with torch.no_grad():
        output = model(text_tensor)
        _, predicted = torch.max(output, 1)
        emotion = list(label_mapping.keys())[list(label_mapping.values()).index(predicted.item())]
        return emotion

# Prediction function for Logistic Regression
def predict_emotion_logistic(text, vectorizer, model):
    text_vec = vectorizer.transform([text])
    return model.predict(text_vec)[0]

# Load the dataset
data_path = 'Emotion_final_with_predictions.csv'  # Ensure the path is correct for deployment
data = load_data(data_path)

# Prepare data
X = data['Text']
y = data['Emotion']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Map emotions to numerical labels for PyTorch
label_mapping = {emotion: idx for idx, emotion in enumerate(y.unique())}
y_train_mapped = y_train.map(label_mapping)
y_test_mapped = y_test.map(label_mapping)

# Train logistic regression model
vectorizer_logistic, logistic_regression_model = train_logistic_regression(X_train, y_train)

# Load PyTorch model
vectorizer_pytorch = TfidfVectorizer(max_features=8000)
vectorizer_pytorch.fit(X_train)  # Same vectorizer used for training
pytorch_model = load_pytorch_model(input_size=8000, output_size=len(label_mapping))

# Calculate total parameters of the PyTorch model
total_params = calculate_total_params(pytorch_model)
print(f'Total Parameters in PyTorch Model: {total_params}')

# Example complex sentence for emotion prediction
complex_sentence = "im happy that he behaving like idots"

# Loop to make multiple predictions (you can adjust how many times you want to loop)
for _ in range(3):  # Replace 3 with however many predictions you want to make
    predicted_emotion_pytorch = predict_emotion_pytorch(complex_sentence, pytorch_model, vectorizer_pytorch, label_mapping)
    print(f'Predicted Emotion (PyTorch Neural Network): {predicted_emotion_pytorch}')

    predicted_emotion_logistic = predict_emotion_logistic(complex_sentence, vectorizer_logistic, logistic_regression_model)
    print(f'Predicted Emotion (Logistic Regression): {predicted_emotion_logistic}')

# Evaluate Logistic Regression model
X_test_vec_logistic = vectorizer_logistic.transform(X_test)
y_pred_logistic = logistic_regression_model.predict(X_test_vec_logistic)
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
print(f'Logistic Regression Accuracy: {accuracy_logistic:.4f}')
print(classification_report(y_test, y_pred_logistic))
