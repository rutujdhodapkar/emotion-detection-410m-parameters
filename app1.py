import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

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

# Vectorization using TF-IDF
vectorizer_pytorch = TfidfVectorizer(max_features=8000)
vectorizer_pytorch.fit(X_train)  # Fit the vectorizer on the training data
X_train_vec = vectorizer_pytorch.transform(X_train).toarray()  # Transform training data
X_test_vec = vectorizer_pytorch.transform(X_test).toarray()      # Transform test data

# Load PyTorch model
pytorch_model = load_pytorch_model(input_size=8000, output_size=len(label_mapping))

# Training loop
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(pytorch_model.parameters(), lr=0.001)
num_epochs = 10  # You can adjust the number of epochs

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_vec, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train_mapped.values, dtype=torch.long).to(device)

# Training the model
for epoch in range(num_epochs):
    pytorch_model.train()
    optimizer.zero_grad()  # Zero the gradients
    outputs = pytorch_model(X_train_tensor)  # Forward pass
    loss = criterion(outputs, y_train_tensor)  # Compute loss
    loss.backward()  # Backward pass
    optimizer.step()  # Update weights
    
    if (epoch + 1) % 1 == 0:  # Print every epoch
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model on test data
pytorch_model.eval()
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test_vec, dtype=torch.float32).to(device)
    test_outputs = pytorch_model(X_test_tensor)
    _, test_predicted = torch.max(test_outputs, 1)
    
# Convert predictions back to emotions
predicted_emotions = [list(label_mapping.keys())[list(label_mapping.values()).index(predicted.item())] for predicted in test_predicted]

# Print evaluation metrics
accuracy = accuracy_score(y_test, predicted_emotions)
print(f'PyTorch Model Accuracy: {accuracy:.4f}')
print(classification_report(y_test, predicted_emotions))

# Example complex sentence for emotion prediction
complex_sentence = "I'm happy that he is behaving like an idiot"

# Make predictions using the PyTorch model
predicted_emotion_pytorch = predict_emotion_pytorch(complex_sentence, pytorch_model, vectorizer_pytorch, label_mapping)
print(f'Predicted Emotion (PyTorch Neural Network): {predicted_emotion_pytorch}')