import os
import torch
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import pickle
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DNN(nn.Module):
    def __init__(self, input_dim, output_dim=2):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class CNN1D(nn.Module):
    def __init__(self, embedding_dim):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(64 * ((embedding_dim // 2) // 2), 64)
        self.fc2 = nn.Linear(64, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(x.size(0), 1, -1)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def load_embeddings(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The specified file does not exist: {file_path}")

    try:
        data = torch.load(file_path)
    except pickle.UnpicklingError:
        raise RuntimeError(f"UnpicklingError: The file might be corrupted or incomplete: {file_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load embeddings from file: {e}")

    if isinstance(data, list):
        adjusted_data = [item[:768] if item.size(0) > 768 else item for item in data]
        data = torch.stack(adjusted_data)
    elif isinstance(data, torch.Tensor) and data.size(1) > 768:
        data = data[:, :768]
    return data.to(device)  # Move data to the specified device

def train_classifier(train_embeddings, train_labels, model_type, epochs=10):
    train_labels_tensor = torch.tensor(train_labels).long().to(device)  # Move labels to the device

    if model_type == "DNN":
        model = DNN(train_embeddings.shape[1])
    elif model_type == "1D-CNN":
        model = CNN1D(train_embeddings.shape[1])
    else:
        raise ValueError("Invalid model type. Choose either 'DNN' or '1D-CNN'.")

    model.to(device)  # Move model to the specified device
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    train_dataset = TensorDataset(train_embeddings, train_labels_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data and labels to the device
            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")
    return model
def main():
    bert_model_name = input("Enter the BERT model name (e.g., bert-base-uncased, bert-base-cased, etc.): ")
    model_type = input("Enter the type of classifier to train (DNN or 1D-CNN): ")

    train_embeddings_file = f"/kaggle/input/processedembedding/XtrainVectorized{bert_model_name}.pt"
    valid_embeddings_file = f"/kaggle/input/processedembedding/XvalVectorized{bert_model_name}.pt"
    train_labels_file = "/kaggle/input/processeddata/trainDataProcessed.csv"
    valid_labels_file = "/kaggle/input/processeddata/valDataProcessed.csv"

    # Load embeddings and ensure they are on the correct device
    train_embeddings = load_embeddings(train_embeddings_file).to(device)
    valid_embeddings = load_embeddings(valid_embeddings_file).to(device)
    train_data = pd.read_csv(train_labels_file)
    valid_data = pd.read_csv(valid_labels_file)

    train_data['label'] = train_data['label'].map({'real': 1, 'fake': 0})
    valid_data['label'] = valid_data['label'].map({'real': 1, 'fake': 0})
    train_labels = train_data['label'].tolist()
    valid_labels = valid_data['label'].tolist()

    model = train_classifier(train_embeddings, train_labels, model_type)

    torch.save(model.state_dict(), f"{model_type}_{bert_model_name}.pth")
    print(f"Model saved to '{model_type}_{bert_model_name}.pth'.")

    model.eval()
    with torch.no_grad():
        if model_type == "DNN":
            outputs = model(valid_embeddings.float())
        elif model_type == "1D-CNN":
            outputs = model(valid_embeddings)
        _, preds_tensor = torch.max(outputs, 1)
        y_pred = preds_tensor.cpu().numpy()  # Move tensor to CPU for numpy conversion

    y_true = np.array(valid_labels)
    print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    main()
