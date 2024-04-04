import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import sys

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

def train_classifier(train_embeddings, train_labels, epochs=10):
    train_labels_tensor = torch.tensor(train_labels).long().to(device)  # Move labels to the device

    model = DNN(train_embeddings.shape[1])
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
    bert_models = ['bert-base-cased', 'bert-base-uncased', 'covid-twitter-bert', 'twhin-bert-base', 'socbert']
    bert_model_name = None

    # Assuming the input is a link from which you want to extract the model name
    link = sys.argv[1]
    # python3 dnn.py .pt-embeddingPathIncludingBERTmodelNAme
    #input("Enter the link: ")

    # Check if any known BERT model names are in the link
    for model in bert_models:
        if model in link:
            bert_model_name = model
            break

    # If a BERT model name is found in the link, proceed
    if bert_model_name:
        print(f"Found BERT model in link: {bert_model_name}")
        train_embeddings_file = link

        # Load embeddings and ensure they are on the correct device
        train_embeddings = load_embeddings(train_embeddings_file).to(device)
        train_labels_file = "trainDataProcessed.csv"
        train_data = pd.read_csv(train_labels_file)

        train_data['label'] = train_data['label'].map({'real': 1, 'fake': 0})
        train_labels = train_data['label'].tolist()

        model = train_classifier(train_embeddings, train_labels)

        # Save the model with a name that includes the BERT model name
        model_save_path = f"DNN_{bert_model_name}.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to '{model_save_path}'.")
    else:
        print("No known BERT model name found in the link.")

if __name__ == "__main__":
    main()
