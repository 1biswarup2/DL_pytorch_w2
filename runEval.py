import pickle
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import sys
from sklearn.metrics import classification_report

# Check if GPU is available and set the default device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    data = torch.load(file_path, map_location=device)

    if isinstance(data, list):
        adjusted_data = [item[:768] if item.size(0) > 768 else item for item in data]
        data = torch.stack(adjusted_data)
    elif isinstance(data, torch.Tensor) and data.size(1) > 768:
        data = data[:, :768]

    return data.to(device)

# def get_embeddings(model, tokenizer, texts, batch_size=16):
#     model = model.to(device)
#     embeddings_list = []

#     for i in range(0, len(texts), batch_size):
#         batch_texts = texts[i:i+batch_size]
#         encoded_batch = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
#         input_ids = encoded_batch['input_ids']
#         attention_mask = encoded_batch['attention_mask']

#         with torch.no_grad():
#             outputs = model(input_ids, attention_mask=attention_mask)

#         embeddings = outputs.last_hidden_state.mean(dim=1)
#         embeddings_list.append(embeddings.cpu())

#     all_embeddings = torch.cat(embeddings_list, dim=0)
#     return all_embeddings.to(device)
def get_embeddings(model, tokenizer, texts, batch_size=16):
    model = model.to(device)
    embeddings_list = []

    # Ensure all inputs are strings and filter out any non-string entries
    clean_texts = [text for text in texts if isinstance(text, str)]

    for i in range(0, len(clean_texts), batch_size):
        batch_texts = clean_texts[i:i+batch_size]
        encoded_batch = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        # Move encoded batch to the correct device
        input_ids = encoded_batch['input_ids'].to(device)
        attention_mask = encoded_batch['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)

        embeddings = outputs.last_hidden_state.mean(dim=1)
        embeddings_list.append(embeddings.cpu())

    all_embeddings = torch.cat(embeddings_list, dim=0)
    return all_embeddings.to(device)


test_data = pd.read_csv("testDataProcessed.csv")
test_texts = test_data['tweet'].tolist()
model_name =sys.argv[1]
pthfile =sys.argv[2]
brtmodel = ""
for brt in ['bert-base-uncased', 'bert-base-cased', 'covid-twitter-bert', 'twhin-bert-base', 'socbert']:
    f = model_name + "_" + brt + ".pth"
    if f in pthfile:
        brtmodel = brt
        break

if brtmodel in ['bert-base-uncased', 'bert-base-cased']:
    TokenizerClass = BertTokenizer
    ModelClass = BertModel
else:
    TokenizerClass = AutoTokenizer
    ModelClass = AutoModel

if brtmodel == 'covid-twitter-bert':
    brtmodel = 'digitalepidemiologylab/covid-twitter-bert-v2'
elif brtmodel == 'twhin-bert-base':
    brtmodel = 'Twitter/twhin-bert-base'
elif brtmodel == 'socbert':
    brtmodel = 'sarkerlab/SocBERT-base'

tokenizer = TokenizerClass.from_pretrained(brtmodel)
model = ModelClass.from_pretrained(brtmodel).to(device)
test_embeddings = get_embeddings(model, tokenizer, test_texts, batch_size=16)
torch.save(test_embeddings, f'XtestVectorized{brtmodel}.pt')

test_embeddings_file = f"XtestVectorized{brtmodel}.pt"
test_embeddings = load_embeddings(test_embeddings_file)
test_data = pd.read_csv("testDataProcessed.csv")
test_data['label'] = test_data['label'].map({'real': 1, 'fake': 0})
test_labels = test_data['label'].tolist()

if model_name == "DNN":
    m = DNN(input_dim=test_embeddings.shape[1])
elif model_name == "1D-CNN":
    m = CNN1D(embedding_dim=test_embeddings.shape[-1])

m = m.to(device)
state_dict = torch.load(pthfile, map_location=device)
m.load_state_dict(state_dict)

m.eval()

with torch.no_grad():
    outputs = m(test_embeddings)
    _, preds_tensor = torch.max(outputs, 1)
    y_pred = preds_tensor.cpu().numpy()

y_true = np.array(test_labels)
print(classification_report(y_true, y_pred))
