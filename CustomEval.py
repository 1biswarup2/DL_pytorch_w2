import pickle
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
import torch
import argparse
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import classification_report
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import emoji
import re
import sys
 ########### ************** usage: python3 CustomEval.py testcustom.csv bert-base-cased .pthfilepath ************* #########



nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def preprocess(df):
    stopWords = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    preprocessed_sentences = []
    for tweet in df['tweet']:
        # Convert emojis to text
        text = emoji.demojize(tweet)

        # Process hashtags: Remove '#' but keep the text
        text = re.sub(r'#(\S+)', r'\1', text)

        # Replace URLs with <URL>
        text = re.sub(r'http\S+|www\S+', '<URL>', text)

        # Remove special characters and numbers, keep spaces
        text = re.sub(r'[^A-Za-z\s]', '', text)

        # Convert camel case in hashtags to spaces
        text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)

        # Tokenization
        tokens = word_tokenize(text)

        # Lowercasing
        tokens = [token.lower() for token in tokens]

        # Stopwords removal
        tokens = [token for token in tokens if token not in stopWords]

        # Lemmatization
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

        # Re-join tokens
        preprocessed_sentences.append(' '.join(tokens))

    return preprocessed_sentences
testcsv=sys.argv[1]
#input("enter the csv test file: ")
dfcustom=pd.read_csv(testcsv)
testcustom=preprocess(dfcustom)
testlabel=dfcustom['label'].tolist()
xtestcustom=pd.DataFrame({'tweet':testcustom,'label':testlabel})

xtestcustom.to_csv("customtestprocessed.csv")

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
    data = torch.load(file_path)

    # Check and adjust the embeddings if they are longer than 768 elements
    if isinstance(data, list):
        # If the data is a list, ensure each item is a tensor of the correct size
        adjusted_data = []
        for item in data:
            if isinstance(item, torch.Tensor):
                # If an embedding is larger than 768, truncate it; otherwise, use it as is
                adjusted_data.append(item[:768] if item.size(0) > 768 else item)
            elif isinstance(item, list):
                # If the item is a list, convert it to a tensor and then check the size
                tensor_item = torch.tensor(item, dtype=torch.float)
                adjusted_data.append(tensor_item[:768] if tensor_item.size(0) > 768 else tensor_item)
        # Stack the adjusted tensors to form a batch
        data = torch.stack(adjusted_data)
    elif isinstance(data, torch.Tensor):
        # If the loaded data is a single tensor, check if it needs to be truncated
        # This assumes the data is 2D: a batch of embeddings
        if data.size(1) > 768:
            data = data[:, :768]

    # Ensure the resulting tensor is on the CPU
    return data.cpu()


# Define 1D CNN model
class CNN1D(nn.Module):
    def __init__(self, embedding_dim):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(64 * ((embedding_dim // 2) // 2), 64)  # Adjusted for the output size after pooling
        self.fc2 = nn.Linear(64, 2)  # 2 classes: real and fake
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # Flatten the tensors for the fully connected layer
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# Function to obtain embeddings in batches directly
def get_embeddings(model, tokenizer, texts, batch_size=16):
    model.eval()  # Put the model in evaluation mode
    embeddings_list = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        encoded_batch = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        input_ids = encoded_batch['input_ids']
        attention_mask = encoded_batch['attention_mask']

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)

        embeddings = outputs.last_hidden_state.mean(dim=1)
        embeddings_list.append(embeddings.cpu())

    all_embeddings = torch.cat(embeddings_list, dim=0)
    return all_embeddings
test_data = pd.read_csv("customtestprocessed.csv")
test_texts = test_data['tweet'].tolist()
brtmodel = sys.argv[2]
#input("Enter the bert model name : ")
pthfile=sys.argv[3]
#input("enter the name of pretrained model  with .pth extension: ")
model_name=""
for mdl in ['DNN','1D-CNN']:
   f=mdl+"_"+brtmodel+".pth"
   if f in pthfile:
     model_name=mdl
     break

    # Initialize the tokenizer and model with the provided model name
if brtmodel in ['bert-base-uncased', 'bert-base-cased']:
      TokenizerClass = BertTokenizer
      ModelClass = BertModel
else:
      TokenizerClass = AutoTokenizer
      ModelClass = AutoModel
if brtmodel=='covid-twitter-bert':
  brtmodel='digitalepidemiologylab/covid-twitter-bert-v2'
elif brtmodel=='twhin-bert-base':
  brtmodel='Twitter/twhin-bert-base'
elif brtmodel=='socbert':
  brtmodel='sarkerlab/SocBERT-base'

tokenizer = TokenizerClass.from_pretrained(brtmodel)
model = ModelClass.from_pretrained(brtmodel)
test_embeddings = get_embeddings(model, tokenizer, test_texts, batch_size=16)
torch.save(test_embeddings, f'XcustomVectorized{brtmodel}.pt')
test_embeddings_file = f"XcustomVectorized{brtmodel}.pt"
test_labels_file = "customtestprocessed.csv"
test_embeddings = load_embeddings(test_embeddings_file)
test_data = pd.read_csv(test_labels_file)
test_data['label'] = test_data['label'].map({'real': 1, 'fake': 0})
test_labels = test_data['label'].tolist()
#pthfile="DNNbert-base-cased.pth"
if model_name == "DNN":
    # Adjust test_embeddings.shape[1] as needed to match your model architecture
    m = DNN(input_dim=test_embeddings.shape[1])
elif model_name == "1D-CNN":
    # For CNN, assuming the embeddings are already in the shape [batch_size, embedding_dim]
    # The CNN model initialization might need to change based on how you defined it
    m = CNN1D(test_embeddings.shape[1])  # This might need adjustment

# Load the state dictionary and update the model
state_dict = torch.load(pthfile, map_location=torch.device('cpu'))
m.load_state_dict(state_dict)

# Set the model to evaluation mode
m.eval()

# Predict and evaluate
with torch.no_grad():
    if model_name == "1D-CNN":
        # For CNN, add an extra dimension to indicate channels if not already done
        test_embeddings = test_embeddings.unsqueeze(1)  # Uncomment if necessary
    outputs = m(test_embeddings.float())

    # Convert outputs to predicted class indices
    _, preds_tensor = torch.max(outputs, 1)
    y_pred = preds_tensor.cpu().numpy()  # Ensure tensor is moved to CPU before converting to NumPy array

    # Ensure y_true is a NumPy array with the correct shape
    y_true = np.array(test_labels)  # Assuming test_labels is a list or a 1D NumPy array of true labels

    # Use y_true and y_pred with the classification report
    print(classification_report(y_true, y_pred))