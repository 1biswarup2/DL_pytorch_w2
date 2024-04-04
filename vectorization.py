import pickle
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
import torch
import argparse
import pandas as pd
import sys


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

def main():
    # parser = argparse.ArgumentParser(description="Generate embeddings for texts using a specified BERT model.")
    # parser.add_argument("train_file", type=str, help="Path to the preprocessed training CSV file.")
    # parser.add_argument("valid_file", type=str, help="Path to the preprocessed validation CSV file.")

    # args = parser.parse_args()

    # Load preprocessed data
    train_data = pd.read_csv("trainDataProcessed.csv")
    valid_data = pd.read_csv("valDataProcessed.csv")

    train_texts = train_data['tweet'].tolist()
    valid_texts = valid_data['tweet'].tolist()

    # Take user input for the model name
    model_name = sys.argv[1]
    #usage: python3 vetorization.py bert-base-cased
    #input("Enter the BERT model name (e.g., bert-base-uncased, bert-base-cased, covid-twitter-bert, twhin-bert-base, socbert): ")

    # Initialize the tokenizer and model with the provided model name
    if model_name in ['bert-base-uncased', 'bert-base-cased']:
        TokenizerClass = BertTokenizer
        ModelClass = BertModel
    else:
        TokenizerClass = AutoTokenizer
        ModelClass = AutoModel

    tokenizer = TokenizerClass.from_pretrained(model_name)
    model = ModelClass.from_pretrained(model_name)

    print("getting train embeddings...")
    train_embeddings = get_embeddings(model, tokenizer, train_texts, batch_size=16)

    print("getting valid embeddings...")
    valid_embeddings = get_embeddings(model, tokenizer, valid_texts, batch_size=16)

    print(train_embeddings.shape)
    print(valid_embeddings.shape)

    # Save the embedding data to .pt files
    torch.save(train_embeddings, f'XtrainVectorized{model_name}.pt')
    torch.save(valid_embeddings, f'XvalVectorized{model_name}.pt')

    print(f"Embeddings for {model_name} saved.")

if __name__ == "__main__":
    main()
