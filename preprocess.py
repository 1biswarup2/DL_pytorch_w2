
import numpy as np
import pandas as pd
import openpyxl
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
#import fasttext
import re
import sys

dataexcel=sys.argv[1]
df=pd.read_excel('data.xlsx')


df.head()


X=df['tweet']
Y=df['label']


df_train,df_tv=train_test_split(df, random_state=104,  test_size=0.20)


df_test,df_valid=train_test_split(df_tv , random_state=104,  test_size=0.50)


X_train, X_tv, Y_train, Y_tv =   train_test_split(X,Y ,random_state=104,test_size=0.20)


X_valid,X_test,Y_valid,Y_test=train_test_split(X,Y , random_state=104,test_size=0.50)


# type(X_train)
# t1=pd.DataFrame(X_train)
# t2=pd.DataFrame(Y_train)
# frames=[t1,t2]
# df_train=pd.concat(frames)
# df_train


df_train.to_csv('train.csv')



df_test.to_csv('test.csv')
df_valid.to_csv('valid.csv')


df_train


df_test


df_valid


import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
# from nltk.corpus import stopwords
#stopwords=stopwords.words('english')


# def super_simple_preprocess(text):
#   # lowercase
#   text = text.lower()
#   # remove non alphanumeric characters
#   text = re.sub('[^A-Za-z0-9 ]+','', text)
#   return text


# stemmer = nltk.stem.porter.PorterStemmer()
# stoplist = nltk.corpus.stopwords.words('english')

# def nltk_preprocess(text, stemmer=stemmer, stoplist=stoplist):
#   text = super_simple_preprocess(text)
#   tokens = text.split()
#   processed_tokens = [stemmer.stem(t) for t in tokens if t not in stoplist]
#   return ' '.join(processed_tokens)
nltk.corpus.stopwords.words('english')

# import string
# def preprocess_tweet(text):

#     # Check characters to see if they are in punctuation
#     nopunc = [char for char in text if char not in string.punctuation]
#     # Join the characters again to form the string.
#     nopunc = ''.join(nopunc)
#     # convert text to lower-case
#     nopunc = nopunc.lower()
#     # remove URLs
#     nopunc = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '', nopunc)
#     nopunc = re.sub(r'http\S+', '', nopunc)
#     # remove usernames
#     nopunc = re.sub('@[^\s]+', '', nopunc)
#     # remove the # in #hashtag
#     nopunc = re.sub(r'#([^\s]+)', r'\1', nopunc)
#     # remove repeated characters
#     nopunc = nltk.word_tokenize(nopunc)
#     # remove stopwords from final word list
#     return [word for word in nopunc if word not in nltk.corpus.stopwords.words('english')]


#preprocess_tweet(nltk_preprocess("Because of Donald Trump's negligence and incompetence:-") )

import  emoji
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity


#stopWords = set(stopwords.words('english'))

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')



# def preprocess(df_x):
#     sent=df_x['tweet']
#     preprocessed_sentences = []
#     for s in sent:
#         res=""
#         for c in s:
#             if c not in string.punctuation:
#                 res+=c


#         if res!="":
#           words = word_tokenize(res)


#         fwords = []
#         for word in words:
#             if word not in stopWords:
#                 fwords.append(WordNetLemmatizer().lemmatize(word) )


#         ps=""
#         ind=1
#         for w in fwords:
#             if ind!=len(fwords):
#                 ps+=(w+' ')
#             else:
#                 ps+=w
#             ind=ind+1

#         preprocessed_sentences.append(ps)
#     return preprocessed_sentences

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



train_preprocessed= preprocess(df_train)
# for s in train_preprocessed:
#     print(s+'\n')


test_preprocessed= preprocess(df_test)
# for s in test_preprocessed:
#     print(s+'\n')


valid_preprocessed= preprocess(df_valid)
# for s in valid_preprocessed:
#     print(s+'\n')
y_train=pd.read_csv("train.csv")
y_train=y_train['label'].tolist()
y_test=pd.read_csv("test.csv")
y_test=y_test['label'].tolist()
y_valid=pd.read_csv("valid.csv")
y_valid=y_valid['label'].tolist()

x_train=pd.DataFrame({'tweet':train_preprocessed,'label':y_train})
x_test=pd.DataFrame({'tweet':test_preprocessed,'label':y_test})
x_valid=pd.DataFrame({'tweet':valid_preprocessed,'label':y_valid})

x_train.to_csv("trainDataProcessed.csv")
x_test.to_csv("testDataProcessed.csv")
x_valid.to_csv("valDataProcessed.csv")


