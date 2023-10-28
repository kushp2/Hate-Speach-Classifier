"""
KPP180001
Kush Patel
CS 4375.003
"""

import torch
from tranfromers import BertModel
from tqdm.notebook import tqdm
import pandas as pd
import string
from transformers import BertTokenizer


# variables
trainData = pd.read_csv("4375train.csv")                                                # dataframe that holds the training data
testData = pd.read_csv("4375test.csv")                                                  # dataframe that holds the testing data
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")                          # tokenizer                                             


# cleans up the dataframe's data. removes columns, punctuation, and captilization
trainData = trainData.drop([2, 3, 4], axis = 1)
testData = testData.drop([2, 3, 4], axis = 1)
trainData[1] = trainData[1].apply(cleanStringData)
testData[1] = testData[1].apply(cleanStringData)

# converts a string to lower case and removes punctuation
def cleanStringData(text):
    text = text.lower()
    for char in string.punctuation:
        text = text.replace(char, '')
    return text

# Tokenize the text data
tokenizedData = tokenizer(trainData[1], return_tensors ='pt')
paddedTokenizedData = transformers.pad_to_max_length(tokenized_text, max_length=128)
truncatedTokenizedData = transformers.truncate_sequences(tokenized_text, max_length=128)


# classifier architecture
class hateSpeachClassifier(torch.nn.Module):
    def __init__(self):
        super (hateSpeachClassifier, self).__init__()

        # loads the model and adds liniar layer with a binary output
        self.bert = BertModel.from_pretrained("bert-base-uncased")                  
        self.linear = torch.nn.Linear(768, 1)

    def forward(self, inputIDs):
        # encodes the input text
        bertOutput = self.bert(inputIDs)

        # gets the last hidden state
        last_hidden_state = bertOutput.last_hidden_state
        pooledOutput = torch.mean(last_hidden_state, dim=1)

        # uses linar layer to predict hate speach
        logits = self.linear(pooledOutput)
        probs = torch.sigmoid(logits)