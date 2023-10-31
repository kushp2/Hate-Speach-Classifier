"""
KPP180001
Kush Patel
CS 4375.003
"""

#imports
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

#loads the test and training data
trainStr = "4375train.csv"
testStr = "4375test.csv"
train_data = pd.read_csv(trainStr, usecols=['example_id', 'sentence', 'label'])
test_data = pd.read_csv(testStr, usecols=['id', 'sentence', 'label'])      

#preprocesses the data
train_data['sentence'] = train_data['sentence'].str.lower()
train_data = train_data.sample(frac=1, random_state=42)
test_data['sentence'] = test_data['sentence'].str.lower()

#splits the data 
x_train = train_data['sentence']
y_train = train_data['label']
x_test = test_data['sentence']
y_test = test_data['label']

#vectorizes the data
vectroizer = TfidfVectorizer()            
x_train_vector = vectroizer.fit_transform(x_train)
x_test_vector = vectroizer.transform(test_data['sentence'])

#trains the model
model = MLPClassifier(hidden_layer_sizes=(100, 10), max_iter=30, early_stopping=True, random_state=42, alpha=0.0001)
model.fit(x_train_vector, y_train)

#prediction from the test set
y_test_pred = model.predict(x_test_vector)
test_report = classification_report(test_data['label'], y_test_pred)

print("Test Report:")
print(test_report)

#saves data to a CSV file
submission = pd.DataFrame({
    'id': test_data['id'],
    'label': y_test_pred
})
submission.to_csv('submission.csv', index=False)
