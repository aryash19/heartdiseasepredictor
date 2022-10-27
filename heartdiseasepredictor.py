"""

IMPORTING THE DEPENDENCIES(SMJH GAYA BHAI )
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

"""DATA PROCESSING

"""

heart_data = pd.read_csv('/content/data.csv')

heart_data.head()

heart_data.shape

heart_data.info()

heart_data.isnull().sum()

heart_data.describe()

heart_data['target'].value_counts()

"""ENCODING THE FEATURES"""

X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

print (X)

print(Y)

"""SPLITING FOR TRAINING AND TESTING DATASET"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

"""MODEL TRAINING"""

model = LogisticRegression()

model.fit(X_train, Y_train)

"""MODEL EVALUATION"""

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

"""ACCURACY ON TRAINING DATASET """

print('Accuracy on Training data : ', training_data_accuracy)

"""ACCURACY ON TESTING DATASET"""

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy on Test data : ', test_data_accuracy)

"""PREDICTION SYSTEM (bhai value ger)"""



input_data = (62,0,0,140,268,0,0,160,0,3.6,0,2,2)


input_data_as_numpy_array= np.asarray(input_data)


input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]== 0):
  print('The Person does not have a Heart Disease')
else:
  print('The Person has Heart Disease')
