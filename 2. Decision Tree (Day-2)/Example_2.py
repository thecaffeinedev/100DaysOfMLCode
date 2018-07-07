#importing libraries 

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import classification_report, confusion_matrix  

#reading the dataset
dataset = pd.read_csv("bill_authentication.csv")  

X = dataset.drop('Class', axis=1)  
y = dataset['Class'] 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)  
#Training and Making Predictions'

from sklearn.tree import DecisionTreeClassifier  
classifier = DecisionTreeClassifier()  
classifier.fit(X_train, y_train)  

y_pred = classifier.predict(X_test) 


print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))