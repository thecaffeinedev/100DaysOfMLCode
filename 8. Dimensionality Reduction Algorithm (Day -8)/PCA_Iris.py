import numpy as np  
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix  
from sklearn.metrics import accuracy_score

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"  
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']  
dataset = pd.read_csv(url, names=names)

X = dataset.drop('Class', 1)  
y = dataset['Class'] 

# Splitting the dataset into the Training set and Test set


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



sc = StandardScaler()  
X_train = sc.fit_transform(X_train)  
X_test = sc.transform(X_test)


pca = PCA()  
X_train = pca.fit_transform(X_train)  
X_test = pca.transform(X_test) 

explained_variance = pca.explained_variance_ratio_  

print(explained_variance)

#Training and Making Predictions
classifier = RandomForestClassifier(max_depth=2, random_state=0)  
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)  

cm = confusion_matrix(y_test, y_pred)  
print(cm)  