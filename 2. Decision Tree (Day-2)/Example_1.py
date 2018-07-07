#We import tree from sklearn and create the model
from sklearn import tree
clf = tree.DecisionTreeClassifier()  

#Then we create the training data for the classifier / decision tree:

#[height, hair-length, voice-pitch]                                             
X = [ [180, 15,0],                                                              
      [167, 42,1],                                                              
      [136, 35,1],                                                              
      [174, 15,0],                                                              
      [141, 28,1]]                                                              

Y = ['man', 'woman', 'woman', 'man', 'woman']

Y = ['man', 'woman', 'woman', 'man', 'woman']                                   

clf = clf.fit(X, Y)                                                             
prediction = clf.predict([[133, 37,1]])                                         
print(prediction)  