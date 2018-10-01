from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
# Initialize a Random Forest classifier with 100 trees
#
def classif(X_train, y_train, X_test, y_test):
    forest = RandomForestClassifier( n_estimators = 100, max_features = 100, oob_score=True, class_weight = "balanced")

    fitted_forest = forest.fit( X_train, y_train )
    y_pred = fitted_forest.predict( X_test )

    print(accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

