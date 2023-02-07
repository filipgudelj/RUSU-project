from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import pickle

heart_data = pd.read_csv('heart_disease.csv')

X = heart_data.drop(columns = 'target', axis = 1)
Y = heart_data['target']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y, random_state = 2)

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 80, num = 10)]
# Number of features to consider at every split
max_features = ['sqrt']
# Maximum number of levels in tree
max_depth = [2,4]
# Minimum number of samples required to split a node
min_samples_split = [2, 5]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2]
# Method of selecting samples for training each tree
bootstrap = [True, False]

param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap} 

rf_param_model = RandomForestClassifier()
rf_model = GridSearchCV(estimator = rf_param_model, param_grid = param_grid, cv = 3, verbose = 2, n_jobs = 4)
rf_model.fit(X_train, Y_train)
pickle.dump(rf_model, open('rf_model.pkl','wb')) 

X_train_prediction = rf_model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on Training data for Random Forest: ', training_data_accuracy)

X_test_prediction = rf_model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on Test data for Random Forest: ', test_data_accuracy)
print()

print("Confusion matrix: ")
confusion_matrix = confusion_matrix(Y_test, X_test_prediction)
print(confusion_matrix)
print()

print("Classification report: ")
report = classification_report(Y_test, X_test_prediction)
print(report)