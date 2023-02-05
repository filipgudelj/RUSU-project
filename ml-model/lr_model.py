from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle

heart_data = pd.read_csv('heart_disease.csv')

X = heart_data.drop(columns = 'target', axis = 1)
Y = heart_data['target']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y, random_state = 2)

lr_model = LogisticRegression(max_iter = 3000)
lr_model.fit(X_train, Y_train)
pickle.dump(lr_model, open('lr_model.pkl','wb')) 

X_train_prediction = lr_model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on Training data for Logistic Regression: ', training_data_accuracy)

X_test_prediction = lr_model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on Test data for Logistic Regression: ', test_data_accuracy)
