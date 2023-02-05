from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import pickle

heart_data = pd.read_csv('heart_disease.csv')

X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y, random_state = 2)

training_accuracy = []
test_accuracy = []
neighbors = range(1, 11)

for number_of_neighbors in neighbors:
    knn = KNeighborsClassifier(n_neighbors = number_of_neighbors)
    knn.fit(X_train, Y_train)
    training_accuracy.append(knn.score(X_train, Y_train))
    test_accuracy.append(knn.score(X_test, Y_test))

plt.plot(neighbors, training_accuracy, label = "Training accuracy")
plt.plot(neighbors, test_accuracy, label = "Test accuracy")
plt.xlabel("Number of neighbors")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

knn_model = KNeighborsClassifier(n_neighbors = 3)
knn_model.fit(X_train, Y_train)
pickle.dump(knn_model, open('knn_model.pkl','wb')) 

X_train_prediction = knn_model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on Training data for K-Nearest Neighbour: ', training_data_accuracy)

X_test_prediction = knn_model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on Test data for K-Nearest Neighbour: ', test_data_accuracy)
