from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

heart_data = pd.read_csv('heart_disease.csv')

print("Broj redova i stupaca: ")
print(heart_data.shape)
print("Informacije o podacima: ")
heart_data.info()
print()

print("Prikaz prva 3 retka: ")
print(heart_data.head(3))
print()
print("Prikaz zadnja 3 retka: ")
print(heart_data.tail(3))
print()

print("Provjera postoje li vrijednosti koje nedostaju: ")
print(heart_data.isnull().sum())
print()

print("Distribucija ciljne varijable: ")
print(heart_data['target'].value_counts())
print()

X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print("Trening podaci: ")
print(X_train.shape)
print("Test podaci: ")
print(X_test.shape)