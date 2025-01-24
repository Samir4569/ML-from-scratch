import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from logistic_regression import LogisticRegression

bc = datasets.load_breast_cancer()

X, y = bc.data, bc.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def accuracy (y, y_pred):
    accuracy = np.sum(y == y_pred) / len(y)
    return accuracy

regressor = LogisticRegression(lr=0.001, n_iters=1000)
regressor.fit(X_train, y_train)
y_predicted = regressor.predict(X_test)

print(accuracy(y_test, y_predicted))