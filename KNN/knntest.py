from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


cmap = ListedColormap(('red', 'lightgreen', 'lightblue'))  # color map for plotting

iris = datasets.load_iris()

X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)




plt.figure()

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap, marker='o', label='training data')
plt.show()



from knn import KNN

clf = KNN(k=6)

clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

acc = np.sum(predictions == y_test) / len(y_test)
print(acc)





