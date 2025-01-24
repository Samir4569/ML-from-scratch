
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets


X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)


from linear_regression import LinearRegression

regressor = LinearRegression(lr=0.03, n_iters=1000) 

regressor.fit(X_train, y_train)

predictions = regressor.predict(X_test)

def mse(y_true, y_predicted):
    return np.mean((y_true-y_predicted)**2)

mse_val = mse(y_test, predictions)
print(mse_val)

cmapper = plt.get_cmap('viridis')

y_predic_line = regressor.predict(X)
plt.scatter(X_train, y_train, color=cmapper(0.9), s=10)
plt.scatter(X_test, y_test, color = cmapper(0.5), s=10)
plt.plot(X, y_predic_line, color = 'black', linewidth = 2)
plt.show()
