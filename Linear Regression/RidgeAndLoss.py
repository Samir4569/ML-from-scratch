import pandas as pd
import numpy as np
import matplotlib
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

data_url = 'https://raw.githubusercontent.com/selva86/datasets/refs/heads/master/BostonHousing.csv'


df = pd.read_csv(data_url)

X = df.iloc[:,:-1]
y = df.iloc[:, -1]

model = LinearRegression()

mse = cross_val_score(model, X, y, scoring = 'neg_mean_squared_error', cv =5)

#print(mse.mean())

ridge = Ridge()

parametrs = {'alpha' : [1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}

ridge_regressor = GridSearchCV(ridge, parametrs,scoring = 'neg_mean_squared_error', cv = 5)
ridge_regressor.fit(X, y)

#print(ridge_regressor.best_params_)
#print(ridge_regressor.best_score_) 

Loss = Lasso()
Loss_regressor = GridSearchCV(Loss, parametrs,scoring = 'neg_mean_squared_error', cv = 5)
Loss_regressor.fit(X, y)

print(Loss_regressor.best_params_)
print(Loss_regressor.best_score_)


