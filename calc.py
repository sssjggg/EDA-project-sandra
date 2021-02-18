import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.formula.api as smf
import statsmodels.api as sm
import pickle

#df = pd.read_pickle('sav.txt')
X = pd.read_pickle('Xpart.csv')
Y = pd.read_pickle('Ypart.csv')

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

## Prediction
model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

## RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))  
print('RMSE: ', rmse)
accu = round((model.score(x_test, y_test)*100), 2)
print('Accuracy: ', accu, '%')# *100 = percenatge of model accurate

## Calculate the new OLS
x_train = sm.add_constant(x_train)
fit_new = sm.OLS(y_train, x_train).fit()
ols_print = fit_new.summary()
print('new OLS: ', ols_print)

## save
with open('summary', 'wb') as f:
    pickle.dump([rmse, fit_new],f)
