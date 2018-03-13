
# coding: utf-8

# import necessary modules
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# read the csv file using pandas 
data = pd.read_csv('/home/ubuntu/Desktop/Advertising/Dataset_advertise.csv', index_col = 0)

# Get all the features in a Dataframe
X = data[['TV', 'Radio', 'Newspaper']]

# Get the response in a Series
y = data['Sales']

# Splitting X and y into two datasets just for train/test split for cross validation
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

# 2. Instantiate a linear regression object: reg
reg = LinearRegression()

# 3. Fit the model with data
reg.fit(X_train, y_train)

#Check coeffecients(positive or negative)
print (reg.intercept_)
print (reg.coef_)

feature_cols = ['TV', 'Radio', 'Newspaper']
list(zip(feature_cols, linreg.coef_))

# 4. Predict the response for a new observation
y_pred = reg.predict(X_test)
# Comparing the predicted response with the true response
list(zip(y_pred, y_test))

# Computing RMSE for the Sales Prediction of the model

# importing metrics for evaluating the model
from sklearn import metrics

# for finding the sqrt
print (np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#calculate accuracy
accuracy=reg.score(X_test,y_test)
print(accuracy)

