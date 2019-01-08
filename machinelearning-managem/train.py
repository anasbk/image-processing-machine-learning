import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing

from sklearn.model_selection import train_test_split




df = pd.read_csv("C:\export\data.csv",delimiter=";")


dataset = df.ix[:,[1,3,4]]

dataset = dataset.values

dflabel = pd.read_csv("C:\export\label.csv",delimiter=";",header=None)

label = dflabel.ix[:,[1,2,3]]

label = label.values

# split X and y into training and testing sets

X_train,X_test,y_train,y_test=train_test_split(dataset,label,test_size=0.20,random_state=0)


# Create linear regression object
regr = linear_model.LinearRegression()



# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)


# save model

filename = 'linear_reg.model'
pickle.dump(regr, open(filename, 'wb'))
