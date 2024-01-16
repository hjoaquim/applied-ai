#!/usr/bin/env python
# coding: utf-8

# # Task 1: Regression (Alcohol)
#
# • Predict alcohol content (red and white datasets merged).
#
# • Use two different regression algorithms

# In[1]:


import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
import joblib

display = print

# In[2]:


# Load the red and white wine datasets
df_red = pd.read_csv(
    "/home/hjoaquim/Documents/applied-ai/wine-analysis/data/red_wine.csv"
)
df_white = pd.read_csv(
    "/home/hjoaquim/Documents/applied-ai/wine-analysis/data/white_wine.csv"
)

# Remove rows with missing values from both datasets
df_red.dropna(inplace=True)
df_white.dropna(inplace=True)

# Merge the datasets
wine_data = pd.concat([df_red, df_white], ignore_index=True)
display(wine_data.head())


# In[3]:


# Separate the features (X) and target (y)
X = wine_data.drop("alcohol", axis=1)
y = wine_data["alcohol"]


# In[4]:


# Split the data into training and test sets
# TODO: check how `random_state` and `shuffle` affect the model
# TODO: check how using cross-validation affects the model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# In[5]:


# Initialize two different regression algorithms
lin_reg = LinearRegression()
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
svr_reg = SVR(kernel="rbf")  # TODO: check other kernels


# In[6]:


# Train models
display(lin_reg.fit(X_train, y_train))
display(rf_reg.fit(X_train, y_train))
display(svr_reg.fit(X_train, y_train))


# In[7]:


# Predict alcohol content with both models
y_pred_lin = lin_reg.predict(X_test)
y_pred_rf = rf_reg.predict(X_test)
y_pred_svr = svr_reg.predict(X_test)


# In[8]:


# Calculate the performance of both models

# Linear Regression
lin_mse = mean_squared_error(y_test, y_pred_lin)
lin_r2 = r2_score(y_test, y_pred_lin)

# Random Forest
rf_mse = mean_squared_error(y_test, y_pred_rf)
rf_r2 = r2_score(y_test, y_pred_rf)

# SVR
svr_mse = mean_squared_error(y_test, y_pred_svr)
svr_r2 = r2_score(y_test, y_pred_svr)

# Print the results
performace_template = "{model}\n\tMSE: {mse:.2f}\n\tR2: {r2:.2f}\n"
print(performace_template.format(model="Linear Regression", mse=lin_mse, r2=lin_r2))
print(performace_template.format(model="Random Forest", mse=rf_mse, r2=rf_r2))
print(performace_template.format(model="SVR", mse=svr_mse, r2=svr_r2))


# # Saving models with `joblib`

# In[9]:

joblib.dump(lin_reg, "models/lin_reg.pkl")
joblib.dump(svr_reg, "models/svr_reg.pkl")
