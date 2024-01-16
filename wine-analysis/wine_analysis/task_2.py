#!/usr/bin/env python
# coding: utf-8

# # Task 2: Classification (Red vs. White)
#
# • Distinguish between red and white wines.
#
# • Use two different classification algorithms (they may, or may not, be the same as in Task 1).

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
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

# Create column color with value 1 for red wine and 0 for white wine
df_red["color"] = 1
df_white["color"] = 0

# Merge the datasets
wine_data = pd.concat([df_red, df_white], ignore_index=True)
display(wine_data.head())


# In[3]:


# TODO: balance the dataset by randomly removing some of the majority class samples.


# In[4]:


# Since we want to predict 'color', separate the features and the target
X = wine_data.drop("color", axis=1)
y = wine_data["color"]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# In[5]:


# Initialize a baseline algorithm using DummyClassifier
# The strategy "most_frequent" will always predict the most frequent class label in the training dataset
dummy_clf = DummyClassifier(strategy="most_frequent", random_state=42)
dummy_clf.fit(X_train, y_train)

# Predict with DummyClassifier
y_pred_dummy = dummy_clf.predict(X_test)

# Evaluate the baseline classifier
accuracy_score_dummy = accuracy_score(y_test, y_pred_dummy)
print(f"Accuracy: {accuracy_score_dummy:.2f}")
print(
    classification_report(
        y_test,
        y_pred_dummy,
        target_names=["withe_wine_0", "red_wine_1"],
        zero_division=0.0,
    )
)
ConfusionMatrixDisplay.from_estimator(
    dummy_clf, X_test, y_test, display_labels=["withe_wine_0", "red_wine_1"]
)
plt.show()


# In[6]:


# Initialize models
log_reg = LogisticRegression(random_state=42, max_iter=1000)
gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Train models
log_reg.fit(X_train, y_train)
gb_clf.fit(X_train, y_train)

# Predict
y_pred_log_reg = log_reg.predict(X_test)
y_pred_gb = gb_clf.predict(X_test)

# Evaluate
log_reg_accuracy = accuracy_score(y_test, y_pred_log_reg)
log_reg_report = classification_report(y_test, y_pred_log_reg)
gb_accuracy = accuracy_score(y_test, y_pred_gb)
gb_report = classification_report(y_test, y_pred_gb)

print("Logistic Regression Accuracy: {:.2f}%".format(log_reg_accuracy * 100))
print(log_reg_report)
print("\nGradient Boosting Accuracy: {:.2f}%".format(gb_accuracy * 100))
print(gb_report)

# Display Confusion Matrix
ConfusionMatrixDisplay.from_estimator(
    log_reg, X_test, y_test, display_labels=["withe_wine_0", "red_wine_1"]
)
plt.title("Logistic Regression")
plt.show()
ConfusionMatrixDisplay.from_estimator(
    gb_clf, X_test, y_test, display_labels=["withe_wine_0", "red_wine_1"]
)
plt.title("Gradient Boosting")
plt.show()


# # Saving models with `joblib`

# In[7]:

joblib.dump(log_reg, "models/task_2_log_reg.pkl")
joblib.dump(gb_clf, "models/task_2_gb_clf.pkl")
