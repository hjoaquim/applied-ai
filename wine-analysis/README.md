# Vinho Verde Wine Analysis

## Overview

This repository contains the analysis of Portuguese "Vinho Verde" wine. The objective is to apply machine learning models to predict wine quality and alcohol content and to classify wines as red or white.

## Project Structure

The project is divided into three main tasks:

1. Task 1: Regression analysis to predict alcohol content.
2. Task 2: Classification to distinguish between red and white wines.
3. Task 3: Classification to predict wine quality.

Each task compares two different algorithms to evaluate their performance.

## Reproducibility

- All methodologies and source code are available.
- Datasets used are open-source and included.

## Getting Started

To replicate the analysis:

1. Clone the repo: `git clone https://github.com/hjoaquim/applied-ai/tree/main/wine-analysis`
2. Navigate to the project: `cd wine-analysis`
3. Install [dependencies](#requirements)
4. Run the notebooks: [exploratory-analysis](https://github.com/hjoaquim/applied-ai/blob/main/wine-analysis/exploratory-analysis.ipynb), [task-1](https://github.com/hjoaquim/applied-ai/blob/main/wine-analysis/task_1.ipynb), [task-2](https://github.com/hjoaquim/applied-ai/blob/main/wine-analysis/task_2.ipynb) and [task-3](https://github.com/hjoaquim/applied-ai/blob/main/wine-analysis/task_3.ipynb); alternatively you can also run the source code (which was compiled from the notebooks using `nbconvert`): [wine-analysis-source](https://github.com/hjoaquim/applied-ai/tree/main/wine-analysis/wine_analysis).

## Run the models

If you want to run the models, those are ready to use out of the box in the [models folder](https://github.com/hjoaquim/applied-ai/tree/main/wine-analysis/models).

Task 1:

- Random Forest: `task_1_rf_reg.pkl`
- Support Vector Regression: `task_1_svr_reg.pkl`

Task 2:

- Gradient Boosting: `task_2_gb_clf.pkl`
- Logistic Regression: `task_2_log_reg.pkl`

Task 3:

- Neural Network: `task_3_nn.pkl`
- Support Vector Machine: `task_3_svm_clf.pkl`

### Example

```python

import joblib

rf_reg = joblib.load('models/task_1_rf_reg.pkl')
rf_reg.predict(X_test)

```

> Note: `X_test` is a pandas dataframe with the features of the test set but, of course, you can use any dataframe or instance with the same features.
> Also, `models/task_1_rf_reg.pkl` should be the path to the model you want to use.


## Requirements

Follow the steps below to install the required packages:

```bash

# Create a virtual environment
conda create -n wine python=3.11

# Install poetry
pip install poetry

# Install dependencies
poetry install

```

## Contributing

Feel free to fork the repository, make changes, and submit a pull request.

## License

This project is open-source and available under the  GNU GENERAL PUBLIC LICENSE.
