import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

def model_selection(train_X, train_y):
    lin_reg = LinearRegression()
    lin_reg.fit(train_X, train_y)
    lin_predictions = lin_reg.predict(train_X)
    lin_rmse = np.sqrt(mean_squared_error(train_y, lin_predictions))

    scores = cross_val_score(lin_reg, train_X, train_y,
                             scoring="neg_mean_squared_error", cv=10)
    lin_rmse_scores = np.sqrt(-scores)

    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(train_X, train_y)
    tree_predictions = tree_reg.predict(train_X)
    tree_rmse = np.sqrt(mean_squared_error(train_y, tree_predictions))

    scores = cross_val_score(tree_reg, train_X, train_y,
                             scoring="neg_mean_squared_error", cv=10)
    tree_rmse_scores = np.sqrt(-scores)

    forest_reg = RandomForestRegressor(n_estimators=10, random_state=42)
    forest_reg.fit(train_X, train_y)
    forest_predictions = forest_reg.predict(train_X)
    forest_rmse = np.sqrt(mean_squared_error(train_y, forest_predictions))

    scores = cross_val_score(forest_reg, train_X, train_y,
                             scoring="neg_mean_squared_error", cv=10)
    forest_rmse_scores = np.sqrt(-scores)

    svm_reg = SVR(kernel="linear")
    svm_reg.fit(train_X, train_y)
    svm_predictions = svm_reg.predict(train_X)
    svm_rmse = np.sqrt(mean_squared_error(train_y, svm_predictions))

    scores = cross_val_score(svm_reg, train_X, train_y,
                             scoring="neg_mean_squared_error", cv=10)
    svm_rmse_scores = np.sqrt(-scores)

    train_rmse = pd.DataFrame(np.c_[lin_rmse, tree_rmse, forest_rmse, svm_rmse],
                              columns=['Linear', 'Decision Tree', 'Random Forest', 'SVM'])

    models = pd.DataFrame(np.c_[lin_rmse_scores, tree_rmse_scores, forest_rmse_scores, svm_rmse_scores],
                          columns=['Linear', 'Decision Tree', 'Random Forest', 'SVM'])

    print('##########  Train RMSE  ##########')
    print(train_rmse)
    print()
    print('##########  Cross Validation Score  ##########')
    print(models.describe())

    sns.boxplot(data=models)
    sns.stripplot(data=models)
    plt.show()
