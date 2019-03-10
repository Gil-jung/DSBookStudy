from collecting import *
from split import *
from preprocessing import *
from model_selection import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

sns.set(style="whitegrid", color_codes=True)

if __name__ == '__main__':
    ##########  Data Collection  ##########
    fetch_housing_data()
    housing = load_housing_data()

    ##########  Data Exploration  ##########
    # print(housing.head())
    # print(housing.info())
    # print(housing['ocean_proximity'].value_counts())   # categorical feature confirm
    # print(housing.describe())                          # numerical feature confirm

    ## histogram confirm ##
    # housing.hist(bins=50, figsize=(20, 15))
    # plt.figure(figsize=(12, 12), dpi=80)
    # plt.subplot(311)
    # sns.distplot(housing['median_income'], bins=50, kde=False)
    # plt.subplot(312)
    # sns.distplot(housing['housing_median_age'], bins=50, kde=False)
    # plt.subplot(313)
    # sns.distplot(housing['median_house_value'], bins=50, kde=False)
    # plt.show()

    ##########  Train/Test Split(General)  ##########
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
    ##########  Train/Test Split(Stratified)  ##########
    housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
    housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
    strat_train_set, strat_test_set = stratsplit(housing, housing["income_cat"], n_splits=1, test_size=0.2, random_state=42)

    ##########  Data Visualization  ##########
    # housing = strat_train_set.copy()
    # sns.jointplot(x="longitude", y="latitude", data=housing, height=6, alpha=0.1)
    # sns.relplot(x="longitude", y="latitude", data=housing, hue='median_house_value',
    #             size=housing["population"] / 100, sizes=(3, 300), alpha=0.4, palette="jet", height=6)
    # attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
    # sns.pairplot(data=housing[attributes], height=4)
    # sns.jointplot(x="median_income", y="median_house_value", data=housing, height=6, alpha=0.1)
    # plt.show()
    #
    # housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
    # housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
    # housing["population_per_household"] = housing["population"] / housing["households"]
    # corr_matrix = housing.corr()
    # corr_matrix["median_house_value"].sort_values(ascending=False)

    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()

    ##########  Data Preprocessing  ##########
    # sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
    # ### Null값 처리 1 : row 제거 ###
    # sample_incomplete_rows.dropna(subset=["total_bedrooms"])
    # ### Null값 처리 2 : column 제거 ###
    # sample_incomplete_rows.drop("total_bedrooms", axis=1)
    # ### Null값 처리 3 : 0, 평균, 중간값 등으로 채우기 ###
    # median = housing["total_bedrooms"].median()
    # sample_incomplete_rows["total_bedrooms"].fillna(median, inplace=True)

    ### Preprocessing Pipeline ###
    housing_num = housing.drop('ocean_proximity', axis=1)
    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(categories='auto'), cat_attribs),
    ])

    housing_prepared = full_pipeline.fit_transform(housing)
    # print(housing_prepared)
    # print(housing_prepared.shape)

    ##########  Model Training & Selection ##########
    # model_selection(housing_prepared, housing_labels)

    ##########  Model Detail Tuning ##########
    param_grid = [
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
    ]
    forest_reg = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error',
                               return_train_score=True, n_jobs=-1)
    grid_search.fit(housing_prepared, housing_labels)
    print('##########  Best Parameter  ##########')
    print(grid_search.best_params_)
    print()
    print('##########  Best Estimator  ##########')
    print(grid_search.best_estimator_)
    print()
    print('##########  Hyper Parameter Search Result  ##########')
    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)
    print()
    print('##########  Feature Importances  ##########')
    feature_importances = grid_search.best_estimator_.feature_importances_
    extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
    cat_encoder = full_pipeline.named_transformers_["cat"]
    cat_one_hot_attribs = list(cat_encoder.categories_[0])
    attributes = num_attribs + extra_attribs + cat_one_hot_attribs
    print(sorted(zip(feature_importances, attributes), reverse=True))
    print()

    ##########  (Option) Preparation & Feature Importances Hyper Parameter Search ##########
    # prepare_select_and_predict_pipeline = Pipeline([
    #     ('preparation', full_pipeline),
    #     ('feature_selection', TopFeatureSelector(feature_importances, k)),
    #     ('forest_reg', RandomForestRegressor(**grid_search.best_params_))
    # ])
    #
    # param_grid = [
    #     {'preparation__num__imputer__strategy': ['mean', 'median', 'most_frequent'],
    #      'feature_selection__k': list(range(1, len(feature_importances) + 1))}
    # ]
    #
    # grid_search_prep = GridSearchCV(prepare_select_and_predict_pipeline, param_grid, cv=5,
    #                                 scoring='neg_mean_squared_error', verbose=2, n_jobs=1)
    # grid_search_prep.fit(housing, housing_labels)
    # print('##########  Best Parameter  ##########')
    # print(grid_search.best_params_)
    # print()
    # print('##########  Hyper Parameter Search Result  ##########')
    # cvres2 = grid_search_prep.cv_results_
    # for mean_score, params in zip(cvres2["mean_test_score"], cvres2["params"]):
    #     print(np.sqrt(-mean_score), params)
    # print()

    ##########  Final Test Result ##########
    final_model = grid_search.best_estimator_
    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()
    X_test_prepared = full_pipeline.transform(X_test)
    final_predictions = final_model.predict(X_test_prepared)
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    print('##########  Final Test Result  ##########')
    print(final_rmse)