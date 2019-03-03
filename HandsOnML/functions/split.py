from sklearn.model_selection import StratifiedShuffleSplit

def stratsplit(dataset, columnn, n_splits=1, test_size=0.2, random_state=42):
    split = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    for train_index, test_index in split.split(dataset, columnn):
        strat_train_set = dataset.loc[train_index]
        strat_test_set = dataset.loc[test_index]
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    return strat_train_set, strat_test_set