import time

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import randint


def statistical_preparation(data, method='standarization'):
    # print(data.corr())
    tmp = data.drop(columns=['diff_win_overall', 'diff_elo', 'diff_break_points', 'diff_hand'])

    y = tmp['winner']
    X = tmp.drop(columns=['p1', 'p2', 'winner'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=420)

    if method == 'standarization':
        scaler = StandardScaler()
    elif method == 'normalization':
        scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, y_train, X_test_scaled, y_test


def testing(X_train, y_train, X_test, y_test):
    models = [LogisticRegression, GaussianNB, KNeighborsClassifier, RandomForestClassifier,
              # SVC
              ]
    # xgboost
    # lightGBM
    # extra trees
    # regularized greedy forest
    # neural networks

    param_grids = [
        {
            'penalty': ['l1', 'l2'],
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['saga'],
            'max_iter': [100, 200, 300]
        },
        {
            'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
        },
        {
            'n_neighbors': [1, 3, 5],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan'],
            'n_jobs': [-1]
        },

        {
            "max_depth": [10, 20, 30, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        },
        # {
        #     'C': [0.1, 1, 10],
        #     'gamma': ['scale', 'auto']
        # }
    ]

    results = pd.DataFrame(0.0, index=[i.__name__ for i in models], columns=['Train', 'Test'])
    for i, j in zip(models, param_grids):
        start = time.time()
        grid = GridSearchCV(i(), j, cv=5)
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        train_acc = best_model.score(X_train, y_train)
        test_acc = best_model.score(X_test, y_test)
        results.loc[i.__name__] = [round(train_acc * 100, 2), round(test_acc * 100, 2)]

        print(f"{i.__name__} ✓ in {round(time.time() - start, 2)}s")

    print(results.sort_values(by=['Test'], ascending=False))


if __name__ == "__main__":
    data = pd.read_csv('data/result.csv')
    X_train, y_train, X_test, y_test = statistical_preparation(data)
    testing(X_train, y_train, X_test, y_test)
