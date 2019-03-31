from plot_mnist import *
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve, roc_auc_score
import numpy as np

np.random.seed(42)

if __name__ == '__main__':
    ##########  Fetch Mnist  ##########
    mnist = fetch_openml('MNIST_784', version=1)
    X, y = mnist["data"], mnist['target']
    print(X.shape)
    print(y.shape)
    y = y.astype(np.int)

    ##########  Plot Mnist  ##########
    # plot_digit(X[36000])

    plt.figure(figsize=(9, 9))
    example_images = np.r_[X[:12000:600], X[13000:30600:600], X[30600:60000:590]]
    plot_digits(example_images, images_per_row=10)

    ##########  Mnist Data Preparation  ##########
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
    shuffle_index = np.random.permutation(60000)
    X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

    ##########  Binary Classification  ##########
    y_train_5 = (y_train == 5)
    y_test_5 = (y_test == 5)

    sgd_clf = SGDClassifier(max_iter=5, random_state=42)
    sgd_clf.fit(X_train, y_train_5)

    some_digit = X[36000]
    y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
    print('###  Confusion matrix  ###')
    print(confusion_matrix(y_train_5, y_train_pred))
    print()
    print('Precision Score : {}'.format(precision_score(y_train_5, y_train_pred)))
    print('Recall Score : {}'.format(recall_score(y_train_5, y_train_pred)))
    print('F1 Score : {}'.format(f1_score(y_train_5, y_train_pred)))

    y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
    precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
    plt.figure(figsize=(8, 4))
    plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

    plt.figure(figsize=(8, 6))
    plot_precision_vs_recall(precisions, recalls)

    fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
    plt.figure(figsize=(8, 6))
    plot_roc_curve(fpr, tpr)

    forest_clf = RandomForestClassifier(n_estimators=10, random_state=42)
    y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")
    y_scores_forest = y_probas_forest[:, 1]
    fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b:', linewidth=2, label="SGD")
    plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")

    print('ROC AUC Score : {}'.format(roc_auc_score(y_train_5, y_scores)))

    ##########  Multilabel Classification  ##########
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
    print("Mnist Accuracy : {}".format(cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")))
    print()

    y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
    conf_mx = confusion_matrix(y_train, y_train_pred)

    print('###  Confusion matrix  ###')
    print(conf_mx)

    plot_confusion_matrix(conf_mx)

    row_sums = conf_mx.sum(axis=1, keepdims=True)
    norm_conf_mx = conf_mx / row_sums

    np.fill_diagonal(norm_conf_mx, 0)
    plot_confusion_matrix(norm_conf_mx)