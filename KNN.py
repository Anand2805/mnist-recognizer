from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df_train = pd.read_csv('./train.csv')
x = np.array(df_train.iloc[:, 1:])  # end index is exclusive
y = np.array(df_train['label'])
X_1, X_test, y_1, y_test = train_test_split(
    x, y, test_size=0.3, random_state=0)


def bestK_PCA():
    print("Begin training...")
    k_best_array = []
    pca_best_array = []
    pca_acc_array = []
    for k in range(1, 11, 2):
        print(k)
        clf = KNeighborsClassifier(n_neighbors=k)
        n_components_array = ([3, 10, 20, 50, 100, 300])
        score_array = np.zeros(len(n_components_array))
        i = 0
        for n_components in n_components_array:
            pca = PCA(n_components=n_components)
            pca.fit(X_1[:18000])
            transform_x = pca.transform(X_1[:18000])
            transform_x_test = pca.transform(X_test[:18000])
            clf.fit(transform_x, y_1[:18000])
            #score_array[i] = clf.score(transform_x ,y_1)
            Y_predicted_test = clf.predict(transform_x_test)
            score_array[i] = accuracy_score(y_test[:18000], Y_predicted_test)
            i = i+1
        j = np.argmax(score_array)
        print("for k = %d max accuracy for n_components = %d is %f%%'" % (k, n_components_array[j],
                                                                          score_array[j]))
        k_best_array.append(k)
        pca_best_array.append(n_components_array[j])
        pca_acc_array.append(score_array[j])
    print(k_best_array)
    print(pca_best_array)
    print(pca_acc_array)
    print("End of training...")
    e = np.argmax(pca_acc_array)
    return k_best_array[e], pca_best_array[e], pca_acc_array[e]


def trainAndPredictData(k, pca_n, queryData, clfKnn, pca):
    print("Begin predict...")
    transform_x_query = pca.transform(queryData.reshape(1, -1))
    predicted = clfKnn.predict(transform_x_query)
    print(predicted)
    return predicted


def getKnnAndTrain(pca_n, k):
    print("Begin after training...")
    pca = PCA(n_components=pca_n)
    pca.fit(X_1)
    transform_x_best = pca.transform(X_1)
    print(transform_x_best.shape)
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(transform_x_best, y_1)
    print("trained clf returned...")
    return clf, pca
