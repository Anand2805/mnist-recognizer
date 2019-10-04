import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

df_train = pd.read_csv('./train.csv')
x = np.array(df_train.iloc[:, 1:]) # end index is exclusive
y = np.array(df_train['label'])
X_1, X_test, y_1, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
#X_1[X_1 > 0] = 1
#X_test[X_test > 0] = 1

def findNEstimators():
    r=range(200,550,50)
    accR_scores = []
    for i in range(200,550,50):
        rfc = RandomForestClassifier(n_estimators=i)
        rfc.fit(X_1[:10000], y_1[:10000])
        predicted = rfc.predict(X_test[:10000])
        accRFC = accuracy_score(y_test[:10000], predicted, normalize=True) * float(100)
        print('\nRF accuracy for n = %d is approx %d%%' % (i, accRFC))
        accR_scores.append(accRFC)
    print(accR_scores)
    iRFC = np.argmax(accR_scores)
    return r[iRFC], accR_scores[iRFC]

def trainAndPredictData(nRFC,queryData):
    print(nRFC)
    rfc = RandomForestClassifier(n_estimators=nRFC)
    rfc.fit(X_1[:5000], y_1[:5000])
    predicted = rfc.predict(queryData.reshape(1, -1))
    print(predicted)
    return predicted