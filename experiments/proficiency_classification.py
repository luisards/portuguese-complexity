import csv
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import metrics
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("../data/NLI-PT_all_features_noNaN.csv", encoding= 'unicode_escape')

X = df.iloc[:, 2:-1].values
y = df.iloc[:, 442].values

X = preprocessing.scale(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param = {'kernel': ('linear', 'poly', 'rbf', 'sigmoid'),
          'C': (1,52,10),
         'degree':[3,8],
         'coef0':[0.001,10,0.5],
         'gamma':('auto', 'scale')}

# Grid search and cross validation for best hyperparameters with SVM
SVModel=SVC()
gridS=GridSearchCV(SVModel, param, cv=5, verbose = 3)
gridS.fit(X_train, y_train)
gridS.cv_results_

#SVM with best parameters
svm_model = SVC(C=10, coef0=0.5, degree=3, gamma='scale', kernel='rbf')
svm_model.fit(X_train, y_train)

predictions = svm_model.predict(X_test)

print(metrics.classification_report(y_test, predictions))
