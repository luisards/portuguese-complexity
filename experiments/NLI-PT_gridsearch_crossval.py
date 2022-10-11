import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold

# Uncomment each feature set and model at a time

"""
# All features
proficiency_df = pd.read_csv("../data/NLI-PT_all_features_new_noNaN.csv", encoding='unicode_escape')
proficiency_df = proficiency_df.drop('ï»¿', axis=1)
X = proficiency_df.iloc[:, 1:-1].values
y = proficiency_df.iloc[:, 444].values
"""

"""
# CFSSubsetEval features
weka_df = pd.read_csv("../data/NLI-PT_cfseval_features.csv", encoding='unicode_escape')
X = weka_df.iloc[:, :-1].values
y = weka_df.iloc[:, 34].values
"""

"""
#Surface features
surface_df = pd.read_csv("../data/NLI-PT_surface_features.csv", encoding='unicode_escape')
X = surface_df.iloc[:, :-1].values
y = surface_df.iloc[:, 17].values
"""

"""
# Lexical features
lexical_df = pd.read_csv("../data/NLI-PT_lexical_features.csv", encoding='unicode_escape')
X = lexical_df.iloc[:, :-1].values
y = lexical_df.iloc[:, 193].values
"""

"""
# Morphosyntactic features
proficiency_df = pd.read_csv("../data/NLI-PT_morphosyntactic_features.csv", encoding='unicode_escape')
X = proficiency_df.iloc[:, :-1].values
y = proficiency_df.iloc[:, 114].values
"""


#Cohesion features
cohesion_df = pd.read_csv("../data/NLI-PT_cohesion_features.csv", encoding='unicode_escape')
X = cohesion_df.iloc[:, :-1].values
y = cohesion_df.iloc[:, 11].values


"""
# Dematto's features
data = pd.read_csv("../data/NLI-PT_demattos_features.csv", encoding='unicode_escape')
data.fillna(0, inplace=True)
X = data.iloc[:, 2:-1].values
y = data.iloc[:, 229].values
"""

#SVM
model = SVC()

params = {
    'kernel': ['rbf'],
    'C': [0.08, 0.1, 1, 10, 50, 100, 1000],
    'gamma': [0.001, 0.01, 0.1, 1, 0.5, 1, 'scale', 'auto']
    }


"""
# Random Forest
model = RandomForestClassifier()

params = {'n_estimators': [2, 4, 8, 16, 32, 64, 100, 200],
          'max_depth': [2, 5, 7, 10, 15, 20, 25]
          }
"""

"""
# Logistic Regression
model = LogisticRegression()
params = {'penalty': ['l1', 'l2'],
          'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
          }
"""

# Begin
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

kf = KFold(n_splits=10, shuffle=False)

scoring = ['accuracy', 'f1_weighted']

gs = GridSearchCV(model, param_grid=params, cv=kf, scoring=scoring, refit='f1_weighted', return_train_score=True,
                  verbose=3)

gs.fit(X_train, y_train)

print('Best params:', gs.best_params_)
print('Best score: ', gs.best_score_)
