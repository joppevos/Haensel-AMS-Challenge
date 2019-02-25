import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix

#models to try out
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier


# load dataset and copy original
df_raw = pd.read_csv('data/sample.csv')
df = df_raw.copy()

# drop the target from the dataframe
train = df[df.columns[:-1]]
target = df.iloc[:,-1]

# see number of classifications
target.value_counts()

# scaling and applying pca
X = StandardScaler().fit_transform(train)
pca = PCA(n_components=5)
pca.fit(X)
print(pca.explained_variance_ratio_)

#sum of variances of all individual principal components and plot it
pca = PCA().fit(X)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');

# encode the target categoricals to labels for coloring.
lb = LabelEncoder()
target = lb.fit_transform(target)


# split up in train and test. we wil use the training set later as validation when cross-folding
X_train, X_test, y_train, y_test = train_test_split(train_n, target, test_size=0.1)

def spotcheck():
    """Creating Models dictionary for spot-checking"""
    models = {'SVM':SVC(), 'logreg': LogisticRegression(), 'RandomForestClassifier': RandomForestClassifier(),
              'Naive-Bayes Gauss': GaussianNB(), 'Naive-Bayes Multi': MultinomialNB(), 'Decision tree': DecisionTreeClassifier(),
             'Knn': KNeighborsClassifier(n_neighbors=20), 'GradientBoost': GradientBoostingClassifier(),
              'AdaBoost':AdaBoostClassifier(base_estimator= DecisionTreeClassifier())}

    # print f1-scoring for each model on test and training
    for key in models:
        score = cross_validate(models[key], X_train[:3000], y_train[:3000], cv=5, scoring='accuracy')
        print(f"{key} test_set_score: {score['test_score'].mean()}")
        print(score['train_score'])


# expose svm to a larger portion of the dataset
svm = SVC(kernel='linear')
svm.fit(X_train[:10000], y_train[:10000])
pred = svm.predict(X_test)
print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))

def plot_fi(fi):
    """
    :return plot feature importance
     """
    return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False, color='b')


m.fit(X_train, y_train)
fi = pd.DataFrame({'cols':X_train.columns, 'imp':m.feature_importances_}
                       ).sort_values('imp', ascending=False)[:50]

# random forest feature importance
plot_fi(fi[:10])

def random_grid():
    """
    creates different hyperparameters for RF
    :return: Dictionary
    """
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=100, stop=500, num=50)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    return random_grid


m = RandomForestClassifier(n_estimators=100)
score = cross_validate(m, X_train, y_train, cv=5, scoring='accuracy')


# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid(), n_iter = 50,
                               cv = 3, verbose=2, random_state=42, n_jobs = -1)

# find out the best parameters after gridsearch
print(rf_random.best_params_)

m = RandomForestClassifier(n_estimators=200, min_samples_split = 10, min_samples_leaf = 1, max_features = 'sqrt',
                          max_depth = 60, bootstrap = True)

score = cross_validate(m, X_train, y_train, cv=3, scoring='accuracy')
print(score)


# not really succesfull. discontinued on RF