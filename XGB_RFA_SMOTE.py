import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
import time
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

#matplotlib seaborn style
sns.set()

df_raw = pd.read_csv('data/sample.csv')
df = df_raw.copy()
# drop the target fromt the dataframe
df = df[df.columns[:-1]]
target = df_raw.B

# convert catorical to numerical
lb = LabelEncoder()
target = lb.fit_transform(target)

# normalize dataframe
min_max_scaler = preprocessing.MinMaxScaler()
scaled = min_max_scaler.fit_transform(df)
df_norm = pd.DataFrame(scaled)

# sum of variances of all individual principal components.
pca = PCA().fit(df_norm)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');

# transform dataset with ideal pca components
pca = PCA(n_components=50).fit(df_norm)
df_pca = pca.transform(df_norm)

xgb_model = xgb.XGBClassifier(objective="multi:softprob", scale_pos_weight=3,random_state=42, n_jobs=-1)

# feature selection based on importance of the splits.
from sklearn.feature_selection import RFECV

rfe = RFECV(estimator=xgb_model,
              step=4,
              cv=StratifiedKFold(
                       n_splits =2,
                       shuffle=False,
                       random_state=101),
              scoring='f1_macro',
              verbose=0)

# time fitting RFE
start = time.time()
rfecv = rfe.fit(df_pca, target)
end = time.time()-start
print('time:', end)

# Summarize the number of features left
print('Optimal number of features: %d' % rfecv.n_features_)
rfecv.grid_scores_.mean()

# transfrom the pca set
df_rfe = rfecv.transform(df_pca)
# reduce to the optimal found columns
df_rfe.shape
# 46 columns are left

# stratify the splits, to have the same distribution
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_rfe, target, test_size=0.1,stratify=target)
X_train.shape

# initialize weights.
weights = np.zeros(len(y_train))
weights[y_train != 2] = 5
weights[y_train == 2] = 1

xgb_model.fit(X_train, y_train, sample_weight=weights)
y_pred = xgb_model.predict(X_test)
print(pd.DataFrame(confusion_matrix(y_test, y_pred)))

# set xbg with grisearch found parameters
xgb_model = xgb.XGBClassifier(objective="multi:softprob", random_state=42, scale_pos_weight=5,
                              seed=42, max_depth=9, min_child_weight=3)
xgb_model.fit(X_train, y_train, sample_weight=weights)

# matrix predictions
y_pred = xgb_model.predict(X_test)
pd.DataFrame(confusion_matrix(y_test, y_pred))

from sklearn.metrics import f1_score
f1_score(y_pred, y_test, average='macro')

print(classification_report(y_test, y_pred))

# apply SMOTE oversampling method
print("Before OverSampling, counts of label '1': {}".format(pd.Series(y_train).value_counts()))
SMOTE(sampling_strategy='minority')
sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_sample(X_train[:1000], y_train.ravel([:1000]))
print('After OverSampling, the shape of train_y: {} \n'.format(pd.Series(y_train_res).value_counts()))

# a try without weights since it is 'balanced' now
xgb_model.fit(X_train_res, y_train_res)

y_pred = xgb_model.predict(X_test)
pd.DataFrame(confusion_matrix(y_test,
                              y_pred))


