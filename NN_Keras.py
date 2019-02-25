from keras.models import Sequential
from keras.layers import Dense
import keras
import pandas as pd
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# read df, get dummies for target
df = pd.read_csv(r'C:\Users\joppe.voss\Desktop\sample.csv')
X = df.iloc[:, :-1]
y = df.iloc[:,-1]
y_dum = pd.get_dummies(y)


def nn_model():
    """
    create and set the model parameters
    :return: nn model
    """
    # create model
    model = Sequential()
    model.add(Dense(8, input_dim=295, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# split data
X_train, X_test, y_train, y_test = train_test_split(X, y_dum)

# train model
m = nn_model()
history = m.fit(X_train, y_train, epochs=30, batch_size=64, verbose=2)

# plot number of epochs against accuracy
plt.figure(figsize=(8, 8))
plt.plot(history.history['acc'])
plt.xlabel('num_epochs')
plt.ylabel('accuracy')

# crossvalidated model and print f1-micro average
estimator = KerasClassifier(build_fn=nn_model, epochs=10, batch_size=64, )
results = cross_val_score(estimator, X, y_dum, cv=3)
print(results.mean())

# try training with scaled data for better fit
scaler = StandardScaler()
scaler.fit(X)
X_s = scaler.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_s, y_dum)

# fit and plot
m = nn_model()
history = m.fit(X_train, y_train, epochs=30, batch_size=64, verbose=2);
plt.figure(figsize=(8, 8))
plt.plot(history.history['acc'])
plt.xlabel('num_epochs')
plt.ylabel('accuracy')

# less epochs needed for the same accuracy. no difference in f1


