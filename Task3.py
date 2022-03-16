from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report

dataset = pd.read_csv('Oranges vs Grapefruit.csv')

dataset = pd.DataFrame(dataset)
dataset['name'] = dataset['name'].map(
    {'orange': 0, 'grapefruit': 1}, na_action=None)

x = dataset[dataset.columns[1:3]].values
y = dataset[dataset.columns[0]].values


# Preparamos la grafica en un meshgrid


x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

svc = svm.SVC(kernel='rbf', C=1, gamma=10)
svc.fit(x, y)

plt.subplot(1, 1, 1)
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel('diameter')
plt.ylabel('weight')
plt.xlim(xx.min(), xx.max())
plt.title('SVC con Kernel rbf')
plt.show()


X_train, X_test, y_train, y_test = train_test_split(
    dataset[dataset.columns[:]].values, dataset[dataset.columns[0]].values, test_size=0.7, train_size=0.2, random_state=123)

# train the model on train set
model = svm.SVC(kernel='rbf', C=1)
model.fit(X_train, y_train)

# print prediction results
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))


# defining parameter range
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf', 'poly', 'sigmoid']}

# Refit an estimator using the best found parameters on the whole dataset.
grid = GridSearchCV(svm.SVC(), param_grid, refit=True, cv=3, verbose=3)

# fitting the model for grid search
grid.fit(X_train, y_train)
# print best parameter after tuning
print(grid.best_params_)

# print how our model looks after hyper-parameter tuning
print(grid.best_estimator_)
grid_predictions = grid.predict(X_test)

# print classification report
print(classification_report(y_test, grid_predictions))
