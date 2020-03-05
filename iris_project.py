# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 16:46:29 2020

@author: Amuleen Gulati
"""
#import library to read data from csv file
from pandas import read_csv

#import library for visualization
from pandas.plotting import scatter_matrix

#import library for graphical representation
from matplotlib import pyplot as plt

#import library to build the model
from sklearn.model_selection import train_test_split

#import libraries or different models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#import libraries for results
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#import libraries for validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

data_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
dataset = read_csv('iris.csv',names = data_names)

#the number of rows and columns in the dataset
print(dataset.shape)

#print first few columns of the dataset
print(dataset.head(20))

#statistical attributes of data including mean, std. deviation etc.
print(dataset.describe())

#number of instances belonging to each class
print(dataset.groupby('class').size())

#box and whisker plots for each attribute of dataset
dataset.plot(kind = 'box', subplots = True, layout = (2,2), sharex = False, sharey = False,)
plt.show()

#histograms of each attribute
dataset.hist()
plt.show()

#scatter plots to show relationship between attributes
scatter_matrix(dataset)
plt.show()

#Split the dataset into train and test data (80,20)
data = dataset.values
X = data[:,:-1]
Y = data[:,-1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1)

#build models
models = []
models.append(('LR', LogisticRegression(solver = 'liblinear', multi_class = 'ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

#run each model in the list
results = []
names = []

for name, model in models:
    kfold = StratifiedKFold(n_splits = 10, random_state = 1, shuffle = True)
    result = cross_val_score(model,X_train,Y_train, cv = kfold, scoring = 'accuracy')
    results.append(result)
    names.append(name)
    print(name, round(result.mean(),2),round(result.std(),2))

#compare algorithms
plt.boxplot(results, labels = names)
plt.title('Model comparison')
plt.show()

#test model to make predictions on validation dataset
model = SVC(gamma = 'auto')
model.fit(X_train, Y_train)
prediction = model.predict(X_test)

#generate results
print('Accuracy: ',round(accuracy_score(Y_test, prediction),2))
print('Confusion Matrix: \n', confusion_matrix(Y_test, prediction))
print('Classification report: \n',classification_report(Y_test, prediction))
