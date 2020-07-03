# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#########

## Import the data, read into the program
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

#########
#########

## Summarize the data, look at the different aspects and values

## Show the shape of the data
print (dataset.shape)

## Show the top contents of the data
print (dataset.head())

## Show the stats of the data
print (dataset.describe())

## Show the different classes of data 
print(dataset.groupby('class').size())

#########
#########

## Visualizing the data
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()

## Playing with different plots
dataset.plot(kind='kde', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()

## Create a histogram of the data
dataset.hist()
pyplot.show()

## Create scatterplots of the data, in order to find structured relationships
## between variables. A diagonal grouping of pairs of attributes suggests
## high correlation and a predictable relationship.
scatter_matrix(dataset)
pyplot.show()

#########
#########

"""
Create models and compare their accuracy on the test data
1. Separate out a validation dataset.
2. Set-up the test harness to use 10-fold cross validation.
3. Build multiple different models to predict species from flower measurements
4. Select the best model.
"""

## Split the data into a train and test set, with X and y variables.
array = dataset.values
X = array[:,:-1]
y = array[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8, random_state=1)

## Using a stratified 10-fold cross validation to estimate model accuracy
## This is when the data is split into 10 part, trained on 9 and tested on 1,
## and repeated for all combos of train_test_split

## Stratified means that each fold in the dataset will have the same 
## distribution of class as the whole dataset. A representative sample.

## 6 different algorithms will be tested on this dataset.
"""
1. Logistic Regression (LR) - Linear
2. Linear Discriminant Analysis (LDA) - Linear
3. K-Nearest Neighbors (KNN). - Nonlinear
4. Classification and Regression Trees (CART). - Nonlinear
5. Gaussian Naive Bayes (NB). - Nonlinear
6. Support Vector Machines (SVM). - Nonlinear
"""


# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()

#########
#########

## Having seen that SVM is the most accurate, we choose this to make a model
## of the dataset for predictions.

model = SVC(gamma='auto')
model.fit(X_train, y_train)
predictions = model.predict(X_test)

## Evaluate the predictions
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
