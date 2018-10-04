# "Multi-class classification with Python"
# By: Luthfi Zharif

# Import library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import pickle

# Import dataset
# X is predictor variable while y is the class variable
# It's assumed that the last variable is class variable, while the other is predictor
dataset = pd.read_csv('dataset.csv')
X = dataset.iloc[:, 0:-1].values
y = dataset.iloc[:, -1].values

# To count number of unique class and its occurences
y_label, y_label_occurences = np.unique(y, return_counts=True)
print('Class Variable contain:', y_label)
print('The count of each class variable', y_label_occurences)

# Split dataset into train & test
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature scaling was done to ease up in finding local minima
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Try to train a SVM model with RBF kernel
# Let's try vanilla build first
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# See train & test accuracy
train_accuracy = classifier.score(X_train, y_train)
test_accuracy = classifier.score(X_test, y_test)
print('Train data accuracy:', train_accuracy)
print('Test data accuracy:', test_accuracy)

# Create Confusion Matrix
from sklearn.metrics import confusion_matrix
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

# Calculate performance metrics for each class
classes = y_label.tolist()
accuracy_class = []
precision_class = []
recall_class =  []
F1_class = []

for label in range(len(classes)):
    column = cm[:, label]
    TP = column[label]
    column_false = np.delete(column,label)
    row = cm[label, :]
    row_false = np.delete(row,label)
    
    FP = np.sum(column_false)
    FN = np.sum(row_false)
    prec = TP / (TP + FP)
    reca = TP / (TP + FN)
    F1 = 2 * prec * reca / (prec + reca)
    accuracy_class.append(TP / (TP + FP + FN))
    precision_class.append(prec)
    recall_class.append(reca)
    F1_class.append(F1)
    
print('Accuracy for each class: ', accuracy_class)
print('Precision for each class: ', precision_class)
print('Recall for each class: ', recall_class)
print('F1 for each class: ', F1_class)

# Let's make confusion matrix more appealing ;)
cmap = plt.cm.Blues
title = "Confusion Matrix"

# Show confusion matrix
plt.figure()
plt.imshow(cm, interpolation='nearest', cmap=cmap)
plt.title(title)
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

fmt = 'd'
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True value')
plt.xlabel('Predicted')
plt.show()

# Save the model (just in case)
filename_noproc = 'model.sav'
filename_scale = 'scale.sav'

with open(filename_noproc, 'wb') as f:
	pickle.dump(classifier, f)
with open(filename_scale, 'wb') as f:
	pickle.dump(sc, f)