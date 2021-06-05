# Random Forest Classification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Training the Random Forest Classification model on the Training set
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import plot_precision_recall_curve

################################# Setup 1 ####################################
classifier1 = RandomForestClassifier(n_estimators = 1, criterion = 'entropy', random_state = 0).fit(X_train, y_train)
scores1 = cross_val_score(classifier1, X_train, y_train, cv=5)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores1.mean(), scores1.std()))

# Predicting a new result
y_pred_1 = classifier1.predict(X_test)

# Making the Confusion Matrix

tn1, fp1, fn1, tp1 = confusion_matrix(y_test, y_pred_1).ravel()
print('True negative: '+str(tn1))
print('True positive: '+str(tp1))
print('False negative: '+str(fn1))
print('False positive: '+str(fp1))


f1_score1 = f1_score(y_test, y_pred_1, average='macro')
print('f1 score: {0:0.2f}'.format(f1_score1))


average_precision1 = average_precision_score(y_test, y_pred_1)
print('Average precision score: {0:0.2f}'.format(average_precision1))


disp1 = plot_precision_recall_curve(classifier1, X_test, y_test)
disp1.ax_.set_title('2-class Precision-Recall curve: ''AP={0:0.2f}'.format(average_precision1))

################################# Setup 2 ####################################
classifier2 = RandomForestClassifier(n_estimators = 15, criterion = 'entropy', random_state = 0).fit(X_train, y_train)
scores2 = cross_val_score(classifier1, X_train, y_train, cv=5)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores2.mean(), scores2.std()))

# Predicting a new result
y_pred_2 = classifier2.predict(X_test)

# Making the Confusion Matrix

tn2, fp2, fn2, tp2 = confusion_matrix(y_test, y_pred_2).ravel()
print('True negative: '+str(tn2))
print('True positive: '+str(tp2))
print('False negative: '+str(fn2))
print('False positive: '+str(fp2))


f1_score2 = f1_score(y_test, y_pred_2, average='macro')
print('f1 score: {0:0.2f}'.format(f1_score2))


average_precision2 = average_precision_score(y_test, y_pred_2)
print('Average precision score: {0:0.2f}'.format(average_precision2))


disp2 = plot_precision_recall_curve(classifier2, X_test, y_test)
disp2.ax_.set_title('2-class Precision-Recall curve: ''AP={0:0.2f}'.format(average_precision2))

################################# Setup 3 ####################################
classifier3 = RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state = 0).fit(X_train, y_train)
scores3 = cross_val_score(classifier3, X_train, y_train, cv=5)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores3.mean(), scores3.std()))

# Predicting a new result
y_pred_3 = classifier3.predict(X_test)

# Making the Confusion Matrix

tn3, fp3, fn3, tp3 = confusion_matrix(y_test, y_pred_3).ravel()
print('True negative: '+str(tn3))
print('True positive: '+str(tp3))
print('False negative: '+str(fn3))
print('False positive: '+str(fp3))


f1_score3 = f1_score(y_test, y_pred_3, average='macro')
print('f1 score: {0:0.2f}'.format(f1_score3))


average_precision3 = average_precision_score(y_test, y_pred_3)
print('Average precision score: {0:0.2f}'.format(average_precision3))


disp3 = plot_precision_recall_curve(classifier3, X_test, y_test)
disp3.ax_.set_title('2-class Precision-Recall curve: ''AP={0:0.2f}'.format(average_precision3))

print(f'& {tn1} & {tp1} & {fn1} & {fp1} & {round(f1_score1,4)} & {round(average_precision1,4)}')
print(f'& {tn2} & {tp2} & {fn2} & {fp2} & {round(f1_score2,4)} & {round(average_precision2,4)}')
print(f'& {tn3} & {tp3} & {fn3} & {fp3} & {round(f1_score3,4)} & {round(average_precision3,4)}')
