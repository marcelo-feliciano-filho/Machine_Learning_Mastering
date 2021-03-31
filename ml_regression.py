from sklearn import linear_model
from pandas import read_csv, DataFrame
from os import path as ospath

root = r'C:\Users\marce\Desktop\PUCPR\MESTRADO EM SISTEMAS DE AUTOMAÇÃO E IA\Courses\Machine learning\Computational works (works in group)\Machine_Learning_Mastering'
cols = ['I0', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'O']
dataset = read_csv(ospath.join(root, r'docs\work_01\qsar_fish_toxicity.csv'), sep=';', names=cols)
clf = linear_model.Lasso(alpha=0.1)

model = clf.fit(dataset[cols[0:5]], dataset[cols[6]])

