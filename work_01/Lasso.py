# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 19:58:06 2021

@author: Bruno, Marcelo & Murillo
"""


# ElasticNet Regression 

 
# Importing the libraries
from os import path as ospath
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.feature_selection import VarianceThreshold
index_a = 1
index_b = 2
root_dir = 'docs'
datasets = ['Airfoil_Self-Noise_Data_Set', 'qsar_aquatic_toxicity', 'qsar_fish_toxicity']

for file in datasets:
    
    print(f'Presenting {file} Benchmark results for LASSO')
        
    # Importing the dataset
    dataset = pd.read_csv(f'{ospath.join(root_dir,file)}.csv',delimiter=";")
    
    #Defining inputs and outputs
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    
    # Transforms input array X into Variance Threshold to remove low-variance features
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    X = sel.fit_transform(X)
    
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
    
    
    # Training the Multiple Linear Regression model on the Training set
    regressor = LassoCV(eps=0.001, n_alphas=50, normalize=True, cv=5, random_state=0).fit(X_train,y_train)
    
    # Regression metrics
    # Coefficient of determination R2 
    R2_train = regressor.score(X_train, y_train)
    print('Coefficient of determination (training): '+str(round(R2_train,5)))
    
    # Mean Squared Error
    y_pred = regressor.predict(X_test)
    MSE = mean_squared_error(y_test,y_pred)
    print('Mean Squared Error (test): '+str(round(MSE,5)))
    
    
    # Coefficient of determination
    R2 = r2_score(y_test,y_pred)
    print('Coefficient of determination (test): '+str(round(R2,5)))
    
    # Mean absolute Error
    MAE = mean_absolute_error(y_test,y_pred)
    print('Mean absolute Error (test): '+str(round(MAE,5)))
    
    # Plotting results
    # Plotting real x predict
    plt.figure(index_a)
    plt.plot(y_test,color='royalblue',label='Real')
    plt.plot(y_pred,color='crimson',label='Predict')
    plt.legend(loc='lower left')
    plt.grid(axis='x', color='0.95')
    plt.grid(axis='y', color='0.95')
    plt.xlabel('Samples')
    plt.xlim(0, len(y_pred))
    plt.title('Comparison between real and predict')
    if file == 'Airfoil_Self-Noise_Data_Set':
        plt.ylabel('decibels (dB)')
    else:
        plt.ylabel('LC50 [-LOG(mol/L)]')
    
    # Plotting Error
    plt.figure(index_b)
    plt.plot(abs(abs(y_test) - abs(y_pred)), color ='limegreen')
    plt.xlabel('Samples')
    plt.xlim(0, len(y_pred))
    plt.title('Residual')
    plt.grid(axis='x', color='0.95')
    plt.grid(axis='y', color='0.95')
    if file == 'Airfoil_Self-Noise_Data_Set':
        plt.ylabel('decibels (dB)')    
    else:
        plt.ylabel('LC50 [-LOG(mol/L)]')
    
    index_a = index_a + 2;
    index_b = index_b + 2;
    
    
