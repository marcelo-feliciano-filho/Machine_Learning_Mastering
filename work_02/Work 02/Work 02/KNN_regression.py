# Extra trees regression


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

# Importing the dataset


dataset = pd.read_csv('qsar_fish_toxicity.csv',delimiter=";")

X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Training the Random Forest Regression model on the whole dataset
from sklearn.neighbors import KNeighborsRegressor

# Metrics
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

################################# Setup 1 ####################################


regressor1 = KNeighborsRegressor(n_neighbors=2).fit(X_train, y_train)

R2_train_1 = cross_val_score(regressor1, X_train, y_train, cv=5)
import statistics

r2_mean_1 = statistics.mean(R2_train_1)
print('Coefficient of determination (training): '+str(round(r2_mean_1,5)))
# Predicting a new result
y_pred_1 = regressor1.predict(X_test)

#Coefficient determination
R2_1 = r2_score(y_test,y_pred_1)
print('Coefficient of determination (test): '+str(round(R2_1,5)))

# MAE
MAE_1 = mean_absolute_error(y_test,y_pred_1)
print('Mean absolute Error (test): '+str(round(MAE_1,5)))

#MSE
MSE_1 = mean_squared_error(y_test,y_pred_1)
print('Mean Squared Error (test): '+str(round(MSE_1,5)))
    

################################# Setup 2 ####################################
regressor2 = KNeighborsRegressor(n_neighbors=4).fit(X_train, y_train)

R2_train_2 = cross_val_score(regressor2, X_train, y_train, cv=5)

r2_mean_2 = statistics.mean(R2_train_2)
print('Coefficient of determination (training): '+str(round(r2_mean_2,5)))
# Predicting a new result
y_pred_2 = regressor2.predict(X_test)

#Coefficient determination
R2_2 = r2_score(y_test,y_pred_2)
print('Coefficient of determination (test): '+str(round(R2_2,5)))

# MAE
MAE_2 = mean_absolute_error(y_test,y_pred_2)
print('Mean absolute Error (test): '+str(round(MAE_2,5)))

#MSE
MSE_2 = mean_squared_error(y_test,y_pred_2)
print('Mean Squared Error (test): '+str(round(MSE_2,5)))
    

################################# Setup 3 ####################################
regressor3 = KNeighborsRegressor(n_neighbors=8).fit(X_train, y_train)

R2_train_3 = cross_val_score(regressor3, X_train, y_train, cv=5)

r2_mean_3 = statistics.mean(R2_train_3)
print('Coefficient of determination (training): '+str(round(r2_mean_3,5)))
# Predicting a new result
y_pred_3 = regressor3.predict(X_test)

#Coefficient determination
R2_3 = r2_score(y_test,y_pred_3)
print('Coefficient of determination (test): '+str(round(R2_3,5)))

# MAE
MAE_3 = mean_absolute_error(y_test,y_pred_3)
print('Mean absolute Error (test): '+str(round(MAE_3,5)))

#MSE
MSE_3 = mean_squared_error(y_test,y_pred_3)
print('Mean Squared Error (test): '+str(round(MSE_3,5)))

###############################################################################
    

fig, axs = plt.subplots(3, sharex=True, sharey=True)
fig.suptitle('Target x Predict')

axs[0].plot(y_test)
axs[0].plot(y_pred_1,'tab:orange')
axs[0].legend("Y""1",loc="lower left")

axs[1].plot(y_test)
axs[1].plot(y_pred_2,'tab:green')
axs[1].legend("Y""2",loc="lower left")
axs[1].set(ylabel='LC50 [-LOG(mol/L)]')

axs[2].plot(y_test)
axs[2].plot(y_pred_3,'tab:red')
axs[2].legend("Y""3",loc="lower left")
axs[2].set(xlabel='Samples')




print(f'& {round(r2_mean_1,4)} & {round(R2_1,4)} & {round(MSE_1,4)} & {round(MAE_1,4)}')
print(f'& {round(r2_mean_2,4)} & {round(R2_2,4)} & {round(MSE_2,4)} & {round(MAE_2,4)}')
print(f'& {round(r2_mean_3,4)} & {round(R2_3,4)} & {round(MSE_3,4)} & {round(MAE_3,4)}')
    




    

