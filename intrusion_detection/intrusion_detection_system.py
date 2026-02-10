# -*- coding: utf-8 -*-
"""week4_notebook.ipynb

This project was originally done on Colab in segments.

Original file is located at
    https://colab.research.google.com/drive/1pwVdirUx370HnPHjhJfs7nul_ai4j_Wp
"""

import pandas as pd

df = pd.read_csv('intrusion_detection/Network_Intrusion_Dataset(1).csv')

# familiarize with table
df.head()

#list all variables, not all were displayed in head
list(df.columns)

#check for missing data
df.isnull().sum()

df.shape

# data types
df.info()

df.describe().transpose()

df = df.drop(columns=df.loc[:, 'Packets_Rx_Dropped':'Packets_Tx_Errors'].columns)
df = df.drop(columns=df.loc[:, 'Delta_Packets_Rx_Dropped':'Delta_Packets_Tx_Errors'].columns)
df = df.drop(['Is_Valid', 'Table_ID', 'Max_Size'], axis=1)
df.describe(include="all").transpose() #include all means both numeric and object

#save dataset
df.to_csv(r'/content/prepared_Network_Intrusion_Dataset.csv', index=False)

# load in prepared dataset
df_prepared = pd.read_csv('/content/prepared_Network_Intrusion_Dataset.csv')

# input is whole data except for the 2 targets
#create a dataframe with all training data except the target column
X = df_prepared.drop(columns=['Traffic_Type','Intrusion_Traffic_Type'])
# here, we select one target variable to model, Traffic_Type
y = df_prepared['Traffic_Type']
#check that the list of input variables
list(X)

from sklearn.model_selection import train_test_split
#split dataset into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=14, stratify=y) #use random state number so that it can be reproduced (same split/shuffle)

#This is to show the number of instances and input features in the training and test sets
print('X_train Instances', X_train.shape)
print('X_test Instances', X_test.shape)

from sklearn.neighbors import KNeighborsClassifier
# Create a KNN classifier
knn = KNeighborsClassifier(n_neighbors = 9) # hyper paramerter = one the data scientist chooses, 9 is a random num

# Fit the classifier to the data
knn.fit(X_train,y_train)
# now the model is trained

#Perform predictions on the test data
y_pred=knn.predict(X_test)

#Create a dataframe for comparing the actual vs predicted results by kNN mode
compare_results_knn_df = pd.DataFrame({'Actual':y_test, 'Predicted': y_pred})
compare_results_knn_df.to_csv(r'/content/knn_pred_comparison.csv', index=True)
compare_results_knn_df

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

"""86% accuracy"""

#Import the packages for costructing the confusion matrix
from sklearn.metrics import confusion_matrix

#Import the packages for plotting the confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay

#Costruct the confusion matrix based on…
#comparing actual values (y_test) vs predicted (y_pred) in test data
cm_knn = confusion_matrix(y_test, y_pred, labels = knn.classes_)

#Plot the confusion matrix
disp_knn_cm = ConfusionMatrixDisplay(cm_knn, display_labels=knn.classes_)
disp_knn_cm.plot()

from sklearn.metrics import RocCurveDisplay
knn_roc = RocCurveDisplay.from_estimator(knn, X_test, y_test)

# Calculating error for K values between 1 and 40
error = []
import numpy as np
import matplotlib.pyplot as plt
# Calculating error for K values between 1 and 40
for i in range(1, 40):
    knn2 = KNeighborsClassifier(n_neighbors=i)
    knn2.fit(X_train, y_train)
    pred_i = knn2.predict(X_test)
    error.append(np.mean(pred_i != y_test))
plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')

"""lowest error rate when k=1, try again with k=1 and view accuracy reports"""

from sklearn.neighbors import KNeighborsClassifier
# Create KNN classifier
knn1 = KNeighborsClassifier(n_neighbors = 1)
# Fit the classifier to the data
knn1.fit(X_train,y_train)
#Perform predictions on the test data
y_pred=knn1.predict(X_test)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
cm_knn1 = confusion_matrix(y_test, y_pred, labels = knn1.classes_)
disp_knn1_cm = ConfusionMatrixDisplay(cm_knn1, display_labels=knn1.classes_)
disp_knn1_cm.plot()

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

"""need to find best combination between distance type and k value to get the best model

use grid search algorithm with k on one side and different distances on the other side. want to find the best best error score to find which will give the least error using k and distance combined

"""

from sklearn.model_selection import GridSearchCV
import numpy as np
#create new a knn model
knn = KNeighborsClassifier()
#create a dictionary of all values we want to test for n_neighbors and distances
param_grid = {'n_neighbors': np.arange(1, 25), 'metric': ['euclidean', 'manhattan']}
#use gridsearch to test all values for n_neighbors
knn_gscv = GridSearchCV(knn, param_grid, cv=5, scoring = 'roc_auc') # you can look for the best of any metric (true positive, error, etc)


#fit model to data - we actually trained it on the test data bc it was included in the test set earlier - oops!

knn_gscv.fit(X, y)

knn_gscv.best_params_

# Perform testing on test dataset
y_pred = knn_gscv.predict(X_test)
# Construct a confusion matrix
cm_knn_gscv = confusion_matrix(y_test, y_pred, labels = knn_gscv.classes_)
disp_knn_gscv_cm = ConfusionMatrixDisplay(cm_knn_gscv, display_labels=knn_gscv.classes_)
disp_knn_gscv_cm.plot()
# Display the classification report
print(classification_report(y_test, y_pred))
