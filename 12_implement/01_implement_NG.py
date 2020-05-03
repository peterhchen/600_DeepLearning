# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Step 1: 
# Importing the database
dataset = pd.read_csv('./Churn_Modelling.csv')

# step 2
# create matrices of the features of dataset and 
# the target variable, which is column 14, labeled as “Exited”.
# RowNumber,CustomerId,Surname,CreditScore,Geography,Gender,Age,
# Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,
# EstimatedSalary,Exited
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values
print ('X[:10]', X[:10])
print ('y[:10]', y[:10])

# step 3
# We make the analysis simpler by encoding string variables. 
# We are using the ScikitLearn function ‘LabelEncoder’ to automatically 
# encode the different labels in the columns with values 
# between 0 to n_classes-1.
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder() 
X[:,1] = labelencoder_X_1.fit_transform(X[:,1]) 
labelencoder_X_2 = LabelEncoder() 
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
print ('X[:10]', X[:10])

# Ste 4:
# Labelling Encoded Data
# We use the same ScikitLearn library and another function called the 
# OneHotEncoder to just pass the column number creating a dummy variable.
#onehotencoder = OneHotEncoder(categorical features = [1])
onehotencoder = OneHotEncoder(handle_unknown='ignore')
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]
print ('X[:10]', X[:10])

# Step 5: Split data into train and test
# Splitting the dataset into the Training set and the Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Step 6:
# Feature Scaling
from sklearn.preprocessing import StandardScaler 
sc = StandardScaler() 
X_train = sc.fit_transform(X_train) 
X_test = sc.transform(X_test)

# Step 7:
# Importing the Keras libraries and packages 
import keras 
from keras.models import Sequential 
from keras.layers import Dense

# Step 8: Initializing Neural Network 
classifier = Sequential()

# Step 9
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', 
activation = 'relu', input_dim = 11))
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', 
activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', 
activation = 'sigmoid'))

# Step 10: Compiling Neural Network 
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Step 11
print ('X_train.shape', X_train.shape)
print ('y_train.shape', y_train.shape)
# classifier.fit(X_train, y_train, batch_size = 10, epochs = 50)

# Step 12
# Predicting the Test set results 
# y_pred = classifier.predict(X_test) 
# y_pred = (y_pred > 0.5)
# new_prediction = classifier.predict(sc.transform
# (np.array([[0.0, 0, 500, 1, 40, 3, 50000, 2, 1, 1, 40000]])))
# new_prediction = (new_prediction > 0.5)
# Step 13
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print (cm)
