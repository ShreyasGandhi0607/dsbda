# Assignment number 6:
# Load the Iris dataset
# Display basic information
# Display statistical information
# Display null values
# Fill the null values
# Feature Engineering through correlation matrix
# Build the Gaussian Naive Bayes Model and find its classification score
# Remove outliers and again see the accuracy of the model

import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 

df  = pd.read_csv('Datasets/iris.csv')

# Basic info

print("\nInfomation \n", df.info())
print("\n Columns  \n", list(df.columns))
print('\nShape of Dataset (row x column): ', df.shape)
print('\nDatatype of attributes (columns):', df.dtypes)
print('\nFirst 5 rows:\n', df.head().T)
print('\nLast 5 rows:\n',df.tail().T)
print('\nAny 5 rows:\n',df.sample(5).T)

# Display statistical information 

print(df.describe())

# check missing values 
print(df.isna().sum())

# fill missing values 

df['SepalLength'] = df['SepalLength'].fillna(df['SepalLength'].mean())
df['PetalWidth'] = df['PetalWidth'].fillna(df['PetalWidth'].mean())


print("Ater filling the missing values \n")
print(df.isna().sum())


def removeoutliers(df , var):
    q1 = df[var].quantile(0.25)
    q3 = df[var].quantile(0.75)
    iqr = q3 - q1 
    high , low = q3 + 1.5*iqr, q1 - 1.5*iqr  
    count = df[(df[var] > high) | (df[var]  < low)][var].count()

    print("Total number of outliers are : ", count)

    df = df[(df[var] <= high)  & (df[var] >= low)]

    return df 


def buildmodel(X, y):
    from sklearn.model_selection import train_test_split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
    
    # Importing the Gaussian Naive Bayes model 
    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB()
    model.fit(X_train, y_train)  # Fit the model to the training data
    
    y_pred = model.predict(X_test)  # Predict the target variable using the test features
    
    from sklearn.metrics import confusion_matrix, classification_report
    cn = confusion_matrix(y_test, y_pred)
    sns.heatmap(cn, annot=True)
    plt.show()

    print(classification_report(y_test, y_pred))


X = df.drop(columns=['Species' , 'Id'])
y = df['Species']

buildmodel(X , y)

# Checking model score after removing outliers
fig, axes = plt.subplots(2,2)
sns.boxplot(data = df, x ='SepalLength', ax=axes[0,0])
sns.boxplot(data = df, x ='SepalWidth', ax=axes[0,1])
sns.boxplot(data = df, x ='PetalLength', ax=axes[1,0])
sns.boxplot(data = df, x ='PetalWidth', ax=axes[1,1])
plt.show()


df = removeoutliers(df, 'SepalWidth')

X = df.drop(columns=['Species' , 'Id'])
y = df['Species']


