# Assignment number 10:

# Load the Iris dataset
# Display basic information
# Display statistical information
# Display null values
# Fill the null values
# Display and iterpret Boxplot & Histogram
# Identify outliers

#---------------------------------------------------------------------------------------

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


# Checking model score after removing outliers
fig, axes = plt.subplots(2,2)
sns.boxplot(data = df, x ='SepalLength', ax=axes[0,0])
sns.boxplot(data = df, x ='SepalWidth', ax=axes[0,1])
sns.boxplot(data = df, x ='PetalLength', ax=axes[1,0])
sns.boxplot(data = df, x ='PetalWidth', ax=axes[1,1])
plt.show()

df = removeoutliers(df, 'SepalWidth')



#---------------------------------------------------------------------------------------
fig, axis = plt.subplots(2,2)
sns.boxplot(ax = axis[0,0], data = df, y='SepalLength')
sns.boxplot(ax = axis[0,1], data = df, y='SepalWidth')
sns.boxplot(ax = axis[1,0], data = df, y='PetalLength')
sns.boxplot(ax = axis[1,1], data = df, y='PetalWidth')
plt.show()
fig, axis = plt.subplots(2,2)
sns.boxplot(ax = axis[0,0], data = df, y='SepalLength', hue='Species')
sns.boxplot(ax = axis[0,1], data = df, y='SepalWidth', hue='Species')
sns.boxplot(ax = axis[1,0], data = df, y='PetalLength', hue='Species')
sns.boxplot(ax = axis[1,1], data = df, y='PetalWidth', hue='Species')
plt.show()
fig, axis = plt.subplots(2,2)

sns.histplot(ax = axis[0,0], data = df, x='SepalLength', multiple = 'dodge', shrink = 0.8, kde = True)
sns.histplot(ax = axis[0,1], data = df, x='SepalWidth', multiple = 'dodge', shrink = 0.8, kde = True)
sns.histplot(ax = axis[1,0], data = df, x='PetalLength', multiple = 'dodge', shrink = 0.8, kde = True)
sns.histplot(ax = axis[1,1], data = df, x='PetalWidth', multiple = 'dodge', shrink = 0.8, kde = True)
plt.show()
fig, axis = plt.subplots(2,2)
sns.histplot(ax=axis[0,0], data=df, x='SepalLength', hue='Species', element='poly', shrink=0.8, kde= True)
sns.histplot(ax=axis[0,1], data = df, x='SepalWidth', hue = 'Species', element = 'poly', shrink = 0.8, kde = True)
sns.histplot(ax=axis[1,0], data = df, x='PetalLength', hue = 'Species', element = 'poly', shrink = 0.8, kde = True)
sns.histplot(ax=axis[1,1], data = df, x='PetalWidth', hue = 'Species', element = 'poly', shrink = 0.8, kde = True)
plt.show()

