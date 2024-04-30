# Assignment number 1:
    # Load the dataset
    # Display basic information
    # Display statistical information
    # Display null values
    # Fill the null values
    # Change datatype of variable
    # Quantization (Encoding): Convert categorical to numerical variable
    # Normalization

import pandas as pd
import numpy as np 

df = pd.read_csv('Datasets/Placement.csv')

# Display basic information
print('\nInformation of Dataset:\n', df.info())
print('\nShape of Dataset (row x column): ', df.shape)
print('\nColumns Name: ', df.columns)
print('\nTotal elements in dataset:', df.size)
print('\nDatatype of attributes (columns):', df.dtypes)
print('\nFirst 5 rows:\n', df.head())
print('\nLast 5 rows:\n',df.tail())
print('\nAny 5 rows:\n',df.sample(5))


print("\n\nDisplay Statistical information !!!!! ")
print(df.describe())

print("\n\n display null values ")
print(df.isna().sum() )   # it will show how many missing value present in each columns


df['gender'].fillna(df['gender'].mode()[0], inplace=True)     # fill gender column with mode value if it is NaN# fill NA/NaN values using different methods# fill NA/NaN values using different methods
df['ssc_p'].fillna(df['ssc_p'].mean(), inplace=True)
df['ssc_b'].fillna(df['ssc_b'].mode()[0], inplace=True)
df['hsc_p'].fillna(df['hsc_p'].mean(), inplace=True)
df['hsc_b'].fillna(df['hsc_b'].mode()[0], inplace=True)
df['hsc_s'].fillna(df['hsc_s'].mode()[0], inplace=True)
df['degree_p'].fillna(df['degree_p'].mean(), inplace=True)
df['degree_t'].fillna(df['degree_t'].mode()[0], inplace=True)
df['mba_p'].fillna(df['mba_p'].mean(), inplace=True)
df['etest_p'].fillna(df['etest_p'].mean(), inplace=True)
df['workex'].fillna(df['workex'].mode()[0], inplace=True)
df['workex'].fillna(df['workex'].mode()[0], inplace=True)
df['specialisation'].fillna(df['specialisation'].mode()[0], inplace=True)




print('Total Number of Null Values in Dataset:', df.isna().sum())

df['sl_no']=df['sl_no'].astype('int8')
print('Change in datatype: ', df['sl_no'].dtypes)


# Converting categorical (qualitative) variable to numeric (quantitative) variable

# 1. Find and replace method
# 2. Label encoding method
# 3. OrdinalEncoder using scikit-learn


df['gender'].replace(['M','F'],[0,1],inplace=True)
df['status'].replace(['Placed','Not Placed'],[1,0],inplace=True)

# Label encoding method
df['ssc_b']=df['ssc_b'].astype('category') #change data type to category
df['ssc_b']=df['ssc_b'].cat.codes

# Ordinal encoder using Scikit-learn
from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder()
df[['hsc_b']]=enc.fit_transform(df[['hsc_b']])


# Normalization of data
# converting the range of data into uniform range
# marks [0-100] [0-1]
# salary [200000 - 200000 per month] [0-1]
# Min-max feature scaling
# minimum value = 0
# maximum value = 1
# when we design model the higher value over powers in the model
df['salary']=(df['salary']-df['salary'].min())/(df['salary'].max()-df['salary'].min())
# (x - min value into that column)/(max value - min value)
# Maximum absolute scaler using scikit-learn
from sklearn.preprocessing import MaxAbsScaler
abs_scaler=MaxAbsScaler()
df[['mba_p']]=abs_scaler.fit_transform(df[['mba_p']])



print('After converting categorical variable to numeric variable: ')
print(df.head().T)

print(df['degree_t'].value_counts())


print(df.head().T)