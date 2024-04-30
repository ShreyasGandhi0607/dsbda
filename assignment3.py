# Assignment number 3:
# Load the dataset
# Display basic information
# Display null values
# Fill the null values
# Display overall statistical information
# Display groupwise statistical information

import numpy as np 
import pandas as pd

df = pd.read_csv('Datasets/Employee_Salary.csv')


# Display basic informatin 

print("The dataset information is like : \n", df.info())
print("\n First 5 elements in the dataset : \n", df.head())
print("\n Last 5 elements in the dataset : \n", df.tail())
print("\n Shape of dataset : \n", df.shape)
print("\n Column names : \n",  list(df.columns))


# find missing values 

print(df.isna().sum())


# fill missing values 

df['Gender']=df['Gender'].fillna(df['Gender'].mode()[0])
df['Experience']=df['Experience'].fillna(df['Experience'].mode()[0])


print("After removing missing values : \n")
print(df.isna().sum())





# Convert categorical to numerical 

df['Gender'] = df['Gender'].replace(['Male','Female'],[1,0])


print(df.head())


print("Show overall statistical information : \n", df.describe())



#---------------------------------------------------------------------------------------
# groupwise statistical information
print('Groupwise Statistical Summary....')
print('\n-------------------------- Experience -----------------------\n')
print(df['Experience'].groupby(df['Gender']).describe())
print('\n-------------------------- Age -----------------------\n')
print(df['Age'].groupby(df['Gender']).describe())
print('\n-------------------------- Salary -----------------------\n')
print(df['Salary'].groupby(df['Gender']).describe())