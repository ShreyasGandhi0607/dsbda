# Assignment number 9:

# Load the Titanic dataset
# Display basic information
# Display statistical information
# Display null values
# Fill the null values
# Display and iterpret boxplot of one variable, two variables and three variables


import pandas as pd 
import numpy as np 
import  matplotlib.pyplot as plt 
import seaborn as sns 

df  = pd.read_csv('Datasets/titanic.csv')

print(df.head())
# Display basic information
print('\nInformation of Dataset:\n', df.info())
print('\nShape of Dataset (row x column): ', df.shape)
print('\nColumns Name: ', df.columns)
print('\nTotal elements in dataset:', df.size)
print('\nDatatype of attributes (columns):', df.dtypes)
print('\nFirst 5 rows:\n', df.head())
print('\nLast 5 rows:\n',df.tail())
print('\nAny 5 rows:\n',df.sample(5))

#  Display statistical information 

print("Statistical information : \n",df.describe())

# check missing values 

print('\n\nChecking for Missing Values:\n\n', df.isnull().sum())



# filling missing values 

df.drop(columns=['Cabin'],inplace=True)

df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])


print('\n\nAfter Handling Null Values Check:\n\n', df.isnull().sum())

#One variable
fig, axes = plt.subplots(1,2)
sns.boxplot(data = df, y ='Age', ax=axes[0])
sns.boxplot(data = df, y ='Fare', ax=axes[1])
plt.show()


# Two variables
fig, axes = plt.subplots(1,3, sharey=True)
sns.boxplot(data = df, x='Sex', y ='Age', hue = 'Sex', ax=axes[0])
sns.boxplot(data = df, x='Pclass', y ='Age', hue = 'Pclass', ax=axes[1])
sns.boxplot(data = df, x='Survived', y ='Age', hue = 'Survived', ax=axes[2])
plt.show()

# Two variables
fig, axes = plt.subplots(1,3, sharey=True)

sns.boxplot(data = df, x='Sex', y ='Fare', hue = 'Sex', ax=axes[0], log_scale = True)
sns.boxplot(data = df, x='Pclass', y ='Fare', hue = 'Pclass', ax=axes[1], log_scale = True)
sns.boxplot(data = df, x='Survived', y ='Fare', hue = 'Survived', ax=axes[2], log_scale = True)
plt.show()

#three variables
fig, axes = plt.subplots(1,2, sharey=True)
sns.boxplot(data = df, x='Sex', y ='Age', hue = 'Survived', ax=axes[0])
sns.boxplot(data = df, x='Pclass', y ='Age', hue = 'Survived', ax=axes[1])
plt.show()

fig, axes = plt.subplots(1,2, sharey=True)
sns.boxplot(data = df, x='Sex', y ='Fare', hue = 'Survived', ax=axes[0], log_scale = True)
sns.boxplot(data = df, x='Pclass', y ='Fare', hue = 'Survived', ax=axes[1], log_scale = True)
plt.show()