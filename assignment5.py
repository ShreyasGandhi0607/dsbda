# Assignment number 5:
# Load the Social network ads dataset
# Display basic information
# Display statistical information
# Display null values
# Fill the null values
# Feature Engineering through correlation matrix
# Build the Logistic Regression Model and find its classification score
# Remove outliers and again see the accuracy of the model


import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 

df = pd.read_csv('Datasets/purchase.csv')


# Basic Information 


print('\nInformation of Dataset:\n', df.info())
print('\nShape of Dataset (row x column): ', df.shape)
print('\nColumns Name: ', df.columns)
print('\nTotal elements in dataset:', df.size)
print('\nDatatype of attributes (columns):', df.dtypes)
print('\nFirst 5 rows:\n', df.head().T)
print('\nLast 5 rows:\n',df.tail().T)
print('\nAny 5 rows:\n',df.sample(5).T)

# There are no misssing values 


# Display statistical information 

df = df.drop('User ID', axis=1)
df.columns = ['Gender', 'Age', 'Salary', 'Purchased']

print(df.describe())


# Label encoding method
df['Gender']=df['Gender'].astype('category')
df['Gender']=df['Gender'].cat.codes

print(df.describe())


sns.heatmap(df.corr(), annot=True)
plt.show()


def removeoutliers(df,var):
    q1 = df[var].quantile(0.25)
    q3 = df[var].quantile(0.75)
    iqr = q3 - q1
    high , low = q3 + 1.5  * iqr, q1 - 1.5 * iqr
    print("Highest allowed in variable:", var, high)
    print("lowest allowed in variable:", var, low) 
    count = df[(df[var]  > high) | (df[var] < low)][var].count()
    print("The outlier in dataframe are " , count)

    df = df[(df[var] <=high)&(df[var] >=low)]
    
    return df


def buildmodel(X,y):
    from sklearn.model_selection import train_test_split

    X_train , X_test , y_train ,y_test = train_test_split(X , y , test_size= 0.2 , random_state=3)
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X_train , y_train)
    y_pred = model.predict(X_test)

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True)
    plt.show()
    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred))




# Choosing input and output variables from correlation matrix
X = df[['Age','Salary']]
Y = df['Purchased']

buildmodel(X, Y)




# Checking model score after removing outliers
fig, axes = plt.subplots(1,2)
sns.boxplot(data = df, x ='Age', ax=axes[0])
sns.boxplot(data = df, x ='Salary', ax=axes[1])
fig.tight_layout()
plt.show()
df = removeoutliers(df, 'Age')
df = removeoutliers(df, 'Salary')
# You can use normalization method to improve the score
# salary -> high range
# age -> low range
# Normalization will smoothe both range salary and age
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

df[['Age','Salary']] = scaler.fit_transform(df[['Age','Salary']])



X = df[['Age','Salary']]
Y = df['Purchased']

buildmodel(X, Y)

