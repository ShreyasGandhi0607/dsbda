# Assignment number 4:
# Load the Boston Housing dataset
# Display basic information
# Display statistical information
# Display null values
# Fill the null values
# Feature Engineering through correlation matrix
# Build the Linear Regression Model and find its accuracy score
# Remove outliers and again see the accuracy of the model


import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 

df  = pd.read_csv('Datasets/Boston.csv')
print(df.head())


df.drop(columns=['Unnamed: 0'],inplace=True)

# Basic Information
print(df.info())
print("\n")
print(list(df.columns))
print("\n")
print("Data shape : ", df.shape)
print('\nTotal elements in dataset:\n', df.size)
print('\nDatatype of attributes (columns):\n', df.dtypes)
print('\nFirst 5 rows:\n', df.head())
print('\nLast 5 rows:\n',df.tail())
print('\nAny 5 rows:\n',df.sample(5))


# Threre are no null values 

print(df.describe())

def removeoutliers(df,var):
    q1 = df[var].quantile(0.25)
    q3 = df[var].quantile(0.75)
    iqr = q3 - q1
    high , low = q3 + 1.5*iqr , q1 - 1.5 * iqr
    print("Highest allowed in variable:", var, high)
    print("lowest allowed in variable:", var, low)  
    count = df[(df[var] > high)  | (df[var] < low)][var].count()  # corrected the OR operator
    print('Total outliers in:',var,':',count)

    df = df[(df[var] <= high)  & (df[var] >= low)]
    return df 



from sklearn.model_selection import train_test_split

X = df.drop(columns=['medv'])
y = df['medv']

def buildmodel(X,y):
    X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2,random_state=3)

    from sklearn.linear_model import LinearRegression

    model = LinearRegression()

    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)

    from sklearn.metrics import  mean_squared_error
    from sklearn.metrics import r2_score
    mse = mean_squared_error(y_test,y_pred)
    r2  = r2_score(y_test,y_pred)
    print(f"MSE : {mse}")
    print(f"R2 Score :   {r2}")


# Heapmap 
sns.heatmap(df.corr(),annot=True)
plt.show()
    
# we observed that lstat, ptratio and rm have high correlation with cost of flat (medv)
# avoid variables which have more internal correlation
# lstat and rm have high internal correlation
# avoid lstat and rm together
# 1. lstat, ptratio
# 2. rm, ptratio
# 3. lstat, rm, ptratio
# #---------------------------------------------------------------------------------------
    
X = df[['ptratio','lstat']] #input variables
Y = df['medv'] #output variable

buildmodel(X,Y)


# Checking model score after removing outliers
fig, axes = plt.subplots(1,2)
sns.boxplot(data = df, x ='ptratio', ax=axes[0])
sns.boxplot(data = df, x ='lstat', ax=axes[1])
fig.tight_layout()
plt.show()


removeoutliers(df , 'ptratio')
removeoutliers(df , 'lstat')
print("After Removing Outliers")

X = df[['ptratio','lstat']]
Y = df['medv']
buildmodel(X,Y)


# after feature engineering selecting 3 variables
# Choosing input and output variables from correlation matrix
X = df[['rm','lstat', 'ptratio']]
Y = df['medv']

buildmodel(X, Y)

# Drop 'chas' column
df.drop(columns=['chas'], inplace=True)

X = df.drop(columns=['medv'])
y = df['medv']
buildmodel(X,y)