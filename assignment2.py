# Assignment number 2:
# Load the dataset
# Display basic information
# Display statistical information
# Display null values
# Fill the null values
# Quantization (Encoding): Convert categorical to numerical variable
# Handle outliers
# Handle skewed data

#---------------------------------------------------------------------------------------
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

df = pd.read_csv('Datasets/academic.csv')

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


# Display statistical information 

print("Statistical information : \n",df.describe())


# Check for missing or NA values

print(df.isna().sum())

# Fill missing values

df['gender']=df['gender'].fillna(df['gender'].mode()[0])   # fill with most frequent value
df['raisedhands']=df['raisedhands'].fillna(df['raisedhands'].mean())

print("After removing the missing values ",df.isna().sum())


# Converting categorical to numeric using Find and replace method
df['Relation']=df['Relation'].astype('category')
df['Relation']=df['Relation'].cat.codes


def detectOutliers(df,var):
    # IQR is a measure of outlier detection that considers the range between the first quartile (25 percent) and third quartile (7
    Q1 = df[var].quantile(0.25)
    Q3 = df[var].quantile(0.75)
    IQR = Q3 - Q1
    high , low = Q3 + 1.5 * IQR , Q1 - 1.5 * IQR


    print("Highest allowed  Value is : ",high )
    print("Lowest allowed Value is : ",low)
    count = df[(df[var] > high) | (df[var] < low)][var].count()
    print("Outliers in var" , count)
    df1 = df[(df[var] < low) | (df[var] > high)]
    print("Outliers : " , len(df1))
    print(df1.T)
    df = df[((df[var] >= low) & (df[var] <= high))] #now filter out data which is not outlier
    return df



# Outliers can be visualized using boxplot
# using seaborn library we can plot the boxplot
# fig, axes = plt.subplots(2,2)
# fig.suptitle('Before removing Outliers')
# sns.boxplot(data = df, x ='raisedhands', ax=axes[0,0])
# sns.boxplot(data = df, x ='VisITedResources', ax=axes[0,1])
# sns.boxplot(data = df, x ='AnnouncementsView', ax=axes[1,0])
# sns.boxplot(data = df, x ='Discussion', ax=axes[1,1])
# plt.show()


#Display and remove outliers
df = detectOutliers(df, 'raisedhands')
# fig, axes = plt.subplots(2,2)
# fig.suptitle('After removing Outliers')
# sns.boxplot(data = df, x ='raisedhands', ax=axes[0,0])
# sns.boxplot(data = df, x ='VisITedResources', ax=axes[0,1])
# sns.boxplot(data = df, x ='AnnouncementsView', ax=axes[1,0])
# sns.boxplot(data = df, x ='Discussion', ax=axes[1,1])
# plt.show()
#---------------------------------------------------------------------------------------

#Display and remove outliers
df = detectOutliers(df, 'raisedhands')

print('---------------- Data Skew Values before Yeo Johnson Transformation ----------------------')
# Calculate skewness before transformation
print('raisedhands: ', df['raisedhands'].skew())
print('VisITedResources: ', df['VisITedResources'].skew())
print('AnnouncementsView: ', df['AnnouncementsView'].skew())
print('Discussion: ', df['Discussion'].skew())

# Plot histograms before transformation
fig, axes = plt.subplots(2, 2)
fig.suptitle('Handling Data Skewness')

sns.histplot(ax=axes[0, 0], data=df['AnnouncementsView'], kde=True)
sns.histplot(ax=axes[0, 1], data=df['Discussion'], kde=True)

# Applying Yeo-Johnson Transformation
from sklearn.preprocessing import PowerTransformer
yeojohnTr = PowerTransformer(standardize=True)
df['AnnouncementsView'] = yeojohnTr.fit_transform(df['AnnouncementsView'].values.reshape(-1, 1))
df['Discussion'] = yeojohnTr.fit_transform(df['Discussion'].values.reshape(-1, 1))

print('---------------- Data Skew Values after Yeo Johnson Transformation ----------------------')
# Calculate skewness after transformation
print('AnnouncementsView: ', df['AnnouncementsView'].skew())
print('Discussion: ', df['Discussion'].skew())

# Plot histograms after transformation
sns.histplot(ax=axes[1, 0], data=df['AnnouncementsView'], kde=True)
sns.histplot(ax=axes[1, 1], data=df['Discussion'], kde=True)

plt.show()
