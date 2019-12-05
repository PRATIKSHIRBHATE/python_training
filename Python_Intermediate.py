# Python Training - numpy and pandas for Data Analysis

# I. numpy
# [Numpy](https://www.numpy.org) is the core library for scientific computing in Python. It provides 
# a high-performance multidimensional array object, and tools for working with these arrays.

# Import convention
import numpy as np

# The main object is the **numpy.array**. Note that `numpy.array` is not the same as the Standard
# Python Library class `array.array`, which only handles one-dimensional arrays and offers less 
# functionality. 

# Introduction
# A sample np.array
a = np.arange(15)
print(a)
type(a)

# Important note: numpy arrays are **homogeneous** (they do not allow for elements of 
# different _data type_, or, in numpy lingo, `dtype`. See rich 
# [documentation on data types](https://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html)
np.array([1, 'a']) # S21: strings of byte length 21

# numpy has a wide variety of supported data types, and manipulating data types in large datasets
# can lead to significant improvements in terms of memory use and performance

# Notice the difference with Standard Python when you just invoke the array:
a

# Standard Python
b = [1,2,3]
b

# Correct way to create a numpy array:
c = np.array([1, 2, 3, 4])
c

# A numpy 2D array with random numbers
np.random.rand(3,2)

# Various attributes
a.shape # returns a tuple

# Returns the data type of elements inside the array
a.dtype

np.zeros((2,2))

np.ones((3)) # shape (3,1) implied

# Identity matrix
np.eye(2)

# Indexing
# Like in Standard Python, there are several ways to access specific elements or subsets of a 
# numpy array: Similar to Python lists, numpy arrays can be **sliced**. Since arrays may be 
# multidimensional, you must specify a slice for each dimension of the array
# Slicing
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
a

# Use slicing to pull out the subarray consisting of the first 2 rows
# and columns 1 and 2
b = a[:2, 1:3]
print(b)

# A slice of an array is a view into the same data, so modifying it
# will modify the original array.
print(a[0, 1])   # Prints "2"
b[0, 0] = 77     # b[0, 0] is the same piece of data as a[0, 1]
print(a[0, 1])   # Prints "77"

row_r1 = a[1, :]    # Rank 1 view of the second row of a
row_r2 = a[1:2, :]  # Rank 2 view of the second row of a
print(row_r1, row_r1.shape)  # Prints "[5 6 7 8] (4,)"
print(row_r2, row_r2.shape)  # Prints "[[5 6 7 8]] (1, 4)"

# We can make the same distinction when accessing columns of an array:
col_r1 = a[:, 1]
col_r2 = a[:, 1:2]
print(col_r1, col_r1.shape)  # Prints "[ 2  6 10] (3,)"
print(col_r2, col_r2.shape)  # Prints "[[ 2]
                             #          [ 6]
                             #          [10]] (3, 1)"


 # Integer Array Indexing
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print(a)
print("")
print(a[0]) # first row
print("")
print(np.array([a[0, 1], a[2, 1]])) # elements (0,1) and (2,1) of a

# Boolean Indexing
a = np.arange(12)
print(a > 2)

# The following method of boolean indexing is **very important** for pandas Series/DataFrames
# manipulations, as we will see later: 
# Subset array elements larger than 2
a[a>2]

# (numpy) Array Math
# array of 10,000 random numbers between 0 and 1
a = np.random.rand(10000)
a

# Transpose
a.T

# Dot Product
b = np.random.rand(10000)
a.dot(b)

# Simple operations on numpy arrays are element-wise:
print(a + b)  # Same as np.add(a,b)
print("")
print(a - b)  # Same as np.subtract(a,b)
print("")
print(a * b)  # Same as np.multiply(a,b)
print("")
print(a / b)  # Same as np.divide(a,b)
print("")
print(a ** b) # Same as np.power(a,b)
print("")
print(np.sqrt(a))

# Other operations: `np.min(), np.max(), np.sum()`, etc.
# Statistics
print('Mean: ', np.mean(a))              # Also: np.nanmean() ignores np.nan values within the array
print('Median: ', np.median(a))          # np.nanmedian()
print('Standard Deviation: ', np.std(a)) # np.nanstd(a)
print('90th percentile: ', np.percentile(a, 90))

# Note: `np.nan` is _like_ `None` in Standard Python, but it is of type `float`!
b = np.array([np.nan, 1])
b.dtype

# There is an abundance of tutorials covering `numpy` in great detail. I will post some links on
# the Confluence page.

# II. pandas
# `pandas` is a Python library written for data manipulation and analysis. In particular, 
# it offers data structures and operations for manipulating numerical tables and time series. 
# Its name comes from the econometric term _panel data_
# Pandas is capable of many tasks including:
# - Reading/writing many different data formats
# - Selecting subsets of data
# - Calculating across rows and down columns
# - Finding and filling missing data
# - Applying operations to independent groups within the data
# - Reshaping data into different forms
# - Combing multiple datasets together
# - Advanced time-series functionality
# - Visualization through matplotlib and seaborn

# Its documentation is extremely thorough (2,500+ pages long!) but, in my opinion, it is not very
# helpful for an audience interested in "real" data analysis: all the data is contrived or randomly
# generated, and the real power of pandas, multiple operations performed in sequence, is not really 
# Voutlined there.

# Import convention
import pandas as pd

# The main data structures in Pandas are the `Series` and the `DataFrame`
# Load a numpy array into a pd.Series
a = np.random.rand(50)
b = pd.Series(a)
b.head() # Returns the first 5 rows, unless specified otherwise

# Load a Python dictionary into a pd.DataFrame (common variable name: `df`)
data = {
    'number': [1, 10, 100, 1000, 500],
    'color': ['Red', 'Green', 'Blue', 'Yellow', 'Black'],
    'price': [2.0, 15.50, 3.42, 64.50, np.nan]
}

df = pd.DataFrame(data)
df

# A `pandas DataFrame` comprises three elements:
# - Index
# - Columns
# - Values
print(df.index) # can be changed with set_index()
print(df.columns) # dtype 'object' is a general string-like data type
type(df.values) # df.values in a numpy array!

# Indexing a pandas DataFrame is an immensely rich topic. Let us just mention that we can manually
# manipulate the `index` (primary key) of the DataFrame.
df.set_index('color')

# Note: `color` can no longer be accessed as a dataframe column
df['color']

# Important note: such operations return **copies** of the original DataFrame, without changing the 
# original variable. In order to change that, you either have to reassign to a new variable, or use
# the `inplace` flag.
df

df.set_index('color', inplace=True)
df

# Working with missing data
# The last row of our dataframe has a `np.nan` (missing) value for `price`. Depending on the 
# specifics of our problem, we might either want to `drop` this row, or impute (`fill`) some value 
# for the missing data:
# Pandas axes: 0 = rows, 1 = columns
df2 = df.dropna(axis=0, how='any') # "Drop all rows where any column has a value of np.nan"
df2

# Let us impute the mean of the remaining values
# Fill all empty "cells" of the DataFrame with the mean value of the "price" column
df3 = df.fillna(df['price'].mean()) 
df3

# Pandas Series have numpy-like math operations (`mean(), median(), sum()`, etc.) - Important 
# distinction: Pandas automatically ignores `np.nan` values during these calculations.
# Get some fast counts of NaN rows for `price` column
print(df['price'].isnull().sum())
print(df['price'].notnull().sum())

# As you can see, a great strength of Pandas is its support for practically limitless
# ** chain operations **. Use carefully - and wisely: it might be easier to perform many operations 
# on one go, but it is harder to debug (if a chain operation fails, you need to figure out which 
# step failed)
# Import a sample dataset from a csv file
df = pd.read_csv('/Users/nikolaos/Desktop/sample_data.csv') # Mike, change filepath
df

# df.describe() automatically subsets the numerical columns of the dataframe
df.describe()

# Chained indexing
# Example 1
# Subset the dataframe, and then return one column of the subset
df[['Names', 'state', 'color']]['state']
df.set_index('Names', inplace=True)

# Example 2
# Location indexing: .loc is primarily label based
df.loc[['Niko', 'Aaron']][['height', 'color']]

# Example 3
# Location indexing: .iloc is primarily integer position based
df.iloc[2:5].iloc[:, -3:] # Returns rows 2,3,4, and then the last three columns of this subset

# Example 4
# Boolean indexing
df[df['state'] == 'TX'][['color', 'score']]

# All these examples are correct pandas code, but it is not _idiomatic_ Let us rewrite them in 
# idiomatic pandas code

# Example 1
df['state'] # No need to subset a dataframe when the desired end result is a single column

# Example 2
df.loc[['Niko', 'Aaron'], ['height', 'color']]

# Example 3
df.iloc[2:5, -3:]

# Example 4
df.loc[df['state'] == 'TX', ['color', 'score']]

# Value sorting
df.sort_values(by='score', ascending=False) # Remember: this does not alter original df

# pandas aggregations: the split-apply-combine framework
# One of the greatest strengths of pandas is the blazing fast speed it can perform aggregations on 
# large DataFrames. `groupby` comes straight from the SQL world. Such aggregations on pandas are 
# usually understood in three distinct steps:
# - The **split** step involves breaking up and grouping a DataFrame depending on the value of the 
# specified key.
# - The **apply** step involves computing some function, usually an aggregate, transformation, or 
# filtering, within the individual groups.
# - The **combine** step merges the results of these operations into an output.
df.groupby('state')['score'].mean()

# Another way to perform the operation above:
df.groupby('state')['score'].agg('mean')

# Careful: curly brackets needed when you want multiple columns as output
df.groupby('state')[['score', 'age']].agg({'mean', 'median'})

# You can write short python functions to use as arguments in `apply` or `agg` for customized results
def add_10(s):
    return s + 10

new_df = df.groupby('state')['score'].apply(add_10).apply(pd.Series)
# the last `apply` makes new_df look "nicer". like a pd.DataFrame

new_df.columns = ['inflated_score']
new_df

# Another example of groupby
# reset_index brings back default indexing
df.groupby('state')['score'].sum().rename('Total Score').reset_index() 

# There are three different `apply` instances in the split-apply-combine process:
# - aggregate, if we want to get a single value for each group
# - filter, if we want to get a subset of the input rows
# - transform, if we want to get a new value for each input row

# The `groupby` examples we have seen so far were instances of the `aggregate` method:

# Aggregate
df.groupby('state')['score'].agg('mean')
# same as df.groupby('state')['score'].mean()
# same as df.groupby('state')['score'].aggregate('mean')

# Filter with lambda function (short function defined on the spot instead of `def...`)
df.groupby('state').filter(lambda x : x['score'].max() > 4)

# Transform - adds new column with same value for each group
df['scaled_score'] = df.groupby('state')['score'].transform(lambda x : x/x.mean())
df

# Pandas is a very powerful library for data manipulation and analysis in Python. Because the operations under the hood are **vectorized**, there is almost **NEVER** a need to iterate over a DataFrame using a `for` loop.

# Understanding `pandas` and the underlying structures of `numpy` is a very important "prerequisite" before moving on to `scikit-learn` for predictive analytics. 

# III. scikit-learn
# Scikit-learn is a python module for machine learning and is built on top of the numpy, scipy, and matplotlib modules. It comes loaded with lots of features and components. Below are a few of them to help you understand:
# - Supervised learning algorithms
# - Unsupervised learning algorithms
# - Cross-validation
# - Feature extraction
# - Dimensionality reduction
# - Preprocessing
# - Model selection and evaluation

# It's very well documented online and the official documentation can be found here: http://scikit-learn.org/stable/documentation.html

# Load the data
# import pandas
import pandas as pd

# import the titanic file 
filepath = "P:\\Python Training\\titanic.csv"
titanic = pd.read_csv(filepath)
titanic.head(10)

# Preprocessing - http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing
# Accoiunt for missing data
# view the dataset to see if there are any missing variables
titanic[titanic['age'].isnull()].shape

# find exactly where the null values are
titanic.isnull().sum()

# replace null values in the age column
age_mean = titanic['age'].mean()  # get the mean age used to fill in missing age values
titanic['age'].fillna(age_mean, axis=0, inplace=True)

# drop the remaining missing variables
titanic.dropna(axis=0, inplace=True)

# Replace the index with passenger ID
# set the index to the pid variable
titanic.set_index('pid', drop=True, inplace=True)
titanic.head()

# Encode categorical variables
# creating dummy vaiables for categorical variables
titanic_new = pd.get_dummies(titanic)
titanic_new.head(10)

# Split the explanatory and repsonse variables
# create an array of the column names
titanic_names = titanic_new.columns.values
titanic_names

# store the resonse and explanatory variables
y = titanic_new['survived']
x = titanic_new[titanic_names[1:]]

x.head()

y.head()

# Creating the training and testing dataset
# Cross Validation
# Learning the parameters of a prediction function and testing it on the same data is a methodological mistake: a model that would just repeat the labels of the samples that it has just seen would have a perfect score but would fail to predict anything useful on yet-unseen data. This situation is called overfitting. To avoid it, it is common practice when performing a (supervised) machine learning experiment to hold out part of the available data as a test set x_test, y_test

# split the dataset into training and testing using train test split
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape )

# Logistic Regression

# Logistic regression, despite its name, is a linear model for classification rather than regression. Logistic regression is also known in the literature as logit regression, maximum-entropy classification (MaxEnt) or the log-linear classifier. In this model, the probabilities describing the possible outcomes of a single trial are modeled using a logistic function.

# import linear model
from sklearn import linear_model

# create logistic regression object
log = linear_model.LogisticRegression()

# train the model using the trainging datasets
model = log.fit(x_train, y_train)

# make the predicitions using the testing set
predictions = log.predict(x_test)
print(predictions)

# get the accuracy of the model
model.score(x_test, y_test)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, predictions)
confusion_matrix

# create the classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))

# K-fold cross validation
# Is this model any good? It helps to verify the effectiveness fo the alogrithm using KFold. This will split our data into k buckets, then run the alogirthm using a different bucket as the test set for each iteration.

from sklearn.cross_validation import cross_val_score, cross_val_predict

# perform 10-fold cross validation
scores = cross_val_score(model, x, y, cv = 10)
print(scores)

# get the mean accuracy of all of the models
scores.mean()