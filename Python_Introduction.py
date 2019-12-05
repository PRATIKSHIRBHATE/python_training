# Introduction to Python
# Evolent Health Data Science & Development
# By: Michael Voss

# The introduction to python lesson is aimed to give the user an overview of python and provide 
# them with the skills to start working in python.

#######################################################
# 1. Data Types and Variable Assignment
#######################################################
#######################################################
# 1.1 Numbers
#######################################################
    
# Number data types store numeric values. Number objects are created when you assign a value to 
# them. In python, you assign a variable a value using the = operator. Integers can be whole numbers 
# that can be positive, negative or 0
25 - 10

# Floating-Point numbers or a float is a real number that can contain a fractional point 
# (think decimal point).
9.0 + 116.42

# The Boolean data type can be one of two values, either TRUE or FALSE
500 > 100

# Number data types can also be assigned to variables. In Python, you do not have to declare
# variables before you assign them.
# Assign var1 the value of 1
var1 = 1
print(var1)

# Assign the result to a variable
var2 = 9.0 + 116.42
print(var2)

# Delete a variable
del var1

#######################################################
# 1.2 Strings
#######################################################
# Stings are amongst the most popular data types in Python. 
# They can be created by simply enclosing characters in single('') or double("") quotes. 
# They can contain letters, numbers, and symbols.

# Example of a single and double quote string
'This is a string in single quotes'

"This is a string in double quotes"

# Like numbers, strings can also be assigned to variables
ml = 'machine learning'

# You can use the print function to display the string
print(ml)

# There are many built-in methods which allow you to manipulate strings.
# The upper() method will capitalize your string
ml.upper()

# The lower() method will make your string all lowercase
ml.lower()

# Creating empty strings
empty = ""

# The len() method will return the length of your string
len(ml)

# The type() method will return the type of your data
type(ml)

# To see what methods Python provides for a datatype, use the dir and help commands: 
dir(ml)

# You can use help() on any built in method
help(ml.find)

# Accessing Individual Characters in a String
ml.find('m')

#######################################################
# 2. Built-in Structures
#######################################################
#######################################################
#2.1 Lists
#######################################################
# Lists are a data structure you can use to store a collection of different information as a 
# sequence under a single variable name. The items are separated by a comma, and enclosed inside a 
# square bracket [ ].

# Let's define a list
variable = [item_1, item_2, item_3, ..., itemN]

# A list can also be empty
empty_list = []

# Example list
fruits = ['apple', 'orange', 'pear', 'banana']

# Access items in the list by index
fruits[0]

# Python also allows negative-indexing from the back of the list. For instance, fruits[-1] will 
# access the last element 'banana': 
fruits[-1]

# A list does not have to be a fixed length either. You can add items to the end of a list any time
# you like.

# The pop() method will remove the last item in the list
fruits.pop()

# View the new list
fruits

# The append() method will add in a new item at the end of the list
fruits.append('grapefruit')

# View the new list
fruits

# Search for an item in a list using index
fruits.index("orange")

# Insert a new item into the list by an index
fruits.insert(1, "peach")

# View the updated list
fruits

# We can use the + operator to do list concatenation: 
other_fruits = ['kiwi', 'strawberry']
fruits + other_fruits

# We can also index multiple adjacent elements using the slice operator. 
# For instance fruits[1:3] which returns a list containing the elements at position 1 and 2. 
# In general fruits[start:stop] will get the elements in start, start+1, ..., stop-1. 
# We can also do fruits[start:] which returns all elements starting from the start index. 
# Also fruits[:end] will return all elements before the element at position end: 

# Get the first and second items in the fruits list
fruits[0:2]

# Get the first, second, and third items in the fruits list
fruits[:3]

# Get the third and fourth items in the fruits list
fruits[2:]

# Similar to strings, use the len() method to get the length of a list
len(fruits)

# The items stored in lists can be any Python data type. For instance we can have lists of lists: 
# Example of a list of lists
list_of_lists = [['a', 'b', 'c'],[1, 2, 3],['one', 'two', 'three']] 

# Index the list of lists
list_of_lists[1][2] 

# The reverse() method will reverse the items in the list
list_of_lists.reverse()

# View the reversed list
list_of_lists

# A list index behaves like any other variable name. It can be used to access as well as assign 
# values.

# Change "apple" to "grape"
fruits[0] = "grape"

# View the new list
fruits

#######################################################
#2.2 Tuples
#######################################################
# A data structure similar to the list is the tuple, which is like a list except that it is 
# immutable once it is created (i.e., you cannot change its content once created). 
# Note that tuples are surrounded with parentheses while lists have square brackets. 

# Tuples are created similar to lists, instead they use the () to store values
pair = (3, 5)

# Similar to lists, you can index a tuple
pair[0]

# However, since they are immutable, you cannot assign new items to them
pair[1] = 6

#######################################################
# 2.3 Sets
#######################################################
# A set is another data structure that serves as an unordered list with no duplicate items. 
# Below, we show how to create a set, add things to the set, test if an item is in the set, and 
# perform common set operations (difference, intersection, union): 

# Create a list of shapes
shapes = ['circle', 'square', 'triangle', 'circle']

# Create a set of the shapes list
set_of_shapes = set(shapes)

# View the set
set_of_shapes

# Add items to a set
set_of_shapes.add("rhombus")

# View new set
set_of_shapes

# Remove items from a set
set_of_shapes.remove("circle")

# View new set
set_of_shapes

# Test if a value is in the set
'square' in set_of_shapes

# Create two new sets
A = {1, 2, 3, 4, 5}
B = {4, 5, 6, 7, 8}

# Use the union operator to get a combined set of all elements
A.union(B)

# Use the intersection operator to get a set that has common elements in both sets
A.intersection(B)

# Use the difference operator to get the elements in A that are not in B
A.difference(B)

#######################################################
# 2.4 Dictionaries (Dicts)
#######################################################
# The last built-in data structure is the dictionary which stores a map from one type of object
# (the key) to another (the value). The key must be an immutable type (string, number, or tuple). 
# The value can be any Python data type. 

# Example of the syntax of a dictionary
d = {'key1': value1, 'key2': value2, "key3": value3}

# Let's make a dictionary
student_grades = {'Mike': 82.3, 'Dan': 70.2, 'Sam': 95.4, 'Amber': 90.5}

# Getting the grade for a particular student
student_grades['Mike']

# Changing the value of an existing key
student_grades['Mike'] = 75.6

# View the new dictionary
student_grades

# Deleting entries from a dictionary using the pop() module
student_grades.pop('Dan')

# View the new dictionary
student_grades

# Adding new entires to a dictionary
student_grades['Ashley'] = 88.7

# View the new dictionary
student_grades

# View the keys in the dictionary
student_grades.keys()

# View the values in the dictionary
student_grades.values()

# View both the keys and values in the dictionary
student_grades.items()

# You can also get the length of a dictionary
len(student_grades)

# Check if a key is in a dictionary
'Sam' in student_grades

# Iterate through the keys in a dictionary using a for loop
for key in student_grades:
    print(key)

# Can get the same output by using the keys() module
for key in student_grades.keys():
    print(key)

# Iterate through the values in a dictionary using the values() module
for value in student_grades.values():
    print(value)

# Iterate through the key-value pairs in a dictionary
for key in student_grades:
    print(key, ":", student_grades[key])

# Iterate through the key-value pairs in a dictionary using the items() module
for key,value in student_grades.items():
    print(key, ":", value)

# Reference vs. Shallow Copy vs. Deep Copy
# The difference between shallow and deep copying is only relevant for compound objects 
# (objects that contain other objects, like lists or class instances):
# A shallow copy constructs a new compound object and then (to the extent possible) inserts 
# references into it to the objects found in the original.
# A deep copy constructs a new compound object and then, recursively, inserts copies into it of 
# the objects found in the original.
# Documentation:
# https://docs.python.org/3.6/library/copy.html

# Reference assignment makes d1 and d2 point to the same object
d1 = {'a': 1, 'b': 2, 'c': [3, 4, 5]}
d2 = d1
print(d1, d2)

# Changes in d1 will also change d2
d2['d'] = 5
print(d1,d2)

# Shallow coping makes d1 and d2 two isolated objects,
d1 = {'a': 1, 'b': 2, 'c': [3, 4, 5]}
d2 = d1.copy()
print(d1, d2)

# Changes in d1 will not change d2
d1['d'] = 5
print(d1,d2)

# However changes to a list nested in d1 will change d2
d1['c'].append(6)
print(d1,d2)

# Deep copy will make not only d2 a seperate object, but all the key, values inside the dict as 
# well.
d1 = {'a': 1, 'b': 2, 'c': [3, 4, 5]}
d2 = copy.deepcopy(d1)
print(d1, d2)

# Changes in d1 will not change d2
d1['d'] = 5
print(d1,d2)

# Changes to a list nested in d1 will not change in d2
d1['c'].append(6)
print(d1,d2)

#######################################################
# 3. Creating Functions
#######################################################
# What are functions? Functions are a set of actions that we group together and give a name to. 
# We can define our own functions, which allows the user to "teach" python a new behavior.

# Let's define a function
def function_name(arg1, arg2, arg3, .... argN):
    # Do whatever we want this function to do using argument_1 and argument_2
    return

# Use function_name to call the function
function_name(value1, value2, value3, .... valueN)

# Example: Here is how we will make a simple function to create weight from pounds to kilograms.

def pounds_to_kilograms(pounds): 
    kilograms = pounds/2.20462 # Conversion of pounds to kilograms
    return(kilograms)
    
# set pounds equal to 100
pounds = 100

# Call the function
pounds_to_kilograms(pounds)

#######################################################
# 3.1 If/Else Statements in Function
#######################################################
# function to determine if something is over 100Kg
def is_it_over_100_kg(pounds):
    kilograms = pounds_to_kilograms(pounds)
    if kilograms > 100:
        return("Over 100 Kilograms")
    else:
        return("Under 100 Kilograms")
    
is_it_over_100_kg(200)

#######################################################
# 3.2 For Loop in Function
#######################################################
def P2Kmultiple(pounds):
    output = []
    for weight in pounds:
        kilograms = weight/2.20462
        output.append(kilograms)
    return(output)

input = [1, 100, 700, 20]

P2Kmultiple(input)


#######################################################
# 4. Importing Packages and Modules
#######################################################
# Import the os module that lets you use operating system commands
import os

# Import both the pandas and numpy modules 
import pandas as pd
import numpy as np

#######################################################
# 5. Data Import and Export
#######################################################
# To begin working with data in python, we must first load it into the python environment. 
# Python allows you to import both data stored on your local machine or connect directly to SQL 
# server and other databases.

#######################################################
# 5.1 Establishing a Working Directory
#######################################################
# Before we load any data into python, we must first know what our working directory is:
# print the current working directory
os.getcwd()

# If we wanted to change our working directory, we would use the following command:
# change the current working directory to python training folder
os.chdir("P:\\Python Training\\")

# print new current working directory
os.getcwd()

# For this exercise, we will work with a dataset called "Auto" which includes information about 
# various types of cars. Store the location of the auto text file in a variable
filepath = "P:\\Python Training\\auto.txt"

#######################################################
# 5.2 Loading Data Into Python
#######################################################
# The most common way to read files into python is by using the pandas module (Python Panels Data 
# Package).Pandas is included in Anaconda and will not need to be installed separately.
import pandas as pd
import numpy as py

# Importing text files using pandas
auto = pd.read_table(filepath, sep="\t")

# importing csv files using pandas
mydata = pd.read_csv(filepath)

# importing excel files using pandas
mydata = pd.excel(filepath, sheetname="Data 1")

# importing SAS files using pandas (reads in .sastbdat files)
mydata = pd.read_sas(filepath)

#######################################################
# 5.3 Exporting and Writing to a File
#######################################################
# export to a text file
auto.to_csv(filepath, sep='\t')

# export to a csv file
auto.to_csv(filepath, sep=',')

# Exporting datasets to an excel file
# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('pandas_simple.xlsx', engine='xlsxwriter')

# Convert the dataframe to an XlsxWriter Excel object.
auto.to_excel(writer, sheet_name='Sheet1')

writer.save()

#######################################################
# 6. Exploring Data and Data Frames
#######################################################

# To get information about the data frame, use the function info()
auto.info()

# To view a small sample of a series or data frame, use the head() and tail() functions
auto.head()

auto.tail()

# You can also view the data frame by simply typing it's name
auto

# To get basic summary statistics about the data we can use the describe() function
auto.describe()

auto.describe(include = 'all')

auto['mpg'].describe()

# If we just want a specific summary statistic we can use simple commands
auto.mean()

# Get the mean for the mpg column
auto['mpg'].mean()

# Get the names of all the columns
auto.columns

# Get a condensed version of the values in the dataframe
auto.values

# Transpose the data
auto.T

# You can get the correlation of the data frame by using the corr() function
auto.corr()

# You can get the covariance matrix of the data frame by using the cov() function
auto.cov()

#######################################################
# 6.1 Simple Plotting
#######################################################
import matplotlib.pyplot as plt

# Create a Scatter Plot
auto.plot.scatter('weight','mpg')

# Create another scatter plot
auto.plot.scatter('horsepower','mpg')

# Create a histogram
plt.hist(auto["mpg"], bins=8, alpha=0.5)
plt.xlabel('MPG')
plt.ylabel('Count')
plt.title('Histogram of MPG')
plt.axis([0, 50, 0, 100])
plt.grid(True)

# Create a boxplot
auto.boxplot(column=["mpg","acceleration"], return_type='axes')






























