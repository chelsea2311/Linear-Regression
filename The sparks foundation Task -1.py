#!/usr/bin/env python
# coding: utf-8

# # NAME - CHELSEA MAHENDRA RODRIGUES
# 
# Email id : chelsearod2311@gmail.com
# 
# Linkedin Profile:  https://www.linkedin.com/in/chelsea-rodrigues-9a401a214
# 
# Github Profile: https://github.com/chelsea2311
# 
# Task 1: Prediction using Supervised ML GRIP @The sparks foundation
# 
# Used Linear Regression Model
# 
# There are two features given in dataset. Using Hours feature we have to predict the scores of the student.
# 
# Linear Regression with Python

# Import Libraries

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Data

# In[3]:


student_data = pd.read_csv('http://bit.ly/w-data')


# Check out the data

# In[4]:


student_data.head()


# Describing the data

# In[5]:


student_data.describe()


# Dimension of the DataFrame

# In[6]:


student_data.shape


# Exploratory Data Analysis(EDA)

# In[7]:


student_data.plot(x = 'Hours' , y = 'Scores',style = '*')
plt.title("Hours Studied vs Percentage Score")
plt.xlabel("Hours student studied")
plt.ylabel("% of student's score")
plt.show()


# Splitting the data into input and output

# In[8]:


x = student_data.iloc[:,:-1].values
y = student_data.iloc[:,1].values


# Training a Linear Regression Model

# Let's now begin to train out regression model! We will first need to split up our data into an X array that contains the features to train on, and y array with the target variable, in this case the scores column.

# Train Test Split

# In[9]:


from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)


# Linear Regression 

# In[10]:


from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression()
linear_regression.fit(x_train, y_train)


# Visualizing the training data

# In[11]:


plt.scatter(x_train, y_train, color='purple')
plt.plot(x_train, linear_regression.predict(x_train), color='green')
plt.title("Hours Studied vs Percentage Score")
plt.xlabel("Hours student studied")
plt.ylabel("% of student's score")
plt.show()           


# Prediction from our Model

# In[12]:


y_pred = linear_regression.predict(x_test)
y_pred


# Model Evaluation

# Comparing the Actual and Predicted Value(in tabular form)

# In[13]:


df = pd.DataFrame({'Actual value': y_test, 'Ptredicted value': y_pred})
df


# Visualizing the difference between Actual and Predicted value

# In[14]:


df.plot(kind = "bar" , figsize = (6,6))
plt.title("Hours Studied vs Percentage Score")
plt.xlabel("Hours student studied")
plt.ylabel("% of student's score")
plt.show()


# Estimating training and test score

# In[15]:


print("Training Score: ",linear_regression.score(x_train,y_train))
print("Test Score: ",  linear_regression.score(x_test,y_test))


# Prediction score if a student studies for 9.25 hrs/day

# In[16]:


hours = [9.25]
ans = linear_regression.predict([hours])
print("Hours student study = {}".format(hours))
print("Predicted score of student = {}".format(ans[0]))


# Regression Evaluation Metrics

# Finding the residuals :It is very important to calculate the performance of the model

# In[18]:


from sklearn import metrics
print("Mean Absolute Error => ", metrics.mean_absolute_error(y_test,y_pred))
print("Mean Square Error => ", metrics.mean_squared_error(y_test,y_pred))
print("Root mean Squared Error => ",np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


# Result

# An approximate 93 (percent) is achieved by student if he studies for 9.25 hrs/day.

# Thank you..!
