# -*- coding: utf-8 -*-
"""Madeline_Gilmore_Notebook2.ipynb

Originally done on Colab in segments.

Original file is located at
    https://colab.research.google.com/drive/1OweVj67gUVjCzUelQdl2DvCbUHTPDSJo
"""

# get library for loading data
import pandas as pd

# load prepared data
data = pd.read_csv('Mall_Customers.csv')

#how many instances and attributes/features
data.shape

#library that lets you plot
import plotly.express as px

# make histogram of the spending score data
spending_score_fig = px.histogram(data, x='Spending_Score')

# show the histogram
spending_score_fig.show()

# make bar chart for customer sastisfaction data; bar charts are for categorical values
Customer_Satisfaction_fig = px.bar(data, x='Customer_Satisfaction')

# show the bar graph
Customer_Satisfaction_fig.show()

# scatter plot of spending score and salary data
# x and y values = graph[feature you want to specify]
Spending_Salary_Association_fig = px.scatter(x=data['Spending_Score'], y=data['Salary'])
Spending_Salary_Association_fig.show()

"""there are 5 groups in this graph (quads and middle)

problem = people who make a lot of money with low spending score, they arent spending enough money with us

need to investigate these people and figure out why.

we can do this with customer satisfaction, color the points in red to see who isnt satisfied
"""

#see customer satisfaction and age correlation
Spending_Score_Satisfaction_fig = px.histogram(data, x='Age', color='Customer_Satisfaction')
Spending_Score_Satisfaction_fig.show()

#use overlay to allow us to see overlaps

Spending_Score_Satisfaction_fig = px.histogram(data, x='Age', color='Customer_Satisfaction', barmode='overlay')
Spending_Score_Satisfaction_fig.show()

#heat map based on how high the salary is and their satisfaction rates while also seeing how many people are satisfied vs unsatisfied

Spending_Score_Satisfaction_fig = px.bar(data, x='Customer_Satisfaction', color='Salary')
Spending_Score_Satisfaction_fig.show()

# see first 5 instances; sex is 0 and 1, needs to be changed to male and female for graph clarity
data.head()

# remap sex feature
data['Sex'] = data['Sex'].map({1:'Male', 0:'Female'})

# bar graph to see whether males or females are more sastisfied
Spending_Score_Satisfaction_fig = px.bar(data, x='Customer_Satisfaction', color='Sex')
Spending_Score_Satisfaction_fig.show()

# how many people at each age (x and y) and are they satisfied (color)

Spending_Score_Satisfaction_fig = px.histogram(data, x='Age', color='Customer_Satisfaction', barmode="group")
Spending_Score_Satisfaction_fig.show()

# see spending score vs salary color coded by satifaction
# high salaries are not satisfied which is why theyre not spending

Spending_Score_Salary_Satisfaction_fig = px.scatter(data, x="Spending_Score", y="Salary", color="Customer_Satisfaction")
Spending_Score_Salary_Satisfaction_fig.show()

# outliers are inconsistant data points
#let’s load your prepared dataset
data = pd.read_csv('/content/Prepared_Mall_Customers.csv')
data.describe().transpose()
# We used transpose to make the columns rows and the rows columns to turn the table

data.head()

# Drop unnecessary variables and rename your dataset
df = data.drop(columns=(['ID', 'NewColumn']))
df.describe()

# can use a histogram to see outliers
salary_fig = px.histogram(df, x='Salary')
salary_fig.show()

#points outside the fences (lines on each side) are outliers; 137k is an outlier but we dont know how many there are because they could be overlapping
Age_fig = px.box(df, x='Salary')
Age_fig.show()

Age_Salary_Scatter_fig = px.scatter(x=df['Age'], y=df['Salary'])
Age_Salary_Scatter_fig.show()

# we can see here that there are actually 2 people with 137k

#pandas.quantile() function, we can create a simple Python function that takes in our column from the data frame and outputs the outliers

def find_outliers_IQR(df):
  q1=df.quantile(0.25)
  q3=df.quantile(0.75)
  IQR=q3-q1
  outliers = df[((df<(q1-1.5*IQR))|(df>(q3+1.5*IQR)))]
  return outliers

outliers = find_outliers_IQR(df['Salary'])
print("number of outliers: "+ str(len(outliers)))
outliers

# now we have to verify our outliers, only remove them if they are not reasonable

# we've decided to remove them; use .drop()

df.drop(df.index[[199,198]], inplace=True)

# check if they were actually dropped: use IQR again, or box plot or anywhere else where they would be

outliers = find_outliers_IQR(df['Salary'])
print("number of outliers: "+ str(len(outliers)))
outliers

# get instances with missing values
df.isnull()

# this is whole data frame wiht T/F if the values is there, true means missing value

#To find the percentage of missing data per variable
df.isna().sum()/len(df)*100

# the is a small amount of missing data so delete it, call this complete case analysis
# only do complete case analysis if we have 5% or less missing data because we like to retain data

"""simple imputation - replace missing values with the mean (average), mode, or median (even random within a limit or range)"""

df_Complete_Case = df.dropna()
df_Complete_Case

# get the means for the different features

Mean_Salary = df['Salary'].mean()
Mean_Spending_Score = df['Spending_Score'].mean()
Mean_Age = df['Age'].mean()

# fill in the means for the features

df['Salary'].fillna(Mean_Salary, inplace=True)
df['Spending_Score'].fillna(Mean_Spending_Score, inplace=True)
df['Age'].fillna(Mean_Age, inplace=True)


# this warning means that the package is outdated and we cant edit original data without a copy in the future

# data is now clean, now save the new data
df.to_csv(r'/content/Clean_Mall_Customers.csv', index=False) # new name, dont overwrite another

import pandas as pd
#let’s load prepared dataset
df1 = pd.read_csv('Mall_Customer_Insights_Data_Preparation/Mall_Customers.csv')
df1.head()

import pandas as pd
#let’s load prepared dataset
df2 = pd.read_csv('Mall_Customer_Insights_Data_Preparation/Mall_Customers_Additional.csv')
df2.head()

Merged_Mall_df = df1.merge(df2, on='CustomerID')
Merged_Mall_df.head()

Merged_Mall_df.to_csv("Merged_Mall_Data.csv", index=False)

import pandas as pd
#let’s load prepared dataset
df = pd.read_csv('/content/Merged_Mall_Data.csv')
df.describe()

from sklearn.preprocessing import StandardScaler
#drop unnecessary numeric and non-numeric variables
df_numeric = df.drop(columns=(['CustomerID', 'Gender']))
ss = StandardScaler()
df_scaled = ss.fit_transform(df_numeric)
df_scaled

df_scaled = pd.DataFrame(df_scaled,columns = df_numeric.columns)
df_scaled.head()

df_scaled.describe()

from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
df_mms = mms.fit_transform(df_numeric)
df_mms

df_mms = pd.DataFrame(df_mms,columns = df_numeric.columns)
df_mms.head()

df_mms.describe()