#!/usr/bin/env python
# coding: utf-8

# # Practical example,using SKLEARN CARS COMPANY 

# # Importing the relevant libraries

# In[2]:


# For this practical example we will need the following libraries and modules
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
sns.set()


# # Loading the raw data

# In[6]:


raw_data = pd.read_csv('C:/Users/rawad/OneDrive/Desktop/aws Restart course/Udemy Data Science Course/exercise/1.04.+Real-life+example.csv')


# In[4]:


data


# # Preprocessing

# # Exploring the descriptive statistics of the variables
# Descriptive statistics are very useful for initial exploration of the variables
# By default, only descriptives for the numerical variables are shown
# To include the categorical ones, you should specify this with an argument
# Note that categorical variables don't have some types of numerical descriptives
# and numerical variables don't have some types of categorical descriptives
# In[7]:


raw_data.describe(include='all')


# # Determining the variables of interest
# in this practical example we will proceed without applying the "MODEL". FOR THEORICAL LEARNING AND REFERENCE THE MODEL Models can be used to make predictions, estimate parameters, and test hypotheses. There are different types of models, such as linear regression, logistic regression, polynomial regression, and more complex models such as neural networks and decision trees. The choice of model depends on the nature of the problem and the data available.


# In[8]:


data = raw_data.drop(['Model'],axis=1)


# In[9]:


data.describe(include='all')


# # Dealing with missing values
# data.isnull() # shows a df with the information whether a data point is null 
# Since True = the data point is missing, while False = the data point is not missing, we can sum them
# This will give us the total number of missing values feature-wise
# In[10]:


data.isnull().sum()


# In[11]:


# Let's simply drop all missing values
# This is not always recommended, however, when we remove less than 5% of the data, it is okay
data_no_mv = data.dropna(axis=0)


# In[12]:


# Let's check the descriptives without the missing values
data_no_mv.describe(include='all')


# # EXPLORING THE PDF 
# A great step in the data exploration is to display the probability distribution function (PDF) of a variable
# The PDF will show us how that variable is distributed 
# This makes it very easy to spot anomalies, such as outliers
# The PDF is often the basis on which we decide whether we want to transform a feature
# In[23]:


sns.distplot(data_no_mv['Price'])


# # Dealing with outliers
# Obviously there are some outliers present 

# Without diving too deep into the topic, we can deal with the problem easily by removing 0.5%, or 1% of the problematic samples
# Here, the outliers are situated around the higher prices (right side of the graph)
# Logic should also be applied
# This is a dataset about used cars, therefore one can imagine how $300,000 is an excessive price

# Outliers are a great issue for OLS, thus we must deal with them in some way
# It may be a useful exercise to try training a model without removing the outliers

# In[24]:


# Let's declare a variable that will be equal to the 99th percentile of the 'Price' variable
q = data_no_mv['Price'].quantile(0.99)
# Then we can create a new df, with the condition that all prices must be below the 99 percentile of 'Price'
data_1 = data_no_mv[data_no_mv['Price']<q]
# In this way we have essentially removed the top 1% of the data about 'Price'
data_1.describe(include='all')


# In[25]:


# We can check the PDF once again to ensure that the result is still distributed in the same way overall
# however, there are much fewer outliers
sns.distplot(data_1['Price'])


# In[26]:


# We can treat the other numerical variables in a similar way
sns.distplot(data_no_mv['Mileage'])


# In[28]:





# In[29]:


# This plot looks kind of normal, doesn't it?
sns.distplot(data_2['Mileage'])


# In[30]:


q = data_1['Mileage'].quantile(0.99)
data_2 = data_1[data_1['Mileage']<q]


# In[ ]:


# The situation with engine volume is very strange
# In such cases it makes sense to manually check what may be causing the problem
# In our case the issue comes from the fact that most missing values are indicated with 99.99 or 99
# There are also some incorrect entries like 75


# In[31]:


sns.distplot(data_no_mv['EngineV'])


# In[32]:


# A simple Google search can indicate the natural domain of this variable
# Car engine volumes are usually (always?) below 6.5l
# This is a prime example of the fact that a domain expert (a person working in the car industry)
# may find it much easier to determine problems with the data than an outsider
data_3 = data_2[data_2['EngineV']<6.5]


# In[33]:


# Following this graph, we realize we can actually treat EngineV as a categorical variable

sns.distplot(data_3['EngineV'])


# In[34]:


# Finally, the situation with 'Year' is similar to 'Price' and 'Mileage'
# However, the outliers are on the low end
sns.distplot(data_no_mv['Year'])


# In[35]:


# I'll simply remove them
q = data_3['Year'].quantile(0.01)
data_4 = data_3[data_3['Year']>q]


# In[36]:


# Here's the new result
sns.distplot(data_4['Year'])

NOTE : # When we remove observations, the original indexes are preserved
# If we remove observations with indexes 2 and 3, the indexes will go as: 0,1,4,5,6
# That's very problematic as we tend to forget about it (later you will see an example of such a problem)

# Finally, once we reset the index, a new column will be created containing the old index (just in case)
# We won't be needing it, thus 'drop=True' to completely forget about it
# In[37]:


data_cleaned = data_4.reset_index(drop=True)


# In[38]:


# Let's see what's left
data_cleaned.describe(include='all')


# # Checking the OLS assumptions
# Here we decided to use some matplotlib code, without explaining it
# But since Price is the 'y' axis of all the plots, it made sense to plot them side-by-side (so we can compare them)
# In[39]:


f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize =(15,3)) #sharey -> share 'Price' as y
ax1.scatter(data_cleaned['Year'],data_cleaned['Price'])
ax1.set_title('Price and Year')
ax2.scatter(data_cleaned['EngineV'],data_cleaned['Price'])
ax2.set_title('Price and EngineV')
ax3.scatter(data_cleaned['Mileage'],data_cleaned['Price'])
ax3.set_title('Price and Mileage')

plt.show()


# In[40]:


# From the subplots and the PDF of price, we can easily determine that 'Price' is exponentially distributed
# A good transformation in that case is a log transformation
sns.distplot(data_cleaned['Price'])


# # Relaxing the assumptions

# In[41]:


# Let's transform 'Price' with a log transformation
log_price = np.log(data_cleaned['Price'])

# Then we add it to our data frame
data_cleaned['log_price'] = log_price
data_cleaned


# In[42]:


# Let's check the three scatters once again
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize =(15,3))
ax1.scatter(data_cleaned['Year'],data_cleaned['log_price'])
ax1.set_title('Log Price and Year')
ax2.scatter(data_cleaned['EngineV'],data_cleaned['log_price'])
ax2.set_title('Log Price and EngineV')
ax3.scatter(data_cleaned['Mileage'],data_cleaned['log_price'])
ax3.set_title('Log Price and Mileage')


plt.show()


# The relationships show a clear linear relationship
# This is some good linear regression material
# Alternatively we could have transformed each of the independent variables
# In[43]:


# Since we will be using the log price variable, we can drop the old 'Price' one
data_cleaned = data_cleaned.drop(['Price'],axis=1)


# # Multicollinearity

# In[44]:


# Let's quickly see the columns of our data frame
data_cleaned.columns.values

# sklearn does not have a built-in way to check for multicollinearity
# one of the main reasons is that this is an issue well covered in statistical frameworks and not in ML ones
# surely it is an issue nonetheless, thus we will try to deal with it

# Here's the relevant module
# full documentation: http://www.statsmodels.org/dev/_modules/statsmodels/stats/outliers_influence.html#variance_inflation_factor
# In[45]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

# To make this as easy as possible to use, we declare a variable where we put
# all features where we want to check for multicollinearity
# since our categorical data is not yet preprocessed, we will only take the numerical ones
# In[46]:


variables = data_cleaned[['Mileage','Year','EngineV']]


# In[47]:


# we create a new data frame which will include all the VIFs
# note that each variable has its own variance inflation factor as this measure is variable specific (not model specific)
vif = pd.DataFrame()

NOTE ABOUT VIF :stands for Variance Inflation Factor. It is a measure of multicollinearity, which is the degree to which the independent variables (features) in a linear regression model are correlated with each other. VIF measures the inflation of the variance of the estimated regression coefficient of a variable due to multicollinearity in the model. A VIF value of 1 indicates that there is no multicollinearity among the independent variables, whereas a value greater than 1 suggests that there is a high degree of multicollinearity. Typically, a VIF value above 5 or 10 is considered to be a cause for concern, as it suggests that the variable may be too highly correlated with other variables in the model.
# In[48]:


# here we make use of the variance_inflation_factor, which will basically output the respective VIFs 
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
# Finally, I like to include names so it is easier to explore the result
vif["Features"] = variables.columns


# In[49]:


# Let's explore the result
vif


# In[50]:


# Since Year has the highest VIF, I will remove it from the model
# This will drive the VIF of other variables down!!! 
# So even if EngineV seems with a high VIF, too, once 'Year' is gone that will no longer be the case
data_no_multicollinearity = data_cleaned.drop(['Year'],axis=1)


# # Create dummy variables
# 
# # To include the categorical data in the regression, let's create dummies
# # There is a very convenient method called: 'get_dummies' which does that seemlessly
# # It is extremely important that we drop one of the dummies, alternatively we will introduce multicollinearity

# In[52]:


data_with_dummies = pd.get_dummies(data_no_multicollinearity, drop_first=True)

THEORICAL EXPLINATION ABOUT using Dummies technique: In statistical modeling and data analysis, categorical variables need to be represented numerically for use in regression analysis or machine learning models. One common technique is to convert categorical variables into dummy variables. Dummy variables are binary (0 or 1) variables representing the presence or absence of a particular categorical variable level. For example, if we have a categorical variable "color" with three levels - red, blue, and green - we can create three dummy variables "red", "blue", and "green". Each observation in the dataset will be represented by a 0 or 1 in each of these dummy variables, indicating whether the observation has that particular color level. This way, we can include categorical variables in our regression models as numerical predictors.
# In[53]:


# Here's the result
data_with_dummies.head()


# # Rearrange a bit
# To make our data frame more organized, we prefer to place the dependent variable in the beginning of the df
# Since each problem is different, that must be done manually
# We can display all possible features and then choose the desired order
# In[54]:


data_with_dummies.columns.values


# In[55]:


# To make the code a bit more parametrized, let's declare a new variable that will contain the preferred order
# If you want a different order, just specify it here
# Conventionally, the most intuitive order is: dependent variable, indepedendent numerical variables, dummies
cols = ['log_price', 'Mileage', 'EngineV', 'Brand_BMW',
       'Brand_Mercedes-Benz', 'Brand_Mitsubishi', 'Brand_Renault',
       'Brand_Toyota', 'Brand_Volkswagen', 'Body_hatch', 'Body_other',
       'Body_sedan', 'Body_vagon', 'Body_van', 'Engine Type_Gas',
       'Engine Type_Other', 'Engine Type_Petrol', 'Registration_yes']


# In[56]:


# To implement the reordering, we will create a new df, which is equal to the old one but with the new order of features
data_preprocessed = data_with_dummies[cols]
data_preprocessed.head()


# # Linear regression model

# # Declare the inputs and the targets

# In[57]:


# The target(s) (dependent variable) is 'log price'
targets = data_preprocessed['log_price']

# The inputs are everything BUT the dependent variable, so we can simply drop it
inputs = data_preprocessed.drop(['log_price'],axis=1)


# # Scale the data

# In[58]:


# Import the scaling module
from sklearn.preprocessing import StandardScaler

# Create a scaler object
scaler = StandardScaler()
# Fit the inputs (calculate the mean and standard deviation feature-wise)
scaler.fit(inputs)


# In[60]:


# Scale the features and store them in a new variable (the actual scaling procedure)
inputs_scaled = scaler.transform(inputs)


# # Train Test Split

# In[61]:


# Import the module for the split
from sklearn.model_selection import train_test_split

# Split the variables with an 80-20 split and some random state
# To have the same split as mine, use random_state = 365
x_train, x_test, y_train, y_test = train_test_split(inputs_scaled, targets, test_size=0.2, random_state=365)


# # Create the regression

# In[62]:


# Create a linear regression object
reg = LinearRegression()
# Fit the regression with the scaled TRAIN inputs and targets
reg.fit(x_train,y_train)


# In[63]:


# Let's check the outputs of the regression
# I'll store them in y_hat as this is the 'theoretical' name of the predictions
y_hat = reg.predict(x_train)


# In[64]:


# The simplest way to compare the targets (y_train) and the predictions (y_hat) is to plot them on a scatter plot
# The closer the points to the 45-degree line, the better the prediction
plt.scatter(y_train, y_hat)
# Let's also name the axes
plt.xlabel('Targets (y_train)',size=18)
plt.ylabel('Predictions (y_hat)',size=18)
# Sometimes the plot will have different scales of the x-axis and the y-axis
# This is an issue as we won't be able to interpret the '45-degree line'
# We want the x-axis and the y-axis to be the same
plt.xlim(6,13)
plt.ylim(6,13)
plt.show()


# In[65]:


# Another useful check of our model is a residual plot
# We can plot the PDF of the residuals and check for anomalies
sns.distplot(y_train - y_hat)

# Include a title
plt.title("Residuals PDF", size=18)

# In the best case scenario this plot should be normally distributed
# In our case we notice that there are many negative residuals (far away from the mean)
# Given the definition of the residuals (y_train - y_hat), negative values imply
# that y_hat (predictions) are much higher than y_train (the targets)
# This is food for thought to improve our model


# In[66]:


# Find the R-squared of the model
reg.score(x_train,y_train)

# Note that this is NOT the adjusted R-squared
# in other words... find the Adjusted R-squared to have the appropriate measure :)


# # Finding the weights and bias

# In[67]:


# Obtain the bias (intercept) of the regression
reg.intercept_


# In[68]:


# Obtain the weights (coefficients) of the regression
reg.coef_

# Note that they are barely interpretable if at all


# In[69]:


# Create a regression summary where we can compare them with one-another
reg_summary = pd.DataFrame(inputs.columns.values, columns=['Features'])
reg_summary['Weights'] = reg.coef_
reg_summary

In linear regression, weight (also known as coefficient or beta) represents the change in the target variable for each unit change in the corresponding feature while holding all other features constant.

For example, the weight for "Mileage" is -0.448713. This means that for each unit increase in mileage, the predicted price of the car is expected to decrease by 0.448713 units, holding all other features constant.

Similarly, the weight for "EngineV" is 0.209035, which means that for each unit increase in engine volume, the predicted price of the car is expected to increase by 0.209035 units, holding all other features constant.

The signs of the weights can also be important in determining the direction of the relationship between the feature and the target variable. A positive weight indicates a positive relationship (as the feature increases, the target variable is expected to increase), while a negative weight indicates a negative relationship (as the feature increases, the target variable is expected to decrease).
# In[70]:


# Check the different categories in the 'Brand' variable
data_cleaned['Brand'].unique()

# In this way we can see which 'Brand' is actually the benchmark


# # Testing
# Once we have trained and fine-tuned our model, we can proceed to testing it
# Testing is done on a dataset that the algorithm has never seen
# Luckily we have prepared such a dataset
# Our test inputs are 'x_test', while the outputs: 'y_test' 
# We SHOULD NOT TRAIN THE MODEL ON THEM, we just feed them and find the predictions
# If the predictions are far off, we will know that our model overfitted
# In[71]:


y_hat_test = reg.predict(x_test)


# In[72]:


# Create a scatter plot with the test targets and the test predictions
# You can include the argument 'alpha' which will introduce opacity to the graph
plt.scatter(y_test, y_hat_test, alpha=0.2)
plt.xlabel('Targets (y_test)',size=18)
plt.ylabel('Predictions (y_hat_test)',size=18)
plt.xlim(6,13)
plt.ylim(6,13)
plt.show()


# In[73]:


# Finally, let's manually check these predictions
# To obtain the actual prices, we take the exponential of the log_price
df_pf = pd.DataFrame(np.exp(y_hat_test), columns=['Prediction'])
df_pf.head()


# In[74]:


# We can also include the test targets in that data frame (so we can manually compare them)
df_pf['Target'] = np.exp(y_test)
df_pf

# Note that we have a lot of missing values
# There is no reason to have ANY missing values, though
# This suggests that something is wrong with the data frame / indexing


# In[75]:


# Let's overwrite the 'Target' column with the appropriate values
# Again, we need the exponential of the test log price
df_pf['Target'] = np.exp(y_test)
df_pf


# In[77]:


# Additionally, we can calculate the difference between the targets and the predictions
# Note that this is actually the residual (we already plotted the residuals)
df_pf['Residual'] = df_pf['Target'] - df_pf['Prediction']

# Since OLS is basically an algorithm which minimizes the total sum of squared errors (residuals),
# this comparison makes a lot of sense


# In[78]:


# Finally, it makes sense to see how far off we are from the result percentage-wise
# Here, we take the absolute difference in %, so we can easily order the data frame
df_pf['Difference%'] = np.absolute(df_pf['Residual']/df_pf['Target']*100)
df_pf


# In[79]:


# Exploring the descriptives here gives us additional insights
df_pf.describe()


# In[81]:


# Sometimes it is useful to check these outputs manually
# To see all rows, we use the relevant pandas syntax
pd.options.display.max_rows = 999
# Moreover, to make the dataset clear, we can display the result with only 2 digits after the dot 
pd.set_option('display.float_format', lambda x: '%.2f' % x)
# Finally, we sort by difference in % and manually check the model
df_pf.sort_values(by=['Difference%'])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




