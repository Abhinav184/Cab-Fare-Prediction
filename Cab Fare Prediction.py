#!/usr/bin/env python
# coding: utf-8

# ## Objective: 

# You are a cab rental start-up company. You have successfully run the pilot project and now want to launch your cab service across the country. You have collected the historical data from your pilot project and now have a requirement to apply analytics for fare prediction. You need to design a system that predicts the fare amount for a cab ride in the city.

# In[1]:


#Importing required libraries
import os #getting access to input files
import pandas as pd # Importing pandas for performing EDA
import numpy as np  # Importing numpy for Linear Algebric operations
import matplotlib.pyplot as plt # Importing for Data Visualization
import seaborn as sns # Importing for Data Visualization
from collections import Counter 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression #ML algorithm
from sklearn.model_selection import train_test_split #splitting dataset
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from pprint import pprint
from sklearn.model_selection import GridSearchCV    

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Setting the working directory

os.chdir("E:/edWisor/Cab Fare Prediction")
print(os.getcwd())


# In[3]:


#Loading the data:
train  = pd.read_csv("E:/edWisor/Cab Fare Prediction/train_cab.csv",na_values={"pickup_datetime":"43"})
test   = pd.read_csv("E:/edWisor/Cab Fare Prediction/test.csv")


# ## Understanding the data :

# In[4]:


train.head() #checking first five rows of the training dataset


# In[5]:


test.head() #checking first five rows of the test dataset


# In[6]:


print("shape of training data is: ",train.shape) #checking the number of rows and columns in training data
print("shape of test data is: ",test.shape) #checking the number of rows and columns in test data


# In[7]:


train.dtypes #checking the data-types in training dataset


# Here we can see pickup datetime and fare amount is of object type. So we need to change the data type of both.

# In[8]:


test.dtypes #checking the data-types in test dataset


# In[9]:


train.describe() 


# In[10]:


test.describe()


# ## Data Cleaning & Missing Value Analysis :

# In[11]:


#Convert fare_amount from object to numeric
train["fare_amount"] = pd.to_numeric(train["fare_amount"],errors = "coerce")  #Using errors=’coerce’. It will replace all non-numeric values with NaN.


# In[12]:


train.dtypes


# In[13]:


train.shape


# In[14]:


train.dropna(subset= ["pickup_datetime"])   #dropping NA values in datetime column


# In[15]:


# Here pickup_datetime variable is in object so we need to change its data type to datetime
train['pickup_datetime'] =  pd.to_datetime(train['pickup_datetime'], format='%Y-%m-%d %H:%M:%S UTC')


# In[16]:


### we will saperate the Pickup_datetime column into separate field like year, month, day of the week, etc

train['year'] = train['pickup_datetime'].dt.year
train['Month'] = train['pickup_datetime'].dt.month
train['Date'] = train['pickup_datetime'].dt.day
train['Day'] = train['pickup_datetime'].dt.dayofweek
train['Hour'] = train['pickup_datetime'].dt.hour
train['Minute'] = train['pickup_datetime'].dt.minute


# In[17]:


train.dtypes #Re-checking datatypes after conversion


# In[18]:


test["pickup_datetime"] = pd.to_datetime(test["pickup_datetime"],format= "%Y-%m-%d %H:%M:%S UTC")


# In[19]:


### we will saperate the Pickup_datetime column into separate field like year, month, day of the week, etc

test['year'] = test['pickup_datetime'].dt.year
test['Month'] = test['pickup_datetime'].dt.month
test['Date'] = test['pickup_datetime'].dt.day
test['Day'] = test['pickup_datetime'].dt.dayofweek
test['Hour'] = test['pickup_datetime'].dt.hour
test['Minute'] = test['pickup_datetime'].dt.minute


# In[20]:


test.dtypes #Re-checking test datatypes after conversion


# #### Observations :
# 
# 1. An outlier in pickup_datetime column of value 43
# 2. Passenger count should not exceed 6(even if we consider SUV)
# 3. Latitudes range from -90 to 90. Longitudes range from -180 to 180
# 4. Few missing values and High values of fare and Passenger count are present. So, decided to remove them.

# Checking the Datetime Variable : 

# In[21]:


#removing datetime missing values rows
train = train.drop(train[train['pickup_datetime'].isnull()].index, axis=0)
print(train.shape)
print(train['pickup_datetime'].isnull().sum())


# Checking the passenger count variable : 

# In[22]:


train["passenger_count"].describe()


# We can see maximum number of passanger count is 5345 which is actually not possible. So reducing the passenger count to 6 (even if we consider the SUV)

# In[23]:


train = train.drop(train[train["passenger_count"]> 6 ].index, axis=0)


# In[24]:


#Also removing the values with passenger count of 0.
train = train.drop(train[train["passenger_count"] == 0 ].index, axis=0)


# In[25]:


train["passenger_count"].describe()


# In[26]:


train["passenger_count"].sort_values(ascending= True)


# In[27]:


#removing passanger_count missing values rows
train = train.drop(train[train['passenger_count'].isnull()].index, axis=0)
print(train.shape)
print(train['passenger_count'].isnull().sum())


# There is one passenger count value of 0.12 which is not possible. Hence we will remove fractional passenger value

# In[28]:


train = train.drop(train[train["passenger_count"] == 0.12 ].index, axis=0)
train.shape


# Next checking the Fare Amount variable :

# In[29]:


##finding decending order of fare to get to know whether the outliers are present or not
train["fare_amount"].sort_values(ascending=False)


# In[30]:


Counter(train["fare_amount"]<0)


# In[31]:


train = train.drop(train[train["fare_amount"]<0].index, axis=0)
train.shape


# In[32]:


##make sure there is no negative values in the fare_amount variable column
train["fare_amount"].min()


# In[33]:


#Also remove the row where fare amount is zero
train = train.drop(train[train["fare_amount"]<1].index, axis=0)
train.shape


# In[34]:


#Now we can see that there is a huge difference in 1st 2nd and 3rd position in decending order of fare amount
# so we will remove the rows having fare amounting more that 454 as considering them as outliers

train = train.drop(train[train["fare_amount"]> 454 ].index, axis=0)
train.shape


# In[35]:


# eliminating rows for which value of "fare_amount" is missing
train = train.drop(train[train['fare_amount'].isnull()].index, axis=0)
print(train.shape)
print(train['fare_amount'].isnull().sum())


# In[36]:


train["fare_amount"].describe()


# Now checking the pickup lattitude and longitude :

# In[37]:


#Lattitude----(-90 to 90)
#Longitude----(-180 to 180)

# we need to drop the rows having  pickup lattitute and longitute out the range mentioned above

#train = train.drop(train[train['pickup_latitude']<-90])
train[train['pickup_latitude']<-90]
train[train['pickup_latitude']>90]


# In[38]:


#Hence dropping one value of >90
train = train.drop((train[train['pickup_latitude']<-90]).index, axis=0)
train = train.drop((train[train['pickup_latitude']>90]).index, axis=0)


# In[39]:


train[train['pickup_longitude']<-180]
train[train['pickup_longitude']>180]


# In[40]:


train[train['dropoff_latitude']<-90]
train[train['dropoff_latitude']>90]


# In[41]:


train[train['dropoff_longitude']<-180]
train[train['dropoff_longitude']>180]


# In[42]:


train.shape


# In[43]:


train.isnull().sum()


# In[44]:


test.isnull().sum()


# ### Now we have successfully cleared our both datasets. Thus proceeding for further operations:

# Calculating distance based on the given coordinates :
# 

# In[45]:


#As we know that we have given pickup longitute and latitude values and same for drop. 
#So we need to calculate the distance Using the haversine formula and we will create a new variable called distance
from math import radians, cos, sin, asin, sqrt

def haversine(a):
    lon1=a[0]
    lat1=a[1]
    lon2=a[2]
    lat2=a[3]
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c =  2 * asin(sqrt(a))
    # Radius of earth in kilometers is 6371
    km = 6371* c
    return km
# 1min 


# In[46]:


train['distance'] = train[['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']].apply(haversine,axis=1)


# In[47]:


test['distance'] = test[['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']].apply(haversine,axis=1)


# In[48]:


train.head()


# In[49]:


test.head()


# In[50]:


train.nunique()


# In[51]:


test.nunique()


# In[52]:


##finding decending order of fare to get to know whether the outliers are presented or not
train['distance'].sort_values(ascending=False)


# As we can see that top 23 values in the distance variables are very high It means more than 8000 Kms distance they have travelled
# Also just after 23rd value from the top, the distance goes down to 127, which means these values are showing some outliers
# We need to remove these values
# 

# In[53]:


Counter(train['distance'] == 0)


# In[54]:


Counter(test['distance'] == 0)


# In[55]:


Counter(train['fare_amount'] == 0)


# In[56]:


###we will remove the rows whose distance value is zero

train = train.drop(train[train['distance']== 0].index, axis=0)
train.shape


# In[57]:


#we will remove the rows whose distance values is very high which is more than 129kms
train = train.drop(train[train['distance'] > 130 ].index, axis=0)
train.shape


# In[58]:


train.head()


# Now we have splitted the pickup date time variable into different varaibles like month, year, day etc so now we dont need to have that pickup_Date variable now. Hence we can drop that, Also we have created distance using pickup and drop longitudes and latitudes so we will also drop pickup and drop longitudes and latitudes variables.

# In[59]:


drop = ['pickup_datetime', 'pickup_longitude', 'pickup_latitude','dropoff_longitude', 'dropoff_latitude', 'Minute']
train = train.drop(drop, axis = 1)


# In[60]:


train.head()


# In[61]:


train['passenger_count'] = train['passenger_count'].astype('int64')
train['year'] = train['year'].astype('int64')
train['Month'] = train['Month'].astype('int64')
train['Date'] = train['Date'].astype('int64')
train['Day'] = train['Day'].astype('int64')
train['Hour'] = train['Hour'].astype('int64')


# In[62]:


train.dtypes


# In[63]:


drop_test = ['pickup_datetime', 'pickup_longitude', 'pickup_latitude','dropoff_longitude', 'dropoff_latitude', 'Minute']
test = test.drop(drop_test, axis = 1)


# In[64]:


test.head()


# In[65]:


test.dtypes


# # Data Visualization :

# Visualization of following:
# 
# 1. Number of Passengers effects the the fare
# 2. Pickup date and time effects the fare
# 3. Day of the week does effects the fare
# 4. Distance effects the fare

# In[66]:


# Count plot on passenger count
plt.figure(figsize=(15,7))
sns.countplot(x="passenger_count", data=train)


# In[67]:


#Relationship beetween number of passengers and Fare

plt.figure(figsize=(15,7))
plt.scatter(x=train['passenger_count'], y=train['fare_amount'], s=10)
plt.xlabel('No. of Passengers')
plt.ylabel('Fare')
plt.show()


# In[68]:


passenger_grp = train.groupby('passenger_count')
passenger_grp['fare_amount'].agg(['mean', 'count'])


# In[69]:


plt.figure(figsize=(15,7))
passenger_grp['fare_amount'].mean().plot.bar(color='g')
plt.title('Average fare by Passenger count')
plt.show()


# ### Observations :
#    By seeing the above plots we can easily conclude that:
# 1. single travelling passengers are most frequent travellers.
# 2. At the sametime we can also conclude that highest Fare are coming from single & double travelling passengers.

# In[70]:


#Relationship between date and Fare
plt.figure(figsize=(15,7))
plt.scatter(x=train['Date'], y=train['fare_amount'], s=10)
plt.xlabel('Date')
plt.ylabel('Fare')
plt.show()


# In[71]:


plt.figure(figsize=(15,7))
train.groupby(train["Hour"])['Hour'].count().plot(kind="bar")
plt.show()


# Lowest cabs at 5 AM and highest at and around 7 PM i.e the office rush hours

# In[72]:


#Relationship between Time and Fare
plt.figure(figsize=(15,7))
plt.scatter(x=train['Hour'], y=train['fare_amount'], s=10)
plt.xlabel('Hour')
plt.ylabel('Fare')
plt.show()


# From the above plot We can observe that the cabs taken at 7 am and 23 Pm are the costliest. 
# Hence we can assume that cabs taken early in morning and late at night are costliest
# 

# In[73]:


#impact of Day on the number of cab rides
plt.figure(figsize=(15,7))
sns.countplot(x="Day", data=train)


# Observation :
# The day of the week does not seem to have much influence on the number of cabs ride

# In[74]:


#Relationships between day and Fare
plt.figure(figsize=(15,7))
plt.scatter(x=train['Day'], y=train['fare_amount'], s=10)
plt.xlabel('Day')
plt.ylabel('Fare')
plt.show()


# The highest fares seem to be on a Sunday, Monday and Thursday, and the low on Wednesday and Saturday. May be due to low demand of the cabs on saturdays the cab fare is low and high demand of cabs on sunday and monday shows the high fare prices

# In[75]:


#Relationship between distance and fare 
plt.figure(figsize=(15,7))
plt.scatter(x = train['distance'],y = train['fare_amount'],c = "g")
plt.xlabel('Distance')
plt.ylabel('Fare')
plt.show()


# It is quite obvious that distance will effect the amount of fare

# # Feature Scaling :

# In[76]:


#Normality check of training data is uniformly distributed or not-

for i in ['fare_amount', 'distance']:
    print(i)
    sns.distplot(train[i],bins='auto',color='green')
    plt.title("Distribution for Variable "+i)
    plt.ylabel("Density")
    plt.show()


# In[77]:


#since skewness of target variable is high, apply log transform to reduce the skewness-
train['fare_amount'] = np.log1p(train['fare_amount'])

#since skewness of distance variable is high, apply log transform to reduce the skewness-
train['distance'] = np.log1p(train['distance'])


# In[78]:


#Normality Re-check to check data is uniformly distributed or not after log transformartion

for i in ['fare_amount', 'distance']:
    print(i)
    sns.distplot(train[i],bins='auto',color='green')
    plt.title("Distribution for Variable "+i)
    plt.ylabel("Density")
    plt.show()


# Here we can see bell shaped distribution. Hence our continous variables are now normally distributed, we will use not use any  Feature Scalling technique. i.e, Normalization or Standarization for our training data

# In[79]:


#Normality check for test data is uniformly distributed or not-

sns.distplot(test['distance'],bins='auto',color='green')
plt.title("Distribution for Variable "+i)
plt.ylabel("Density")
plt.show()


# In[80]:


#since skewness of distance variable is high, apply log transform to reduce the skewness-
test['distance'] = np.log1p(test['distance'])


# In[81]:


#rechecking the distribution for distance
sns.distplot(test['distance'],bins='auto',color='green')
plt.title("Distribution for Variable "+i)
plt.ylabel("Density")
plt.show()


# As we can see a bell shaped distribution. Hence our continous variables are now normally distributed, we will use not use any  Feature Scalling technique. i.e, Normalization or Standarization for our test data

# # Applying ML ALgorithms: 

# In[82]:


##train test split for further modelling
X_train, X_test, y_train, y_test = train_test_split( train.iloc[:, train.columns != 'fare_amount'], 
                         train.iloc[:, 0], test_size = 0.20, random_state = 1)


# In[83]:


print(X_train.shape)
print(X_test.shape)


# ### Linear Regression Model :

# In[84]:


# Building model on top of training dataset
fit_LR = LinearRegression().fit(X_train , y_train)


# In[85]:


#prediction on train data
pred_train_LR = fit_LR.predict(X_train)


# In[86]:


#prediction on test data
pred_test_LR = fit_LR.predict(X_test)


# In[87]:


##calculating RMSE for test data
RMSE_test_LR = np.sqrt(mean_squared_error(y_test, pred_test_LR))

##calculating RMSE for train data
RMSE_train_LR= np.sqrt(mean_squared_error(y_train, pred_train_LR))


# In[88]:


print("Root Mean Squared Error For Training data = "+str(RMSE_train_LR))
print("Root Mean Squared Error For Test data = "+str(RMSE_test_LR))


# In[89]:


#calculate R^2 for train data
from sklearn.metrics import r2_score
r2_score(y_train, pred_train_LR)


# In[90]:


r2_score(y_test, pred_test_LR)


# ### Decision tree Model : 

# In[91]:


fit_DT = DecisionTreeRegressor(max_depth = 2).fit(X_train,y_train)


# In[92]:


#prediction on train data
pred_train_DT = fit_DT.predict(X_train)

#prediction on test data
pred_test_DT = fit_DT.predict(X_test)


# In[93]:


##calculating RMSE for train data
RMSE_train_DT = np.sqrt(mean_squared_error(y_train, pred_train_DT))

##calculating RMSE for test data
RMSE_test_DT = np.sqrt(mean_squared_error(y_test, pred_test_DT))


# In[94]:


print("Root Mean Squared Error For Training data = "+str(RMSE_train_DT))
print("Root Mean Squared Error For Test data = "+str(RMSE_test_DT))


# In[95]:


## R^2 calculation for train data
r2_score(y_train, pred_train_DT)


# In[96]:


## R^2 calculation for test data
r2_score(y_test, pred_test_DT)


# ### Random Forest Model :

# In[97]:


fit_RF = RandomForestRegressor(n_estimators = 200).fit(X_train,y_train)


# In[98]:


#prediction on train data
pred_train_RF = fit_RF.predict(X_train)
#prediction on test data
pred_test_RF = fit_RF.predict(X_test)


# In[99]:


##calculating RMSE for train data
RMSE_train_RF = np.sqrt(mean_squared_error(y_train, pred_train_RF))
##calculating RMSE for test data
RMSE_test_RF = np.sqrt(mean_squared_error(y_test, pred_test_RF))


# In[100]:


print("Root Mean Squared Error For Training data = "+str(RMSE_train_RF))
print("Root Mean Squared Error For Test data = "+str(RMSE_test_RF))


# In[101]:


## calculate R^2 for train data

r2_score(y_train, pred_train_RF)


# In[102]:


#calculate R^2 for test data
r2_score(y_test, pred_test_RF)


# ### Gradient Boosting :

# In[103]:


fit_GB = GradientBoostingRegressor().fit(X_train, y_train)


# In[104]:


#prediction on train data
pred_train_GB = fit_GB.predict(X_train)

#prediction on test data
pred_test_GB = fit_GB.predict(X_test)


# In[105]:


##calculating RMSE for train data
RMSE_train_GB = np.sqrt(mean_squared_error(y_train, pred_train_GB))
##calculating RMSE for test data
RMSE_test_GB = np.sqrt(mean_squared_error(y_test, pred_test_GB))


# In[106]:


print("Root Mean Squared Error For Training data = "+str(RMSE_train_GB))
print("Root Mean Squared Error For Test data = "+str(RMSE_test_GB))


# In[107]:


#calculate R^2 for test data
r2_score(y_test, pred_test_GB)


# In[108]:


#calculate R^2 for train data
r2_score(y_train, pred_train_GB)


# # Optimizing the results with parameters tuning :

# In[109]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state = 42)
from pprint import pprint
# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(rf.get_params())


# In[110]:


##Random Hyperparameter Grid


# In[111]:


from sklearn.model_selection import train_test_split,RandomizedSearchCV


# In[112]:


##Random Search CV on Random Forest Model

RRF = RandomForestRegressor(random_state = 0)
n_estimator = list(range(1,20,2))
depth = list(range(1,100,2))

# Create the random grid
rand_grid = {'n_estimators': n_estimator,
               'max_depth': depth}

randomcv_rf = RandomizedSearchCV(RRF, param_distributions = rand_grid, n_iter = 5, cv = 5, random_state=0)
randomcv_rf = randomcv_rf.fit(X_train,y_train)
predictions_RRF = randomcv_rf.predict(X_test)

view_best_params_RRF = randomcv_rf.best_params_

best_model = randomcv_rf.best_estimator_

predictions_RRF = best_model.predict(X_test)

#R^2
RRF_r2 = r2_score(y_test, predictions_RRF)
#Calculating RMSE
RRF_rmse = np.sqrt(mean_squared_error(y_test,predictions_RRF))

print('Random Search CV Random Forest Regressor Model Performance:')
print('Best Parameters = ',view_best_params_RRF)
print('R-squared = {:0.2}.'.format(RRF_r2))
print('RMSE = ',RRF_rmse)


# In[113]:


gb = GradientBoostingRegressor(random_state = 42)
from pprint import pprint
# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(gb.get_params())


# In[114]:


##Random Search CV on gradient boosting model

gb = GradientBoostingRegressor(random_state = 0)
n_estimator = list(range(1,20,2))
depth = list(range(1,100,2))

# Create the random grid
rand_grid = {'n_estimators': n_estimator,
               'max_depth': depth}

randomcv_gb = RandomizedSearchCV(gb, param_distributions = rand_grid, n_iter = 5, cv = 5, random_state=0)
randomcv_gb = randomcv_gb.fit(X_train,y_train)
predictions_gb = randomcv_gb.predict(X_test)

view_best_params_gb = randomcv_gb.best_params_

best_model = randomcv_gb.best_estimator_

predictions_gb = best_model.predict(X_test)

#R^2
gb_r2 = r2_score(y_test, predictions_gb)
#Calculating RMSE
gb_rmse = np.sqrt(mean_squared_error(y_test,predictions_gb))

print('Random Search CV Gradient Boosting Model Performance:')
print('Best Parameters = ',view_best_params_gb)
print('R-squared = {:0.2}.'.format(gb_r2))
print('RMSE = ', gb_rmse)


# In[115]:


from sklearn.model_selection import GridSearchCV    
## Grid Search CV for random Forest model
regr = RandomForestRegressor(random_state = 0)
n_estimator = list(range(11,20,1))
depth = list(range(5,15,2))

# Create the grid
grid_search = {'n_estimators': n_estimator,
               'max_depth': depth}

## Grid Search Cross-Validation with 5 fold CV
gridcv_rf = GridSearchCV(regr, param_grid = grid_search, cv = 5)
gridcv_rf = gridcv_rf.fit(X_train,y_train)
view_best_params_GRF = gridcv_rf.best_params_

#Apply model on test data
predictions_GRF = gridcv_rf.predict(X_test)

#R^2
GRF_r2 = r2_score(y_test, predictions_GRF)
#Calculating RMSE
GRF_rmse = np.sqrt(mean_squared_error(y_test,predictions_GRF))

print('Grid Search CV Random Forest Regressor Model Performance:')
print('Best Parameters = ',view_best_params_GRF)
print('R-squared = {:0.2}.'.format(GRF_r2))
print('RMSE = ',(GRF_rmse))


# In[116]:


## Grid Search CV for gradinet boosting
gb = GradientBoostingRegressor(random_state = 0)
n_estimator = list(range(11,20,1))
depth = list(range(5,15,2))

# Create the grid
grid_search = {'n_estimators': n_estimator,
               'max_depth': depth}

## Grid Search Cross-Validation with 5 fold CV
gridcv_gb = GridSearchCV(gb, param_grid = grid_search, cv = 5)
gridcv_gb = gridcv_gb.fit(X_train,y_train)
view_best_params_Ggb = gridcv_gb.best_params_

#Apply model on test data
predictions_Ggb = gridcv_gb.predict(X_test)

#R^2
Ggb_r2 = r2_score(y_test, predictions_Ggb)
#Calculating RMSE
Ggb_rmse = np.sqrt(mean_squared_error(y_test,predictions_Ggb))

print('Grid Search CV Gradient Boosting regression Model Performance:')
print('Best Parameters = ',view_best_params_Ggb)
print('R-squared = {:0.2}.'.format(Ggb_r2))
print('RMSE = ',(Ggb_rmse))


# # Prediction of fare from provided test dataset :

# We have already cleaned and processed our test dataset along with our training dataset. Hence we will be predicting using grid search CV for random forest model

# In[118]:



## Grid Search CV for random Forest model
regr = RandomForestRegressor(random_state = 0)
n_estimator = list(range(11,20,1))
depth = list(range(5,15,2))

# Create the grid
grid_search = {'n_estimators': n_estimator,
             'max_depth': depth}

## Grid Search Cross-Validation with 5 fold CV
gridcv_rf = GridSearchCV(regr, param_grid = grid_search, cv = 5)
gridcv_rf = gridcv_rf.fit(X_train,y_train)
view_best_params_GRF = gridcv_rf.best_params_

#Apply model on test data
predictions_GRF_test_Df = gridcv_rf.predict(test)


# In[119]:


predictions_GRF_test_Df


# In[121]:


test['Predicted_fare'] = predictions_GRF_test_Df


# In[123]:


test.head()


# In[ ]:


test_df.to_csv('test_df.csv')

