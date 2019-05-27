#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 11:31:09 2018

@author: inderjeetsingh
"""

import numpy as np 
import pandas as pd 
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

train= pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

joined = train.append(test,ignore_index=True)
print(joined.info())
print(train.isnull().values.any())
print(test.isnull().values.any())
print(train.describe())

# Root mean square log error function is used for evaluation.
def rmsle(y, y_,convertExp=True):
    if convertExp:
        y = np.exp(y),
        y_ = np.exp(y_)
    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))
    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))
    calc = (log1 - log2) ** 2
    return np.sqrt(np.mean(calc))

# Generating month,hours and weekday columns from time series
    
joined['Month'] = joined['datetime'].map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').month)
joined['Hour'] = joined['datetime'].map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').hour)
joined['Weekday'] = joined['datetime'].map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').weekday())


datetimecol = test["datetime"]

joined.drop('datetime',inplace=True,axis=1)
train = joined[ joined['count'].notnull()]
test = joined[ joined['count'].isnull()]


# Visualizations, inspired by  Vivek Srinivasan's kernal.

fig,axes= plt.subplots(2,2,figsize=(20,20))
sns.boxplot(data=train, y='count',x='weather',ax=axes[0,0])
sns.boxplot(data=train, y='count',x='season',ax=axes[0,1])
sns.boxplot(data=train, y='count',x='Weekday',ax=axes[1,0])
sns.boxplot(data=train, y='count',x='Month',ax=axes[1,1])
plt.savefig('Boxplot.png',dpi=900)
plt.show()
Month_grouped = train[['count','Month']].groupby('Month',as_index=False).mean()
Hour_season_grouped = (train.groupby(["Hour","season"],as_index=False)["count"].mean())
Hour_Weekday_grouped = (train.groupby(["Hour","Weekday"],as_index=False)["count"].mean())
Hour_Month_grouped = (train.groupby(["Hour","Month"],as_index=False)["count"].mean())
Hour_weather_grouped = (train.groupby(["Hour","weather"],as_index=False)["count"].mean())

fig,axes= plt.subplots(5,1,figsize=(30,20))
sns.barplot( x = Month_grouped['Month'], y = Month_grouped['count'], ax=axes[0])
sns.pointplot( y = Hour_season_grouped['count'], x = Hour_season_grouped['Hour'],  hue=Hour_season_grouped['season'], join=True,ax=axes[1])
sns.pointplot( y = Hour_Weekday_grouped['count'], x = Hour_Weekday_grouped['Hour'],  hue=Hour_Weekday_grouped['Weekday'], join=True,ax=axes[2])
sns.pointplot( y = Hour_Month_grouped['count'], x = Hour_Month_grouped['Hour'],  hue=Hour_Month_grouped['Month'], join=True,ax=axes[3])
sns.pointplot( y = Hour_weather_grouped['count'], x = Hour_weather_grouped['Hour'],  hue=Hour_weather_grouped['weather'], join=True,ax=axes[4])
plt.savefig('barplot-pointplot.png',dpi=900)
plt.show()

plt.figure()
sns.heatmap(data=train.corr(), linewidths=2)
plt.savefig('heatmap.png',dpi=900)
plt.show()

""" Removing Outliers """
train['humidity']=train['humidity'].replace([0],61) 
train['windspeed']=train['windspeed'].replace([56.9969],20)

# As casual+register = count and atemp is highly correlated with temp so we will drop them all :P
test.drop(['casual','registered','count','atemp'],inplace=True,axis=1)
train.drop(['casual','registered','atemp'],inplace=True,axis=1)
X = train.drop('count',axis=1)
y = train['count']

# "train_test_split"
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)



# Gradient Boosted Regression
gbreg = GradientBoostingRegressor()
param_grid = {'n_estimators': [1000,2500],'learning_rate':[0.5,0.3,0.2,0.1,0.01,0.001,0.0001]}
CV_gbreg = GridSearchCV(estimator=gbreg,param_grid=param_grid,cv=5)
CV_gbreg.fit(X=X_train,y=np.log1p(y_train))
print("Best Paramters for Gradient boosted regression: ",CV_gbreg.best_params_)
preds = CV_gbreg.predict(X=X_test)
print ("RMSLE Value for Gradient Boosted Regression: ",  rmsle( np.exp([max(x,0) for x in preds]) ,y_test,False))
predictions = CV_gbreg.predict(test)
for i in range(len(predictions)):
    if predictions[i]<=0:
        predictions[i]=0
pd.DataFrame({'Datetime': datetimecol, 'count': predictions}).to_csv("bike_gbr.csv", index=False)


# Lasso Regression

lasso = Lasso()
alpha = [0.001,0.005,0.01,0.3,0.1,0.3,0.5,0.7]
lasso_param_grid = {'alpha':alpha, 'max_iter':[1000]}
grid_lasso = GridSearchCV(lasso, lasso_param_grid, cv=5)
grid_lasso.fit(X=X_train,y=np.log1p(y_train))
preds = grid_lasso.predict(X=X_test)
print ("Best Paramters for Lasso Regression: ",grid_lasso.best_params_)
print ("RMSLE Value for Lasso regression: ",  rmsle( np.exp([max(x,0) for x in preds]) ,y_test,False))
predictions = grid_lasso.predict(test)
for i in range(len(predictions)):
    if predictions[i]<=0:
        predictions[i]=0
pd.DataFrame({'Datetime': datetimecol, 'count': predictions}).to_csv("bike_lasso.csv", index=False)


# Random Forest Regression

rfreg = RandomForestRegressor()
param_grid = {'n_estimators': np.arange(495,500)}
grid_random_forest = GridSearchCV(rfreg, param_grid, cv=5)
grid_random_forest.fit(X=X_train, y= np.log1p(y_train))
preds = grid_random_forest.predict(X=X_test)
print("Best Paramters for Random Forest Regression: ", grid_random_forest.best_params_)
print("RMSLE Value for Random forest regression: ",  rmsle( np.exp([max(x,0) for x in preds]) ,y_test,False))
predictions = grid_random_forest.predict(test)
for i in range(len(predictions)):
    if predictions[i]<=0:
        predictions[i]=0
pd.DataFrame({'Datetime': datetimecol, 'count': predictions}).to_csv("bike_rfreg.csv", index=False)


# Ridge Regression

ridge = Ridge()
ridge_params = {'max_iter':[3000],'alpha':[0.1,1,2,3,4,10,30,100,200,300,400,800,900,1000]}
grid_ridge = GridSearchCV(ridge, ridge_params, cv=5)
grid_ridge.fit(X_train, np.log1p(y_train))
preds = grid_ridge.predict(X=X_test)
print ("Best Paramters for Ridge Regression: ",grid_ridge.best_params_)
print ("RMSLE Value For Ridge Regression: ",rmsle( np.exp([max(x,0) for x in preds]) ,y_test,False))
predictions = grid_ridge.predict(test)
for i in range(len(predictions)):
    if predictions[i]<=0:
        predictions[i]=0
pd.DataFrame({'Datetime': datetimecol, 'count': predictions}).to_csv("bike_ridge.csv", index=False)


# Linear Regression

linreg = LinearRegression()
linreg.fit(X=X_train, y=np.log1p(y_train))
y_pred = linreg.predict(X=X_test)
print ("RMSLE Value for linear regression: ",  rmsle( np.exp([max(x,0) for x in preds]) ,y_test,False))
predictions = linreg.predict(test)
for i in range(len(predictions)):
    if predictions[i]<=0:
        predictions[i]=0
pd.DataFrame({'Datetime': datetimecol, 'count': predictions}).to_csv("bike_linreg.csv", index=False)
















