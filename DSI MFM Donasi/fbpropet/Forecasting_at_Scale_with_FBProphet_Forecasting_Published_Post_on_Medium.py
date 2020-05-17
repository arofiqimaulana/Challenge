#!/usr/bin/env python
# coding: utf-8

# # Import Library

# In[ ]:


# If this is your first time of using FBProphet
# run this block
get_ipython().system('pip install fbprophet')
get_ipython().system('pip install plotly --upgrade')
get_ipython().system('pip install chart-studio')


# In[ ]:


def configure_plotly_browser_state():
  import IPython
  display(IPython.core.display.HTML('''
        <script src="/static/components/requirejs/require.js"></script>
        <script>
          requirejs.config({
            paths: {
              base: '/static/base',
              plotly: 'https://cdn.plot.ly/plotly-1.5.1.min.js?noext',
            },
          });
        </script>
        '''))


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
from scipy import stats

from fbprophet import Prophet
import logging
logging.getLogger().setLevel(logging.ERROR)

import matplotlib.pyplot as plt
import datetime as dt
import io

import plotly as py
from plotly.offline import iplot, init_notebook_mode
import plotly.graph_objs as go

from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[159]:


# import files
# in this case we can upload the data file here
from google.colab import files
uploaded = files.upload()


# In[ ]:


# read uploaded file
medium_posts_df = pd.read_csv(io.BytesIO(uploaded['medium_posts.csv']), delimiter='\t')


# In[ ]:


# count number of missing values for each field
def missing_values_table(df): 
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        return mis_val_table_ren_columns 


# In[162]:


print(missing_values_table(medium_posts_df))


# In[ ]:


# drop all the columns beside these two columns that we need it later
# remove duplicates posts
medium_posts_df = medium_posts_df[['published', 'url']].dropna().drop_duplicates()

# change published into datetime, so we can use it for FBProphet
medium_posts_df['published'] = pd.to_datetime(medium_posts_df['published']).dt.tz_localize(None)


# In[164]:


# check top 5 data
medium_posts_df.sort_values(by=['published']).head(5)


# There is no way for medium post which published on 1970, since we know that public release year for Medium was in 2012 (August 15th 2012)

# In[165]:


# Let's check the max date for published date
medium_posts_df.published.max()


# In[166]:


# Slice the date from 2012-08-15 to 2017-06-27
medium_posts_df = medium_posts_df[(medium_posts_df['published'] >= '2012-08-15') & 
                                  (medium_posts_df['published'] < '2017-06-26')].sort_values(by=['published'])
medium_posts_df.head(5)


# In[167]:


# Since the level of granularity for the data is on timestamp level
# We'd like to change it into daily level, to make it easier to create the model
# to predict daily level of data

medium_posts_daily_df = medium_posts_df.groupby('published')[['url']].count()
medium_posts_daily_df.columns = ['count_posts']
medium_posts_daily_df.head()

# resample the data into daily 1 day bins :

medium_posts_daily_df = medium_posts_daily_df.resample('D').apply(sum)
medium_posts_daily_df.head(5)


# # Exploratory Data Analysis

# In[ ]:


# custom function to visualize
def viz_df(df, title=''):
    common_kw = dict(x=df.index, mode='lines')
    data = [go.Scatter(y=df[c], name=c, **common_kw) for c in df.columns]
    layout = dict(title=title)
    fig = dict(data=data, layout=layout)
    iplot(fig, show_link=False)


# In[169]:


configure_plotly_browser_state()
init_notebook_mode(connected=False)

viz_df(medium_posts_daily_df, 'Daily Post in Medium August 2012 - June 2017')


# Let's take a look on weekly level of the data
# to grasp better quality of trend

# In[170]:


medium_posts_weekly_df = medium_posts_daily_df.resample('W').apply(sum)
medium_posts_weekly_df.tail(5)


# In[63]:


configure_plotly_browser_state()
init_notebook_mode(connected=False)

# We'd like to get better understand about the trend and seasonality
# As on this part we will try to plot the data on weekly level
viz_df(medium_posts_weekly_df, 'Weekly Post in Medium, August 2012 - June 2017')


# Since the initial data from 2012 up to end of 2014 it has very low number of post in daily, so we will ommit the date up to initial of 2015

# In[171]:


medium_posts_daily_df = medium_posts_daily_df.loc[medium_posts_daily_df.index >= '2015-01-01']
medium_posts_daily_df.head(5)


# # Modeling Part

# We arrive at modeling part. But before get in touch with the model, FBProphet requires us to transform the data into its format
# 
# ```
# ds -> (date / datetime / timestamp) 
# y -> is numeric value that we will try to forecast / predict
# ```
# so we need to rename our existing columns into ds and y
# 
# There will be some of the parameters of FBProphet that I'd like to introduce to you guys, I won't cover all of them since it needs more time, I will cover most used parameters on FBProphet ;)
# 
# ```
# *   Which growth that we will use (linear or logistics)
# *   Which model that we will use (additive or multiplicative)
# *   Changepoint (Automatatic and Manual approach)
# *   Trend Flexibility (How to adjust the flexibility)
# *   Specifying custom seasonalities
# *   Seasonality Prior Scale (L1 Regularization Rate)
# *   Uncertainty intervals (in trend)
# *   Forecast components
# ```
# 
# 

# In[172]:


# Change the columns name format as required by FBProphet
# Refer to the explanation above

medium_model_df = medium_posts_daily_df.reset_index()
medium_model_df.columns = ['ds', 'y']
medium_model_df.tail()


# In[173]:


# We will split the dataset to measure the performance later
n_pred = 30
train_medium_df = medium_model_df[:-n_pred]
train_medium_df.tail(5)


# By default Prophet model will use these parameters
# ```
# Prophet(growth='linear', 
#         changepoints=None, 
#         n_changepoints=25, 
#         changepoint_range=0.8, 
#         yearly_seasonality='auto', 
#         weekly_seasonality='auto', 
#         daily_seasonality='auto', 
#         holidays=None, 
#         seasonality_mode='additive', 
#         seasonality_prior_scale=10.0, 
#         holidays_prior_scale=10.0, 
#         changepoint_prior_scale=0.05, 
#         mcmc_samples=0, 
#         interval_width=0.8, 
#         uncertainty_samples=1000, 
#         stan_backend=None)
# ```
# 

# In[174]:


# Instatiate the Prophet model object
# Fit the training data into prophet model
# First attempt we will use default parameters

medium_prophet = Prophet()
medium_prophet.fit(train_medium_df)


# In[175]:


# This method of make_future_dataframe will helps us to create N size of date
medium_future = medium_prophet.make_future_dataframe(periods=n_pred)
medium_future.tail(5)


# In[176]:


medium_forecast = medium_prophet.predict(medium_future)
medium_forecast.tail(5)


# By default, Prophet will use additive model instead of multiplicative model, so you will find the values of multiplicative are zero
# 
# You will find the predicted / forecasted value on **yhat** column and alos you can find the **upper and bottom (interval width)** for each weekly, yearly seasonalities

# In[177]:


medium_prophet.plot(medium_forecast)


# In[178]:


# We want to observe how's the component
# for each level of forecast
medium_prophet.plot_components(medium_forecast)


# # Model Performance

# In[ ]:


def result_comparison(historical, forecast):
  """
  Joining our actual data and forecasted data
  to see the gap between those two
  """
  return forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']].join(historical.set_index('ds'))


# In[180]:


medium_comparison_df = result_comparison(medium_model_df, medium_forecast)
medium_comparison_df.tail(5)


# In[181]:


configure_plotly_browser_state()
init_notebook_mode(connected=False)

py.offline.iplot([
    go.Scatter(x=train_medium_df['ds'], y=train_medium_df['y'], name='y'),
    go.Scatter(x=medium_forecast['ds'], y=medium_forecast['yhat'], name='yhat'),
    go.Scatter(x=medium_forecast['ds'], y=medium_forecast['yhat_upper'], fill='tonexty', mode='none', name='upper'),
    go.Scatter(x=medium_forecast['ds'], y=medium_forecast['yhat_lower'], fill='tonexty', mode='none', name='lower')
])


# As shown on the above graph, it looks that the model is not performing that well
# huge gap between actual data and forecasted data on the initial 2015 till Q4 2015, as the model going further huge gap can be found again on the initial of 2017. Despite of that, still Prophet can learn the trend from the data

# In[182]:


print("MAE yhat\t: {}\nMAE yhat_lower: {}\nMAE yhat_upper: {}".format(
    mean_absolute_error(medium_comparison_df['y'].values, medium_comparison_df['yhat']),
    mean_absolute_error(medium_comparison_df['y'].values, medium_comparison_df['yhat_lower']),
    mean_absolute_error(medium_comparison_df['y'].values, medium_comparison_df['yhat_upper'])))


# # Model Tuning (Parameter)

# In[ ]:


medium_train_df_2 = train_medium_df.copy().reset_index()


# In[ ]:


# by default prophet would create changepoints for 25 datapoints
# within range of 80% of the initial datapoints
# so in this section we would try to create our own changepoints
# we will create changepoints by defining all the datapoints outside IQR as chagepoints

def create_changepoints(df):
  q1 = df['y'].quantile(0.25)
  q3 = df['y'].quantile(0.75)
  iqr = q3 - q1
  return df[(df['y'] < q1 - (1.5 * iqr)) | (df['y'] > q3 + (1.5 * iqr))]


# In[185]:


# print the changepoints
changepoints_df = create_changepoints(medium_train_df_2)
changepoints_df.head()


# In[186]:


configure_plotly_browser_state()
init_notebook_mode(connected=False)

# Create a trace
trace = go.Scatter(
    x = medium_train_df_2['ds'],
    y = medium_train_df_2['y'],
    mode = 'lines',
    name = 'actual data'
)
trace_cp = go.Scatter(
    x = changepoints_df['ds'],
    y = changepoints_df['y'],
    mode = 'markers',
    name = 'changepoint'
)

data = [trace,trace_cp]
fig = go.Figure(data=data)
py.offline.iplot(fig)


# As shown on the above graph, we plot which datapoints that
# we considered changepoints compared to actual data
# this is only simple method, if you have good approach for creating the changepoints you can try your approach ;)
# 
# To create the changepoints, you can event assign it manually by creating a Series or DataFrame for which date that you consider as changepoints

# In[187]:


# creating new model of Prophet
medium_prophet_2 = Prophet(growth= 'linear', 
                           seasonality_mode = 'multiplicative',
                           changepoints= changepoints_df['ds'],
                           daily_seasonality= False,
                           weekly_seasonality= False,
                           yearly_seasonality= False).add_seasonality(
                               name='monthly',
                               period=30.5,
                               fourier_order=15,
                               prior_scale=15
                           ).add_seasonality(
                               name='weekly',
                               period=7,
                               fourier_order=10,
                               prior_scale=20
                           ).add_seasonality(
                               name='yearly',
                               period=365.25,
                               fourier_order=20
                           ).add_seasonality(
                               name='quarterly',
                               period=365.25/4,
                               fourier_order=5,
                               prior_scale=15
                           )
medium_prophet_2.fit(medium_train_df_2)


# In[188]:


# create future dataframe with N size of datapoints
medium_future_2 = medium_prophet_2.make_future_dataframe(periods=n_pred)
medium_future_2.tail(3)


# In[189]:


# predicting
medium_forecast_2 = medium_prophet_2.predict(medium_future_2)
medium_forecast_2.tail(5)


# # Model Performance (after hyper parameters tuning)

# In[190]:


# stitch the predicted value with actual value 
# after we made a lil bit of changes on the parameters
medium_comparison_df_2 = result_comparison(medium_model_df, medium_forecast_2)
medium_comparison_df_2.tail(5)


# In[219]:


# Print the evaluation metrics for the predicted value
print("MAE yhat\t: {}".format(
    mean_absolute_error(medium_comparison_df_2['y'].values, medium_comparison_df_2['yhat'])))


# After we tune our model with a set of parameters
# We get better performance compared to before adjust the parameters

# # Transformation (Box Cox)

# In this part we will try to transform the value using Box Cox
# So idea is to make our datapoints normalized and less of noise
# Eventhough FBProphet is good at handling noises, outlier, etc
# having your custom solution over the model is not that bad tho

# In[ ]:


# create function to inverse the transformed value
# we will use this function later to inverse the value
# after transformed using Box Cox

def inverse_box_cox(y, lambda_):
  return np.exp(y) if lambda_ == 0 else np.exp(np.log(lambda_ * y + 1) / lambda_)


# In[193]:


# tranformed the copied dataset on the section 2
boxcox_medium_df = medium_train_df_2.copy().set_index('ds')
boxcox_medium_df.head(3)


# In[ ]:


# BoxCox transformed data
boxcox_medium_df['y'], lambda_prophet = stats.boxcox(boxcox_medium_df['y'])
boxcox_medium_df.reset_index(inplace=True)


# In[210]:


print(missing_values_table(boxcox_medium_df))


# # New Model with Transformed Data

# In[212]:


medium_prophet_3 = Prophet(growth= 'linear', 
                           seasonality_mode = 'multiplicative',
                           changepoints= changepoints_df['ds'],
                           daily_seasonality= False,
                           weekly_seasonality= False,
                           yearly_seasonality= False).add_seasonality(
                               name='monthly',
                               period=30.5,
                               fourier_order=15,
                               prior_scale=15
                           ).add_seasonality(
                               name='weekly',
                               period=7,
                               fourier_order=10,
                               prior_scale=20
                           ).add_seasonality(
                               name='yearly',
                               period=365.25,
                               fourier_order=20
                           ).add_seasonality(
                               name='quarterly',
                               period=365.25/4,
                               fourier_order=5,
                               prior_scale=15
                           )
medium_prophet_3.fit(boxcox_medium_df)


# In[213]:


# create future dataframe with N size of datapoints
medium_future_3 = medium_prophet_3.make_future_dataframe(periods=n_pred)
medium_future_3.tail(3)


# In[214]:


# predicting
medium_forecast_3 = medium_prophet_3.predict(medium_future_3)
medium_forecast_3.tail(5)


# In[ ]:


# in this part we will inverse the boxcox value into the original value
# using lambda value that we have instatiated before
for column in ['yhat', 'yhat_lower', 'yhat_upper']:
    medium_forecast_3[column] = inverse_box_cox(medium_forecast_3[column], 
                                               lambda_prophet)


# # Model Performances after Transforming 

# In[220]:


medium_comparison_df_3 = result_comparison(medium_model_df, medium_forecast_3)
medium_comparison_df_3.head(3)


# In[221]:


# Print the evaluation metrics for the predicted value
print("MAE yhat\t: {}".format(
    mean_absolute_error(medium_comparison_df_3['y'].values, medium_comparison_df_3['yhat'])))


# It's decreased a lil bit after we applied the BoxCox transformation

# In[229]:


configure_plotly_browser_state()
init_notebook_mode(connected=False)

"""
This graph shows us actual data for the daily Medium Posts
"""

py.offline.iplot([
    go.Scatter(x=train_medium_df['ds'], y=train_medium_df['y'], name='y'),
])


# In[228]:


configure_plotly_browser_state()
init_notebook_mode(connected=False)

"""
This graph shows us actual data for the daily Medium Posts
and the predicted / forecasted value using FBProphet after applying
BoxCox Transformation and tuning the parameters
"""

py.offline.iplot([
    go.Scatter(x=train_medium_df['ds'], y=train_medium_df['y'], name='y'),
    go.Scatter(x=medium_forecast_3['ds'], y=medium_forecast_3['yhat'], name='yhat'),
    go.Scatter(x=medium_forecast_3['ds'], y=medium_forecast_3['yhat_upper'], fill='tonexty', mode='none', name='upper'),
    go.Scatter(x=medium_forecast_3['ds'], y=medium_forecast_3['yhat_lower'], fill='tonexty', mode='none', name='lower')
])


# # Hack Session

# After learning together how to use FBProphet on Python
# Let's compete each other to find hyper parameter for the model
# The FBProphet Developer created this package to allow us doing
# black box on the model, so we can tune the model explicitly
