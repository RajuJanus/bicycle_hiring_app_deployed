# Import Libraries

import pandas as pd
import numpy as np
from datetime import datetime as dt, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import holidays
from prophet import Prophet
import joblib


import streamlit as st

header = st.container()

features = st.container()
dataset = st.container()

# Read the dataset

bikehiring_from_2010  =  pd.read_excel('data/streamlit_bikehiring.xlsx', engine='openpyxl')

 

# Read the coordinates file for the stations and load the trained model

coordinates = pd.read_csv('data/start_location_coordinates.csv')
m= joblib.load('data/my_model')


# Make first header and give information to the users

with header:
    st.title("Time series analysis and predictions on Bike Hiring in London(UK)")
    st.text(
    """
     Do you want to hire a bike in London?
     Let's be honest London is a crowded city! 
     Whether you are a tourist or even if you want to start a bike rental business in London, 
     then you are at the right place!
 
     You will get predictions about hired bicycle traffic in the city!
    
     How do I make a prediction?
     I use a machine learning algorithm called facebook Prophet for time-series analysis and prediction.
    
     In this project, we will also analyze the data about trends of bicycle traffic in London by using other python libraries.
     We will explore seasonality patterns.
     And most importantly, you will know how many bicycles might be on the streets in future dates/months/years, etc.
    
    """)
   
   
  # Ask user input and provide results
   
    st.subheader('How many bicycle will be hired tomorrow? ')
    d = dt.today() + timedelta(days=1)

    future_date = pd.DataFrame({'ds':[d]})
    forcast_date = m.predict(future_date)
    forcast_to_show = forcast_date[['ds','yhat', 'yhat_upper', 'yhat_lower']]
    forcast_to_show.yhat = int(round(forcast_to_show.yhat))
    forcast_to_show.yhat_upper = int(round(forcast_to_show.yhat_upper,0))
    forcast_to_show.yhat_lower = int(round(forcast_to_show.yhat_lower,0))
    forcast_to_show.ds = forcast_to_show.ds.dt.date
    forcast_to_show.columns = ['Date', 'Expected Number of hires', 'Upper limit of hires', 'Lower limit of hires']
   
    st.write(forcast_to_show)


    st.subheader('Do you want to know predition fora specific date? ')
    d = st.date_input("""Select a date and see the prediction in the table below""")

    future_date = pd.DataFrame({'ds':[d]})

    forcast_date = m.predict(future_date)
    forcast_to_show = forcast_date[['ds','yhat', 'yhat_upper', 'yhat_lower']]
    forcast_to_show.yhat = int(round(forcast_to_show.yhat,0))
    forcast_to_show.yhat_upper = int(round(forcast_to_show.yhat_upper,0))
    forcast_to_show.yhat_lower = int(round(forcast_to_show.yhat_lower,0))
    forcast_to_show.ds = forcast_to_show.ds.dt.date
    forcast_to_show.columns = ['Your selected date', 'Expected Number of hires', 'Upper limit of hires', 'Lower limit of hires']
   
    st.write(forcast_to_show)
    
    st.subheader('Select a range of days for prediction')
    sel_col, disp_col = st.columns(2)
    max_depth = sel_col.slider('Select the range of prediction starting from 2019-12-31', min_value = 365, max_value = 365*10, value =365)
    
    future = m.make_future_dataframe(periods = max_depth, freq = 'D')
    st.text("""
     In the graph: Black dots are true values (number of hired bicycles on that day, from available data)
     Deep blue line: Prediction by the model
     Light blue regions indicates predicted upper and lower values.   
     
     """)
    forcast = m.predict(future)
    plot1 = m.plot(forcast)
    st.write(plot1)

# Show users some features

with features:
    st.header("Let's see some features")

    st.subheader('Location of the all bicycle stations')
    
    st.map(coordinates)
    st.subheader('Yearly Bicycle Hire from 2011')

    yearly_bike_hiring= pd.DataFrame(bikehiring_from_2010.groupby(pd.Grouper(key="ds", freq="Y")).sum())
    st.line_chart(yearly_bike_hiring.y)

    st.subheader('Average Monthly Bike Hire from 2011 to Present')
    Monthly_hire = pd.DataFrame(bikehiring_from_2010.groupby(bikehiring_from_2010['ds'].dt.strftime('%m %B'))['y'].mean().round(0).sort_index())

    st.bar_chart(Monthly_hire)
    st.subheader('Monthly Average Temperature in London( in Degrees)')

    Monthly_average_temperature = pd.DataFrame(bikehiring_from_2010.groupby(bikehiring_from_2010['ds'].dt.strftime('%m %B'))['Temperature'].mean().round(0).sort_index())

    st.bar_chart(Monthly_average_temperature)

# Give users information about the dataset
    
with dataset:

    st.header("Information about the dataset")
    st.text("""
    The  dataset can be found in London data store 
    (https://data.london.gov.uk/dataset/number-bicycle-hires) 
    which contains the daily number of bicycle hire from 2010 to present. 
    I also added temperature profile of London on those dates which can be found in Kaggle 
    (https://www.kaggle.com/sudalairajkumar/daily-temperature-of-major-cities)

    To get the coordinates of the bicycle stations, I used the publicly available dataset 'London Bicycle Hire' in Bigquery.

    (https://console.cloud.google.com/marketplace/product/greater-london-authority/london-bicycles?project=fit-shift-332509) 
    This data contains the number of hires and location information  about the
     London's Santander Cycle Hire Scheme from 01/2015 to 06/2017.

    Have a look on the dataset here.


    """)
    st.write(bikehiring_from_2010.head())

# This section is for cross validation. I commented out this to make the things simple. Future I can add this feature too.
    
#     st.subheader('We can have a look on how good is our model in prediction with cross validation')
#     st.text("""We plot our Mean absolute error by training the model for two years
#      and seeing the predition of next 365 days""")

#     from prophet.diagnostics import cross_validation, performance_metrics
#     cv_results = cross_validation(model = m, initial= '731 days', horizon = '365 days')
#     performance = performance_metrics(cv_results)
#     from prophet.plot import plot_cross_validation_metric
#     fig3 = plot_cross_validation_metric(cv_results, metric = 'mape')
#     st.write(fig3)
