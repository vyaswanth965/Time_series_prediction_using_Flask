#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 15:22:22 2019

@author: vudumula
"""

import os
from flask import jsonify
import pandas as pd
from fbprophet import Prophet
import numpy as np 
from datetime import datetime
import pickle
from flask import Flask, render_template, request
app = Flask(__name__)


def fill_in_missing_dates(df,idx, date_col_name , fill_value = 0):

    df.set_index(date_col_name,drop=True,inplace=True)
    df.index = pd.DatetimeIndex(df.index)

    df = df.reindex(idx,fill_value=fill_value)
    df[date_col_name] = pd.DatetimeIndex(df.index)

    return df

df_original = pd.DataFrame()

for file in os.listdir('./data'):
   dataFrame = pd.read_csv('{}/{}'.format('./data',file))
   df_original = df_original.append(dataFrame)


df_original['csv_scheddate'] = pd.to_datetime(df_original['csv_scheddate'], format='%m/%d/%y')
df_original['VisitCount'] = pd.to_numeric(df_original['VisitCount'])

@app.route('/')
def home():
   return render_template('home.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':

      visit_id = request.form['visit_id']      
      start_date = datetime.strptime(request.form['start_date'], '%Y/%m/%d')
      end_date = datetime.strptime(request.form['end_date'], '%Y/%m/%d')
      #options = request.form['options']
      #print(options)

      if start_date>end_date:
         return 'start_date should be less than end_date'
      
      df = df_original[df_original['sc_code'] == visit_id]
      df.drop(["sc_code"],axis = 1,inplace = True)

      future = pd.date_range(start_date, end_date).to_frame()
      future = future.rename(columns={0: 'ds'})


      Models_abs_path = os.path.abspath('./Models')
      filename = Models_abs_path+'/'+visit_id+'.sav'

      if  os.path.exists(filename):
         model = pickle.load(open(filename, 'rb'))
      else:
         if len(df)>900:

            train_df = df.rename(columns={ 'csv_scheddate':'ds','VisitCount':'y'})
            idx = pd.date_range(df_original.csv_scheddate.min(),df_original.csv_scheddate.max())
            train_df = fill_in_missing_dates(train_df,idx,'ds',train_df.y.mean())
            train_df = train_df[:-30]

            model = Prophet(seasonality_mode="multiplicative",\
                  daily_seasonality=False,weekly_seasonality=False,\
                  yearly_seasonality=False).add_seasonality(name="monthly",period=30.5,fourier_order=20)\
                  .add_seasonality(name='daily',period=1,fourier_order=20).\
                  add_seasonality(name='weekly',period=7,fourier_order=20)\
                  .add_seasonality(name='yearly',period=365.25,fourier_order=20).\
                  add_seasonality(name="quarterly",period=365.25/4,fourier_order=20)   
            model.add_country_holidays(country_name='US')

            model.fit(train_df)

            if not os.path.exists(Models_abs_path):
               os.makedirs(Models_abs_path)

            pickle.dump(model, open(filename, 'wb'))
         else:
            return 'no sufficient records to train model for this visit code'   


      forecast = model.predict(future)
      forecast['yhat'] = np.round(np.where(forecast.yhat > 0, forecast.yhat, forecast.yhat_upper))
      forecast = forecast.iloc[:,[0,-1]]
      forecast = forecast.rename(columns={ 'ds':'date','yhat':'predicted_count'})

      if start_date > df.csv_scheddate.max():

         return render_template('result.html',  tables=[forecast.to_html(classes='data')], titles=forecast.columns.values)         

      else:

         df = df.rename(columns={'csv_scheddate': 'date', 'VisitCount': 'actual_count'}) 
         forecast = pd.merge(forecast,  df,on='date', how='left')

         forecast['error'] = forecast['actual_count'] - forecast['predicted_count']
         forecast['error_percentage'] = 100 * forecast['error'] / forecast['actual_count']
         error_mean = lambda error_name: np.mean(np.abs(forecast[error_name]))
         MAPE = str(error_mean('error_percentage'))
         MAE = str(error_mean('error'))

         return render_template('result.html',  tables=[forecast.to_html(classes='data')], titles=['MAE : '+MAE, 'MAPE : '+MAPE])


if __name__ == '__main__':
   app.run(host=os.getenv('IP', '0.0.0.0'), 
            port=int(os.getenv('PORT', 4000)))       
        