import json
import pickle

import numpy as np
import pandas as pd

from sklearn import metrics

def load_future_wind_speed():
    # load data from knmi
    df_knmi = pd.read_csv('https://cdn.knmi.nl/knmi/map/page/klimatologie/gegevens/daggegevens/etmgeg_269.zip', 
                    na_values='     ', 
                    header=44, 
                    parse_dates=['YYYYMMDD']).rename(columns=str.strip)

    # rename columns
    df_knmi.rename(columns={'YYYYMMDD': 'ds', 'FG': 'y'}, inplace=True)
    # keep only timestamp and wind speed
    df_knmi = df_knmi[['ds', 'y']]
    # devide wind speed by 10
    df_knmi['y'] = df_knmi['y'] / 10

    return df_knmi.loc[(df_knmi['ds'] > '2020-12-31') & (df_knmi['ds'] < '2021-04-15')]

def load_y_true_pred_df(forecast_wind):
    df_knmi = load_future_wind_speed()
    df = pd.concat([df_knmi, forecast_wind.loc[(forecast_wind['ds'] > '2020-12-31') & (forecast_wind['ds'] < '2021-04-15'), ['yhat', 'yhat_lower', 'yhat_upper']]], axis=1)
    return df.rename(columns={'y': 'y_true', 'yhat': 'y_pred'})

# load in trained model
m_wind = pickle.load(open('./assets/models/prophet.pickle', mode='rb'))

# load in data
df_future = pd.read_csv('./assets/predictions/wind_forecast.csv')
df_ytrue_ypred = load_y_true_pred_df(df_future)

print(df_ytrue_ypred.tail())

mean_squared_error = metrics.mean_squared_error(df_ytrue_ypred['y_true'], df_ytrue_ypred['y_pred'])
print(f'MSE: {mean_squared_error}')

root_mean_squared_error = metrics.mean_squared_error(df_ytrue_ypred['y_true'], df_ytrue_ypred['y_pred'], squared=False)
print(f'RMSE: {root_mean_squared_error}')

mean_error = np.mean(np.abs(df_ytrue_ypred['y_true']-df_ytrue_ypred['y_pred']))
print(f'ME: {mean_error}')

r2_score = metrics.r2_score(df_ytrue_ypred['y_true'], df_ytrue_ypred['y_pred'])
print(f'R2: {r2_score}')

in_range = df_ytrue_ypred.loc[(df_ytrue_ypred['y_true'] >= df_ytrue_ypred['yhat_lower']) & (df_ytrue_ypred['y_true'] <= df_ytrue_ypred['yhat_upper']), 'y_true'].count()
total = df_ytrue_ypred.shape[0]
points_in_range = in_range/total

print(f'The percentage of points in the lower/upper limit is: {points_in_range}')


with open('./assets/metrics/metrics.json', mode='w') as output_file:
    json.dump(
        dict(
            mse=mean_squared_error,
            rmse=root_mean_squared_error,
            me=mean_error,
            r2=r2_score,
            points_in_range=points_in_range
        ),
        output_file
    )