import pickle

import pandas as pd

from fbprophet import Prophet

from config import Config

# encode month to season
def add_season_encoding(ds):
    month = ds.month 
    if month == 12 or month == 1 or month == 2:
        return 3
    if month == 3 or month == 4 or month == 5:
        return 1
    if month == 6 or month == 7 or month == 8:
        return 2
    if month == 9 or month == 10 or month == 11:
        return 0

# main dataset
df = pd.read_csv(f'{Config.DATASET_PATH}/lelystad_final_features_numerical.csv', parse_dates=['YYYYMMDD'])

# encode the season to numerical values
df = pd.concat([df, pd.get_dummies(df['Season'])], axis=1)
df.drop(['Season'], axis=1, inplace=True)

print(df.tail())

# Pressure and Evaporation
seasons = ['Autumn', 'Spring', 'Summer', 'Winter']

# create dataframes for forecasting pressure and evaporation
df_ds_pressure = df[['YYYYMMDD', 'Autumn', 'Spring', 'Summer', 'Winter', 'MeanPress']]
df_ds_evaporation = df[['YYYYMMDD', 'Autumn', 'Spring', 'Summer', 'Winter', 'PotEvap']]

df_ds_pressure.rename(columns={'YYYYMMDD': 'ds', 'MeanPress': 'y'}, inplace=True)
df_ds_evaporation.rename(columns={'YYYYMMDD': 'ds', 'PotEvap': 'y'}, inplace=True)

# create and train model to forecast the pressure and evaporation
m_pressure = Prophet()
m_evaporation = Prophet()

for season in seasons:
    m_pressure.add_regressor(season)
    m_evaporation.add_regressor(season)

m_pressure.fit(df_ds_pressure)
m_evaporation.fit(df_ds_evaporation)

# forecast the pressure and evaporation
future_pressure = m_pressure.make_future_dataframe(periods=365)
future_evaporation = m_evaporation.make_future_dataframe(periods=365)

future_pressure.loc[(future_pressure['ds'].dt.month >= 9) & (future_pressure['ds'].dt.month <= 11), 'Autumn'] = 1
future_pressure.loc[(future_pressure['ds'].dt.month >= 3) & (future_pressure['ds'].dt.month <= 5), 'Spring'] = 1
future_pressure.loc[(future_pressure['ds'].dt.month >= 6) & (future_pressure['ds'].dt.month <= 8), 'Summer'] = 1
future_pressure.loc[(future_pressure['ds'].dt.month == 12) | (future_pressure['ds'].dt.month == 1) | (future_pressure['ds'].dt.month == 2), 'Winter'] = 1

future_evaporation.loc[(future_pressure['ds'].dt.month >= 9) & (future_pressure['ds'].dt.month <= 11), 'Autumn'] = 1
future_evaporation.loc[(future_pressure['ds'].dt.month >= 3) & (future_pressure['ds'].dt.month <= 5), 'Spring'] = 1
future_evaporation.loc[(future_pressure['ds'].dt.month >= 6) & (future_pressure['ds'].dt.month <= 8), 'Summer'] = 1
future_evaporation.loc[(future_pressure['ds'].dt.month == 12) | (future_pressure['ds'].dt.month == 1) | (future_pressure['ds'].dt.month == 2), 'Winter'] = 1

future_pressure.fillna(0, inplace=True)
future_evaporation.fillna(0, inplace=True)

future_pressure[seasons] = future_pressure[seasons].astype(int)
future_evaporation[seasons] = future_evaporation[seasons].astype(int)

forecast_pressure = m_pressure.predict(future_pressure)
forecast_evaporation = m_evaporation.predict(future_evaporation)

# Wind

# create dataframes for forecasting wind speed
df_ds_wind = df[['YYYYMMDD', 'Autumn', 'Spring', 'Summer', 'Winter', 'MeanPress', 'PotEvap', 'MeanWind']]
df_ds_wind.rename(columns={'YYYYMMDD': 'ds', 'MeanWind': 'y'}, inplace=True)

# create and train model to forecast the wind speed
m_wind = Prophet()

for season in seasons:
    m_wind.add_regressor(season)

m_wind.add_regressor('MeanPress')
m_wind.add_regressor('PotEvap')

m_wind.fit(df_ds_wind)

# forecast the wind speed
future_wind = m_wind.make_future_dataframe(periods=365)

future_wind.loc[(future_pressure['ds'].dt.month >= 9) & (future_pressure['ds'].dt.month <= 11), 'Autumn'] = 1
future_wind.loc[(future_pressure['ds'].dt.month >= 3) & (future_pressure['ds'].dt.month <= 5), 'Spring'] = 1
future_wind.loc[(future_pressure['ds'].dt.month >= 6) & (future_pressure['ds'].dt.month <= 8), 'Summer'] = 1
future_wind.loc[(future_pressure['ds'].dt.month == 12) | (future_pressure['ds'].dt.month == 1) | (future_pressure['ds'].dt.month == 2), 'Winter'] = 1

future_wind['MeanPress'] = forecast_pressure['yhat']
future_wind['PotEvap'] = forecast_evaporation['yhat']

future_wind.fillna(0, inplace=True)

future_wind[seasons] = future_pressure[seasons].astype(int)

forecast_wind = m_wind.predict(future_wind)

pickle.dump(m_wind, open('./assets/models/prophet.pickle', mode='wb'))

# save wind speed forecast
forecast_wind.to_csv('./assets/predictions/wind_forecast.csv', index=False, mode='w')
print(forecast_wind.tail())

while (date := input('Enter a date to view the wind speed: (exit) ')) != 'exit':
    print(f'{date}: {forecast_wind.loc[(forecast_wind["ds"] == date), "yhat"]}')

    print('Min/Max')
    print(f'{date}: {forecast_wind.loc[(forecast_wind["ds"] == date), "yhat_lower"]}')
    print(f'{date}: {forecast_wind.loc[(forecast_wind["ds"] == date), "yhat_upper"]}')

