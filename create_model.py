import pickle

import pandas as pd

from sklearn.preprocessing import LabelEncoder

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
df['Season'] = LabelEncoder().fit_transform(df['Season'])

# Pressure and Evaporation

# create dataframes for forecasting pressure and evaporation
df_ds_pressure = df[['YYYYMMDD', 'Season', 'MeanPress']]
df_ds_evaporation = df[['YYYYMMDD', 'Season', 'PotEvap']]

df_ds_pressure.rename(columns={'YYYYMMDD': 'ds', 'MeanPress': 'y'}, inplace=True)
df_ds_evaporation.rename(columns={'YYYYMMDD': 'ds', 'PotEvap': 'y'}, inplace=True)

# create and train model to forecast the pressure and evaporation
m_pressure = Prophet()
m_evaporation = Prophet()

m_pressure.add_regressor('Season')
m_evaporation.add_regressor('Season')

m_pressure.fit(df_ds_pressure)
m_evaporation.fit(df_ds_evaporation)

# forecast the pressure and evaporation
future_pressure = m_pressure.make_future_dataframe(periods=365)
future_evaporation = m_evaporation.make_future_dataframe(periods=365)

future_pressure['Season'] = future_pressure['ds'].apply(add_season_encoding)
future_evaporation['Season'] = future_evaporation['ds'].apply(add_season_encoding)

forecast_pressure = m_pressure.predict(future_pressure)
forecast_evaporation = m_evaporation.predict(future_evaporation)


# Wind

# create dataframes for forecasting wind speed
df_ds_wind = df[['YYYYMMDD', 'Season', 'MeanPress', 'PotEvap', 'MeanWind']]
df_ds_wind.rename(columns={'YYYYMMDD': 'ds', 'MeanWind': 'y'}, inplace=True)

# create and train model to forecast the wind speed
m_wind = Prophet()

m_wind.add_regressor('Season')
m_wind.add_regressor('MeanPress')
m_wind.add_regressor('PotEvap')

m_wind.fit(df_ds_wind)

# forecast the wind speed
future_wind = m_wind.make_future_dataframe(periods=365)

future_wind['Season'] = future_wind['ds'].apply(add_season_encoding)
future_wind['MeanPress'] = forecast_pressure['yhat']
future_wind['PotEvap'] = forecast_evaporation['yhat']

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

