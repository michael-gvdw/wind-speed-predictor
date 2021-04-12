import json
import pickle

import pandas as pd

from sklearn import metrics

# load in trained model
m_wind = pickle.load(open('./assets/models/prophet.pickle', mode='rb'))

# load in data
df_histoy = pd.read_csv('./assets/data/lelystad_final_features_numerical.csv')
df_histoy.rename(columns={'YYYYMMDD': 'ds', 'MeanWind': 'y'}, inplace=True)

df_future = pd.read_csv('./assets/predictions/wind_forecast.csv')

# create dataframe for evaluation
df_score = df_histoy[['ds', 'y']]
df_score['yhat'] = df_future['yhat']

print(df_score.tail())

