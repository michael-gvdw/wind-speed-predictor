import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from config import Config


df = pd.read_csv(str(Config.ORIGINAL_DATASET_FILE_PATH), parse_dates=['YYYYMMDD'])
print(df.tail())


# Adding columns as part of feature engineering 

# create day month and year columns
df['Day'] = df['YYYYMMDD'].dt.day
df['Month'] = df['YYYYMMDD'].dt.month
df['Year'] = df['YYYYMMDD'].dt.year

# create day season column
df.loc[(df['Month'] == 1) | (df['Month'] == 2) | (df['Month'] == 12), 'Season'] = 'Winter'
df.loc[(df['Month'] == 3) | (df['Month'] == 4) | (df['Month'] == 5), 'Season'] = 'Spring'
df.loc[(df['Month'] == 6) | (df['Month'] == 7) | (df['Month'] == 8), 'Season'] = 'Summer'
df.loc[(df['Month'] == 9) | (df['Month'] == 10) | (df['Month'] == 11), 'Season'] = 'Autumn'

# create WindCategory column
wind_categories = ['Light Air', 'Light Breeze', 'Gentle Breeze', 'Moderate Breeze', 
                   'Fresh Breeze', 'Strong Breeze', 'Moderate Gale', 'Fresh Gale', 
                   'Strong Gale', 'Whole Gale', 'Storm', 'Hurricane']

df.loc[(df['MeanWind'] <= 2.0), 'WindCategory'] = wind_categories[0]
df.loc[(df['MeanWind'] > 2.0) & (df['MeanWind'] <= 3.5), 'WindCategory'] = wind_categories[1]
df.loc[(df['MeanWind'] > 3.5) & (df['MeanWind'] <= 5.5), 'WindCategory'] = wind_categories[2]
df.loc[(df['MeanWind'] > 5.5) & (df['MeanWind'] <= 8.5), 'WindCategory'] = wind_categories[3]
df.loc[(df['MeanWind'] > 8.5) & (df['MeanWind'] <= 10.5), 'WindCategory'] = wind_categories[4]
df.loc[(df['MeanWind'] > 10.5) & (df['MeanWind'] <= 13.5), 'WindCategory'] = wind_categories[5]
df.loc[(df['MeanWind'] > 13.5) & (df['MeanWind'] <= 16.5), 'WindCategory'] = wind_categories[6]
df.loc[(df['MeanWind'] > 16.5) & (df['MeanWind'] <= 20.5), 'WindCategory'] = wind_categories[7]
df.loc[(df['MeanWind'] > 20.5) & (df['MeanWind'] <= 23.5), 'WindCategory'] = wind_categories[8]
df.loc[(df['MeanWind'] > 23.5) & (df['MeanWind'] <= 27.5), 'WindCategory'] = wind_categories[9]
df.loc[(df['MeanWind'] > 27.5) & (df['MeanWind'] <= 31.5), 'WindCategory'] = wind_categories[10]
df.loc[(df['MeanWind'] > 31.5), 'WindCategory'] = wind_categories[11]

print(df.tail())

# save current dataset
df.to_csv(f'{Config.DATASET_PATH}/lelystad.csv', index=False, mode='w')


# we are interested only in the mean values
df = df[['YYYYMMDD', 'Year', 'Month', 'Day', 'Season', 
         'MeanTemp', 'MeanHum', 'MeanPress', 'WindDir', 
         'MeanWind', 'WindCategory',
         'SunshineDur', 'RainDur', 
         'RainAmount',  'MinVis', 
         'MaxVis', 'Cloudness',  
         'PotEvap']]

# get mean evaporation
df['MeanVis'] = (df['MaxVis'] + df['MinVis']) / 2

df.drop('MaxVis', axis=1, inplace=True)
df.drop('MinVis', axis=1, inplace=True)

# save current dataset
df.to_csv(f'{Config.DATASET_PATH}/lelystad_main_features.csv', index=False, mode='w')

# filtering out columns

# correlation of not important columns
print(df[['SunshineDur', 'RainDur', 'RainAmount', 'MeanVis', 'PotEvap', 'MeanWind', 'Cloudness']].corr())

# keep features that we believe to add value
df = df[['YYYYMMDD', 'Season', 'PotEvap', 'MeanTemp', 'MeanHum', 'MeanPress', 'WindDir', 'MeanWind', 'WindCategory']] 
df.to_csv(f'{Config.DATASET_PATH}/lelystad_main_features.csv', index=False, mode='w')

# correlation of our final columns
df_temp = df.copy()
df_temp['Season'] = LabelEncoder().fit_transform(list(df_temp['Season']))
print(df_temp[['Season', 'PotEvap', 'MeanTemp', 'MeanHum', 'MeanPress', 'WindDir', 'MeanWind',]].corr())

# keep features that we believe to add value
df = df[['YYYYMMDD', 'Season', 'PotEvap', 'MeanTemp', 'MeanPress', 'MeanWind', 'WindCategory']]

# fill missing values

# mean values for each month
mean_values = []
mean_values.append(dict(df['MeanWind'].groupby(df['YYYYMMDD'].dt.month).mean()))
mean_values.append(dict(df['MeanTemp'].groupby(df['YYYYMMDD'].dt.month).mean()))
mean_values.append(dict(df['MeanPress'].groupby(df['YYYYMMDD'].dt.month).mean()))
mean_values.append(dict(df['PotEvap'].groupby(df['YYYYMMDD'].dt.month).mean()))

# fill the values
col = ['MeanWind', 'MeanTemp', 'MeanPress', 'PotEvap']

for i in range(4):
    for j in range(1, 13):
        filters = (df['YYYYMMDD'].dt.month == j) & (np.nan_to_num(df[col[i]]) == 0.0)
        df.loc[filters, col[i]] = df.loc[filters, col[i]].fillna(mean_values[i][j])

df['WindCategory'].fillna('Gentle Breeze', inplace=True)

print(df.isna().sum())

# final save of datasets

# final main features
final_features_numerical = ['YYYYMMDD', 'Season', 'PotEvap', 'MeanTemp', 'MeanPress', 'MeanWind']
final_features_categorical = ['YYYYMMDD', 'Season', 'PotEvap', 'MeanTemp', 'MeanPress', 'WindCategory']

df = df[['YYYYMMDD', 'Season', 'PotEvap', 'MeanTemp', 'MeanPress', 'MeanWind', 'WindCategory']]
df_numerical = df[final_features_numerical]
df_categorical = df[final_features_categorical]

# save datasets
df_numerical.to_csv(f'{Config.DATASET_PATH}/lelystad_final_features_numerical.csv', index=False, mode='w')
df_categorical.to_csv(f'{Config.DATASET_PATH}/lelystad_final_features_categorical.csv', index=False, mode='w')
df.to_csv(f'{Config.DATASET_PATH}/lelystad_final_features.csv', index=False, mode='w')

