import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import Config

Config.ORIGINAL_DATASET_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
Config.DATASET_PATH.mkdir(parents=True, exist_ok=True)

# load dataset into pandas dataframe
df = pd.read_csv('https://cdn.knmi.nl/knmi/map/page/klimatologie/gegevens/daggegevens/etmgeg_269.zip', 
                 na_values='     ', 
                 header=44, 
                 parse_dates=['YYYYMMDD']).rename(columns=str.strip)

# convert column names to more descriptive ones
df.rename(columns={
                'DDVEC': 'WindDir',
                'FG': 'MeanWind',
                'FHX': 'MaxWind',
                'FHXH': 'MaxWindHour',
                'FHN': 'MinWind',
                'FHNH': 'MinWindHour',
                'FXX': 'MaxWindGust',
                'FXXH': 'MaxWindGustHour',
                'TG': 'MeanTemp',
                'TN': 'MinTemp',
                'TNH': 'MinTempHour',
                'TX': 'MaxTemp',
                'TXH': 'MaxTempHour',
                'SQ': 'SunshineDur',
                'Q': 'Radiation',
                'DR': 'RainDur',
                'RH': 'RainAmount',
                'RHX': 'MaxRainAmount',
                'RHXH': 'MaxRainAmountHour',
                'PG': 'MeanPress',
                'PX': 'MaxPress',
                'PXH': 'MaxPressHour',
                'PN': 'MinPress',
                'PNH': 'MinPressHour',
                'VVN': 'MinVis',
                'VVNH': 'MinVisHour',
                'VVX': 'MaxVis',
                'VVXH': 'MaxVisHour',
                'NG': 'Cloudness',
                'UG': 'MeanHum',
                'UX': 'MaxHum',
                'UXH': 'MaxHumHour',
                'UN': 'MinHum',
                'UNH': 'MinHumHour',
                'EV24': 'PotEvap'
}, inplace=True)

# devide column values by 10 because their are in 0.1
#  Example:
#      1 mean 0.1 degree C
#      15 means 1.5 degrees C
#      250 means 25.0 degrees C
columns_div = ['MeanWind', 'MaxWind', 'MaxWindHour', 'MinWind', 
    'MinWindHour', 'MaxWindGustHour', 'MeanTemp', 
    'MinTemp', 'MinTempHour', 'MaxTemp', 
    'MaxTempHour', 'SunshineDur', 'RainDur', 
    'RainAmount', 'MaxRainAmount', 'MaxRainAmountHour', 
    'MeanPress', 'MaxPress', 'MaxPressHour',
    'MinPress', 'MinPressHour']

df[columns_div] = df[columns_div] / 10  

# save in directory
df.to_csv(str(Config.ORIGINAL_DATASET_FILE_PATH), index=False, mode='w')

