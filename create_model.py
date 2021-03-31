import pandas as pd

from fbprophet import Prophet
df_timeseries = pd.DataFrame()

df = pd.read_csv('./assets/data/lelystad_main_features.csv', parse_dates=['YYYYMMDD'])
df_timeseries[['ds', 'y']] = df[['YYYYMMDD', 'MeanWind']]
print(df_timeseries.tail())

model = Prophet()
model.fit(df_timeseries)

future = model.make_future_dataframe(periods=365)
future.tail()

forecasts = model.predict(future)
print(forecasts.shape)


fig1 = model.plot(forecasts)
fig1.show()
temp = input()
# print(fig1)

# for i in range(11000, 11050):
#     print(forecasts.iloc[i])
#     print()

