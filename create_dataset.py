import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import Config

Config.ORIGINAL_DATASET_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
Config.DATASET_PATH.mkdir(parents=True, exist_ok=True)

# load dataset into pandas dataframe
df = pd.read_csv("https://cdn.knmi.nl/knmi/map/page/klimatologie/gegevens/daggegevens/etmgeg_269.zip",
                 skiprows=[i for i in range(47)])

# replace empty string values with -999 to represent missing values
df.replace('     ', -999, inplace=True)

# convert to right data type
df = df.astype('int')

# save in directory
df.to_csv(str(Config.ORIGINAL_DATASET_FILE_PATH), index=False, mode='w')

