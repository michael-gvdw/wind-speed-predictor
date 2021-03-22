import pandas as pd
from config import Config

# load dataset into pandas dataframe
df = pd.read_csv("https://cdn.knmi.nl/knmi/map/page/klimatologie/gegevens/daggegevens/etmgeg_269.zip",
                 skiprows=[i for i in range(47)])

# replace empty string values with -999 to represent missing values
df.replace('     ', -999, inplace=True)

# convert to right data type
df = df.astype('int')

# save in directory
df.to_csv(str(Config.ORIGINAL_DATASET_FILE_PATH), index=False, mode='w')