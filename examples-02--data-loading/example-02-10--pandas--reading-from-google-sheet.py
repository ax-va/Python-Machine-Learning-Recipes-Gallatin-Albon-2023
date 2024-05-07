import pandas as pd

# Export the Google Sheet as a CSV by using "/export?format=csv"
url = "https://docs.google.com/spreadsheets/d/1ehC-9otcAuitqnmWksqt1mOrTRCL38dv0K9UjhwzTOA/export?format=csv"

df = pd.read_csv(url)
df.head(2)
#    integer            datetime  category
# 0        5  2015-01-01 0:00:00         0
# 1        5  2015-01-01 0:00:01         0

df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 100 entries, 0 to 99
# Data columns (total 3 columns):
#  #   Column    Non-Null Count  Dtype
# ---  ------    --------------  -----
#  0   integer   100 non-null    int64
#  1   datetime  100 non-null    object
#  2   category  100 non-null    int64
# dtypes: int64(2), object(1)
# memory usage: 2.5+ KB
