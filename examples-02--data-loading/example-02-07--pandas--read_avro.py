import requests
import pandavro as pdx

url = 'https://machine-learning-python-cookbook.s3.amazonaws.com/data.avro'

r = requests.get(url)

with open('../data/data.avro', 'wb') as file:
    file.write(r.content)

df = pdx.read_avro('../data/data.avro')
df.head(2)
#    integer             datetime  category
# 0        5  2015-01-01 00:00:00         0
# 1        5  2015-01-01 00:00:01         0

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
