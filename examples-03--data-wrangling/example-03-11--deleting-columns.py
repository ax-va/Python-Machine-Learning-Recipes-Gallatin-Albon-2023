import pandas as pd

url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

df = pd.read_csv(url)
df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 1313 entries, 0 to 1312
# Data columns (total 6 columns):
#  #   Column    Non-Null Count  Dtype
# ---  ------    --------------  -----
#  0   Name      1313 non-null   object
#  1   PClass    1313 non-null   object
#  2   Age       756 non-null    float64
#  3   Sex       1313 non-null   object
#  4   Survived  1313 non-null   int64
#  5   SexCode   1313 non-null   int64
# dtypes: float64(1), int64(2), object(3)
# memory usage: 61.7+ KB

# Delete column
df.drop('Age', axis=1).info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 1313 entries, 0 to 1312
# Data columns (total 5 columns):
#  #   Column    Non-Null Count  Dtype
# ---  ------    --------------  -----
#  0   Name      1313 non-null   object
#  1   PClass    1313 non-null   object
#  2   Sex       1313 non-null   object
#  3   Survived  1313 non-null   int64
#  4   SexCode   1313 non-null   int64
# dtypes: int64(2), object(3)
# memory usage: 51.4+ KB

# Drop column by number
df.drop(df.columns[2], axis=1).info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 1313 entries, 0 to 1312
# Data columns (total 5 columns):
#  #   Column    Non-Null Count  Dtype
# ---  ------    --------------  -----
#  0   Name      1313 non-null   object
#  1   PClass    1313 non-null   object
#  2   Sex       1313 non-null   object
#  3   Survived  1313 non-null   int64
#  4   SexCode   1313 non-null   int64
# dtypes: int64(2), object(3)
# memory usage: 51.4+ KB

# Drop columns
df.drop(['Age', 'Sex', 'SexCode'], axis=1).info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 1313 entries, 0 to 1312
# Data columns (total 3 columns):
#  #   Column    Non-Null Count  Dtype
# ---  ------    --------------  -----
#  0   Name      1313 non-null   object
#  1   PClass    1313 non-null   object
#  2   Survived  1313 non-null   int64
# dtypes: int64(1), object(2)
# memory usage: 30.9+ KB

# Drop column by numbers
df.drop([df.columns[2], df.columns[3], df.columns[5]], axis=1).info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 1313 entries, 0 to 1312
# Data columns (total 3 columns):
#  #   Column    Non-Null Count  Dtype
# ---  ------    --------------  -----
#  0   Name      1313 non-null   object
#  1   PClass    1313 non-null   object
#  2   Survived  1313 non-null   int64
# dtypes: int64(1), object(2)
# memory usage: 30.9+ KB
