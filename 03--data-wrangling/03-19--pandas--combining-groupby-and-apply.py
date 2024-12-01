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

# Combine groupby and apply:
# Group rows, apply function to groups
df.groupby('Sex').apply(lambda x: x.count())
#         Name  PClass  Age  Sex  Survived  SexCode
# Sex
# female   462     462  288  462       462      462
# male     851     851  468  851       851      851
