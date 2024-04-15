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

df.head(3)
#                                   Name PClass   Age     Sex  Survived  SexCode
# 0         Allen, Miss Elisabeth Walton    1st  29.0  female         1        1
# 1          Allison, Miss Helen Loraine    1st   2.0  female         0        1
# 2  Allison, Mr Hudson Joshua Creighton    1st  30.0    male         0        0

# Group rows by the values of the column 'Sex', calculate mean of each group
df.groupby('Sex').mean(numeric_only=True)
#               Age  Survived  SexCode
# Sex
# female  29.396424  0.666667      1.0
# male    31.014338  0.166863      0.0

# Group by something and then apply a function to each of those groups
df.groupby('Survived')['Name'].count()
# Survived
# 0    863
# 1    450
# Name: Name, dtype: int64

df.groupby('Survived')['Sex'].count()
# Survived
# 0    863
# 1    450
# Name: Sex, dtype: int64

df.groupby(['Sex','Survived'])['Age'].mean()
# Sex     Survived
# female  0           24.901408
#         1           30.867143
# male    0           32.320780
#         1           25.951875
# Name: Age, dtype: float64
