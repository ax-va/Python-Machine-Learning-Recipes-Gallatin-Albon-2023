"""
Using:
- isnull(), an alias for isna()
- notnull(), an alias for notna()
- pd.read_csv(url, na_values=('NONE', -999)) to recognize missing values
- fillna() to fill missing values
"""
import numpy as np
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

df[df['Age'].isnull()].head(3)
#                             Name PClass  Age     Sex  Survived  SexCode
# 12  Aubert, Mrs Leontine Pauline    1st  NaN  female         1        1
# 13      Barkworth, Mr Algernon H    1st  NaN    male         1        0
# 14            Baumann, Mr John D    1st  NaN    male         0        0

df[df['Age'].isnull()].count()
# Name        557
# PClass      557
# Age           0
# Sex         557
# Survived    557
# SexCode     557
# dtype: int64

df['Age'].isnull().head(3)
# 0    False
# 1    False
# 2    False
# Name: Age, dtype: bool

# (whole number of passengers)
df['Age'].isnull().count()  # isnull() is an alias for isna()
# 1313

df[df['Age'].notnull()].count()  # notnull() is an alias for notna()
# Name        756
# PClass      756
# Age         756
# Sex         756
# Survived    756
# SexCode     756
# dtype: int64

# Replace values with NaN
df['Sex'] = df['Sex'].replace('male', np.nan)
df[df['Sex'].notnull()].count()
# Name        462
# PClass      462
# Age         288
# Sex         462
# Survived    462
# SexCode     462
# dtype: int64

# Load data, recognize missing values
df = pd.read_csv(url, na_values=('NONE', -999))

# Get a single null row
na_entry = df[df["Age"].isna()].head(1)
#                             Name PClass  Age     Sex  Survived  SexCode
# 12  Aubert, Mrs Leontine Pauline    1st  NaN  female         1        1

# Use fillna() to fill with the median value
na_entry.fillna(df["Age"].median())
#                             Name PClass   Age     Sex  Survived  SexCode
# 12  Aubert, Mrs Leontine Pauline    1st  28.0  female         1        1
