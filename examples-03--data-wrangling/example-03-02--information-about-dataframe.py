import pandas as pd

url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

df = pd.read_csv(url)
#                                                Name PClass    Age     Sex  Survived  SexCode
# 0                      Allen, Miss Elisabeth Walton    1st  29.00  female         1        1
# 1                       Allison, Miss Helen Loraine    1st   2.00  female         0        1
# 2               Allison, Mr Hudson Joshua Creighton    1st  30.00    male         0        0
# 3     Allison, Mrs Hudson JC (Bessie Waldo Daniels)    1st  25.00  female         0        1
# 4                     Allison, Master Hudson Trevor    1st   0.92    male         1        0
# ...                                             ...    ...    ...     ...       ...      ...
# 1308                             Zakarian, Mr Artun    3rd  27.00    male         0        0
# 1309                         Zakarian, Mr Maprieder    3rd  26.00    male         0        0
# 1310                               Zenni, Mr Philip    3rd  22.00    male         0        0
# 1311                               Lievens, Mr Rene    3rd  24.00    male         0        0
# 1312                                 Zimmerman, Leo    3rd  29.00    male         0        0
#
# [1313 rows x 6 columns]

# Show two rows
df.head(2)
#                            Name PClass   Age     Sex  Survived  SexCode
# 0  Allen, Miss Elisabeth Walton    1st  29.0  female         1        1
# 1   Allison, Miss Helen Loraine    1st   2.0  female         0        1

# Show dimensions
df.shape
# (1313, 6)

# Show statistics
df.describe()
#               Age     Survived      SexCode
# count  756.000000  1313.000000  1313.000000
# mean    30.397989     0.342727     0.351866
# std     14.259049     0.474802     0.477734
# min      0.170000     0.000000     0.000000
# 25%     21.000000     0.000000     0.000000
# 50%     28.000000     0.000000     0.000000
# 75%     39.000000     1.000000     1.000000
# max     71.000000     1.000000     1.000000

# Show info
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
