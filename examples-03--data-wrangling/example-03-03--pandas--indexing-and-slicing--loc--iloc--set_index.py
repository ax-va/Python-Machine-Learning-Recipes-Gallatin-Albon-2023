import pandas as pd

df = pd.read_csv('../data/titanic.csv')

# Select first row
df.iloc[0]
# Name        Allen, Miss Elisabeth Walton
# PClass                               1st
# Age                                 29.0
# Sex                               female
# Survived                               1
# SexCode                                1
# Name: 0, dtype: object

# Select three rows
df.iloc[2:5]
#                                             Name PClass    Age     Sex  Survived  SexCode
# 2            Allison, Mr Hudson Joshua Creighton    1st  30.00    male         0        0
# 3  Allison, Mrs Hudson JC (Bessie Waldo Daniels)    1st  25.00  female         0        1
# 4                  Allison, Master Hudson Trevor    1st   0.92    male         1        0

# Get all rows up to a point
df.iloc[:5]
#                                             Name PClass    Age     Sex  Survived  SexCode
# 0                   Allen, Miss Elisabeth Walton    1st  29.00  female         1        1
# 1                    Allison, Miss Helen Loraine    1st   2.00  female         0        1
# 2            Allison, Mr Hudson Joshua Creighton    1st  30.00    male         0        0
# 3  Allison, Mrs Hudson JC (Bessie Waldo Daniels)    1st  25.00  female         0        1
# 4                  Allison, Master Hudson Trevor    1st   0.92    male         1        0

# Set index
df = df.set_index('Name')  # 'Name' is excluded as column
df.head()
#                                               PClass    Age     Sex  Survived  SexCode
# Name
# Allen, Miss Elisabeth Walton                     1st  29.00  female         1        1
# Allison, Miss Helen Loraine                      1st   2.00  female         0        1
# Allison, Mr Hudson Joshua Creighton              1st  30.00    male         0        0
# Allison, Mrs Hudson JC (Bessie Waldo Daniels)    1st  25.00  female         0        1
# Allison, Master Hudson Trevor                    1st   0.92    male         1        0

df.loc['Allen, Miss Elisabeth Walton']
# PClass         1st
# Age           29.0
# Sex         female
# Survived         1
# SexCode          1
# Name: Allen, Miss Elisabeth Walton, dtype: object

df = df.reset_index()
df.head()
#                                             Name PClass    Age     Sex  Survived  SexCode
# 0                   Allen, Miss Elisabeth Walton    1st  29.00  female         1        1
# 1                    Allison, Miss Helen Loraine    1st   2.00  female         0        1
# 2            Allison, Mr Hudson Joshua Creighton    1st  30.00    male         0        0
# 3  Allison, Mrs Hudson JC (Bessie Waldo Daniels)    1st  25.00  female         0        1
# 4                  Allison, Master Hudson Trevor    1st   0.92    male         1        0

# Set index
df = df.set_index(df['Name'])  # 'Name' is included as column
df.head()
#                                                                                         Name PClass    Age     Sex  Survived  SexCode
# Name
# Allen, Miss Elisabeth Walton                                    Allen, Miss Elisabeth Walton    1st  29.00  female         1        1
# Allison, Miss Helen Loraine                                      Allison, Miss Helen Loraine    1st   2.00  female         0        1
# Allison, Mr Hudson Joshua Creighton                      Allison, Mr Hudson Joshua Creighton    1st  30.00    male         0        0
# Allison, Mrs Hudson JC (Bessie Waldo Daniels)  Allison, Mrs Hudson JC (Bessie Waldo Daniels)    1st  25.00  female         0        1
# Allison, Master Hudson Trevor                                  Allison, Master Hudson Trevor    1st   0.92    male         1        0

# Show row
df.loc['Allen, Miss Elisabeth Walton']
# Name        Allen, Miss Elisabeth Walton
# PClass                               1st
# Age                                 29.0
# Sex                               female
# Survived                               1
# SexCode                                1
# Name: Allen, Miss Elisabeth Walton, dtype: object
