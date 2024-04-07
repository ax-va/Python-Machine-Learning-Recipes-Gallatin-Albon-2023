import pandas as pd

url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

df = pd.read_csv(url)
df.head()
#                                             Name PClass    Age     Sex  Survived  SexCode
# 0                   Allen, Miss Elisabeth Walton    1st  29.00  female         1        1
# 1                    Allison, Miss Helen Loraine    1st   2.00  female         0        1
# 2            Allison, Mr Hudson Joshua Creighton    1st  30.00    male         0        0
# 3  Allison, Mrs Hudson JC (Bessie Waldo Daniels)    1st  25.00  female         0        1
# 4                  Allison, Master Hudson Trevor    1st   0.92    male         1        0

# Delete rows with Sex equal male
df[df['Sex'] != 'male'].head(3)
#                                             Name PClass   Age     Sex  Survived  SexCode
# 0                   Allen, Miss Elisabeth Walton    1st  29.0  female         1        1
# 1                    Allison, Miss Helen Loraine    1st   2.0  female         0        1
# 3  Allison, Mrs Hudson JC (Bessie Waldo Daniels)    1st  25.0  female         0        1

# Delete single row
df[df['Name'] != 'Allison, Miss Helen Loraine'].head(4)
#                                             Name PClass    Age     Sex  Survived  SexCode
# 0                   Allen, Miss Elisabeth Walton    1st  29.00  female         1        1
# 2            Allison, Mr Hudson Joshua Creighton    1st  30.00    male         0        0
# 3  Allison, Mrs Hudson JC (Bessie Waldo Daniels)    1st  25.00  female         0        1
# 4                  Allison, Master Hudson Trevor    1st   0.92    male         1        0

# Delete row with index equal to 0
df[df.index != 0].head(2)

# Drop rows
df.drop([0, 1], axis=0).head(3)
#                                             Name PClass    Age     Sex  Survived  SexCode
# 2            Allison, Mr Hudson Joshua Creighton    1st  30.00    male         0        0
# 3  Allison, Mrs Hudson JC (Bessie Waldo Daniels)    1st  25.00  female         0        1
# 4                  Allison, Master Hudson Trevor    1st   0.92    male         1        0
