import pandas as pd

url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

df = pd.read_csv(url)
df.head(3)
#                                   Name PClass   Age     Sex  Survived  SexCode
# 0         Allen, Miss Elisabeth Walton    1st  29.0  female         1        1
# 1          Allison, Miss Helen Loraine    1st   2.0  female         0        1
# 2  Allison, Mr Hudson Joshua Creighton    1st  30.0    male         0        0

# Replace values, show three rows
df['Sex'].replace("female", "Woman").head(3)
# 0    Woman
# 1    Woman
# 2     male
# Name: Sex, dtype: object

# Replace "female" and "male" with "Woman" and "Man"
df['Sex'].replace(["female", "male"], ["Woman", "Man"]).head(3)
# 0    Woman
# 1    Woman
# 2      Man
# Name: Sex, dtype: object

# Replace across the entire DataFrame instance
df.replace(1, "One").head(3)
#                                   Name PClass   Age     Sex Survived SexCode
# 0         Allen, Miss Elisabeth Walton    1st  29.0  female      One     One
# 1          Allison, Miss Helen Loraine    1st   2.0  female        0     One
# 2  Allison, Mr Hudson Joshua Creighton    1st  30.0    male        0       0

# Replace values with regex
df.replace(r".st", "First", regex=True)
#                                                Name PClass    Age     Sex  Survived  SexCode
# 0                      Allen, Miss Elisabeth Walton  First  29.00  female         1        1
# 1                       Allison, Miss Helen Loraine  First   2.00  female         0        1
# 2               Allison, Mr Hudson Joshua Creighton  First  30.00    male         0        0
# 3     Allison, Mrs Hudson JC (Bessie Waldo Daniels)  First  25.00  female         0        1
# 4                   Allison, MFirster Hudson Trevor  First   0.92    male         1        0
# ...                                             ...    ...    ...     ...       ...      ...
# 1308                             Zakarian, Mr Artun    3rd  27.00    male         0        0
# 1309                         Zakarian, Mr Maprieder    3rd  26.00    male         0        0
# 1310                               Zenni, Mr Philip    3rd  22.00    male         0        0
# 1311                               Lievens, Mr Rene    3rd  24.00    male         0        0
# 1312                                 Zimmerman, Leo    3rd  29.00    male         0        0
