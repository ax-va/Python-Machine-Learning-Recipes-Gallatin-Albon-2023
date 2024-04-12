import pandas as pd

url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

df = pd.read_csv(url)

# Drop only rows with full matching in all the columns
print("Number Of Rows In The Original DataFrame:", len(df))
# Number Of Rows In The Original DataFrame: 1313
print("Number Of Rows After Deduping:", len(df.drop_duplicates()))
# Number Of Rows After Deduping: 1313

# Drop duplicates in subset
df.drop_duplicates(subset=['Sex'])  # default: keep="first"
#                                   Name PClass   Age     Sex  Survived  SexCode
# 0         Allen, Miss Elisabeth Walton    1st  29.0  female         1        1
# 2  Allison, Mr Hudson Joshua Creighton    1st  30.0    male         0        0

df.drop_duplicates(subset=['Sex'], keep="first")
#                                   Name PClass   Age     Sex  Survived  SexCode
# 0         Allen, Miss Elisabeth Walton    1st  29.0  female         1        1
# 2  Allison, Mr Hudson Joshua Creighton    1st  30.0    male         0        0

df.drop_duplicates(subset=['Sex'], keep="last")
#                      Name PClass   Age     Sex  Survived  SexCode
# 1307  Zabour, Miss Tamini    3rd   NaN  female         0        1
# 1312       Zimmerman, Leo    3rd  29.0    male         0        0

# Return a boolean series denoting whether a row is a duplicate or not
df.duplicated()
# 0       False
# 1       False
# 2       False
# 3       False
# 4       False
#         ...
# 1308    False
# 1309    False
# 1310    False
# 1311    False
# 1312    False
# Length: 1313, dtype: bool

df.duplicated(subset=["Sex"])  # default: keep="first"
# 0       False
# 1        True
# 2       False
# 3        True
# 4        True
#         ...
# 1308     True
# 1309     True
# 1310     True
# 1311     True
# 1312     True
# Length: 1313, dtype: bool

df.duplicated(subset=["Sex"], keep="last")
# 0        True
# 1        True
# 2        True
# 3        True
# 4        True
#         ...
# 1308     True
# 1309     True
# 1310     True
# 1311     True
# 1312    False
# Length: 1313, dtype: bool
