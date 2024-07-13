import collections
import pandas as pd

url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

df = pd.read_csv(url)
df.head(3)
#                                   Name PClass   Age     Sex  Survived  SexCode
# 0         Allen, Miss Elisabeth Walton    1st  29.0  female         1        1
# 1          Allison, Miss Helen Loraine    1st   2.0  female         0        1
# 2  Allison, Mr Hudson Joshua Creighton    1st  30.0    male         0        0

# Rename column
df.rename(columns={'PClass': 'Passenger Class'}).head(3)
#                                   Name Passenger Class   Age     Sex  Survived  SexCode
# 0         Allen, Miss Elisabeth Walton             1st  29.0  female         1        1
# 1          Allison, Miss Helen Loraine             1st   2.0  female         0        1
# 2  Allison, Mr Hudson Joshua Creighton             1st  30.0    male         0        0

# Rename columns, show two rows
df.rename(columns={'PClass': 'Passenger Class', 'Sex': 'Gender'}).head(3)
#                                   Name Passenger Class   Age  Gender  Survived  SexCode
# 0         Allen, Miss Elisabeth Walton             1st  29.0  female         1        1
# 1          Allison, Miss Helen Loraine             1st   2.0  female         0        1
# 2  Allison, Mr Hudson Joshua Creighton             1st  30.0    male         0        0

# If you want to rename all columns at once
column_names = collections.defaultdict(str)
# defaultdict(str, {})

# Create keys
for name in df.columns:
    column_names[name]

column_names
# defaultdict(str,
#             {'Name': '',
#              'PClass': '',
#              'Age': '',
#              'Sex': '',
#              'Survived': '',
#              'SexCode': ''})
