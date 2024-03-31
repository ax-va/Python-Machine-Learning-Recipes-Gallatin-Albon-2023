import pandas as pd

url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

# Load data
df = pd.read_csv(url)

# Show top two rows where column 'sex' is 'female'
df[df['Sex'] == 'female'].head(2)
#                            Name PClass   Age     Sex  Survived  SexCode
# 0  Allen, Miss Elisabeth Walton    1st  29.0  female         1        1
# 1   Allison, Miss Helen Loraine    1st   2.0  female         0        1

# Select all the rows where the passenger is a female 65 or older
mask = (df['Sex'] == 'female') & (df['Age'] >= 65)
df[mask]
#                                                  Name PClass   Age     Sex  Survived  SexCode
# 73  Crosby, Mrs Edward Gifford (Catherine Elizabet...    1st  69.0  female         1        1
