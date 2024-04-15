import pandas as pd

url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

df = pd.read_csv(url)

# Print first two names uppercased
for name in df['Name'][0:2]:
    print(name.upper())
# ALLEN, MISS ELISABETH WALTON
# ALLISON, MISS HELEN LORAINE

# list comprehension:
# Show first two names uppercased
[name.upper() for name in df['Name'][0:2]]
# ['ALLEN, MISS ELISABETH WALTON', 'ALLISON, MISS HELEN LORAINE']

# A more Pythonic solution would use the pandas apply method
