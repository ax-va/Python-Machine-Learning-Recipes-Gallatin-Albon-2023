import pandas as pd

url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

df = pd.read_csv(url)


# Create function
def make_uppercase(x: str) -> str:
    return x.upper()

# Apply function, show two rows
df['Name'].apply(make_uppercase)[0:2]
# 0    ALLEN, MISS ELISABETH WALTON
# 1     ALLISON, MISS HELEN LORAINE
# Name: Name, dtype: object
