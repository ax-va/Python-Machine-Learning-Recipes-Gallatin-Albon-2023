import pandas as pd

url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

df = pd.read_csv(url)

# Select first row
df.iloc[0]
# Name        Allen, Miss Elisabeth Walton
# PClass                               1st
# Age                                 29.0
# Sex                               female
# Survived                               1
# SexCode                                1
# Name: 0, dtype: object
