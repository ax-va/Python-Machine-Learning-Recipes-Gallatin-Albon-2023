"""
Both unique and value_counts are useful for manipulating and exploring categorical columns.
"""
import pandas as pd

url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

df = pd.read_csv(url)

# Select unique values
df['Sex'].unique()
# array(['female', 'male'], dtype=object)

# Alternatively
df['Sex'].value_counts()
# Sex
# male      851
# female    462
# Name: count, dtype: int64

df['PClass'].unique()
# array(['1st', '2nd', '*', '3rd'], dtype=object)

# Show counts
df['PClass'].value_counts()
# PClass
# 3rd    711
# 1st    322
# 2nd    279
# *        1
# Name: count, dtype: int64

# Count the number of unique values
df['PClass'].nunique()
# 4
