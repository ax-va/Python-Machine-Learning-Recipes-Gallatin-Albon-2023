import pandas as pd

url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

df = pd.read_csv(url)

# Calculate statistics
print('Maximum:', df['Age'].max())
print('Minimum:', df['Age'].min())
print('Mean:', df['Age'].mean())
print('Median:', df['Age'].median())
print('Sum:', df['Age'].sum())
print('Count:', df['Age'].count())
# Maximum: 71.0
# Minimum: 0.17
# Mean: 30.397989417989418
# Median: 28.0
# Sum: 22980.88
# Count: 756

# See also:
# - var (variance)
# - std (standard deviation)
# - kurt (kurtosis)
# - skew (skewness)
# - sem (standard error of the mean)
# - mode (mode)
# - value counts, and other methods

# Apply these methods to the whole DataFrame
df.count()
# Name        1313
# PClass      1313
# Age          756
# Sex         1313
# Survived    1313
# SexCode     1313
# dtype: int64
