import pandas as pd

url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

df = pd.read_csv(url)

# Get the minimum of every column
df.agg("min")
# Name        Abbing, Mr Anthony
# PClass                       *
# Age                       0.17
# Sex                     female
# Survived                     0
# SexCode                      0
# dtype: object

# Apply specific functions to specific sets of columns: mean Age, min and max SexCode
df.agg({"Age": ["mean"], "SexCode": ["min", "max"]})
#             Age  SexCode
# mean  30.397989      NaN
# min         NaN      0.0
# max         NaN      1.0

# Aggregate functions to groups to get more specific, descriptive statistics:
# Get the number of people who survived / didn't survive in each class
df.groupby(
    ["PClass", "Survived"]
).agg(
    {"Survived": ["count"]}
).reset_index()
#   PClass Survived
#                   count
# 0      *        0     1
# 1    1st        0   129
# 2    1st        1   193
# 3    2nd        0   160
# 4    2nd        1   119
# 5    3rd        0   573
# 6    3rd        1   138
