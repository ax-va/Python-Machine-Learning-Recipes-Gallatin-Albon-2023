import pandas as pd

df = pd.DataFrame(
    data={
        "Date 1": [pd.Timestamp('01-01-2017'), pd.Timestamp('01-04-2017')],
        "Date 2": [pd.Timestamp('01-01-2017'), pd.Timestamp('01-06-2017')],
    }
)
#       Date 1     Date 2
# 0 2017-01-01 2017-01-01
# 1 2017-01-04 2017-01-06

# Calculate duration between features
df["Date 2"] - df["Date 1"]
# 0   0 days
# 1   2 days
# dtype: timedelta64[ns]

df["Date 1"] - df["Date 2"]
# 0    0 days
# 1   -2 days
# dtype: timedelta64[ns]

# Remove the days output and keep only the numerical value
pd.Series(timedelta.days for timedelta in (df["Date 2"] - df["Date 1"]))
# 0    0
# 1    2
# dtype: int64
