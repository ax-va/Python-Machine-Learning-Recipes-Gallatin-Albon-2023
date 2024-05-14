import pandas as pd

df = pd.DataFrame(
    data={
        "date": pd.date_range('1/1/2001', periods=150, freq='W'),
    }
)
#           date
# 0   2001-01-07
# 1   2001-01-14
# 2   2001-01-21
# 3   2001-01-28
# 4   2001-02-04
# ..         ...
# 145 2003-10-19
# 146 2003-10-26
# 147 2003-11-02
# 148 2003-11-09
# 149 2003-11-16
#
# [150 rows x 1 columns]

# Create features for year, month, day, hour, and minute
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['hour'] = df['date'].dt.hour
df['minute'] = df['date'].dt.minute

df
#           date  year  month  day  hour  minute
# 0   2001-01-07  2001      1    7     0       0
# 1   2001-01-14  2001      1   14     0       0
# 2   2001-01-21  2001      1   21     0       0
# 3   2001-01-28  2001      1   28     0       0
# 4   2001-02-04  2001      2    4     0       0
# ..         ...   ...    ...  ...   ...     ...
# 145 2003-10-19  2003     10   19     0       0
# 146 2003-10-26  2003     10   26     0       0
# 147 2003-11-02  2003     11    2     0       0
# 148 2003-11-09  2003     11    9     0       0
# 149 2003-11-16  2003     11   16     0       0
#
# [150 rows x 6 columns]