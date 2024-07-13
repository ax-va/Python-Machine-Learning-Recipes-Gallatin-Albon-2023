"""
Get the day of the week for each date.
"""
import pandas as pd

dates = pd.Series(pd.date_range("2/2/2002", periods=3, freq="ME"))  # ME = the end of the month
# 0   2002-02-28
# 1   2002-03-31
# 2   2002-04-30
# dtype: datetime64[ns]

# Show days of the week
dates.dt.day_name()
# 0    Thursday
# 1      Sunday
# 2     Tuesday
# dtype: object

# Show days of the week (Monday = 0)
dates.dt.weekday
# 0    3
# 1    6
# 2    1
# dtype: int32
