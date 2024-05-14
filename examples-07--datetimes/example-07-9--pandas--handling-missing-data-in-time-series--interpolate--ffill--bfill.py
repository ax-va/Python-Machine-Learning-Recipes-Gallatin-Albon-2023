"""
Replace missing values with:
- interpolating with interpolate() (available for np.nan but not for pd.NA)
- forward filling with ffill()
- backfilling with bfill()
"""
import numpy as np
import pandas as pd

time_index = pd.date_range("01/01/2010", periods=5, freq="ME")

df = pd.DataFrame(
    data={"sales": [1.0, 2.0, np.nan, np.nan, 5.0]},
    index=time_index,
)
#             sales
# 2010-01-31    1.0
# 2010-02-28    2.0
# 2010-03-31    NaN
# 2010-04-30    NaN
# 2010-05-31    5.0

# Interpolate missing values
df.interpolate()  # default: method="linear"
#             sales
# 2010-01-31    1.0
# 2010-02-28    2.0
# 2010-03-31    3.0
# 2010-04-30    4.0
# 2010-05-31    5.0

df.interpolate(method="quadratic")
#                sales
# 2010-01-31  1.000000
# 2010-02-28  2.000000
# 2010-03-31  3.059808
# 2010-04-30  4.038069
# 2010-05-31  5.000000

# Use "limit" to restrict the number of interpolated values
# and "limit_direction" to set whether to interpolate values
# forward from the last known value before the gap or vice versa
df.interpolate(limit=1, limit_direction="forward")
#             sales
# 2010-01-31    1.0
# 2010-02-28    2.0
# 2010-03-31    3.0
# 2010-04-30    NaN
# 2010-05-31    5.0

# Replace missing values with the last known value (i.e., forward filling)
df.ffill()
#             sales
# 2010-01-31    1.0
# 2010-02-28    2.0
# 2010-03-31    2.0
# 2010-04-30    2.0
# 2010-05-31    5.0

# Replace missing values with the latest known value (i.e., backfilling)
df.bfill()
#             sales
# 2010-01-31    1.0
# 2010-02-28    2.0
# 2010-03-31    5.0
# 2010-04-30    5.0
# 2010-05-31    5.0
