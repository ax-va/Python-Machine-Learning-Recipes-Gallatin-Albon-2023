"""
Lagging a feature = Creating a feature that is lagged n time periods:
- with Pandas' shift
"""
import pandas as pd

df = pd.DataFrame(
    data={
        "date": pd.date_range("1/1/2001", periods=5, freq="D"),
        "price": [1.1, 2.2, 3.3, 4.4, 5.5],
    }
)
#         date  price
# 0 2001-01-01    1.1
# 1 2001-01-02    2.2
# 2 2001-01-03    3.3
# 3 2001-01-04    4.4
# 4 2001-01-05    5.5

# Lagged values by one row
df["previous_day_price"] = df["price"].shift(1)
df
#         date  price  previous_day_price
# 0 2001-01-01    1.1                 NaN
# 1 2001-01-02    2.2                 1.1
# 2 2001-01-03    3.3                 2.2
# 3 2001-01-04    4.4                 3.3
# 4 2001-01-05    5.5                 4.4
