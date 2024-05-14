"""
Calculate statistics for rolling (also called moving) time windows in time series data
->
Smooth time series data to dampen the effect of short-term fluctuations.

For example, if we have a time window of three months. The rolling mean then is calculated as follows:
1. mean(January, February, March)
2. mean(February, March, April)
3. mean(March, April, May)
4. ...

Statistics examples for rolling time windows:
- max()
- mean()
- count()
- rolling correlation corr()

See also:
pandas.DataFrame.rolling
https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html

What are Moving Average or Smoothing Techniques?
https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc42.htm
"""
import pandas as pd

time_index = pd.date_range("01/01/2010", periods=5, freq="ME")
# DatetimeIndex(['2010-01-31', '2010-02-28', '2010-03-31', '2010-04-30',
#                '2010-05-31'],
#               dtype='datetime64[ns]', freq='ME')

df = pd.DataFrame(
    data={"price": range(5)},
    index=time_index
)
#             price
# 2010-01-31      0
# 2010-02-28      1
# 2010-03-31      2
# 2010-04-30      3
# 2010-05-31      4

# Calculate rolling mean
df.rolling(window=2).mean()
#             price
# 2010-01-31    NaN
# 2010-02-28    0.5
# 2010-03-31    1.5
# 2010-04-30    2.5
# 2010-05-31    3.5
