import pandas as pd
import numpy as np

# Create date range
time_index = pd.date_range('01/01/2024', periods=100000, freq='30s')
# DatetimeIndex(['2024-01-01 00:00:00', '2024-01-01 00:00:30',
#                '2024-01-01 00:01:00', '2024-01-01 00:01:30',
#                '2024-01-01 00:02:00', '2024-01-01 00:02:30',
#                '2024-01-01 00:03:00', '2024-01-01 00:03:30',
#                '2024-01-01 00:04:00', '2024-01-01 00:04:30',
#                ...
#                '2024-02-04 17:15:00', '2024-02-04 17:15:30',
#                '2024-02-04 17:16:00', '2024-02-04 17:16:30',
#                '2024-02-04 17:17:00', '2024-02-04 17:17:30',
#                '2024-02-04 17:18:00', '2024-02-04 17:18:30',
#                '2024-02-04 17:19:00', '2024-02-04 17:19:30'],
#               dtype='datetime64[ns]', length=100000, freq='30s')

df = pd.DataFrame(index=time_index)
# Empty DataFrame
# Columns: []
# Index: [2024-01-01 00:00:00, 2024-01-01 00:00:30, ...

# Create column of random values
df['Sale_Amount'] = np.random.randint(1, 10, 100000)
#                      Sale_Amount
# 2024-01-01 00:00:00            3
# 2024-01-01 00:00:30            6
# 2024-01-01 00:01:00            4
# 2024-01-01 00:01:30            4
# 2024-01-01 00:02:00            3
# ...                          ...
# 2024-02-04 17:17:30            6
# 2024-02-04 17:18:00            7
# 2024-02-04 17:18:30            9
# 2024-02-04 17:19:00            3
# 2024-02-04 17:19:30            3
#
# [100000 rows x 1 columns]

# Group rows by week, calculate sum per week
df.resample('W').sum()
#             Sale_Amount
# 2024-01-07       100842
# 2024-01-14       100758
# 2024-01-21       100360
# 2024-01-28       100594
# 2024-02-04        96055

# resample requires the index to be a datetime-like value

# Group by two weeks, calculate mean
df.resample('2W').mean()
#             Sale_Amount
# 2024-01-07     5.002083
# 2024-01-21     4.988046
# 2024-02-04     4.975936

# Group by month with the "right" label (the end of the month), count rows
df.resample('M').count()
#             Sale_Amount
# 2024-01-31        89280
# 2024-02-29        10720

# resample returns by default the label of the right "edge" (the last label) of the time group

# Group by month with the "left" label (a day before the beginning of the month), count rows
df.resample('M', label='left').count()
#             Sale_Amount
# 2023-12-31        89280
# 2024-01-31        10720
