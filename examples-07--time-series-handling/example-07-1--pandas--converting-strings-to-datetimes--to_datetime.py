"""
Convert strings representing dates and times to time series data.
"""
import numpy as np
import pandas as pd

datetime_strings = np.array(
    [
        '03-04-2005 11:35 PM',
        '23-05-2010 12:01 AM',
        '04-09-2009 09:09 PM',
    ]
)

# Convert to datetimes
[pd.to_datetime(dt_str, format='%d-%m-%Y %I:%M %p') for dt_str in datetime_strings]
# [Timestamp('2005-04-03 23:35:00'),
#  Timestamp('2010-05-23 00:01:00'),
#  Timestamp('2009-09-04 21:09:00')]

# Add an argument to the errors parameter to handle problems:
# if errors="coerce" and an error occurs -> no exception will be raised and set the value to NaT (not a time)
[pd.to_datetime(dt_str, format="%d-%m-%Y %I:%M %p", errors="coerce") for dt_str in datetime_strings]
# [Timestamp('2005-04-03 23:35:00'),
#  Timestamp('2010-05-23 00:01:00'),
#  Timestamp('2009-09-04 21:09:00')]

# some common date and time formatting codes:
# code          description                         example
# %Y            full year                           2012
# %m            month with zero padding             08
# %d            day with zero padding               01
# %I            12-hour clock with zero padding     06
# %p            AM or PM                            AM
# %M            minutes with zero padding           05
# %S            seconds with zero padding           05

# See also:
# Python strftime cheatsheet
# https://strftime.org/
