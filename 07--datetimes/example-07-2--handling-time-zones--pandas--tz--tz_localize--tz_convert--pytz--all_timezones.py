"""
Add or change time zone information.
"""
import pandas as pd
from pytz import all_timezones

# Add a time zone using tz during creation
pd.Timestamp('2017-05-01 06:00:00', tz='Europe/London')
# Timestamp('2017-05-01 06:00:00+0100', tz='Europe/London')

# Add a time zone to a previously created datetime using tz_localize
dt = pd.Timestamp('2017-05-01 06:00:00')
# Timestamp('2017-05-01 06:00:00')

dt_london = dt.tz_localize('Europe/London')
# Timestamp('2017-05-01 06:00:00+0100', tz='Europe/London')

# Change time zone
dt_london.tz_convert('Africa/Abidjan')
# Timestamp('2017-05-01 05:00:00+0000', tz='Africa/Abidjan')

# Apply tz_localize and tz_convert to every Series entry
dates = pd.Series(pd.date_range('2/2/2002', periods=3, freq='ME'))  # ME = end of month
# 0   2002-02-28
# 1   2002-03-31
# 2   2002-04-30
# dtype: datetime64[ns]

dates.dt.tz_localize('Africa/Abidjan')
# 0   2002-02-28 00:00:00+00:00
# 1   2002-03-31 00:00:00+00:00
# 2   2002-04-30 00:00:00+00:00
# dtype: datetime64[ns, Africa/Abidjan]

all_timezones[0:3]
# ['Africa/Abidjan', 'Africa/Accra', 'Africa/Addis_Ababa']

len(all_timezones)
# 596
