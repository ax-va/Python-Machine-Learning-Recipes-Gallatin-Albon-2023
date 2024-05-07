"""
This example requires you to start a running SQL instance locally
that mimics a remote server on localhost

See:
https://github.com/kylegallatin/mysql-db-example

To start:
$ sudo docker run -it -p 3306:3306 -e MYSQL_ALLOW_EMPTY_PASSWORD=yes kylegallatin/ml-python-cookbook-mysql --secure-file-priv=/

To stop:
$ top
Find PID of mysql, then
$ sudo kill <PID>
"""
import pymysql
import pandas as pd

# Create a DB connection
conn = pymysql.connect(
    host='localhost',
    user='root',
    password="",
    db='db',
)

df = pd.read_sql("select * from data", conn)
df.head(2)
#    integer            datetime  category
# 0        5 2015-01-01 00:00:00         0
# 1        5 2015-01-01 00:00:01         0

df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 100 entries, 0 to 99
# Data columns (total 3 columns):
#  #   Column    Non-Null Count  Dtype
# ---  ------    --------------  -----
#  0   integer   100 non-null    int64
#  1   datetime  100 non-null    datetime64[ns]
#  2   category  100 non-null    int64
# dtypes: datetime64[ns](1), int64(2)
# memory usage: 2.5 KB

# read_sql() is a function created for convenience that may call
# read_sql_query() or read_sql_table() depending on the input we provide
