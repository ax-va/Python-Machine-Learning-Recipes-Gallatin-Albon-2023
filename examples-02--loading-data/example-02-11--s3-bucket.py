import pandas as pd

s3_uri = "s3://machine-learning-python-cookbook/data.csv"

# # Set AWS credentials
# ACCESS_KEY_ID = "x"
# SECRET_ACCESS_KEY = "x"

dataframe = pd.read_csv(
    s3_uri,
    # storage_options={
    #     "key": ACCESS_KEY_ID,
    #     "secret": SECRET_ACCESS_KEY,
    # }
)

dataframe.head(2)
#    integer             datetime  category
# 0        5  2015-01-01 00:00:00         0
# 1        5  2015-01-01 00:00:01         0
