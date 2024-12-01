import pandas as pd

my_dict = {
    "Name": ['Jacky Jackson', 'Steven Stevenson'],
    "Age": [38, 25],
    "Driver": [True, False],
}

df = pd.DataFrame(my_dict)
#                Name  Age  Driver
# 0     Jacky Jackson   38    True
# 1  Steven Stevenson   25   False

# Add a column for eye color
df["Eyes"] = ["Brown", "Blue"]
df
#                Name  Age  Driver   Eyes
# 0     Jacky Jackson   38    True  Brown
# 1  Steven Stevenson   25   False   Blue



