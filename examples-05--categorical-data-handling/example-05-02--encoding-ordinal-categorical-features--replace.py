import pandas as pd

df = pd.DataFrame(
    {
        "Score": ["Low", "Low", "Medium", "Medium", "High"]
    }
)
#     Score
# 0     Low
# 1     Low
# 2  Medium
# 3  Medium
# 4    High

# Create mapper
scale_mapper = {
    "Low": 1,
    "Medium": 2,
    "High": 3,
}

# Replace feature values with scale
df["Score"].replace(scale_mapper)
# 0    1
# 1    1
# 2    2
# 3    2
# 4    3
# Name: Score, dtype: int64

# # # for non-equal intervals
df = pd.DataFrame(
    {
        "Score": ["Low", "Low", "Medium", "Medium", "High", "Barely More Than Medium"]
    }
)
#                      Score
# 0                      Low
# 1                      Low
# 2                   Medium
# 3                   Medium
# 4                     High
# 5  Barely More Than Medium

scale_mapper = {
    "Low": 1,
    "Medium": 2,
    "Barely More Than Medium": 2.1,
    "High": 4,
}

df["Score"].replace(scale_mapper)
# 0    1.0
# 1    1.0
# 2    2.0
# 3    2.0
# 4    4.0
# 5    2.1
# Name: Score, dtype: float64
