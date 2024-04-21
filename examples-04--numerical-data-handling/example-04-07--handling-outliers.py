"""
Mark outliers und reduce their impact on the data distribution by taking the logarithm of features.
"""
import numpy as np
import pandas as pd

houses = pd.DataFrame(
    data={
        'Price': [534433, 392333, 293222, 4322032],
        'Bathrooms': [2, 3.5, 2, 116],
        'Square_Feet': [1500, 2500, 1500, 48000],
    }
)
#      Price  Bathrooms  Square_Feet
# 0   534433        2.0         1500
# 1   392333        3.5         2500
# 2   293222        2.0         1500
# 3  4322032      116.0        48000

houses[houses['Bathrooms'] < 20]
#     Price  Bathrooms  Square_Feet
# 0  534433        2.0         1500
# 1  392333        3.5         2500
# 2  293222        2.0         1500

np.where(houses["Bathrooms"] < 20, 0, 1)
# array([0, 0, 0, 1])

# Then, mark outliers and include the "Outlier" feature
houses["Outlier"] = np.where(houses["Bathrooms"] < 20, 0, 1)
houses
#      Price  Bathrooms  Square_Feet  Outlier
# 0   534433        2.0         1500        0
# 1   392333        3.5         2500        0
# 2   293222        2.0         1500        0
# 3  4322032      116.0        48000        1

# Transform the feature to dampen the effect of the outlier
houses["Log_Square_Feet"] = [np.log(x) for x in houses["Square_Feet"]]
houses
#      Price  Bathrooms  Square_Feet  Outlier  Log_Square_Feet
# 0   534433        2.0         1500        0         7.313220
# 1   392333        3.5         2500        0         7.824046
# 2   293222        2.0         1500        0         7.313220
# 3  4322032      116.0        48000        1        10.778956
