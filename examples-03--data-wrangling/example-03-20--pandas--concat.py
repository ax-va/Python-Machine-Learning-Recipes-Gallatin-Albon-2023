import pandas as pd

data_a = {
    'id': ['1', '2', '3'],
    'first': ['Alex', 'Amy', 'Allen'],
    'last': ['Anderson', 'Ackerman', 'Ali'],
}
df_a = pd.DataFrame(data_a)
#   id  first      last
# 0  1   Alex  Anderson
# 1  2    Amy  Ackerman
# 2  3  Allen       Ali


data_b = {
    'id': ['4', '5', '6'],
    'first': ['Billy', 'Brian', 'Bran'],
    'last': ['Bonder', 'Black', 'Balwner'],
}
df_b = pd.DataFrame(data_b)
#   id  first     last
# 0  4  Billy   Bonder
# 1  5  Brian    Black
# 2  6   Bran  Balwner

# Concatenate along the row axis (axis=0)
pd.concat([df_a, df_b], axis=0)
#   id  first      last
# 0  1   Alex  Anderson
# 1  2    Amy  Ackerman
# 2  3  Allen       Ali
# 0  4  Billy    Bonder
# 1  5  Brian     Black
# 2  6   Bran   Balwner

# Concatenate along the column axis (axis=1)
pd.concat([df_a, df_b], axis=1)
#   id  first      last id  first     last
# 0  1   Alex  Anderson  4  Billy   Bonder
# 1  2    Amy  Ackerman  5  Brian    Black
# 2  3  Allen       Ali  6   Bran  Balwner
