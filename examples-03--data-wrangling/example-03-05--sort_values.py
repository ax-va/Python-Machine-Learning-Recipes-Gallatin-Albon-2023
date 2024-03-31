import pandas as pd

url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

df = pd.read_csv(url)
# Sort the dataframe by age, show two rows
df.sort_values(by=["Age"]).head(3)
#                                        Name PClass   Age     Sex  Survived  SexCode
# 763  Dean, Miss Elizabeth Gladys (Millvena)    3rd  0.17  female         1        1
# 751  Danbom, Master Gilbert Sigvard Emanuel    3rd  0.33    male         0        0
# 544          Richards, Master George Sidney    2nd  0.80    male         1        0

df.sort_values(by=["Age"], ascending=False).head(3)
#                            Name PClass   Age   Sex  Survived  SexCode
# 505  Mitchell, Mr Henry Michael    2nd  71.0  male         0        0
# 119    Goldschmidt, Mr George B    1st  71.0  male         0        0
# 9        Artagaveytia, Mr Ramon    1st  71.0  male         0        0
