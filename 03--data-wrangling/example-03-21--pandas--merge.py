import pandas as pd

employee_data = {
    'employee_id': ['1', '2', '3', '4'],
    'name': ['Amy Jones', 'Allen Keys', 'Alice Bees', 'Tim Horton'],
}
df_employees = pd.DataFrame(employee_data)
#   employee_id        name
# 0           1   Amy Jones
# 1           2  Allen Keys
# 2           3  Alice Bees
# 3           4  Tim Horton

sales_data = {
    'employee_id': ['3', '4', '5', '6'],
    'total_sales': [4376, 2512, 2345, 3855],
}
df_sales = pd.DataFrame(sales_data)
#   employee_id  total_sales
# 0           3         4376
# 1           4         2512
# 2           5         2345
# 3           6         3855

# Merge DataFrames to inner join
pd.merge(df_employees, df_sales, on='employee_id')
#   employee_id        name  total_sales
# 0           3  Alice Bees         4376
# 1           4  Tim Horton         2512

# Merge DataFrames to outer join
pd.merge(df_employees, df_sales, on='employee_id', how='outer')
#   employee_id        name  total_sales
# 0           1   Amy Jones          NaN
# 1           2  Allen Keys          NaN
# 2           3  Alice Bees       4376.0
# 3           4  Tim Horton       2512.0
# 4           5         NaN       2345.0
# 5           6         NaN       3855.0

# Merge DataFrames to left join
pd.merge(df_employees, df_sales, on='employee_id', how='left')
#   employee_id        name  total_sales
# 0           1   Amy Jones          NaN
# 1           2  Allen Keys          NaN
# 2           3  Alice Bees       4376.0
# 3           4  Tim Horton       2512.0

# Specify the column name in each DataFrame to merge on
pd.merge(df_employees, df_sales, left_on='employee_id', right_on='employee_id')
#   employee_id        name  total_sales
# 0           3  Alice Bees         4376
# 1           4  Tim Horton         2512

# If we want to merge on the indexes, we can replace the left_on and right_on with left_index=True and right_index=True
