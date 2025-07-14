# 5. Загрузите файл из указанной директории -
# '/content/sample_data/california_housing_test.csv'. Находится в папке
# sample_data. Преобразуйте столбец housing_median_age к
# целочисленному типу из вещественного, т.е к типу int64. Выведите
# получившийся результат.


import pandas as pd

df = pd.read_csv('sample_data//customers-int.csv',header=0)

df['Subscription Date'] = df['Subscription Date'].astype('int64')
print(df)
print(df.dtypes)