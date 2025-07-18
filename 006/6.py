# 3. Создайте таблицу товаров из магазина электроники, назовите её df3.
# Должно быть 5 строк и 3 столбца: ['Наименование товара', 'Цена',
# 'Остаток'].
# shape(5, 3).
# После того, как создали таблицу добавьте
# ещё одну строку с умной колонкой(Алиса, Сири..).
# shape(6, 3).
# Результат выведите на экран.
# 4. Отсортируйте получившуюся в п.3 таблицу df3, по столбцу "цена". При
# этом индексы должны идти по порядку. (По возрастанию)
# 5. Сохраните получившуюся в п.4 таблицу в excel формате. Загрузите
# получившийся результат и выведите на экран.


import pandas as pd

df = pd.DataFrame({
    'Наименование товара': ['Товар1', 'Товар2', 'Товар3', 'Товар4', 'Товар5'],
    'Цена': [100, 200, 150, 300, 250],
    'Остаток': [10, 5, 8, 2, 7]
})
print(df.shape)
df.loc[len(df)] = ['Алиса',150,3]
df.sort_values('Цена').to_excel('output.xlsx')



