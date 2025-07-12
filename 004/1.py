# Создайте список с числами от 15 до 30 включительно с помощью цикла
# for.


test_list = []

for i in  range(15,31):
    test_list.append(i)



# 2. Защитите от изменений этот список, преобразовав его в иную структуру
# данных.

print(type(test_list))
print(test_list)
test_list = tuple(test_list)

print(type(test_list))
print(test_list)


# 3. Продемонстрируйте неизменяемость данных в ней попыткой изменить
# предпоследний элемент, при этом взяв его по отрицательному индексу.
print(test_list[-2])
test_list[-2] = 666

print(test_list)