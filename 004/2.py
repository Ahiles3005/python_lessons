# 1. Создайте произвольный список длиной 5 с данными целочисленного типа (числа от 0 до 100).
# 2. Измените этот список так, что те числа, которые больше 50 - должны увеличиться на 100, а остальные - возвестись в квадрат.


start_list = [15, 30, 45, 60, 75]

print(start_list)
for index, item in enumerate(start_list):
    if item > 50:
        start_list[index] += 100
    else:
        start_list[index] **= 2

print(start_list)
