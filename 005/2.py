# 2. Создайте текстовый файл с именем "sample.txt". Запишите в него текст
# '== Data Science ==', а потом прочитайте его в верхнем регистре.
# Текст не должен уезжать. Пример:
# print(res.upper())
# == DATA SCIENCE ==



with open('2.txt','w') as file:
    file.write('== Data Science ==')
    file.close()

with open('2.txt','r') as file:
    res = file.read()
    print(res.upper())