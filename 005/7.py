# 2. Создайте в директории /content папку my_folder и запишите в ней 10
# файлов. Пример вывода:
# Папка my_folder создана в текущей директории.
# Файл file_1.txt создан и записан.
# Файл file_2.txt создан и записан.
# Файл file_3.txt создан и записан.
# Файл file_4.txt создан и записан.
# Файл file_5.txt создан и записан.
# Файл file_6.txt создан и записан.
# Файл file_7.txt создан и записан.
# Файл file_8.txt создан и записан.
# Файл file_9.txt создан и записан.
# Файл file_10.txt создан и записан.
# Вернулись в исходную директорию.


import os

myd_dir = os.getcwd()

new_folder = myd_dir + "\\my_folder"

if os.path.isdir(new_folder):
    print('Папка my_folder уже создана.')
    exit()
else:
    os.mkdir(new_folder)
    print('Папка my_folder создана в текущей директории.')

for i in range(1, 11):
    file_name = f"file_{i}.txt"
    if open(new_folder + '/' + file_name, 'a'):
        print(f'Файл {file_name} создан и записан')

print('Вернулись в исходную директорию.')
