# Пользователь вводит пароль с клавиатуры. Если введенный пароль
# оказывается правильным (в контексте задачи вы знаете правильный
# пароль), то выведите на экран "Добро пожаловать". Иначе выведите на
# экран "Вам отказано в доступе"


current_password = 123456789

input_password = int(input("Введите пароль: "))


if current_password==input_password:
    print('Добро пожаловать')
else:
    print('Вам отказано в доступе')