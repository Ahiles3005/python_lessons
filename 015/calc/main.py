from fastapi import FastAPI

app = FastAPI()


# При помощи фреймворка fastapi создайте API калькулятора.
# Ваш API должен иметь 4 функции-обработчика, которые выполняют операции суммы,
# разности, деления и умножения. Пользователь, при отправке post запроса с 2
# числами должен получить результат выполнения операции.


@app.get('/')
def root():
    return {'message': 'Hello fastApi'}


@app.get('/summ')
def summ(one: int = 0, two: int = 10):
    return one + two

@app.get('/raz')
def raz(one: int = 0, two: int = 10):
    return one - two

@app.get('/delett')
def delett(one: int = 0, two: int = 10):
    return one / two

@app.get('/umnoj')
def umnoj(one: int = 0, two: int = 10):
    return one * two
