
# Напишите функцию, которая принимает ввод от пользователя и возвращает ответ от модели GPT.
# Ваша функция должна использовать код, представленный в материалах, и должна быть способна
# обрабатывать любой ввод пользователя.


import openai
import os
from openai import OpenAI



os.environ["OPENAI_API_KEY"] = openai_key
openai.api_key = openai_key


def get_gpt_response(user_input):
    client = OpenAI()  # создается экземпляр класса OpenAI
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": user_input}],
    )

    return stream.choices[0].message.content


text = input('Введите вопрос в gpt: ')

result = get_gpt_response(text)

print(result)
