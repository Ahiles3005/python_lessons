

#Напишите функцию, которая принимает ввод от пользователя и возвращает ответ от модели GPT,
# а также подсчитывает количество токенов. Ваша функция должна быть способна обрабатывать
# любой ввод пользователя.


import openai
import os
from openai import OpenAI
from langchain_openai import ChatOpenAI
from  langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage
)



os.environ["OPENAI_API_KEY"] = openai_key
openai.api_key = openai_key


def get_gpt_response(user_input):
    client = OpenAI()  # создается экземпляр класса OpenAI
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": user_input}],
    )

    messages = [
        HumanMessage(content=user_input)
    ]
    chat = ChatOpenAI(temperature=0)
    count_token = chat.get_num_tokens_from_messages(messages)

    return {'Ответ':stream.choices[0].message.content,'Токены':count_token}



text = input('Введите вопрос в gpt: ')

result = get_gpt_response(text)

print(result)
