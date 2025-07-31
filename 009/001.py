# На основе регламента контроля и взыскания дебиторской задолженности создайте нейро-ассистента для ответов на
# вопросы сотрудников по данной базе знаний.


import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import requests
import openai
from langchain.docstore.document import Document
from openai import OpenAI

load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]


def load_document_text(url: str) -> str:
    # Extract the document ID from the URL
    match_ = re.search('/document/d/([a-zA-Z0-9-_]+)', url)
    if match_ is None:
        raise ValueError('Invalid Google Docs URL')
    doc_id = match_.group(1)

    # Download the document as plain text
    response = requests.get(f'https://docs.google.com/document/d/{doc_id}/export?format=txt')
    response.raise_for_status()
    text = response.text

    return text


def answer_index(system, topic, search_index, verbose=1):
    # Поиск релевантных отрезков из базы знаний
    docs = search_index.similarity_search(topic, k=4)
    if verbose: print('\n ===========================================: ')
    message_content = re.sub(r'\n{2}', ' ', '\n '.join([f'\nОтрывок документа №{i+1}\n=====================' + doc.page_content + '\n' for i, doc in enumerate(docs)]))
    if verbose: print('message_content :\n ======================================== \n', message_content)
    client = OpenAI()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Ответь на вопрос сотрудника компании. Не упоминай документ с информацией для ответа сотруднику в ответе. Документ с информацией для ответа сотруднику: {message_content}\n\nВопрос сотрудника: \n{topic}"}
    ]

    if verbose: print('\n ===========================================: ')

    completion = client.chat.completions.create(
        model="gpt-4o-mini",                              # заполните поле необходимым значением
        messages=messages,
        temperature=0                            # заполните поле необходимым значением
    )
    answer = completion.choices[0].message.content
    return answer  # возвращает ответ


data_from_url= load_document_text('https://docs.google.com/document/d/1UD9lSQy6PcCejUy3k5ARdUBNS-GEhXqOjnMLMZ6r488/')

source_chunks=[]
splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
for chunk in splitter.split_text(data_from_url):
    source_chunks.append(Document(page_content=chunk, metadata={"meta":"data"}))
print("Общее количество чанков: ", len(source_chunks))


# Инициализирум модель эмбеддингов
embeddings = OpenAIEmbeddings()


db = FAISS.from_documents(source_chunks,embeddings)


system=("Ты-консультант в компании Simble, ответь на вопрос клиента на основе документа с информацией. Не придумывай ничего от себя, отвечай максимально по документу. Не упоминай Документ с информацией для ответа клиенту. Клиент ничего не должен знать про Документ с информацией для ответа клиенту"
        " Твой основной контингент это люди из дагестана, поэтому отвечай как типичный дагестанец, со всякими их словечками, подколами, терминами, жаргонами.")

topic = input('введите вопрос')
result = answer_index(system,topic,db)
print(result)