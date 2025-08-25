from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from openai import OpenAI
import re
import os


def get_text() -> str:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(BASE_DIR, "text.txt")
    with open(path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content


def answer_index(user_question, db):
    docs = db.similarity_search(user_question, k=6)
    message_content = re.sub(r'\n{2}', ' ', '\n '.join(
        [f'\nОтрывок документа №{i + 1}\n=====================' + doc.page_content + '\n' for i, doc in
         enumerate(docs)]))

    client = OpenAI()
    messages = [
        {"role": "system", "content": 'Ты консультант помогающий людям найти верный ответ'},
        {"role": "user",
         "content": f"Ответь на вопрос сотрудника компании. Не упоминай документ с информацией для ответа сотруднику в ответе. Документ с информацией для ответа сотруднику: {message_content}\n\nВопрос сотрудника: \n{user_question}"}
    ]

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0
    )
    answer = completion.choices[0].message.content
    return answer


def get_chunks_db(text_db):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
    source_chunks = []
    for chunk in splitter.split_text(text_db):
        source_chunks.append(Document(page_content=chunk, metadata={"meta": "data"}))
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(source_chunks, embeddings)
    return db



