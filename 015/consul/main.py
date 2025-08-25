from fastapi import FastAPI, Form
import os
from dotenv import load_dotenv
import requests
import openai
from openai import OpenAI
from .helper import get_text
from .helper import answer_index
from .helper import get_chunks_db
from fastapi.responses import HTMLResponse

load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]

app = FastAPI()


# На основании базы знаний с ПРАВИЛАМИ СТРАХОВАНИЯ ОТВЕТСТВЕННОСТИ АЭРОПОРТОВ И АВИАЦИОННЫХ ТОВАРОПРОИЗВОДИТЕЛЕЙ
# создайте нейро-консультанта, который бы отвечал на вопросы клиентов по информации, содержащейся в представленном
# документе (ДЗ из занятия 6). Создайте API этого нейро-консультанта. Документ с базой знаний можно найти по
# ссылке: файл ПРАВИЛА СТРАХОВАНИЯ ОТВЕТСТВЕННОСТИ.docx
#



@app.get("/", response_class=HTMLResponse)
async def read_root():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(BASE_DIR, "template.html")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content=content)


@app.post('/answer')
def answer(text: str = Form(...)):
    text_db = get_text()
    db_chunks = get_chunks_db(text_db)
    return answer_index(text, db_chunks)

