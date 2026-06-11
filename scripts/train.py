import os
from dotenv import load_dotenv
from openai import OpenAI

# Загружаем твой ключ
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Обрати внимание: мы стучимся по адресу API Groq!
client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1", 
)

# Запрашиваем модели и печатаем их ID
models = client.models.list()

print("🚀 Доступные модели Groq прямо сейчас:")
for m in models.data:
    print(f" - {m.id}")