import os
import json
import time
import urllib.request
import tarfile
from pathlib import Path
import pandas as pd # 🛑 Используем pandas вместо datasets
from dotenv import load_dotenv
from groq import Groq
from tqdm import tqdm

# 1. Загружаем ключи
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("ОШИБКА: Ключ GROQ_API_KEY не найден в .env!")

client = Groq(api_key=api_key)

# 2. ХАКЕРСКИЙ ПУТЬ: Скачиваем сырые данные напрямую с серверов Facebook
print("📥 Обходим Hugging Face и качаем оригинальный датасет...")
url = "https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz"
project_root = Path(__file__).resolve().parent.parent
raw_data_dir = project_root / "data" / "raw"
processed_data_dir = project_root / "data" / "processed"
tar_path = raw_data_dir / "empatheticdialogues.tar.gz"
extract_dir = raw_data_dir / "empatheticdialogues"
raw_data_dir.mkdir(parents=True, exist_ok=True)
processed_data_dir.mkdir(parents=True, exist_ok=True)

# Скачиваем и распаковываем, если еще не делали этого
if not (extract_dir / "train.csv").exists():
    print("⏳ Загрузка архива (около 28 МБ)...")
    urllib.request.urlretrieve(url, tar_path)
    print("📦 Распаковка архива...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=raw_data_dir)

# Читаем CSV-файл напрямую
print("📊 Обработка данных через pandas...")
df = pd.read_csv(extract_dir / "train.csv", on_bad_lines='skip')

# ФИЛЬТРАЦИЯ: Убираем пустые строки и слишком короткие огрызки текста
valid_prompts = [str(p) for p in df['prompt'].dropna() if len(str(p).split()) > 10]

print("🧹 Извлекаем уникальные жизненные ситуации...")
unique_situations = list(set(valid_prompts))
target_situations = unique_situations[:5000] # Берем 300 для прогона




# 3. Системный промпт (ПЕРЕДЕЛАН ПОД JSON MODE)
SYSTEM_PROMPT = """You are an AI with deep emotional intelligence and profound empathy.
The user will share a raw, real-life situation they are going through.
Your goal is to make them feel profoundly heard, seen, and validated. Do NOT interrogate them like a clinician. Do NOT give unsolicited advice.

CRITICAL LANGUAGE REQUIREMENT:
The user's situation is in English, but your ENTIRE output MUST BE in natural, highly expressive, and empathetic Russian language. No English words allowed in the final output!

Your work has 2 strict steps, outputted as a JSON object:

STEP 1 (field "thought"): Perform an 'Empathy Map' (max 4-5 sentences in Russian):
- Sensory & Somatic: What does it physically feel like to be in their shoes right now?
- Core Emotion: What is the deepest underlying emotion here?
- Validation Strategy: How can I mirror this emotion?
Write in the third person ("Пользователь чувствует...").

STEP 2 (field "output"): Provide a deeply validating, warm, and human response in Russian.
- Share the emotional weight of their situation.
- Ask EXACTLY ONE gentle, open-ended question to help them reflect.
- NEVER advise them to break up or quit.
- NEVER use lists or bullet points.

OUTPUT STRICTLY IN THIS JSON FORMAT:
{
  "thought": "your empathy map analysis in Russian",
  "output": "your deep empathetic response to the user in Russian"
}
"""

output_file = processed_data_dir / "DeepEmpathy_dataset.jsonl"

# 4. УМНЫЙ СТАРТ: Запоминаем, что уже сгенерировали
seen_instructions = set()
if os.path.exists(output_file):
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                seen_instructions.add(data["instruction"].strip())
            except:
                pass
    print(f"🔄 Найдено {len(seen_instructions)} уже готовых записей. Продолжаем работу...")

print(f"🚀 Начинаем генерацию! Данные будут сохранены в {output_file}")

# 5. Основной цикл (режим 'a' - дозапись)
with open(output_file, "a", encoding="utf-8") as f:
    for situation in tqdm(target_situations, desc="Проживание ситуаций"):
        situation = situation.strip()
        
        # Пропускаем, если уже делали
        if situation in seen_instructions:
            continue
            
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"My situation: {situation}"}
                ],
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                temperature=0.70,
                max_tokens=2048, # Увеличили лимит для русского языка
                response_format={"type": "json_object"} # 🛑 ВРУБИЛИ JSON MODE
            )
            
            # Достаем JSON напрямую
            response_json = json.loads(chat_completion.choices[0].message.content)
            
            thought = response_json.get("thought", "Ошибка анализа.")
            output = response_json.get("output", "").strip()
            
            if not output:
                print("\n⚠️ Пустой ответ, пропускаем...")
                continue
                
            # Собираем финальный словарь
            dataset_entry = {
                "instruction": situation,
                "thought": thought,
                "output": output
            }
            
            f.write(json.dumps(dataset_entry, ensure_ascii=False) + "\n")
            seen_instructions.add(situation)
            
            time.sleep(3.5)
            
        except Exception as e:
            print(f"\n⚠️ Ошибка генерации: {e}")
            time.sleep(10)

print("\n✅ Готово! Твоя модель получила огромную дозу эмоционального интеллекта.")
