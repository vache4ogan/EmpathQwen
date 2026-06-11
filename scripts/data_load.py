import os
import json
import time
from dotenv import load_dotenv
from datasets import load_dataset
from groq import Groq
from tqdm import tqdm

# 1. Загружаем секретный ключ из файла .env
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("ОШИБКА: Ключ GROQ_API_KEY не найден. Проверь файл .env!")

# Инициализируем клиента Groq
client = Groq(api_key=api_key)

# 2. Скачиваем сырые данные (берем 50 штук для тестового прогона)
print("📥 Скачиваем сырые истории с Reddit...")
raw_dataset = load_dataset("Amod/mental_health_counseling_conversations", split="train")

# 🧹 ФИЛЬТРАЦИЯ: Убираем мусор
# Оставляем только те ситуации, где пользователь написал больше 30 слов
print("🧹 Отсеиваем короткие приветствия...")
filtered_dataset = raw_dataset.filter(lambda x: len(str(x['Context']).split()) > 30)

target_dataset = filtered_dataset

# 3. Системный промпт (Душа нашей будущей модели)
SYSTEM_PROMPT = """Ты — гениальный ИИ-психотерапевт. Твоя цель: помочь пользователю глубоко проанализировать его жизненную ситуацию с помощью сократовского диалога, без раздачи банальных советов.

ВНИМАНИЕ: Независимо от того, на каком языке написана ситуация пользователя, весь твой мыслительный процесс и финальный ответ ДОЛЖНЫ БЫТЬ СТРОГО НА РУССКОМ ЯЗЫКЕ. Никаких английских слов в ответе!

Твоя работа делится на 2 строгих этапа, которые нужно записать в соответствующие поля JSON:

ЭТАП 1: СКРЫТЫЙ АНАЛИЗ (поле "thought")
Проведи краткий клинический разбор ситуации (максимум 4-5 предложений):
- Какие когнитивные искажения (чёрно-белое мышление, катастрофизация и т.д.) присутствуют у пользователя?
- Какие скрытые эмоции (страх, стыд, чувство вины) стоят за его словами?
Пиши этот анализ от третьего лица (например: "Пользователь чувствует...", "Возможно, здесь присутствует..."). Не обращайся к пользователю в этом поле!

ЭТАП 2: ЭМПАТИЧНЫЙ ОТВЕТ (поле "output")
Напиши теплый, поддерживающий ответ пользователю.
- Отрази его чувства и валидируй его боль (максимум 2-3 предложения).
- Задай РОВНО ОДИН глубокий, открытый сократовский вопрос, который заставит его задуматься о своих скрытых мотивах.
- НИКОГДА не советуй расстаться или "все бросить".
- НИКОГДА не используй списки, маркеры или лекционный тон.

ВЫДАЙ ОТВЕТ СТРОГО В ФОРМАТЕ JSON следующей структуры:
{
  "thought": "твой скрытый анализ от третьего лица",
  "output": "твой эмпатичный ответ пользователю с одним вопросом"
}
"""

output_file = "EmpathLLaMa_dataset.jsonl"
print(f"🚀 Начинаем генерацию! Данные будут сохранены в {output_file}")

seen_instructions = set()


# Считываем уже готовые идеальные строки, чтобы не делать их заново
if os.path.exists(output_file):
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                seen_instructions.add(data["instruction"].strip())
            except:
                pass
    print(f"🔄 Найдено {len(seen_instructions)} уникальных готовых записей. Пропускаем их...")
# 🛑 ИСПРАВЛЕНИЕ 1: Правильная обрезка Hugging Face датасета

# Открываем файл для добавления новых строк
with open(output_file, "a", encoding="utf-8") as f:
    for item in tqdm(filtered_dataset, desc="Генерация датасета"):
        user_situation = item['Context'].strip()
        
        # Защита от дубликатов: если история уже есть в файле — пропускаем
        if user_situation in seen_instructions:
            continue
            
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"User situation: {user_situation}"}
                ],
                # 🛑 Врубаем 17-миллиардную Llama 4
                model="meta-llama/llama-4-scout-17b-16e-instruct", 
                temperature=0.6,
                max_tokens=4096,
            )
            
            full_response = chat_completion.choices[0].message.content
            
            # Парсим ответ
            if "<think>" in full_response and "</think>" in full_response:
                parts = full_response.split("</think>")
                thought = parts[0].replace("<think>", "").strip()
                output = parts[1].strip()
            else:
                thought = "Ошибка парсинга тегов."
                output = full_response.strip()
            
            # Защита от пустых ответов
            if not output:
                print("\n⚠️ Модель выдала пустой ответ, пропускаем...")
                continue
                
            dataset_entry = {
                "instruction": user_situation,
                "thought": thought,
                "output": output
            }
            
            f.write(json.dumps(dataset_entry, ensure_ascii=False) + "\n")
            seen_instructions.add(user_situation)
            
            time.sleep(3.5) 
            
        except Exception as e:
            print(f"\n⚠️ Ошибка API: {e}")
            time.sleep(10)


print("\n✅ Готово! Открой файл", output_file, "и оцени результат.")



