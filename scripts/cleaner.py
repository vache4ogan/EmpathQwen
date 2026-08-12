import json
import re
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
processed_data_dir = project_root / "data" / "processed"
input_file = processed_data_dir / "DeepEmpathy_dataset.jsonl"
cleaned_file = processed_data_dir / "EDeepEmpathy_dataset_cleaned.jsonl"

good_lines = 0
bad_lines = 0
duplicate_lines = 0
contaminated_lines = 0 # Новый счетчик для языкового мусора

seen_instructions = set() # Сюда будем сохранять уникальные истории

print("🧹 Начинаем терминальную чистку датасета (пустота + дубликаты + английский в ответах)...")

with open(input_file, "r", encoding="utf-8") as infile, \
     open(cleaned_file, "w", encoding="utf-8") as outfile:
    
    for line in infile:
        try:
            data = json.loads(line)
            instruction = data.get("instruction", "").strip()
            output = data.get("output", "").strip()
            
            # 1. Проверяем на пустоту
            if output == "":
                bad_lines += 1
                continue
                
            # 2. Проверяем на дубликаты (уже видели такую историю?)
            if instruction in seen_instructions:
                duplicate_lines += 1
                continue
            
            # 3. ЩИТ ОТ ГАЛЛЮЦИНАЦИЙ: Ищем латинские буквы в финальном ответе
            # Если есть хотя бы одна английская буква (a-z, A-Z) — выбраковываем
            if re.search(r'[a-zA-Z]', output):
                contaminated_lines += 1
                continue
                
            # Если всё ок — сохраняем и запоминаем историю
            seen_instructions.add(instruction)
            outfile.write(line)
            good_lines += 1
            
        except json.JSONDecodeError:
            bad_lines += 1 # Если строка вообще сломалась

print(f"✅ Готово!")
print(f"💎 Сохранено уникальных и кристально чистых строк: {good_lines}")
print(f"🗑️ Вырезано пустых и кривых: {bad_lines}")
print(f"👯 Убито дубликатов: {duplicate_lines}")
print(f"🇬🇧 Вырезано из-за английских слов/артефактов в ответе: {contaminated_lines}")
