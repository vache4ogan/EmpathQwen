import os
import torch
import gc
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ============================================
# 1. ОЧИСТКА ПАМЯТИ И ОКРУЖЕНИЯ
# ============================================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
gc.collect()
torch.cuda.empty_cache()

# ============================================
# 2. ЗАГРУЗКА МОДЕЛИ И ТОКЕНИЗАТОРА
# ============================================
model_id = "Qwen/Qwen2.5-7B-Instruct"  # Можешь поменять на 7B/8B, если запускаешь не на T4

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ============================================
# 3. НАСТРОЙКА LoRA (ЧИСТЫЙ PEFT)
# ============================================
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# ============================================
# 4. ПОДГОТОВКА И ТОКЕНИЗАЦИЯ ТВОИХ ДАННЫХ
# ============================================
# 🛑 Укажи имя своего итогового файла (например, final_empathy_dataset.jsonl)
dataset = load_dataset("json", data_files="/kaggle/input/datasets/va4heo/empath/final_dataset.jsonl", split="train")

# Шаг 1: Форматируем твой JSONL в единую текстовую строку
def format_func(x):
    # Модель будет учиться воспроизводить всю цепочку: Ситуация -> Мысли -> Ответ
    text = f"Ситуация: {x['instruction']}\nМысли: {x['thought']}\nОтвет: {x['output']}{tokenizer.eos_token}"
    return {"text": text}

dataset = dataset.map(format_func, remove_columns=dataset.column_names)

# Шаг 2: Токенизируем
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=1024) # Увеличили до 1024 под русский язык

tokenized_dataset = dataset.map(tokenize_function, remove_columns=["text"])

# Шаг 3: Создаем дата-коллатор
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ============================================
# 5. ПАРАМЕТРЫ ОБУЧЕНИЯ
# ============================================
training_args = TrainingArguments(
    output_dir="./final_psychologist_model",
    per_device_train_batch_size=2,          # Оставляем 2 для 16GB VRAM
    gradient_accumulation_steps=4,        # Эмулируем батч = 8
    max_steps=500,                        # Можно увеличить, если данных стало больше (например, 1000)
    learning_rate=2e-4,
    fp16=True,
    bf16=False,
    logging_steps=10,
    save_strategy="no",
    report_to="none",
    dataloader_num_workers=0,
    optim="paged_adamw_8bit",
)

# ============================================
# 6. ЗАПУСК ТРЕНЕРА
# ============================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

print("🚀 ТРЕНЕР НАСТРОЕН НА ТВОИ ДАННЫЕ. СТАРТУЕМ!")
trainer.train()

# ============================================
# 7. СОХРАНЕНИЕ
# ============================================
trainer.model.save_pretrained("./my_psychologist_final")
tokenizer.save_pretrained("./my_psychologist_final")
print("✅ ПОБЕДА! Модель обучена на твоем датасете, адаптеры сохранены в './my_psychologist_final'")