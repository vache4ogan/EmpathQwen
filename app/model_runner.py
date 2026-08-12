import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

class ModelRunner:
    def __init__(self):
        # Пути к базовой модели и твоим сохраненным адаптерам
        # Замени "Qwen/Qwen2.5-7B-Instruct" на точную базу, если использовал другую (например, не Instruct-версию)
        self.base_model_name = "Qwen/Qwen2.5-7B-Instruct" 
        self.lora_weights_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../my_psychologist_final")
        )
        
        print("📥 Загрузка токенайзера...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, trust_remote_code=True)
        
        print("📥 Загрузка базовой модели (в 4/8 bit для экономии VRAM)...")
        # Если запускаешь на мощной карте, можно убрать load_in_8bit=True
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=True, 
            trust_remote_code=True
        )
        
        print("🔗 Подключение LoRA адаптеров...")
        self.model = PeftModel.from_pretrained(self.base_model, self.lora_weights_path)
        self.model.eval() # Переводим в режим инференса
        print("✅ Модель успешно собрана и готова к работе!")

    def generate_response(self, full_prompt: str) -> str:
        """Принимает собранный текст со всей историей, возвращает ТОЛЬКО новый ответ модели"""
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.7,
                do_sample=True,
                eos_token_id=self.tokenizer.eos_token_id,
                # Стоп-строки, чтобы модель не генерировала реплики за человека
                stop_strings=["Пользователь:", "\n\nПользователь"],
                tokenizer=self.tokenizer
            )
            
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Отрезаем промпт, оставляя только то, что сгенерировала модель сейчас
        new_generation = full_text[len(full_prompt):].strip()
        return new_generation
