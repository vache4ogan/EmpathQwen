import os
import tempfile


import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class ModelRunner:
    def __init__(self):
        # Пути к базовой модели и твоим сохраненным адаптерам
        # Замени "Qwen/Qwen2.5-7B-Instruct" на точную базу, если использовал другую (например, не Instruct-версию)
        self.base_model_name = "Qwen/Qwen2.5-7B-Instruct" 
        self.lora_weights_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../my_psychologist_final")
        )
        self.offload_folder = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../.model_offload")
        )
        self.temp_folder = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../.model_tmp")
        )
        os.makedirs(self.offload_folder, exist_ok=True)
        os.makedirs(self.temp_folder, exist_ok=True)
        tempfile.tempdir = self.temp_folder
        self.max_input_tokens = int(os.getenv("MODEL_MAX_INPUT_TOKENS", "2048"))
        self.max_new_tokens = int(os.getenv("MODEL_MAX_NEW_TOKENS", "256"))
        
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA недоступна для PyTorch. Проверь NVIDIA-драйвер "
                "и CUDA-сборку torch командой: "
                "python -c \"import torch; print(torch.cuda.is_available())\""
            )

        print("📥 Загрузка токенайзера...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
        )
        self.tokenizer.truncation_side = "left"
        
        print("📥 Загрузка базовой модели в 4-bit NF4...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            llm_int8_enable_fp32_cpu_offload=True,
        )

        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            max_memory={0: "3200MiB", "cpu": "5GiB"},
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            offload_folder=self.offload_folder,
            offload_state_dict=True,
            offload_buffers=True,
            trust_remote_code=True,
        )
        
        print("🔗 Подключение LoRA адаптеров...")
        self.model = PeftModel.from_pretrained(
            self.base_model,
            self.lora_weights_path,
            offload_folder=self.offload_folder,
            low_cpu_mem_usage=True,
        )
        self.model.eval()  # Переводим в режим инференса
        print("✅ Модель успешно собрана и готова к работе!")

    def generate_response(self, full_prompt: str) -> str:
        """Принимает собранный текст со всей историей, возвращает ТОЛЬКО новый ответ модели"""
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(self.model.get_input_embeddings().weight.device)
        prompt_length = inputs["input_ids"].shape[1]
        
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=0.7,
                do_sample=True,
                eos_token_id=self.tokenizer.eos_token_id,
                # Стоп-строки, чтобы модель не генерировала реплики за человека
                stop_strings=["Пользователь:", "\n\nПользователь"],
                tokenizer=self.tokenizer
            )
            
        generated_tokens = outputs[0, prompt_length:]
        return self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()
