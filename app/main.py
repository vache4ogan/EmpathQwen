from model_runner import ModelRunner
from guardrails import check_safety, CRISIS_MESSAGE
from memory import ConversationMemory
import sys

def main():
    print("🚀 Запуск EmpathQwen Core...")
    # Инициализируем модель и память
    runner = ModelRunner()
    memory = ConversationMemory()
    
    print("\n🛋️ Сессия психотерапии открыта. Напишите 'выход' для завершения.\n")
    
    while True:
        try:
            user_message = input("👤 Вы: ").strip()
            
            if not user_message:
                continue
                
            if user_message.lower() in ['выход', 'exit', 'quit', 'стоп']:
                print("👋 До свидания. Берегите себя!")
                break
            
            # 1. Проверка безопасности (Guardrails)
            if not check_safety(user_message):
                print(f"\n🛋️ Психолог: {CRISIS_MESSAGE}\n")
                print("-" * 50)
                # При триггере суицида лучше очистить память сессии, чтобы сбросить контекст
                memory.clear()
                continue
            
            # 2. Добавление сообщения в память и получение контекста
            memory.add_user_message(user_message)
            full_prompt = memory.get_full_context()
            
            # 3. Генерация ответа моделью
            print("🧠 ИИ думает...")
            ai_raw_response = runner.generate_response(full_prompt)
            
            # 4. Сохраняем ответ модели (вместе с Рассуждением) в память для сохранения контекста
            memory.add_ai_response(ai_raw_response)
            
            # 5. Красивый вывод пользователю (прячем внутреннее рассуждение, если нужно)
            if "Ответ:" in ai_raw_response:
                thought_part, answer_part = ai_raw_response.split("Ответ:", 1)
                # Мы выводим рассуждение в консоль для дебага стартапа, но юзеру в ТГ потом слать не будем
                print(f"\n🔍 [Внутренний анализ]: {thought_part.strip()}")
                print(f"\n🛋️ Психолог: {answer_part.strip()}\n")
            else:
                print(f"\n🛋️ Психолог: {ai_raw_response}\n")
                
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n👋 Программа принудительно завершена.")
            sys.exit(0)

if __name__ == "__main__":
    main()