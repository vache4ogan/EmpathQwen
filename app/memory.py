from prompts import START_PROMPT_TEMPLATE, CONTINUE_PROMPT_TEMPLATE

class ConversationMemory:
    def __init__(self):
        self.history = ""

    def add_user_message(self, message: str):
        if self.history == "":
            self.history = START_PROMPT_TEMPLATE.format(situation=message)
        else:
            self.history += CONTINUE_PROMPT_TEMPLATE.format(user_message=message)

    def add_ai_response(self, ai_response: str):
        # Добавляем сгенерированный рассуждающий ответ в историю
        self.history += " " + ai_response

    def get_full_context(self) -> str:
        return self.history

    def clear(self):
        self.history = ""