from threading import Lock

from app.guardrails import CRISIS_MESSAGE, check_safety
from app.memory import ConversationMemory
from app.model_runner import ModelRunner


class ChatService:
    def __init__(self, model: ModelRunner):
        self.model = model
        self.sessions: dict[str, ConversationMemory] = {}
        self.generation_lock = Lock()

    def send_message(self, session_id: str, message: str) -> tuple[str, bool]:
        memory = self.sessions.setdefault(
            session_id,
            ConversationMemory(),
        )

        if not check_safety(message):
            memory.clear()
            return CRISIS_MESSAGE, True

        memory.add_user_message(message)
        prompt = memory.get_full_context()

        # Одна модель не должна одновременно обрабатывать
        # несколько запросов на одной GPU.
        with self.generation_lock:
            raw_response = self.model.generate_response(prompt)

        memory.add_ai_response(raw_response)

        if "Ответ:" in raw_response:
            _, answer = raw_response.split("Ответ:", 1)
            return answer.strip(), False

        return raw_response.strip(), False

    def clear_session(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)