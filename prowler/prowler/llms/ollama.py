import requests
from prowler.llms.base import LLM, APIConnectionError, RateLimitError, APIStatusError

class Ollama(LLM):
    def __init__(self, model: str, base_url: str, system_prompt: str = "") -> None:
        super().__init__(system_prompt)
        self.api_url = base_url
        self.model = model

    def create_messages(self, user_prompt: str):
        return user_prompt

    def send_message(self, user_prompt: str, max_tokens: int):
        payload = {
            "model": self.model,
            "prompt": user_prompt,
            "options": {"temperature": 1, "system": self.system_prompt},
            "stream": False,
        }
        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if response.status_code == 429:
                raise RateLimitError("Request was rate-limited") from e
            elif response.status_code >= 500:
                raise APIConnectionError("Server could not be reached") from e
            else:
                raise APIStatusError(response.status_code, response.json()) from e

    def get_response(self, response):
        return response["response"]
