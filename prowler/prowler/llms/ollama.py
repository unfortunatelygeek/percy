import requests
from prowler.llms.base import LLM, APIConnectionError, RateLimitError, APIStatusError
from prowler.core.logger import log  # Ensure logging is imported

class Ollama(LLM):
    def __init__(self, model: str, base_url: str, system_prompt: str = "") -> None:
        super().__init__(system_prompt)
        self.api_url = base_url
        self.model = model
        log.info(f"Ollama initialized with model: {self.model}, API URL: {self.api_url}")

    def create_messages(self, user_prompt: str):
        return user_prompt

    def send_message(self, user_prompt: str, max_tokens: int):
        payload = {
            "model": self.model,
            "prompt": user_prompt,
            "options": {"temperature": 1, "system": self.system_prompt},
            "stream": False,
        }
        log.debug(f"Sending request to Ollama: {payload}")

        try:
            response = requests.post(self.api_url, json=payload)
            log.debug(f"Ollama raw response: {response.status_code} - {response.text}")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            log.error(f"Ollama API request failed: {str(e)}")
            if response is not None:
                log.error(f"Ollama Response: {response.status_code} - {response.text}")

            if response and response.status_code == 429:
                raise RateLimitError("Request was rate-limited") from e
            elif response and response.status_code >= 500:
                raise APIConnectionError("Server could not be reached") from e
            else:
                raise APIStatusError(response.status_code, response.text) from e

    def get_response(self, response):
        log.debug(f"Extracting response from Ollama output: {response}")
        return response["response"]
