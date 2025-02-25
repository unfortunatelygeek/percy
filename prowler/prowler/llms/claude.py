import anthropic
from prowler.llms.base import LLM, APIConnectionError, RateLimitError, APIStatusError

class Claude(LLM):
    def __init__(self, model: str, base_url: str, system_prompt: str = "") -> None:
        super().__init__(system_prompt)
        self.client = anthropic.Anthropic(max_retries=3, base_url=base_url)
        self.model = model

    def create_messages(self, user_prompt: str):
        return [{"role": "user", "content": user_prompt}]

    def send_message(self, messages, max_tokens: int):
        try:
            return self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=self.system_prompt,
                messages=messages
            )
        except anthropic.APIConnectionError as e:
            raise APIConnectionError("Server could not be reached") from e
        except anthropic.RateLimitError as e:
            raise RateLimitError("Request was rate-limited") from e
        except anthropic.APIStatusError as e:
            raise APIStatusError(e.status_code, e.response) from e

    def get_response(self, response):
        return response.content[0].text.replace("\n", "")
