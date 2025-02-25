import google.generativeai as genai
import os
from prowler.llms.base import LLM, APIConnectionError, RateLimitError, APIStatusError

class Gemini(LLM):
    def __init__(self, model: str, base_url: str, system_prompt: str = "") -> None:
        super().__init__(system_prompt)
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel(model) 

    def create_messages(self, user_prompt: str):
        return f"{self.system_prompt}\n{user_prompt}"  # Gemini expects plain text, not dict messages, this is new for me, JS usually handles types (not)

    def send_message(self, messages, max_tokens: int = 512):
        try:
            response = self.model.generate_content(messages, generation_config={"max_output_tokens": max_tokens})
            return response
        except APIConnectionError as e:
            raise APIConnectionError("Server could not be reached") from e
        except RateLimitError as e:
            raise RateLimitError("Request was rate-limited") from e
        except APIStatusError as e:
            raise APIStatusError(e.status_code, e.response) from e

    def get_response(self, response):
        return response.text  
