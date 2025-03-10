import google.generativeai as genai
import os
from prowler.llms.base import LLM, APIConnectionError, RateLimitError, APIStatusError
from prowler.core.logger import log  # Import logging

class Gemini(LLM):
    def __init__(self, model: str, base_url: str, system_prompt: str = "") -> None:
        super().__init__(system_prompt)
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel(model)
        self.model_name = model  # Store model name for logging
        log.info(f"Gemini initialized with model: {model}, API key configured")

    def create_messages(self, user_prompt: str):
        full_prompt = f"{self.system_prompt}\n{user_prompt}"
        log.debug(f"Created Gemini prompt with system prompt and user input")
        return full_prompt  # Gemini expects plain text, not dict messages

    def send_message(self, messages, max_tokens: int = 512):
        log.debug(f"Sending request to Gemini model: {self.model_name}")
        log.debug(f"Generation config: max_output_tokens={max_tokens}")
        
        try:
            response = self.model.generate_content(
                messages, 
                generation_config={"max_output_tokens": max_tokens}
            )
            log.debug(f"Received response from Gemini")
            return response
        except APIConnectionError as e:
            log.error(f"Gemini API connection error: {str(e)}")
            raise APIConnectionError("Server could not be reached") from e
        except RateLimitError as e:
            log.error(f"Gemini rate limit exceeded: {str(e)}")
            raise RateLimitError("Request was rate-limited") from e
        except APIStatusError as e:
            log.error(f"Gemini API error: status_code={getattr(e, 'status_code', 'unknown')}")
            raise APIStatusError(getattr(e, 'status_code', 500), getattr(e, 'response', '')) from e
        except Exception as e:
            log.error(f"Unexpected error with Gemini API: {str(e)}")
            raise

    def get_response(self, response):
        log.debug(f"Extracting text from Gemini response")
        return response.text