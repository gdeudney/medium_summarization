# prompt_opt/client.py
import logging
import lmstudio as lms
from typing import List, Dict, Any
from tenacity import retry, wait_exponential, stop_after_attempt

class ModelClient:
    """A client for interacting with a local LM Studio server."""
    def __init__(self, **kwargs):
        # LM Studio connection is handled by the library, so no base_url/api_key needed.
        # We'll cache loaded models to avoid reloading them on every call.
        self.loaded_models: Dict[str, Any] = {}
        self.api_calls = 0
        # Token tracking is removed as it's not provided by the local server.
        
    def _get_model(self, model_identifier: str):
        """Loads a model or retrieves it from the cache."""
        if model_identifier not in self.loaded_models:
            logging.info(f"Loading LM Studio model for the first time: {model_identifier}")
            self.loaded_models[model_identifier] = lms.llm(model_identifier)
        return self.loaded_models[model_identifier]

    @retry(wait=wait_exponential(multiplier=1, min=2, max=30), stop=stop_after_attempt(3))
    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        response_format: Dict[str, Any] | None = None, # Note: This is now ignored
    ) -> str:
        """
        Generates a response using LM Studio's .respond() method.
        Note: The 'response_format' parameter is specific to OpenAI APIs and will be ignored.
        Your prompt must be engineered to reliably produce JSON for the jury.
        """
        self.api_calls += 1
        
        # 1. Get the model from the server
        llm = self._get_model(model)

        # 2. Adapt the input format from OpenAI's `messages` to a simple string
        if not messages or "content" not in messages[0]:
            raise ValueError("Input 'messages' list is malformed or empty.")
        prompt_text = messages[0]["content"]

        # 3. Prepare the configuration for LM Studio
        config = {
            "temperature": temperature,
            "max_tokens": max_tokens, # LM Studio uses this key for max generation length
        }

        # 4. Generate the response
        response = llm.respond(prompt_text, config=config)
        
        if not response:
            raise ValueError("LM Studio returned an empty response.")
        
        # The response is the direct string content
        return response.strip()