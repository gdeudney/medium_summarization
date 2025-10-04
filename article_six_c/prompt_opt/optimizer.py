# prompt_opt/optimizer.py
import json
from jinja2 import Template
from .client import ModelClient

class Optimizer:
    def __init__(self, client: ModelClient, model_name: str, template_str: str):
        self.client = client
        self.model = model_name
        self.tmpl = Template(template_str)

    def improve(
        self,
        old_prompt: str,
        article_sample: str,
        generated_summary: str,
        jury_verdict: dict,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Returns the raw text of the improved prompt from the optimizer LLM."""
        # Ensure jury_verdict is serializable
        serializable_verdict = {k: v for k, v in jury_verdict.items() if isinstance(v, (str, int, float, bool, list, dict))}
        
        rendered = self.tmpl.render(
            article_sample=article_sample,
            old_prompt=old_prompt,
            generated_summary=generated_summary,
            jury_verdict=json.dumps(serializable_verdict, indent=2),
        )
        return self.client.chat_completion(
            model=self.model,
            messages=[{"role": "user", "content": rendered}],
            temperature=temperature,
            max_tokens=max_tokens,
        )