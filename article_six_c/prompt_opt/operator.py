# prompt_opt/operator.py
import logging
from jinja2 import Template
from .client import ModelClient

class Operator:
    def __init__(self, client: ModelClient, model_name: str, template_str: str):
        self.client = client
        self.model = model_name
        self.tmpl = Template(template_str)

    def run_on_dataset(
        self,
        prompt_text: str,
        dataset: List[str],
        temperature: float,
        max_tokens: int,
    ) -> List[str]:
        """Generates an output for each item in the dataset."""
        outputs = []
        for i, article in enumerate(dataset):
            logging.info(f"Operator processing dataset item {i+1}/{len(dataset)}...")
            rendered = self.tmpl.render(prompt_text=prompt_text, article=article)
            try:
                output = self.client.chat_completion(
                    model=self.model,
                    messages=[{"role": "user", "content": rendered}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                outputs.append(output)
            except Exception as e:
                logging.error(f"Operator failed on item {i+1}: {e}", exc_info=True)
                outputs.append(f"ERROR: Operator failed to generate output.")
        return outputs