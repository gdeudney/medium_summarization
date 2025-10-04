# prompt_opt/jury.py
import json
import logging
from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np
from jinja2 import Template
from .client import ModelClient
from .config import CalibrationItem, Criterion

@dataclass
class JuryResult:
    raw_verdict: Dict[str, Any]
    total_score: float

class BaseJuryScorer:
    """Shared scoring logic for calibration and evaluation."""
    def __init__(self, client: ModelClient, template_str: str, criteria: List[Criterion]):
        self.client = client
        self.tmpl = Template(template_str)
        self.criteria = criteria
        self.criteria_names = [c.name for c in criteria]
        self.criteria_weights = {c.name: c.weight for c in criteria}

    def _score_one(self, model: str, article: str, summary: str, temperature: float) -> float:
        rendered_prompt = self.tmpl.render(criteria=self.criteria, article=article, summary=summary)
        raw_json = self.client.chat_completion(
            model=model,
            messages=[{"role": "user", "content": rendered_prompt}],
            temperature=temperature,
            max_tokens=1024, # Jury responses should be concise
            response_format={"type": "json_object"},
        )
        parsed = json.loads(raw_json)
        
        total_weighted_score = 0
        for name in self.criteria_names:
            entry = parsed.get(name) or parsed.get(name.lower())
            if entry is None:
                raise ValueError(f"Juror '{model}' verdict missing required criterion: '{name}'")

            rank = 0
            if isinstance(entry, dict):
                rank = int(entry.get("rank", 0))
            elif isinstance(entry, (int, float)):
                rank = int(entry)
            
            total_weighted_score += rank * self.criteria_weights[name]
        return float(total_weighted_score)

class JuryCalibrator(BaseJuryScorer):
    def run(self, juror_models: List[str], golden_set: List[CalibrationItem], temp: float) -> Dict[str, float]:
        bias_profile = {}
        logging.info("--- Starting Jury Calibration ---")
        for model in juror_models:
            model_scores, truth_scores = [], []
            for item in golden_set:
                try:
                    score = self._score_one(model, item.source_text, item.output, temp)
                    model_scores.append(score)
                    truth_scores.append(item.ground_truth_score)
                except Exception as e:
                    logging.warning(f"Calibration scoring failed for {model}: {e}")

            if not model_scores:
                bias = 0.0
                logging.error(f"Juror '{model}' failed to score any calibration items.")
            else:
                bias = np.mean(model_scores) - np.mean(truth_scores)
            
            bias_profile[model] = bias
            logging.info(f"Juror '{model}' has a bias of {bias:+.2f}")
        return bias_profile

class JuryEvaluator(BaseJuryScorer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.juror_models = []
        self.bias_profile: Dict[str, float] = {}
        self.temperature = 0.0

    def configure(self, juror_models: List[str], bias_profile: Dict[str, float], temp: float):
        self.juror_models = juror_models
        self.bias_profile = bias_profile
        self.temperature = temp

    def evaluate(self, article: str, summary: str) -> Dict[str, JuryResult]:
        results = {}
        for model in self.juror_models:
            verdict, total_score = {}, 0.0
            try:
                raw_score = self._score_one(model, article, summary, self.temperature)
                model_bias = self.bias_profile.get(model, 0.0)
                normalized_score = raw_score - model_bias
                total_score = normalized_score
            except Exception as e:
                verdict = {"error": str(e)}
                total_score = 0.0
                logging.warning(f"Evaluation by juror '{model}' failed: {e}")
            results[model] = JuryResult(raw_verdict=verdict, total_score=total_score)
        return results