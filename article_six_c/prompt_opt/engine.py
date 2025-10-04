# prompt_opt/engine.py
import re
import time
import sys
import argparse
import logging
import json
import csv
from pathlib import Path
import numpy as np
from dotenv import load_dotenv

from .config import EngineConfig
from .client import ModelClient
from .operator import Operator
from .jury import JuryCalibrator, JuryEvaluator
from .optimizer import Optimizer
from .logger import RunLogger

class PromptOptimizationEngine:
    PROMPT_TAG_RE = re.compile(r"<prompt>(.*?)</prompt>", re.DOTALL)

    def __init__(self, config: EngineConfig):
        self.cfg = config
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.cfg.run_dir / f"{self.cfg.run_name}_{timestamp}"
        self.logger = RunLogger(self.run_dir, self.cfg.run_name)
        
        self.client = ModelClient(
            base_url=str(self.cfg.provider.base_url),
            api_key=self.cfg.provider.get_api_key()
        )
        self.calibrator = JuryCalibrator(self.client, self.cfg.prompts.jury, self.cfg.evaluation_criteria)
        self.jury = JuryEvaluator(self.client, self.cfg.prompts.jury, self.cfg.evaluation_criteria)
        self.operator = Operator(self.client, self.cfg.models.operator, self.cfg.prompts.operator)
        self.optimizer = Optimizer(self.client, self.cfg.models.optimizer, self.cfg.prompts.optimizer)

        self.best_prompt = self.cfg.initial_prompt
        self.best_score = -1.0
        self.no_improvement_streak = 0

    def _load_dataset(self) -> List[str]:
        path = self.cfg.dataset.path
        fmt = self.cfg.dataset.format
        docs = []
        if fmt == "text":
            files = list(path.glob("*.txt")) + list(path.glob("*.md"))
            for file in files:
                docs.append(file.read_text(encoding="utf-8"))
        elif fmt == "csv":
            with open(path, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    docs.append(row[self.cfg.dataset.csv_column])
        elif fmt == "jsonl":
            with open(path, mode='r', encoding='utf-8') as f:
                for line in f:
                    docs.append(json.loads(line)[self.cfg.dataset.jsonl_key])
        if not docs:
            raise ValueError(f"No documents loaded from dataset path: {path}")
        return docs
    
    def _validate_prompt(self, prompt: str) -> bool:
        for rule in self.cfg.validation.prompt_validation_rules:
            try:
                if not eval(rule, {"prompt": prompt, "len": len}):
                    logging.warning(f"Prompt failed validation rule: '{rule}'")
                    return False
            except Exception as e:
                logging.error(f"Error evaluating validation rule '{rule}': {e}")
                return False
        return True

    @classmethod
    def _extract_new_prompt(cls, raw_text: str) -> str | None:
        if not raw_text: return None
        m = cls.PROMPT_TAG_RE.search(raw_text)
        return m.group(1).strip() if m else None

    def run(self) -> None:
        # --- PHASE 1: CALIBRATION ---
        bias_profile = {}
        if self.cfg.calibration_set:
            bias_profile = self.calibrator.run(
                self.cfg.models.jurors,
                self.cfg.calibration_set,
                self.cfg.model_params.jury.temperature,
            )
        else:
            logging.warning("No calibration set found. Scores will not be normalized.")
        self.jury.configure(self.cfg.models.jurors, bias_profile, self.cfg.model_params.jury.temperature)
        
        # --- PHASE 2: OPTIMIZATION ---
        logging.info(f"--- Starting Optimization against dataset: {self.cfg.dataset.path} ---")
        dataset = self._load_dataset()
        logging.info(f"Loaded {len(dataset)} documents from dataset.")

        current_prompt = self.cfg.initial_prompt
        iteration = 1
        opt_cfg = self.cfg.optimization

        while iteration <= opt_cfg.max_iterations:
            logging.info("="*60)
            logging.info(f"ITERATION {iteration}/{opt_cfg.max_iterations}")

            if not self._validate_prompt(current_prompt):
                logging.error("Current prompt is invalid. Aborting run.")
                break

            summaries = self.operator.run_on_dataset(
                current_prompt, dataset, **self.cfg.model_params.operator.model_dump()
            )
            
            per_item_results = []
            for i, article in enumerate(dataset):
                juror_results = self.jury.evaluate(article, summaries[i])
                avg_item_score = np.mean([res.total_score for res in juror_results.values()])
                per_item_results.append({
                    "score": avg_item_score, "summary": summaries[i],
                    "verdicts": {name: res.raw_verdict for name, res in juror_results.items()}
                })

            aggregated_score = np.mean([res['score'] for res in per_item_results])
            logging.info(f"Aggregated Score for Iteration: {aggregated_score:.2f}")

            if aggregated_score > self.best_score:
                logging.info(f"🎉 New best score! ({aggregated_score:.2f})")
                self.best_score = aggregated_score
                self.best_prompt = current_prompt
                self.no_improvement_streak = 0
            else:
                self.no_improvement_streak += 1
                logging.info(f"No improvement in score. Streak: {self.no_improvement_streak}")
            
            iter_data = {
                "iteration": iteration, "prompt": current_prompt,
                "aggregated_score": aggregated_score, "is_best": self.best_prompt == current_prompt,
                "per_item_results": per_item_results
            }

            if self.best_score >= opt_cfg.target_score:
                logging.info(f"Target score of {opt_cfg.target_score} reached!")
                self.logger.log_iteration_md(iter_data)
                self.logger.log_iteration_jsonl(iter_data)
                break
            
            if self.no_improvement_streak >= opt_cfg.convergence_streak:
                logging.info(f"Convergence reached after {opt_cfg.convergence_streak} iterations with no improvement. Stopping.")
                break

            # --- Optimizer Step ---
            sample_idx = np.argmin([r['score'] for r in per_item_results]) # pick the worst item
            optimizer_response = self.optimizer.improve(
                old_prompt=current_prompt, article_sample=dataset[sample_idx],
                generated_summary=per_item_results[sample_idx]['summary'],
                jury_verdict=per_item_results[sample_idx]['verdicts'],
                **self.cfg.model_params.optimizer.model_dump()
            )
            iter_data["optimizer_response"] = optimizer_response
            
            new_prompt = self._extract_new_prompt(optimizer_response)
            current_prompt = new_prompt if new_prompt else self.best_prompt

            self.logger.log_iteration_md(iter_data)
            self.logger.log_iteration_jsonl(iter_data)
            
            # --- Cost & Rate Limit Checks ---
            if self.client.api_calls >= self.cfg.cost_controls.max_api_calls:
                logging.warning("Max API calls reached. Stopping run.")
                break
            
            time.sleep(self.cfg.cost_controls.rate_limit_delay)
            iteration += 1

        success = self.best_score >= opt_cfg.target_score
        self.logger.log_final_summary(self.best_prompt, self.best_score, success)
        logging.info(f"Find full logs and artifacts in: {self.run_dir}")

if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run the Prompt Optimization Engine.")
    parser.add_argument("config_path", type=str, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    try:
        cfg = EngineConfig.from_yaml(args.config_path)
        engine = PromptOptimizationEngine(cfg)
        engine.run()
    except Exception:
        logging.exception("A fatal error occurred during the run.")
        sys.exit(1)