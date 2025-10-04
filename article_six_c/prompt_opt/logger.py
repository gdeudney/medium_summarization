# prompt_opt/logger.py
import json
import logging
from pathlib import Path

class RunLogger:
    """Configures logging and handles structured file outputs for a run."""
    def __init__(self, base_dir: Path, run_name: str):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.md_log_path = self.base_dir / "optimization_log.md"
        self.jsonl_results_path = self.base_dir / "results.jsonl"

        # Configure root logger
        log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)

        # File handler
        file_handler = logging.FileHandler(self.base_dir / "run.log")
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        root_logger.addHandler(console_handler)

        with self.md_log_path.open("w", encoding="utf-8") as f:
            f.write(f"# Prompt Optimization Run: {run_name}\n\n")

    def log_iteration_md(self, iter_data: dict) -> None:
        """Logs a summary of an iteration to a markdown file."""
        with self.md_log_path.open("a", encoding="utf-8") as f:
            f.write(f"## --- ITERATION {iter_data['iteration']} ---\n\n")
            f.write(f"**Aggregated Score:** `{iter_data['aggregated_score']:.2f}`\n\n")
            f.write(f"**Prompt Used:**\n```\n{iter_data['prompt']}\n```\n\n")
            if iter_data.get('optimizer_response'):
                f.write(f"**Optimizer Full Response:**\n```\n{iter_data['optimizer_response']}\n```\n\n")

    def log_iteration_jsonl(self, iter_data: dict) -> None:
        """Logs detailed iteration data to a JSONL file."""
        with self.jsonl_results_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(iter_data) + "\n")

    def log_final_summary(self, best_prompt: str, best_score: float, success: bool) -> None:
        logging.info("=" * 60)
        logging.info("--- FINAL RESULT ---")
        logging.info(f"Status: {'✅ SUCCESS' if success else '❌ FAILURE'}")
        logging.info(f"Best Score Achieved: {best_score:.2f}")
        logging.info(f"Best Prompt:\n{best_prompt}")
        logging.info("=" * 60)
        
        with self.md_log_path.open("a", encoding="utf-8") as f:
            f.write("\n---\n# FINAL RESULT\n\n")
            f.write(f"**Status:** {'✅ SUCCESS' if success else '❌ FAILURE'}\n")
            f.write(f"**Best Score Achieved:** `{best_score:.2f}`\n\n")
            f.write(f"**Best Prompt:**\n```\n{best_prompt}\n```\n")