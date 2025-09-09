import os
import openai
from openai import OpenAI
import json
from dotenv import load_dotenv
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# --- Configuration ---
# A main directory to store all experimental runs for this benchmark
RUNS_DIR = Path("./jury_benchmark_runs")
# Directory containing the summaries to be evaluated
SUMMARIES_DIR = Path("./test_summaries")
# The number of times each juror evaluates each summary to test for consistency
NUM_EVAL_RUNS_PER_SUMMARY = 5
# The generic model identifier used by the LM Studio server.
# This is typically 'local-model'. Check your LM Studio server logs to confirm.
LMSTUDIO_API_MODEL_IDENTIFIER = "local-model"


# --- Load Environment Variables ---
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_APIKEY")

# --- 1. Define Model Lists ---
# For OpenRouter
OPENROUTER_JUROR_MODELS = [
    "z-ai/glm-4-32b",
    "z-ai/glm-4.5-air",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3-30b-a3b-instruct-2507",
    "deepseek/deepseek-r1-0528-qwen3-8b",
    "mistralai/magistral-small-2506",
    "nousresearch/hermes-4-70b",
    "qwen/qwen3-235b-a22b-2507",
    "baidu/ernie-4.5-21b-a3b",
    "thudm/glm-4.1v-9b-thinking",
    "thudm/glm-4-32b",
    "qwen/qwen3-4b:free"
]

# For LM Studio
# IMPORTANT: You must manually load the model in the LM Studio UI that corresponds to the identifier below.
LMSTUDIO_JUROR_MODELS = [
    "gpt-oss-20b",
    "qwen/qwen3-4b"
    # Add other local model identifiers here
]

# --- 2. Define the Core Components & Prompts ---

# The article that was summarized. This is the reference text for the jury.
article_filepath = Path("./LLM evaluation fundatmentals.md")
print(f"--- Loading Source Article from: {article_filepath} ---")
try:
    SOURCE_ARTICLE = article_filepath.read_text(encoding="utf-8")
    print("Article loaded successfully.")
except FileNotFoundError:
    print(f"FATAL ERROR: The file '{article_filepath}' was not found.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred while reading the file: {e}")
    exit()

# --- 3. Load Test Summaries from Disk ---

def load_summaries_from_disk(summaries_path: Path) -> dict:
    """Loads all .md summaries from a given directory, sorted alphabetically."""
    print(f"--- Loading Test Summaries from: {summaries_path} ---")
    if not summaries_path.is_dir():
        print(f"FATAL ERROR: Summaries directory '{summaries_path}' not found.")
        print("Please create it and add your .md summary files (e.g., '01_summary_bad.md').")
        exit()

    summaries = {}
    sorted_files = sorted(summaries_path.glob("*.md"))
    
    if not sorted_files:
        print(f"FATAL ERROR: No .md files found in '{summaries_path}'.")
        exit()
        
    for filepath in sorted_files:
        try:
            summary_id = filepath.stem
            summary_content = filepath.read_text(encoding="utf-8").strip()
            if summary_content:
                summaries[summary_id] = summary_content
                print(f"  - Loaded summary: {summary_id}")
            else:
                print(f"  - WARNING: Skipping empty file: {filepath.name}")
        except Exception as e:
            print(f"!! Could not read file {filepath}: {e}")
    
    return summaries

TEST_SUMMARIES = load_summaries_from_disk(SUMMARIES_DIR)


# The prompt for the "Jury" LLM.
JURY_PROMPT = """
You are a meticulous and impartial AI quality analyst, acting as a judge.
Your task is to conduct a reference-free evaluation of a machine-generated summary.
You will compare the provided 'Summary to Evaluate' directly against the 'Original Source Text'.
Your evaluation must be strict, objective, and detailed.

**Evaluation Criteria:**

1.  **Faithfulness (Rank 1-5):** How factually accurate is the summary compared to the source text? (1: Complete garbage, 5: Flawless)
2.  **Coherence (Rank 1-5):** How well-written, logical, and easy to understand is the summary? (1: Incoherent, 5: Exceptionally clear)
3.  **Conciseness (Rank 1-5):** Does the summary avoid fluff, redundancy, and irrelevant details? (1: Very verbose, 5: Perfectly succinct)
4.  **Coverage (Rank 1-5):** How well does the summary capture the most critical points of the source text? (1: Misses most key info, 5: Excellent coverage)

**EVIDENCE:**

**1. Original Source Text:**
---
{original_text}
---

**2. Summary to Evaluate:**
---
{summary_to_evaluate}
---

**YOUR VERDICT:**

**Output Format (JSON only):**
Your entire response must be a single, valid JSON object. Do not include any text or formatting before or after the JSON.
{{
  "faithfulness": {{ "rank": <integer>, "reasoning": "<string>" }},
  "coherence": {{ "rank": <integer>, "reasoning": "<string>" }},
  "conciseness": {{ "rank": <integer>, "reasoning": "<string>" }},
  "coverage": {{ "rank": <integer>, "reasoning": "<string>" }},
  "overall_assessment": "<string>"
}}
"""

# --- 4. Define the Evaluation and Logging Functions ---

def evaluate_summary_with_juror(client, api_model_name, display_model_name, summary, run_dir, summary_id, eval_run_num):
    """Runs a single evaluation of one summary by one juror model."""
    print(f"   - Evaluating with [{display_model_name}], run {eval_run_num}/{NUM_EVAL_RUNS_PER_SUMMARY}...")
    raw_response_content = ""
    try:
        is_local = "localhost" in str(client.base_url)
        response = client.chat.completions.create(
            model=api_model_name, # Use the API-specific model name
            messages=[{"role": "user", "content": JURY_PROMPT.format(original_text=SOURCE_ARTICLE, summary_to_evaluate=summary)}],
            temperature=0.0,
            # Forcing JSON output is more reliable with cloud APIs
            response_format={"type": "json_object"} if not is_local else None,
        )
        raw_response_content = response.choices[0].message.content
        
        # Clean the response to extract only the JSON object
        json_str = raw_response_content.strip().lstrip("```json").rstrip("```")
        verdict_json = json.loads(json_str)

        if not isinstance(verdict_json, dict):
            raise TypeError("Verdict is not a dictionary.")
            
        # Use the display name for logging to keep folders clear
        sanitized_juror_name = display_model_name.replace("/", "_").replace(":", "_")
        verdict_path = run_dir / f"{sanitized_juror_name}"
        verdict_path.mkdir(exist_ok=True)
        file_path = verdict_path / f"{summary_id}_run_{eval_run_num}.json"
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(verdict_json, f, indent=2, ensure_ascii=False)
            
        return verdict_json

    except Exception as e:
        print(f"   - !! Evaluation Error for [{display_model_name}]: {e}")
        print(f"   - !! Raw response content that caused error: '{raw_response_content}'")
        return {"error": str(e), "raw_response": raw_response_content}

def parse_verdict(verdict):
    """Parses scores from a single verdict JSON."""
    scores = {}
    total_score = 0
    criteria = ["faithfulness", "coherence", "conciseness", "coverage"]
    
    if "error" in verdict:
        return {criterion: 0 for criterion in criteria}, 0
        
    for criterion in criteria:
        rank = verdict.get(criterion, {}).get('rank', 0)
        scores[criterion] = int(rank) if rank is not None else 0
        total_score += scores[criterion]
        
    return scores, total_score

def run_evaluation_rounds(eval_client, display_model_name, api_model_name, all_results, run_dir, is_local=False):
    """Runs the core evaluation loops for a given model."""
    if is_local:
        input(f">>> Please ensure model corresponding to '{display_model_name}' is loaded in LM Studio. Press Enter to continue...")

    for summary_id, summary_text in TEST_SUMMARIES.items():
        print(f" >> Testing [{display_model_name}] against: {summary_id}")
        
        run_scores, run_criteria_scores = [], []
        for i in range(1, NUM_EVAL_RUNS_PER_SUMMARY + 1):
            verdict = evaluate_summary_with_juror(eval_client, api_model_name, display_model_name, summary_text, run_dir, summary_id, i)
            criteria_scores, total_score = parse_verdict(verdict)
            
            if "error" not in verdict and total_score > 0:
                print(f"     - Run {i} Score: {total_score}/20")
                run_scores.append(total_score)
                run_criteria_scores.append(criteria_scores)
            else:
                print(f"   - Invalid or error verdict for run {i}. It will be excluded from stats.")
            time.sleep(2)
        
        if not run_scores:
            print(f" >> No valid scores for {summary_id} with {display_model_name}. Skipping.")
            continue

        mean_score = np.mean(run_scores)
        std_dev = np.std(run_scores)
        df_criteria = pd.DataFrame(run_criteria_scores)
        mean_criteria = df_criteria.mean().to_dict()

        print(f" >> Results for {summary_id}: Mean Score = {mean_score:.2f}, Std Dev = {std_dev:.2f}")

        all_results.append({
            "juror_model": display_model_name,
            "summary_id": summary_id,
            "mean_score": mean_score,
            "std_dev": std_dev,
            "mean_faithfulness": mean_criteria.get('faithfulness', 0),
            "mean_coherence": mean_criteria.get('coherence', 0),
            "mean_conciseness": mean_criteria.get('conciseness', 0),
            "mean_coverage": mean_criteria.get('coverage', 0)
        })

# --- 5. The Main Execution Logic ---

def main():
    """Main function to orchestrate the juror benchmarking process."""
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / f"benchmark_{run_timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n--- All benchmark outputs will be saved in: {run_dir} ---")
    
    all_results = []
    
    # --- Run LM Studio Benchmark First ---
    print("\n--- INITIALIZING LM STUDIO BENCHMARK ---")
    try:
        lmstudio_client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        for model_id in LMSTUDIO_JUROR_MODELS:
            print(f"\n{'='*20} BENCHMARKING LOCAL JUROR: {model_id} {'='*20}")
            run_evaluation_rounds(lmstudio_client, model_id, LMSTUDIO_API_MODEL_IDENTIFIER, all_results, run_dir, is_local=True)
    except openai.APIConnectionError:
        print("\n!! Could not connect to LM Studio server at http://localhost:1234.")
        print("!! Please ensure LM Studio is running and the server is started.")
        print("!! Skipping LM Studio benchmark.")
    except Exception as e:
        print(f"\n!! An unexpected error occurred during the LM Studio benchmark: {e}")
        print("!! Skipping LM Studio benchmark.")

    # --- Run OpenRouter Benchmark Second ---
    print("\n--- INITIALIZING OPENROUTER BENCHMARK ---")
    if not OPENROUTER_API_KEY:
        print("!! OPENROUTER_APIKEY not found. Skipping OpenRouter benchmark.")
    else:
        openrouter_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )
        for model_id in OPENROUTER_JUROR_MODELS:
            print(f"\n{'='*20} BENCHMARKING OPENROUTER JUROR: {model_id} {'='*20}")
            # For OpenRouter, the display name and API name are the same
            run_evaluation_rounds(openrouter_client, model_id, model_id, all_results, run_dir)

    # --- 6. Final Report Generation ---
    if not all_results:
        print("\nNo results were generated from any server. Exiting.")
        return

    results_df = pd.DataFrame(all_results)
    csv_path = run_dir / "juror_benchmark_summary.csv"
    results_df.to_csv(csv_path, index=False)
    
    print(f"\n\n{'='*20} BENCHMARK COMPLETE {'='*20}")
    print(f"Summary report saved to: {csv_path}")
    
    print("\n--- Juror Mean Score by Summary Quality ---")
    pivot_df = results_df.pivot_table(index="juror_model", columns="summary_id", values="mean_score")
    print(pivot_df.to_string(float_format="%.2f"))

    print("\n--- Juror Consistency (Standard Deviation) by Summary Quality ---")
    pivot_std_dev = results_df.pivot_table(index="juror_model", columns="summary_id", values="std_dev")
    print(pivot_std_dev.to_string(float_format="%.2f"))

if __name__ == "__main__":
    main()

