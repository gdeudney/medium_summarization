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
import lmstudio as lms
import re

# --- Configuration ---
print("NOTE: This script uses the 'lmstudio' library. Please install with: pip install lmstudio")

# A main directory to store all experimental runs for this benchmark
RUNS_DIR = Path("./llm_scorer_runs")
# Directory containing the articles to be evaluated
ARTICLES_DIR = Path("./test_articles")
# The number of times each model evaluates each article to test for consistency
NUM_EVAL_RUNS_PER_ARTICLE = 5

# --- Load Environment Variables ---
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_APIKEY")

# --- 1. Define Model Lists ---
# For OpenRouter - add any models you want to test
OPENROUTER_MODELS = [
    "x-ai/grok-4",
    "x-ai/grok-3",
    "google/gemini-2.0-flash-001",
    "google/gemini-2.5-flash",
    "openai/gpt-4",
    "openai/gpt-4-0314",
    "openai/gpt-4.1",
    "openai/gpt-5",
    "moonshotai/kimi-k2",
    "moonshotai/kimi-k2-0905",
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3.7-sonnet",
    "anthropic/claude-sonnet-4",
    "deepseek/deepseek-chat-v3.1",
    "deepseek/deepseek-v3.1-terminus",
    "qwen/qwen3-30b-a3b",
    "qwen/qwen3-30b-a3b-instruct-2507",
    "qwen/qwen3-235b-a22b",
    "qwen/qwen3-235b-a22b-2507",
    "qwen/qwen3-next-80b-a3b-instruct",
]

# For LM Studio
# IMPORTANT: Use the full model identifier from LM Studio's "Local Server" tab.
# Example: "gemma-2-9b-it-gguf/gemma-2-9b-it-q8_0.gguf"
LMSTUDIO_MODELS = [
    #"magistral-small-2506",
    #"mistralai/magistral-small-2509"
]

# --- 2. Define the Core Scoring Prompt ---

SCORING_PROMPT = """
You are a meticulous and impartial AI quality analyst. Your task is to evaluate a given article based on its overall quality, following a structured rubric.

Read the article provided below. For each of the four pillars of quality, provide a score from 1 to 10. Then, calculate an overall score which is the average of the four pillars.

**Evaluation Criteria (The Four Pillars):**
1.  **Clarity & Readability (1-10):** Is the language clear, concise, and easy to understand? Is the structure logical?
2.  **Depth & Insight (1-10):** Does the article provide valuable, non-obvious information or a unique perspective? Does it demonstrate expertise?
3.  **Engagement (1-10):** Is the article interesting and well-paced? Does it successfully hold the reader's attention?
4.  **Structure & Coherence (1-10):** Is the article well-organized with a logical flow? Do the arguments and points connect seamlessly?

**Article to Evaluate:**
---
{article_text}
---

**YOUR EVALUATION:**

**Output Format (JSON only):**
Your entire response must be a single, valid JSON object. Do not output any text or formatting before or after the JSON.
{{
  "clarity_score": <integer>,
  "depth_score": <integer>,
  "engagement_score": <integer>,
  "structure_score": <integer>,
  "score": <average integer score>,
  "reasoning": "<A brief justification for your overall score, referencing the pillars.>"
}}
"""

# --- 3. Load Test Articles from Disk ---

def load_articles_from_disk(articles_path: Path) -> dict:
    """Loads all .md articles from a given directory, sorted alphabetically."""
    print(f"--- Loading Test Articles from: {articles_path} ---")
    if not articles_path.is_dir():
        print(f"FATAL ERROR: Articles directory '{articles_path}' not found.")
        print("Please create it and add your .md article files (e.g., 'article_01.md').")
        exit()

    articles = {}
    sorted_files = sorted(articles_path.glob("*.md"))

    if not sorted_files:
        print(f"FATAL ERROR: No .md files found in '{articles_path}'.")
        exit()

    for filepath in sorted_files:
        try:
            article_id = filepath.stem
            article_content = filepath.read_text(encoding="utf-8").strip()
            if article_content:
                articles[article_id] = article_content
                print(f"  - Loaded article: {article_id}")
            else:
                print(f"  - WARNING: Skipping empty file: {filepath.name}")
        except Exception as e:
            print(f"!! Could not read file {filepath}: {e}")

    return articles

TEST_ARTICLES = load_articles_from_disk(ARTICLES_DIR)

# --- 4. Define the Evaluation and Logging Functions ---

def get_score_from_model(client_or_model, article_text: str):
    """Generic function to get a score from either an OpenAI client or an LM Studio model object."""
    full_prompt = SCORING_PROMPT.format(article_text=article_text)
    if isinstance(client_or_model, OpenAI):
        response = client_or_model.chat.completions.create(
            model=client_or_model.model,
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content
    else:
        response_chunks = client_or_model.chat.completions.create(
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.1,
            stream=False,
        )
        return response_chunks['choices'][0]['message']['content']


def extract_json_from_string(text: str) -> dict | None:
    """Finds and extracts the first valid JSON object from a string."""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            print(f"   - !! JSON Decode Error on string: {json_str}")
            return None
    return None

def parse_full_response(response: dict) -> dict:
    """Parses all scores and reasoning from a single response JSON."""
    if "error" in response or not isinstance(response, dict):
        return {
            "score": -1, "clarity_score": -1, "depth_score": -1,
            "engagement_score": -1, "structure_score": -1,
            "reasoning": f"Error in response: {response.get('raw_response', 'Unknown error')}"
        }

    try:
        score = int(response.get('score', -1))
        return {
            "score": score if 1 <= score <= 10 else -1,
            "clarity_score": int(response.get('clarity_score', -1)),
            "depth_score": int(response.get('depth_score', -1)),
            "engagement_score": int(response.get('engagement_score', -1)),
            "structure_score": int(response.get('structure_score', -1)),
            "reasoning": str(response.get('reasoning', "No reasoning provided."))
        }
    except (ValueError, TypeError):
        return {
            "score": -1, "clarity_score": -1, "depth_score": -1,
            "engagement_score": -1, "structure_score": -1,
            "reasoning": "Type error parsing scores from response."
        }


def run_evaluation_for_model(client_or_model, display_model_name: str, all_results: list, run_dir: Path):
    """Runs the full evaluation suite for a single model."""
    print(f"\n{'='*20} MODEL: {display_model_name} {'='*20}")

    if isinstance(client_or_model, OpenAI):
        client_or_model.model = display_model_name

    model_evaluation_data = [] # To hold all evaluation data for this model

    for article_id, article_text in TEST_ARTICLES.items():
        print(f" >> Evaluating article: {article_id}")

        run_scores = []
        for i in range(1, NUM_EVAL_RUNS_PER_ARTICLE + 1):
            print(f"   - Run {i}/{NUM_EVAL_RUNS_PER_ARTICLE}...", end="", flush=True)
            raw_response_content = ""
            response_json = {}
            try:
                raw_response_content = get_score_from_model(client_or_model, article_text)
                response_json = extract_json_from_string(raw_response_content)
                if response_json is None:
                    raise ValueError("Could not extract a valid JSON object from the response.")
            except Exception as e:
                print(f" FAILED! Error: {e}")
                response_json = {"error": str(e), "raw_response": raw_response_content}

            parsed_data = parse_full_response(response_json)
            parsed_data["article_id"] = article_id
            parsed_data["run"] = i
            model_evaluation_data.append(parsed_data)

            score = parsed_data["score"]
            if score != -1:
                print(f" Score: {score}/10")
                run_scores.append(score)
            else:
                print(" Invalid score or error in response.")

            time.sleep(1)

        if not run_scores:
            print(f" >> No valid scores for {article_id} with {display_model_name}. Skipping stats.")
            continue

        mean_score = np.mean(run_scores)
        std_dev = np.std(run_scores)
        print(f" >> Results for '{article_id}': Mean Score = {mean_score:.2f}, Std Dev = {std_dev:.2f}")

        all_results.append({
            "scorer_model": display_model_name,
            "article_id": article_id,
            "mean_score": mean_score,
            "std_dev": std_dev
        })

    # Save the consolidated evaluation file for the current model
    sanitized_model_name = display_model_name.replace("/", "_").replace(":", "_")
    eval_file_path = run_dir / f"{sanitized_model_name}_evaluations.json"
    print(f" >> Saving all evaluation data for '{display_model_name}' to {eval_file_path}")
    with eval_file_path.open("w", encoding="utf-8") as f:
        json.dump(model_evaluation_data, f, indent=2, ensure_ascii=False)


# --- 5. The Main Execution Logic ---

def main():
    """Main function to orchestrate the article scoring process."""
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / f"benchmark_complex_{run_timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n--- All benchmark outputs will be saved in: {run_dir} ---")

    all_results = []

    # --- Run LM Studio Benchmark ---
    print("\n--- INITIALIZING LM STUDIO BENCHMARK ---")
    if not LMSTUDIO_MODELS:
        print("No models listed in LMSTUDIO_MODELS. Skipping.")
    else:
        try:
            lmstudio_client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
            for model_id in LMSTUDIO_MODELS:
                run_evaluation_for_model(lmstudio_client, model_id, all_results, run_dir)
        except Exception as e:
            print(f"\n!! Could not connect to LM Studio server. Error: {e}")

    # --- Run OpenRouter Benchmark ---
    print("\n--- INITIALIZING OPENROUTER BENCHMARK ---")
    if not OPENROUTER_API_KEY:
        print("!! OPENROUTER_APIKEY not found. Skipping.")
    elif not OPENROUTER_MODELS:
        print("No models in OPENROUTER_MODELS. Skipping.")
    else:
        openrouter_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
        for model_id in OPENROUTER_MODELS:
            run_evaluation_for_model(openrouter_client, model_id, all_results, run_dir)

    # --- 6. Final Report Generation ---
    if not all_results:
        print("\nNo results were generated. Exiting.")
        return

    results_df = pd.DataFrame(all_results)
    csv_path = run_dir / "article_score_summary.csv"
    results_df.to_csv(csv_path, index=False)

    print(f"\n\n{'='*20} BENCHMARK COMPLETE {'='*20}")
    print(f"Summary report saved to: {csv_path}")

    print("\n--- Model Mean Score by Article ---")
    pivot_df = results_df.pivot_table(index="scorer_model", columns="article_id", values="mean_score")
    print(pivot_df.to_string(float_format="%.2f"))

    print("\n--- Model Consistency (Std Dev) by Article ---")
    pivot_std_dev = results_df.pivot_table(index="scorer_model", columns="article_id", values="std_dev")
    print(pivot_std_dev.to_string(float_format="%.2f"))

if __name__ == "__main__":
    main()

