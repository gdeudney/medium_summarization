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
print("NOTE: This script now uses the 'lmstudio' library. Please install with: pip install lmstudio")

# A main directory to store all experimental runs for this benchmark
RUNS_DIR = Path("./jury_benchmark_runs")
# Directory containing the summaries to be evaluated
SUMMARIES_DIR = Path("./test_summaries")
# The number of times each juror evaluates each summary to test for consistency
NUM_EVAL_RUNS_PER_SUMMARY = 5

# --- Load Environment Variables ---
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_APIKEY")

# --- 1. Define Model Lists ---
# For OpenRouter
OPENROUTER_JUROR_MODELS = [
    
    #"qwen/qwen3-235b-a22b-2507"
]

# For LM Studio
# IMPORTANT: Use the full model identifier from LM Studio's "Local Server" tab.
# Example: "gemma-2-9b-it-gguf/gemma-2-9b-it-q8_0.gguf"
LMSTUDIO_JUROR_MODELS = [
    "glm-4.1v-9b-thinking",
    #"ernie-4.5-21b-a3b-pt",
    #"deepseek-r1-0528-qwen3-8b",
    #"qwen3-30b-a3b-instruct-2507",
    #"gpt-oss-20b",
    #"mistralai/magistral-small-2509",
    "magistral-small-2506",
    #"gemma-3-27b-it",
    #"qwq-32b",
    #"qwen3-32b",
    #"qwen3-4b-instruct-2507",
    
    #"glm-z1-9b-0414",
    #"qwen3-30b-a3b-thinking-2507",
    
    #"hermes-4-70b",
    #"glm-4.5-air",
    #"gpt-oss-120b",
    
    #"THUDM_GLM-4-32B-0414",

    #"mistral-small-3.2-24b-instruct-2506"

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
Your task is to conduct a sophisticated, context-aware evaluation of a machine-generated summary. Your analysis will proceed in two stages: Source Analysis and Contextual Evaluation.
Stage 1: Source Analysis
Before evaluating the summary, you must first analyze and classify the 'Original Source Text'. Identify its type based on the framework below:
Is it a 'Technical Document'? A research paper, legal contract, or engineering spec where the goal is a pure, uncut information distillate. For this type, "dry" is a feature, not a bug.
Is it 'Creative Media'? A novel, poem, or opinion piece where the goal is to capture the feeling and the theme, not just the sequence of events.
Is it a 'Hybrid'? Content that blends factual information with a distinct personality or narrative style. It requires a summary to have the Technical Document's accuracy and the Creative Media's soul.
In your response, state the identified source type and briefly justify your classification. This decision will shape your entire evaluation.

Stage 2: Contextual Evaluation
Now, evaluate the 'Summary to Evaluate' using the Four Pillars, interpreting them through the lens of your Stage 1 analysis.
Evaluation Criteria (The Four Pillars):
Faithfulness (Rank 1-5):
For all types: The summary must not invent facts.
Contextual Lens: For a Technical Document, this is paramount and relates to cold, hard data. For Creative Media, this also includes faithfulness to the original's theme and feeling.
Coverage (Rank 1-5):
Contextual Lens: For a Technical Document, did it cover the key specs and factual arguments? For Creative Media, did it capture the core theme and nuance, not just the plot points? For a Hybrid, did it successfully cover both the core information and the unique perspective that gives it power?
Coherence (Rank 1-5):
For all types: The summary must be a readable, logical piece of text, not a "sentence salad".
Contextual Lens: For Creative Media or a Hybrid, coherence also means successfully recreating the tone or vibe of the original. Does the summary capture the source's soul, or is it a robotic clone?
Conciseness (Rank 1-5):
For all types: The summary must get to the point. Every word must pull its weight.
Analysis & Scoring:
For each of the four pillars, provide a detailed analysis that explicitly references your Stage 1 classification. Explain why the summary succeeds or fails for that specific type of source text. Conclude with an overall assessment.


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
Your entire response must be a single, valid JSON object. Do not output thinking tokens. Do not include any text or formatting before or after the JSON.
{{
  "faithfulness": {{ "rank": <integer>, "reasoning": "<string>" }},
  "coherence": {{ "rank": <integer>, "reasoning": "<string>" }},
  "conciseness": {{ "rank": <integer>, "reasoning": "<string>" }},
  "coverage": {{ "rank": <integer>, "reasoning": "<string>" }},
  "overall_assessment": "<string>"
}}
"""

# --- 4. Define the Evaluation and Logging Functions ---

def get_verdict_from_model(client_or_model, summary_text):
    """Generic function to get a verdict from either an OpenAI client or an LM Studio model object."""
    if isinstance(client_or_model, OpenAI):
        # It's an OpenAI-compatible client (like OpenRouter)
        full_prompt_messages = [{"role": "user", "content": JURY_PROMPT.format(original_text=SOURCE_ARTICLE, summary_to_evaluate=summary_text)}]
        response = client_or_model.chat.completions.create(
            model=client_or_model.model, # The model is attached to the client for this pattern
            messages=full_prompt_messages,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content
    else:
        # It's a loaded LM Studio model object
        full_prompt_string = JURY_PROMPT.format(original_text=SOURCE_ARTICLE, summary_to_evaluate=summary_text)
        response = client_or_model.respond(
            full_prompt_string,
            config={"temperature": 0.0}
        )
        # Return the raw PredictionResult object
        return response

def extract_json_from_string(text):
    """Finds and extracts the first valid JSON object from a string."""
    # Use a regex to find the content between the first { and the last }
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            # Try to parse the extracted string as JSON
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None # Return None if parsing fails
    return None


def run_evaluation_for_model(client_or_model, display_model_name, all_results, run_dir):
    """Runs the full evaluation suite for a single model."""
    print(f"\n{'='*20} BENCHMARKING JUROR: {display_model_name} {'='*20}")
    
    # Attach model name to client for consistent access in the helper function
    if isinstance(client_or_model, OpenAI):
        client_or_model.model = display_model_name

    for summary_id, summary_text in TEST_SUMMARIES.items():
        print(f" >> Testing [{display_model_name}] against: {summary_id}")
        
        run_scores, run_criteria_scores = [], []
        for i in range(1, NUM_EVAL_RUNS_PER_SUMMARY + 1):
            print(f"   - Run {i}/{NUM_EVAL_RUNS_PER_SUMMARY}...")
            response_object = None
            verdict = {}
            try:
                # This can return either a string (OpenAI) or a PredictionResult (LM Studio)
                response_object = get_verdict_from_model(client_or_model, summary_text)
                
                # Explicitly convert the response to a string to handle both cases
                raw_response_content = str(response_object)

                # Attempt to extract JSON from the string
                verdict = extract_json_from_string(raw_response_content)
                
                if verdict is None or not isinstance(verdict, dict):
                    raise ValueError("Could not find a valid JSON object in the response.")

            except Exception as e:
                raw_content_for_log = str(response_object) if response_object else ""
                print(f"   - !! Evaluation Error for [{display_model_name}]: {e}")
                print(f"   - !! Raw response content that caused error: '{raw_content_for_log}'")
                verdict = {"error": str(e), "raw_response": raw_content_for_log}
            
            criteria_scores, total_score = parse_verdict(verdict)
            
            if "error" not in verdict and total_score > 0:
                print(f"     - Score: {total_score}/20")
                run_scores.append(total_score)
                run_criteria_scores.append(criteria_scores)
            else:
                print(f"   - Invalid or error verdict for run {i}. It will be excluded from stats.")

            # Save the raw JSON verdict or the error log
            sanitized_juror_name = display_model_name.replace("/", "_").replace(":", "_")
            verdict_path = run_dir / sanitized_juror_name
            verdict_path.mkdir(exist_ok=True)
            file_path = verdict_path / f"{summary_id}_run_{i}.json"
            
            with file_path.open("w", encoding="utf-8") as f:
                json.dump(verdict, f, indent=2, ensure_ascii=False)

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
        # The connection check is now implicit. If the server isn't running,
        # the first call to lms.llm() inside the loop will raise an exception.
        for model_id in LMSTUDIO_JUROR_MODELS:
            try:
                # Automatically load the model using the direct lms.llm() function
                print(f"-> Loading local model via lms.llm(): {model_id}...")
                model = lms.llm(model_id)
                run_evaluation_for_model(model, model_id, all_results, run_dir)
                
            except Exception as e:
                print(f"!! LM Studio Error for model '{model_id}': {e}")
                print("!! Please ensure the model identifier is correct and the model files are available.")
                continue
    except Exception as e:
        print("\n!! Could not connect to LM Studio server.")
        print("!! Please ensure LM Studio is running and the server is started.")
        print(f"!! Error details: {e}")
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
            run_evaluation_for_model(openrouter_client, model_id, all_results, run_dir)

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

