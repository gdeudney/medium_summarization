import os
import openai
import json
from dotenv import load_dotenv
import time
import numpy as np
from pathlib import Path
from datetime import datetime
import csv

# --- Configuration ---
# A main directory to store all validation runs
RUNS_DIR = Path("./validation_runs")
# The number of summaries to generate with the winning prompt
NUM_SUMMARIES_TO_GENERATE = 5
# The number of times the full jury will evaluate each individual summary
NUM_JURY_RUNS_PER_SUMMARY = 5

# --- Load Environment Variables ---
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_APIKEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_APIKEY not found in environment variables. Please set it in your .env file.")

# --- 1. Configure API Client & Models ---
# Client will be configured for OpenRouter
client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# Define the models for each role
OPERATOR_MODEL = "qwen/qwen3-32b"
JUROR_MODELS = [
    "qwen/qwen3-30b-a3b-instruct-2507",
    "deepseek/deepseek-r1-0528-qwen3-8b",
    "mistralai/magistral-small-2506"
]

# --- 2. Define the Core Components & Prompts ---

# The article we will be summarizing in every loop.
article_filepath = Path("./LLM evaluation fundatmentals.md")
print(f"--- Loading Article from: {article_filepath} ---")
try:
    article_to_summarize = article_filepath.read_text(encoding="utf-8")
    print("Article loaded successfully.")
except FileNotFoundError:
    print(f"FATAL ERROR: The file '{article_filepath}' was not found.")
    print("Please make sure the markdown file is in the same directory as the script.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred while reading the file: {e}")
    exit()
    
SOURCE_ARTICLE = article_to_summarize

# The "Winning Prompt" that we will be validating for consistency.
# This prompt was the final output from the optimization script.
WINNING_PROMPT = """
Create a concise, accurate summary of the article that preserves all core information, structural sections, key arguments, and distinctive stylistic elements. The summary must:

- Retain the author’s humorous, conversational tone and **include every named metaphor, analogy, and recurring joke** (e.g., “Digital Darwin Awards,” “squirrel court,” “more cowbell,” “coffee breaks,” “digital gremlins,” “biking to the moon,” “spaghetti and stomachs”) **exactly where they appear in the original context**, as they are critical to both tone and argument.
- Explicitly state the central thesis: that rigorous LLM evaluation is non-negotiable for safety, compliance, product quality, and development efficiency—and that it must be **ongoing, multi-method, and multi-faceted**.
- Cover all major sections in order: motivations for evaluation (safety, compliance, quality, efficiency), automated metrics (accuracy, F1-score, BLEU, ROUGE, METEOR, BERTScore, perplexity), LLM-as-a-judge methods (binary, multi-choice, pairwise, ranking, direct scoring, critique generation), human evaluation, intrinsic vs. extrinsic evaluation, and additional aspects (bias, reliability, non-determinism, efficiency, observability, user satisfaction).
- **Include all named technical concepts and cited sources**, such as: the ArXiv paper on *Real-World Language Model Failures*, Chain-of-Thought (CoT) prompting, Recursive Thought Expansion (RTE), Hierarchical Thought Decomposition (HTD), and the impact of temperature on non-determinism.
- Emphasize that **no single metric suffices**, evaluation must be repeated due to non-determinism, and combining automated, LLM-based, and human evaluation is essential.
- Avoid adding new information, opinions, or interpretations. Do not sacrifice factual completeness or clarity for brevity—but ensure the summary feels like a lively, authentic echo of the original’s voice, not a sterile abstraction.

The result should be both **comprehensive and engaging**: a summary that allows someone to grasp the full scope, evidence, and personality of the article in one read.
"""

# The prompt for the "Jury" LLM, which evaluates the summary.
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
{{
  "faithfulness": {{
    "rank": <integer>,
    "reasoning": "<string>"
  }},
  "coherence": {{
    "rank": <integer>,
    "reasoning": "<string>"
  }},
  "conciseness": {{
    "rank": <integer>,
    "reasoning": "<string>"
  }},
  "coverage": {{
    "rank": <integer>,
    "reasoning": "<string>"
  }},
  "overall_assessment": "<string>"
}}
"""

# --- 3. Define the LLM Interaction Functions ---

def run_operator(summarization_prompt, article):
    """Generates a summary using the provided prompt."""
    print(f">> Running Operator [{OPERATOR_MODEL}] to generate summary...")
    try:
        full_prompt_for_operator = f"{summarization_prompt}\n\n---\n\nArticle to summarize:\n{article}"
        
        response = client.chat.completions.create(
            model=OPERATOR_MODEL,
            messages=[
                {"role": "user", "content": full_prompt_for_operator}
            ],
            # We use a non-zero temperature to test for consistency despite non-determinism.
            temperature=0.7,
            max_tokens=2048
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"!! Operator Error: {e}")
        return None

def run_jury(article, summary, run_dir, summary_num, jury_run_num):
    """Evaluates the summary with a panel of jurors and returns the average score."""
    print(">> Convening the Jury to evaluate summary...")
    all_scores = []
    all_verdicts = {}
    criteria = ["faithfulness", "coherence", "conciseness", "coverage"]
    for juror_model in JUROR_MODELS:
        print(f"   - Juror [{juror_model}] is deliberating...")
        raw_response_content = ""
        try:
            response = client.chat.completions.create(
                model=juror_model,
                messages=[{"role": "user", "content": JURY_PROMPT.format(original_text=article, summary_to_evaluate=summary)}],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            raw_response_content = response.choices[0].message.content
            verdict_json = json.loads(raw_response_content)
            
            score = 0
            if not isinstance(verdict_json, dict):
                raise TypeError(f"Verdict is not a dictionary. Got type: {type(verdict_json)}")

            for criterion in criteria:
                criterion_data = verdict_json.get(criterion, {})
                if isinstance(criterion_data, dict) and 'rank' in criterion_data:
                    score += int(criterion_data.get('rank', 0))
            
            # Save individual juror verdict to a uniquely named file
            sanitized_juror_name = juror_model.replace("/", "_").replace(":", "_")
            jury_output_path = run_dir / f"summary_{summary_num}_jury_run_{jury_run_num}_juror_{sanitized_juror_name}.json"
            with jury_output_path.open("w", encoding="utf-8") as f:
                json.dump(verdict_json, f, indent=2, ensure_ascii=False)

            all_scores.append(score)
            all_verdicts[juror_model] = verdict_json
            print(f"   - Juror [{juror_model}] Score: {score}/20")
        except Exception as e:
            print(f"   - !! Juror Error [{juror_model}]: {e}")
            print(f"   - !! Raw response was: {raw_response_content}")
            all_verdicts[juror_model] = {"error": str(e), "raw_response": raw_response_content}

    if not all_scores:
        return 0, all_verdicts

    average_score = np.mean(all_scores)
    return average_score, all_verdicts

# --- 4. Logging Functions ---

def log_validation_run(log_file, data):
    """Appends the results of a single jury evaluation run to the main log file."""
    with open(log_file, 'a', encoding="utf-8") as f:
        f.write(f"--- SUMMARY {data['summary_num']} / JURY RUN {data['jury_run_num']} ---\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        if data['jury_run_num'] == 1:
             # Only write the summary text once for the first jury run to save space
            f.write(f"**Generated Summary (Full Text):**\n{data['summary_text']}\n\n")
        f.write(f"**Jury Verdicts (Consolidated for this run):**\n{json.dumps(data['verdicts'], indent=2, ensure_ascii=False)}\n\n")
        f.write(f"**AVERAGE SCORE for this run:** {data['score']:.2f} / 20\n\n")
        f.write("="*50 + "\n\n")

def log_summary_average_to_csv(csv_file, summary_num, summary_filename, average_score):
    """Appends a summary's final average score to a CSV file."""
    with open(csv_file, 'a', encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([summary_num, f"{average_score:.2f}", summary_filename])

def log_final_analysis(log_file, all_scores):
    """Appends the final statistical analysis to the log file."""
    mean_score = np.mean(all_scores)
    std_dev = np.std(all_scores)
    
    with open(log_file, 'a', encoding="utf-8") as f:
        f.write("\n" + "="*50 + "\n")
        f.write("--- FINAL STATISTICAL ANALYSIS ---\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Winning Prompt Tested:\n{WINNING_PROMPT}\n\n")
        f.write(f"Total Summaries Generated: {NUM_SUMMARIES_TO_GENERATE}\n")
        f.write(f"Jury Runs Per Summary: {NUM_JURY_RUNS_PER_SUMMARY}\n")
        f.write(f"Total Jury Evaluations Conducted: {len(all_scores)}\n\n")
        f.write(f"**Overall Mean Score: {mean_score:.2f} / 20**\n")
        f.write(f"**Standard Deviation: {std_dev:.2f}**\n")
        f.write("="*50 + "\n")

# --- 5. The Main Validation Loop ---

def main():
    """Main function to run the validation process."""
    # --- Setup Run Directory ---
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / f"validation_{run_timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"--- All outputs will be saved in: {run_dir} ---")
    
    log_file = run_dir / "prompt_validation_log.txt"
    summary_scores_file = run_dir / "summary_scores.csv"

    # Initialize the main log file for this run
    with open(log_file, 'w', encoding="utf-8") as f:
        f.write(f"Prompt Validation Log - Started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Validating Prompt:\n{WINNING_PROMPT}\n\n")

    # Initialize the summary CSV file with headers
    with open(summary_scores_file, 'w', encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["summary_num", "average_score", "summary_text_file"])

    all_final_scores = []

    # --- Outer Loop: Generate N summaries ---
    for i in range(1, NUM_SUMMARIES_TO_GENERATE + 1):
        print(f"\n{'='*20} GENERATING SUMMARY {i}/{NUM_SUMMARIES_TO_GENERATE} {'='*20}")
        
        summary_text = run_operator(WINNING_PROMPT, SOURCE_ARTICLE)
        if not summary_text:
            print(f"!! Halting run due to operator failure on summary {i}.")
            break
        
        summary_filename = f"summary_{i}_text.txt"
        summary_path = run_dir / summary_filename
        summary_path.write_text(summary_text, encoding="utf-8")

        scores_for_this_summary = []
        # --- Inner Loop: Evaluate each summary M times ---
        for j in range(1, NUM_JURY_RUNS_PER_SUMMARY + 1):
            print(f"\n-- Running Jury Evaluation {j}/{NUM_JURY_RUNS_PER_SUMMARY} for Summary {i} --")
            
            score, jury_verdicts = run_jury(SOURCE_ARTICLE, summary_text, run_dir, summary_num=i, jury_run_num=j)
            
            if score > 0:
                all_final_scores.append(score)
                scores_for_this_summary.append(score)
            
            # Log the results of this specific jury run
            log_data = {
                "summary_num": i, "jury_run_num": j, "summary_text": summary_text,
                "score": score, "verdicts": jury_verdicts
            }
            log_validation_run(log_file, log_data)
            time.sleep(2) # Small delay between jury runs
        
        # Calculate and log the average score for the current summary
        if scores_for_this_summary:
            avg_score_for_summary = np.mean(scores_for_this_summary)
            print(f"\n>>>> Average score for Summary {i}: {avg_score_for_summary:.2f} / 20 <<<<\n")
            log_summary_average_to_csv(summary_scores_file, i, summary_filename, avg_score_for_summary)

        print(f"\n{'='*20} COMPLETED ALL JURY RUNS FOR SUMMARY {i} {'='*20}")
        time.sleep(5) # Add a delay between generating each new summary

    # --- Final Analysis ---
    if not all_final_scores:
        print("\n--- No scores were recorded. Cannot perform final analysis. ---")
        return

    mean_score = np.mean(all_final_scores)
    std_dev = np.std(all_final_scores)
    
    print("\n" + "="*25)
    print("--- VALIDATION COMPLETE ---")
    print("="*25)
    print(f"Total Summaries Generated: {NUM_SUMMARIES_TO_GENERATE}")
    print(f"Total Jury Evaluations: {len(all_final_scores)}")
    print(f"\nOverall Mean Score: {mean_score:.2f} / 20")
    print(f"Standard Deviation: {std_dev:.2f}")
    print(f"\nDetailed logs and verdicts saved in: {run_dir}")
    print(f"Summary of scores saved in: {summary_scores_file}")

    # Log the final statistical results to the main log file
    log_final_analysis(log_file, all_final_scores)

if __name__ == "__main__":
    main()
