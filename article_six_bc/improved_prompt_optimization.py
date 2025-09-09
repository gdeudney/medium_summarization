import os
import openai
import json
from dotenv import load_dotenv
import time
import numpy as np
from pathlib import Path
from datetime import datetime
import re

# --- Configuration ---
# A main directory to store all experimental runs
RUNS_DIR = Path("./robust_runs")
# The number of times the jury evaluates each summary to get a stable score
NUM_JURY_RUNS_PER_ITERATION = 5

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
OPTIMIZER_MODEL = "qwen/qwen3-235b-a22b-2507"
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
    exit()
except Exception as e:
    print(f"An unexpected error occurred while reading the file: {e}")
    exit()
    
SOURCE_ARTICLE = article_to_summarize

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

# The prompt for the "Optimizer" LLM, which improves the summarization prompt.
OPTIMIZER_PROMPT = """
You are an expert AI Prompt Engineer. Your task is to improve a summarization prompt based on feedback from an AI Jury.
Analyze the provided information: the original article, the failing prompt, the summary it produced, and the Jury's detailed verdict from multiple evaluation runs.
Your goal is to write a new, improved summarization prompt that directly addresses the weaknesses identified by the Jury.
Explain your reasoning for the changes in a brief analysis, then provide the new prompt. The new prompt must be enclosed within <prompt> tags.

**CONTEXT:**

**1. Original Article:**
---
{original_text}
---

**2. Failing Summarization Prompt:**
---
{failing_prompt}
---

**3. Summary Produced by Failing Prompt:**
---
{summary_produced}
---

**4. AI Jury's Verdicts (Feedback from multiple jurors over several runs):**
---
{jury_verdicts}
---

**YOUR TASK:**

Write a new summarization prompt that will achieve a higher and more consistent score from the Jury.

**Analysis of Failure:**
<Your brief analysis of why the old prompt failed based on the jury's feedback>

**New Improved Prompt:**
<prompt>
<Your new prompt here>
</prompt>
"""

# --- 3. Define the LLM Interaction Functions ---

def run_operator(summarization_prompt, article):
    """Generates a summary using the provided prompt."""
    print(f">> Running Operator [{OPERATOR_MODEL}] to generate summary...")
    try:
        full_prompt_for_operator = f"{summarization_prompt}\n\n---\n\nArticle to summarize:\n{article}"
        
        response = client.chat.completions.create(
            model=OPERATOR_MODEL,
            messages=[{"role": "user", "content": full_prompt_for_operator}],
            temperature=0.7, # Use non-zero temperature to get varied outputs for robust testing
            max_tokens=2048
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"!! Operator Error: {e}")
        return None

def run_jury(article, summary, run_dir, iteration, jury_run_num):
    """Evaluates the summary with a panel of jurors and returns the average score for one run."""
    print(f" >> Convening Jury for run {jury_run_num}/{NUM_JURY_RUNS_PER_ITERATION}...")
    all_scores = []
    all_verdicts = {}
    criteria = ["faithfulness", "coherence", "conciseness", "coverage"]
    for juror_model in JUROR_MODELS:
        print(f"    - Juror [{juror_model}] is deliberating...")
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
            
            sanitized_juror_name = juror_model.replace("/", "_").replace(":", "_")
            jury_output_path = run_dir / f"iteration_{iteration}_jury_run_{jury_run_num}_juror_{sanitized_juror_name}.json"
            with jury_output_path.open("w", encoding="utf-8") as f:
                json.dump(verdict_json, f, indent=2, ensure_ascii=False)

            all_scores.append(score)
            all_verdicts[juror_model] = verdict_json
            print(f"    - Juror [{juror_model}] Score: {score}/20")
        except Exception as e:
            print(f"    - !! Juror Error [{juror_model}]: {e}")
            all_verdicts[juror_model] = {"error": str(e), "raw_response": raw_response_content}

    if not all_scores:
        return 0, all_verdicts

    average_score = np.mean(all_scores)
    return average_score, all_verdicts

def run_optimizer(article, failing_prompt, summary, all_jury_verdicts):
    """Generates a new, improved prompt based on multiple jury runs."""
    print(f">> Running Optimizer [{OPTIMIZER_MODEL}] to generate new prompt...")
    try:
        response = client.chat.completions.create(
            model=OPTIMIZER_MODEL,
            messages=[{"role": "user", "content": OPTIMIZER_PROMPT.format(
                original_text=article,
                failing_prompt=failing_prompt,
                summary_produced=summary,
                jury_verdicts=json.dumps(all_jury_verdicts, indent=2)
            )}],
            temperature=0.7,
            max_tokens=2048
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"!! Optimizer Error: {e}")
        return None

def log_results(log_file, data):
    """Appends the results of a full iteration (including multiple jury runs) to the log file."""
    with open(log_file, 'a', encoding="utf-8") as f:
        f.write(f"--- ITERATION {data['iteration']} ---\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Prompt Used:**\n{data['prompt']}\n\n")
        f.write(f"**Generated Summary:**\n{data['summary']}\n\n")
        f.write(f"**Jury Verdicts (Consolidated from {NUM_JURY_RUNS_PER_ITERATION} runs):**\n{json.dumps(data['verdicts'], indent=2, ensure_ascii=False)}\n\n")
        f.write(f"**Individual Run Scores:** {data['run_scores']}\n")
        f.write(f"**AVERAGE SCORE FOR ITERATION:** {data['average_score']:.2f} / 20\n")
        f.write(f"**Standard Deviation:** {data['std_dev']:.2f}\n\n")
        f.write(f"**Optimizer Full Response:**\n{data.get('optimizer_response', 'N/A')}\n\n")
        f.write("="*50 + "\n\n")

def log_final_result(log_file, data, success):
    """Logs the final outcome of the optimization process."""
    with open(log_file, 'a', encoding="utf-8") as f:
        f.write("\n" + "="*50 + "\n")
        f.write("--- FINAL RESULT ---\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        if success:
            f.write(f"SUCCESS: Target score of {data['target_score']} reached.\n")
            f.write(f"Final Winning Prompt: {data['best_prompt']}\n")
            f.write(f"Final Score: {data['best_score']:.2f}\n")
        else:
            f.write(f"FAILURE: Target score of {data['target_score']} not reached after {data['max_iterations']} iterations.\n")
            f.write(f"Best prompt found: {data['best_prompt']}\n")
            f.write(f"Best score achieved: {data['best_score']:.2f}\n")
        f.write("="*50 + "\n")

# --- 4. The Main Optimization Loop ---

def main():
    """Main function to run the robust optimization loop."""
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / f"run_{run_timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"--- All outputs will be saved in: {run_dir} ---")
    
    log_file = run_dir / "robust_optimization_log.txt"

    with open(log_file, 'w', encoding="utf-8") as f:
        f.write(f"Robust Prompt Optimization Log - Started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    current_prompt = "Summarize this article."
    best_prompt = current_prompt
    best_score = 0
    iteration = 1
    max_iterations = 10
    target_score = 19.0  # Adjusted target score for averaged runs

    while best_score < target_score and iteration <= max_iterations:
        print(f"\n{'='*20} ITERATION {iteration}/{max_iterations} {'='*20}")
        print(f"Current Prompt: '{current_prompt}'")

        # 1. Operator generates a single summary for this iteration
        summary = run_operator(current_prompt, SOURCE_ARTICLE)
        if not summary:
            print("!! Halting loop due to operator failure.")
            break
        
        summary_path = run_dir / f"iteration_{iteration}_summary.txt"
        summary_path.write_text(summary, encoding="utf-8")
        print(f"\nGenerated Summary:\n{summary[:300]}...")

        # 2. Jury evaluates the summary multiple times to get a stable score
        iteration_scores = []
        all_verdicts_for_iteration = []
        for j in range(1, NUM_JURY_RUNS_PER_ITERATION + 1):
            score, jury_verdicts = run_jury(SOURCE_ARTICLE, summary, run_dir, iteration, j)
            if score > 0:
                iteration_scores.append(score)
            all_verdicts_for_iteration.append({f"run_{j}": jury_verdicts})
            time.sleep(2) # Delay between jury runs

        if not iteration_scores:
            print("!! No valid scores from jury. Halting loop.")
            break

        # 3. Calculate the average score for the current prompt
        average_score = np.mean(iteration_scores)
        std_dev = np.std(iteration_scores)
        print(f"\n--- Iteration {iteration} Results ---")
        print(f"Individual Run Scores: {[f'{s:.2f}' for s in iteration_scores]}")
        print(f"AVERAGE SCORE: {average_score:.2f} / 20")
        print(f"Standard Deviation: {std_dev:.2f}")

        # 4. Check for new best score and success
        if average_score > best_score:
            print(f"New best score found! {average_score:.2f} > {best_score:.2f}")
            best_score = average_score
            best_prompt = current_prompt

        if best_score >= target_score:
            print(f"\n*** SUCCESS! Target score of {target_score} reached. ***")
            log_results(log_file, {
                "iteration": iteration, "prompt": current_prompt, "summary": summary,
                "verdicts": all_verdicts_for_iteration, "run_scores": iteration_scores,
                "average_score": average_score, "std_dev": std_dev
            })
            log_final_result(log_file, {"target_score": target_score, "best_prompt": best_prompt, "best_score": best_score}, success=True)
            break

        # 5. If not successful, run the optimizer
        print(f"\nScore is less than {target_score}. Optimizing prompt...")
        optimizer_response = run_optimizer(SOURCE_ARTICLE, current_prompt, summary, all_verdicts_for_iteration)
        if not optimizer_response:
            print("!! Halting loop due to optimizer failure.")
            break
        
        # Log the full data for the completed (but not yet successful) iteration
        log_results(log_file, {
            "iteration": iteration, "prompt": current_prompt, "summary": summary,
            "verdicts": all_verdicts_for_iteration, "run_scores": iteration_scores,
            "average_score": average_score, "std_dev": std_dev,
            "optimizer_response": optimizer_response
        })

        # 6. Parse and update to the new prompt
        match = re.search(r'<prompt>(.*?)</prompt>', optimizer_response, re.DOTALL)
        if match:
            new_prompt = match.group(1).strip()
        else:
            print("!! Optimizer did not return a valid new prompt. Reverting to best prompt.")
            new_prompt = best_prompt

        current_prompt = new_prompt
        iteration += 1
        time.sleep(5)

    if best_score < target_score:
        print(f"\n--- Loop finished after {max_iterations} iterations ---")
        log_final_result(log_file, {"target_score": target_score, "best_prompt": best_prompt, "best_score": best_score, "max_iterations": max_iterations}, success=False)

if __name__ == "__main__":
    main()
