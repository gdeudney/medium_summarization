import os
import json
from dotenv import load_dotenv
import numpy as np
from pathlib import Path
from openai import OpenAI
from datetime import datetime

# NOTE: This script runs the summarizer with the BASE prompt only, using a direct API call.
# It does NOT use any DSPy optimization and is intended to establish a pure baseline score.

# --- Configuration ---
# Load environment variables from a .env file for security
load_dotenv()

# 1. Configure API Keys and Models
openrouter_api_key = os.getenv("OPENROUTER_APIKEY")
if not openrouter_api_key:
    raise ValueError("OPENROUTER_APIKEY not found in environment variables. Please set it in your .env file.")

summarizer_model_id = "qwen/qwen3-32b:free"

JUROR_MODELS = [
    "qwen/qwen3-30b-a3b:free",
    "thudm/glm-4.1v-9b-thinking",
    "deepseek/deepseek-r1-0528-qwen3-8b",
    "mistralai/magistral-small-2506"
]

# --- Base Prompts ---
# This is the initial, unoptimized instruction for the summarizer.
summarizer_base_prompt = """Act as an expert technical writer. Your task is to produce a high-quality, summary synthesizing the key findings from the provided articles.

Your summary should:
1.  **Identify the Core Thesis:** Start by stating the central, unifying theme or main purpose of the articles.
2.  **Extract Key Supporting Points:** Identify the most critical facts, data, or conclusions from the articles and weave them together.
3.  **Maintain Neutral Tone:** Summarize the information objectively, without adding personal opinions or interpretations.
4.  **Be Concise and Fluent:** Ensure the summary is well-written, easy to read, and free of redundant phrases.
5.  **Synthesize, Don't Just List:** Combine the information into a cohesive narrative, not just a list of points from each article.
"""

judge_prompt_template = """
You are a meticulous and impartial AI quality analyst, acting as a judge.
Your task is to conduct a reference-free evaluation of a machine-generated summary.
You will compare the provided 'Summary to Evaluate' directly against the 'Original Source Text'.
Your evaluation must be strict, objective, and detailed.

**Evaluation Criteria:**

1.  **Faithfulness (Rank 1-5):** How factually accurate is the summary compared to the source text?
    - 1: Complete garbage. Contains major fabrications and contradicts the source.
    - 3: Mostly accurate, but has minor inaccuracies or hallucinations.
    - 5: Flawless. Perfectly reflects the facts and nuances of the source text.

2.  **Coherence (Rank 1-5):** How well-written, logical, and easy to understand is the summary?
    - 1: Incoherent and confusing. A jumble of words.
    - 3: Understandable, but the flow is somewhat awkward or disjointed.
    - 5: Exceptionally clear, well-structured, and fluent.

3.  **Conciseness (Rank 1-5):** Does the summary avoid fluff, redundancy, and irrelevant details?
    - 1: Very verbose and full of filler.
    - 3: Reasonably concise but could be more direct.
    - 5: Perfectly succinct. Every word serves a purpose.

4.  **Coverage (Rank 1-5):** How well does the summary capture the most critical points and main ideas of the source text?
    - 1: Misses most of the key information.
    - 3: Captures some main ideas but misses important nuances or secondary points.
    - 5: Excellent coverage of all essential concepts from the source.

**EVIDENCE:**

**1. Original Source Text:**
---
{original_text}
---

**2. Summary to Evaluate (from local model):**
---
{summary_to_evaluate}
---

**YOUR VERDICT:**

Provide your final verdict as a single JSON object. For each criterion, provide the rank and detailed reasoning to justify your score. Base your reasoning solely on the evidence provided.

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

# --- Evaluation Function ---
def evaluate_summary(original_context, summary_to_evaluate, run_dir, pass_name="baseline"):
    """
    Uses a jury of LLMs to evaluate the quality of a generated summary.
    """
    all_scores = []
    print(f"\n--- Convening the Jury for {pass_name} summary ---")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
    )

    for juror_model in JUROR_MODELS:
        print(f"Juror [{juror_model}] is now evaluating...")
        evaluation_string = ""
        
        try:
            prompt = judge_prompt_template.format(
                original_text=original_context,
                summary_to_evaluate=summary_to_evaluate
            )

            completion = client.chat.completions.create(
                model=juror_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=15000,
                temperature=0.2,
            )

            evaluation_string = completion.choices[0].message.content
            cleaned_str = evaluation_string.replace("```json", "").replace("```", "").replace("<answer>", "").replace("</answer>", "").strip()
            eval_data = json.loads(cleaned_str)

            if run_dir:
                sanitized_juror_name = juror_model.replace("/", "_").replace(":", "_")
                jury_output_path = run_dir / f"jury_{pass_name}_juror_{sanitized_juror_name}.json"
                with jury_output_path.open("w", encoding="utf-8") as f:
                    json.dump(eval_data, f, indent=2, ensure_ascii=False)
                print(f"Jury response saved to {jury_output_path}")

            faithfulness = eval_data.get("faithfulness", {}).get("rank", 0)
            coherence = eval_data.get("coherence", {}).get("rank", 0)
            conciseness = eval_data.get("conciseness", {}).get("rank", 0)
            coverage = eval_data.get("coverage", {}).get("rank", 0)

            total_score = faithfulness + coherence + conciseness + coverage
            all_scores.append(total_score)
            print(f"Juror [{juror_model}] Score: {total_score}/20")
            print(f"Juror Feedback: {eval_data.get('overall_assessment', 'N/A')}\n")

        except Exception as e:
            print(f"!! Error with juror {juror_model}: {e}")
            print(f"!! Raw output was: {evaluation_string}")

    if not all_scores:
        return 0
    
    return np.mean(all_scores)


# --- Main Execution ---
def main():
    # --- Setup Run Directory ---
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"./run_unoptimized_{run_timestamp}")
    run_dir.mkdir(exist_ok=True)
    print(f"--- All outputs will be saved in: {run_dir} ---")

    # --- Data Loading ---
    article_filepath = Path("./LLM evaluation fundatmentals.md")
    print(f"--- Loading Article from: {article_filepath} ---")
    try:
        article_to_summarize = article_filepath.read_text(encoding="utf-8")
        print("Article loaded successfully.")
    except FileNotFoundError:
        print(f"FATAL ERROR: The file '{article_filepath}' was not found.")
        print("Please make sure the markdown file is in the same directory as the script.")
        return
    except Exception as e:
        print(f"An unexpected error occurred while reading the file: {e}")
        return

    # --- Generate Summary with Base Prompt ---
    print("\n--- Generating Summary with UNOPTIMIZED Base Prompt ---")
    
    try:
        # Manually construct the full prompt
        full_prompt_sent = f"{summarizer_base_prompt}\n\n---\n\nContext: {article_to_summarize}\nSummary:"

        # Save the prompt that will be sent
        full_prompt_path = run_dir / "unoptimized_prompt.txt"
        with full_prompt_path.open("w", encoding="utf-8") as f:
            f.write(full_prompt_sent)
        print(f"Unoptimized prompt saved to {full_prompt_path}")

        # Call the API directly
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_api_key)
        completion = client.chat.completions.create(
            model=summarizer_model_id,
            messages=[{"role": "user", "content": full_prompt_sent}],
            max_tokens=16096,
            temperature=0.7,
        )
        summary_text = completion.choices[0].message.content

        print(f"\nArticle (first 500 chars):\n{article_to_summarize[:500]}...")
        print(f"\nGenerated Summary:\n{summary_text}")

        # --- Save the generated summary to a file ---
        summary_output_path = run_dir / "unoptimized_summary.txt"
        with summary_output_path.open("w", encoding="utf-8") as f:
            f.write(summary_text)
        print(f"Unoptimized summary saved to {summary_output_path}")

        # --- Evaluate the Unoptimized Summary ---
        final_score = evaluate_summary(article_to_summarize, summary_text, run_dir, pass_name="unoptimized")
        print(f"\nFinal Average Score for Unoptimized Summary: {final_score:.2f}/20")

        # --- Save the final score ---
        score_log_path = run_dir / "final_score.log"
        with score_log_path.open("a", encoding="utf-8") as f:
            f.write(f"{run_timestamp}: Final Average Score = {final_score:.2f}/20\n")
        print(f"Final score logged to {score_log_path}")

    except Exception as e:
        print(f"\n!! An error occurred during the summarization call: {e}")


if __name__ == "__main__":
    main()
