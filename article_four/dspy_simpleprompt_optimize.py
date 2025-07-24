import dspy
import os
import json
from dotenv import load_dotenv
import numpy as np
from pathlib import Path
from openai import OpenAI
from types import SimpleNamespace
from datetime import datetime

# NOTE: We are now using the modern dspy.LM class to initialize models.
# This is a more unified and up-to-date approach.

# --- Configuration ---
# Load environment variables from a .env file for security
load_dotenv()

# 1. Configure the Summarizer LLM (Student) via OpenRouter
# This model will be doing the summarization.
openrouter_api_key = os.getenv("OPENROUTER_APIKEY")
if not openrouter_api_key:
    raise ValueError("OPENROUTER_APIKEY not found in environment variables. Please set it in your .env file.")

# Corrected the summarizer model and increased the token limit as requested.
summarizer_model_id = "openrouter/qwen/qwen3-32b:free"
summarizer_lm = dspy.LM(
    summarizer_model_id,
    api_base="https://openrouter.ai/api/v1",
    api_key=openrouter_api_key,
    max_tokens=16096, # Increased token limit for summarization
    temperature=0.7,
)

# 2. Configure the Jury of LLMs via OpenRouter
# This list contains the models that will act as our jury.
JUROR_MODELS = [
    "qwen/qwen3-30b-a3b:free",
    "thudm/glm-4.1v-9b-thinking",
    "deepseek/deepseek-r1-0528-qwen3-8b",
    "mistralai/magistral-small-2506"
]

# 3. Configure DSPy Settings
# We tell DSPy which model to use for the main task (lm).
dspy.settings.configure(lm=summarizer_lm)


# --- DSPy Signatures ---
# Signatures define the input/output behavior of our prompts.

class Summarize(dspy.Signature):
    """Summarize this article"""
    context = dspy.InputField(desc="A collection of articles combined into a single text, separated by '---'.")
    summary = dspy.OutputField(desc="A single paragraph synthesized summary.")

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

# Counter for tracking optimization passes
optimization_pass_counter = 0

# --- DSPy Metric ---
# This custom metric will guide the optimization process.
# It uses a JURY of LLMs, averages their scores, and returns the result.
def summary_quality_metric(gold, pred, trace=None, run_dir=None, pass_num=None):
    """
    A DSPy metric that uses a jury of LLMs to evaluate the quality of a generated summary.
    It returns an average score from 0 to 20.
    """
    global optimization_pass_counter
    if pass_num is None:
        optimization_pass_counter += 1
        current_pass = optimization_pass_counter
    else:
        current_pass = pass_num

    original_text = gold.context
    summary_to_evaluate = pred.summary

    # --- Save the prompt and summary from the optimization pass ---
    if run_dir:
        # Save the summary
        summary_output_path = run_dir / f"summary_pass_{current_pass}.txt"
        print(f"\n--- Saving optimization pass summary to {summary_output_path} ---")
        try:
            summary_output_path.write_text(summary_to_evaluate, encoding="utf-8")
            print("Intermediate summary saved successfully.")
        except Exception as e:
            print(f"!! Could not save intermediate summary to file: {e}")
        
        # Save the prompt that generated this summary
        if summarizer_lm.history and len(summarizer_lm.history) > 0:
            prompt_sent = summarizer_lm.history[-1]['prompt']
            prompt_path = run_dir / f"prompt_pass_{current_pass}.txt"
            try:
                prompt_path.write_text(prompt_sent, encoding="utf-8")
                print(f"Prompt for pass {current_pass} saved successfully.")
            except Exception as e:
                print(f"!! Could not save prompt for pass {current_pass}: {e}")
    
    all_scores = []
    print("\n--- Convening the Jury ---")

    # Initialize the OpenAI client for OpenRouter
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
    )

    for juror_model in JUROR_MODELS:
        print(f"Juror [{juror_model}] is now evaluating...")
        evaluation_string = "" # Initialize to ensure it's available in the except block
        
        try:
            # Construct the prompt manually
            prompt = judge_prompt_template.format(
                original_text=original_text,
                summary_to_evaluate=summary_to_evaluate
            )

            # Call the API directly using the openai library
            completion = client.chat.completions.create(
                model=juror_model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                max_tokens=15000, # Increased token limit for jurors
                temperature=0.2, # Lower temperature for more deterministic evaluation
            )

            evaluation_string = completion.choices[0].message.content

            # --- DEBUGGING: Print the raw response from the juror ---
            print("\n" + "="*20 + f" RAW RESPONSE FROM {juror_model} " + "="*20)
            print(evaluation_string)
            print("="* (42 + len(juror_model)) + "\n")
            # --- END DEBUGGING ---

            # Add a check for an empty response before trying to parse JSON.
            if not evaluation_string or not evaluation_string.strip():
                print(f"!! Juror {juror_model} returned an empty response. Skipping.")
                continue

            # The output from the jury LLM might have markdown or other tags, so we clean it.
            cleaned_str = evaluation_string.replace("```json", "").replace("```", "")
            cleaned_str = cleaned_str.replace("<answer>", "").replace("</answer>", "").strip()
            
            eval_data = json.loads(cleaned_str)

            # --- Save the jury's JSON response to a file ---
            if run_dir:
                sanitized_juror_name = juror_model.replace("/", "_").replace(":", "_")
                jury_output_path = run_dir / f"jury_pass_{current_pass}_juror_{sanitized_juror_name}.json"
                try:
                    with jury_output_path.open("w", encoding="utf-8") as f:
                        json.dump(eval_data, f, indent=2, ensure_ascii=False)
                    print(f"Jury response saved to {jury_output_path}")
                except Exception as e:
                    print(f"!! Could not save jury response to file: {e}")


            # Calculate the total score for this juror
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
            # Print the raw output in case of an error for debugging
            print(f"!! Raw output was: {evaluation_string}")

    if not all_scores:
        print("!!! All jurors failed to provide a score. Returning 0.")
        return 0

    average_score = np.mean(all_scores)
    print(f"--- Jury Deliberation Complete ---")
    print(f"Final Average Score: {average_score:.2f}/20\n")
    
    # Return the score normalized to a 0-1 scale for the optimizer
    return average_score / 20.0

# --- DSPy Program ---
# This is the module we want to optimize. It just contains the summarizer.
class SummarizationModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.summarizer = dspy.Predict(Summarize)

    def forward(self, context):
        return self.summarizer(context=context)


# --- Optimization ---
# We use BootstrapFewShot to generate and refine prompts.
def main():
    # --- Setup Run Directory ---
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"./run_simple_{run_timestamp}")
    run_dir.mkdir(exist_ok=True)
    print(f"--- All outputs will be saved in: {run_dir} ---")

    # Dictionary to hold all scores for this run
    run_scores = {}

    # --- Data Loading ---
    article_filepath = Path("./LLM evaluation fundatmentals.md")
    print(f"--- Loading Article from: {article_filepath} ---")
    try:
        article_to_summarize = article_filepath.read_text(encoding="utf-8")
        print("Article loaded successfully.")
    except FileNotFoundError:
        print(f"FATAL ERROR: The file '{article_filepath}' was not found.")
        print("Please make sure the markdown file is in the same directory as the script.")
        return # Exit the program if the file is not found
    except Exception as e:
        print(f"An unexpected error occurred while reading the file: {e}")
        return

    # --- Create a Hand-Crafted "Gold Standard" Example Trainset ---
    train_data = [
        dspy.Example(
            context="""Article 1: The Mars Curiosity Rover has found new evidence of ancient 'megafloods' that washed across Gale Crater around 4 billion years ago. Scientists believe the floods were likely triggered by the heat of a meteor impact, which melted ice deposits on the Martian surface.
---
Article 2: Data from NASA's Curiosity Rover indicates that Mars' Gale Crater once hosted long-lasting lakes and river systems. The geological layers suggest that water was present for millions of years, pointing to a more habitable past for the Red Planet.
---
Article 3: Researchers analyzing sediment layers in Gale Crater on Mars have confirmed that the area was subjected to intense flooding. The findings, based on Curiosity Rover's observations, add to the growing body of evidence that early Mars had a thicker atmosphere and liquid water.""",
            summary="NASA's Curiosity Rover has found evidence in Gale Crater of ancient megafloods and long-lasting lakes, suggesting Mars had a more habitable, water-rich past approximately 4 billion years ago. These floods were likely caused by a meteor impact melting surface ice."
        ).with_inputs("context"),
        
        dspy.Example(
            context="""A new study from the University of Helsinki shows that regular sauna use can reduce the risk of cardiovascular disease by up to 50%. The study followed 2,000 men over a 20-year period.
---
Finnish researchers have published findings indicating that the heat stress from saunas improves blood vessel function and lowers blood pressure. The benefits increase with frequency, with 4-7 sessions per week showing the most significant effect.
---
The heat of a sauna can also trigger the release of endorphins, leading to feelings of relaxation and well-being, which contributes to overall heart health, according to a recent Finnish study.""",
            summary="A long-term Finnish study of 2,000 men found that regular sauna use (4-7 times per week) can reduce cardiovascular disease risk by up to 50% by improving blood vessel function, lowering blood pressure, and promoting relaxation."
        ).with_inputs("context"),
        dspy.Example(
            context="""Article 1: A recent study in 'Nature Communications' found that plastic nanoparticles can cross the blood-brain barrier in mice, raising concerns about potential neurological effects in humans.
---
Article 2: Researchers at the University of Vienna discovered that tiny plastic particles, consumed through food and water, can accumulate in the brain within just two hours of ingestion.
---
Article 3: The potential health risks of microplastic and nanoplastic exposure are still being investigated, but initial findings suggest a link to inflammation and neurodegenerative diseases.""",
            summary="Recent studies on mice show that plastic nanoparticles can cross the blood-brain barrier and accumulate in the brain shortly after ingestion, raising concerns about potential links to inflammation and neurodegenerative diseases in humans."
        ).with_inputs("context"),
    ]

    trainset = train_data


    print("\n--- Starting DSPy Prompt Optimization with Provided Examples ---")

    # 1. Set up the optimizer
    metric_with_context = lambda gold, pred, trace: summary_quality_metric(gold, pred, trace, run_dir=run_dir)
    
    optimizer = dspy.BootstrapFewShot(
        metric=metric_with_context, 
        max_bootstrapped_demos=5, 
        max_labeled_demos=3, 
        max_rounds=2
    )

    # 2. Compile the program
    optimized_summarizer_module = optimizer.compile(SummarizationModule(), trainset=trainset)

    print("\n--- Optimization Complete ---")
    print("\n--- Optimized Prompt (Instructions Only) ---")
    optimized_prompt_instructions = optimized_summarizer_module.summarizer.signature.instructions
    print(optimized_prompt_instructions)

    # --- Save the optimized prompt instructions to a file ---
    prompt_output_path = run_dir / "optimized_prompt_instructions.txt"
    print(f"\n--- Saving optimized prompt instructions to {prompt_output_path} ---")
    try:
        prompt_output_path.write_text(optimized_prompt_instructions, encoding="utf-8")
        print("Prompt instructions saved successfully.")
    except Exception as e:
        print(f"!! Could not save prompt instructions to file: {e}")
    
    # --- Save the final optimized prompt instructions to the root directory ---
    final_prompt_path = Path("./final_optimized_prompt.txt")
    print(f"\n--- Saving final optimized prompt instructions to {final_prompt_path} ---")
    try:
        final_prompt_path.write_text(optimized_prompt_instructions, encoding="utf-8")
        print("Final prompt instructions saved successfully.")
    except Exception as e:
        print(f"!! Could not save final prompt instructions to file: {e}")


    print("\n--- Testing the Optimized Summarizer on Your Article ---")
    
    try:
        # Manually reconstruct the full prompt to ensure we can capture and save it
        demos = optimized_summarizer_module.summarizer.demos
        
        full_prompt_sent = ""
        full_prompt_sent += optimized_prompt_instructions + "\n\n---\n\n"
        for demo in demos:
            full_prompt_sent += f"Context: {demo.context}\n"
            full_prompt_sent += f"Summary: {demo.summary}\n\n---\n\n"
        full_prompt_sent += f"Context: {article_to_summarize}\n"
        full_prompt_sent += "Summary:"

        # --- Save the full prompt before sending it to the LLM ---
        full_prompt_path = run_dir / "full_final_prompt.txt"
        print(f"\n--- Saving full prompt sent to LLM to {full_prompt_path} ---")
        try:
            full_prompt_path.write_text(full_prompt_sent, encoding="utf-8")
            print("Full prompt saved successfully.")
        except Exception as e:
            print(f"!! Could not save full prompt to file: {e}")

        # Call the API directly
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_api_key)
        completion = client.chat.completions.create(
            model=summarizer_model_id.replace("openrouter/", ""),
            messages=[{"role": "user", "content": full_prompt_sent}],
            max_tokens=16096,
            temperature=0.7,
        )
        summary_text = completion.choices[0].message.content
        final_summary = SimpleNamespace(summary=summary_text)


        print(f"\nYour Article (first 500 chars):\n{article_to_summarize[:500]}...")
        print(f"\nGenerated Summary:\n{final_summary.summary}")

        # --- Save the final generated summary to a file ---
        summary_output_path = Path("./final_optimized_summary.txt")
        print(f"\n--- Saving final summary to {summary_output_path} ---")
        try:
            summary_output_path.write_text(final_summary.summary, encoding="utf-8")
            print("Summary saved successfully.")
        except Exception as e:
            print(f"!! Could not save summary to file: {e}")


        print("\n--- Final Evaluation of the Optimized Summary ---")
        final_score_normalized = summary_quality_metric(dspy.Example(context=article_to_summarize), final_summary, run_dir=run_dir, pass_num="final")
        final_score_20 = final_score_normalized * 20.0
        print(f"\nFinal Average Score for Optimized Summary: {final_score_20:.2f}/20")
        run_scores["final_optimized_run"] = final_score_20

        # --- Log the Final Score ---
        score_log_path = Path("./run_simple_scores.log")
        print(f"\n--- Logging final score to {score_log_path} ---")
        try:
            with score_log_path.open("a", encoding="utf-8") as f:
                log_entry = f"{run_timestamp}: Final Average Score = {final_score_20:.2f}/20\n"
                f.write(log_entry)
            print("Score logged successfully.")
        except Exception as e:
            print(f"!! Could not log score to file: {e}")

        # --- Save all run scores to a JSON file ---
        run_scores_path = run_dir / "run_summary_scores.json"
        print(f"\n--- Saving all run scores to {run_scores_path} ---")
        try:
            with run_scores_path.open("w", encoding="utf-8") as f:
                json.dump(run_scores, f, indent=2)
            print("Run scores saved successfully.")
        except Exception as e:
            print(f"!! Could not save run scores to file: {e}")


    except Exception as e:
        print(f"\n!! An error occurred during the final summarization call: {e}")


if __name__ == "__main__":
    main()
