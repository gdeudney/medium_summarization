import os
import openai
import json
from dotenv import load_dotenv
import time
import numpy as np
from pathlib import Path
from datetime import datetime

# --- Configuration ---
# A main directory to store all validation runs
RUNS_DIR = Path("./summary_evaluation_runs")
# The number of times the full jury will evaluate the single winning summary
NUM_JURY_RUNS = 5

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

# Define the models for the Jury
JUROR_MODELS = [
    "qwen/qwen3-30b-a3b-instruct-2507",
    "deepseek/deepseek-r1-0528-qwen3-8b",
    "mistralai/magistral-small-2506"
]

# --- 2. Define the Core Components & Prompts ---

# The article that was originally summarized.
article_filepath = Path("./LLM evaluation fundatmentals.md")
print(f"--- Loading Article from: {article_filepath} ---")
try:
    SOURCE_ARTICLE = article_filepath.read_text(encoding="utf-8")
    print("Article loaded successfully.")
except FileNotFoundError:
    print(f"FATAL ERROR: The file '{article_filepath}' was not found.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred while reading the file: {e}")
    exit()

# The specific, static "Winning Summary" text that we are evaluating.
WINNING_SUMMARY_TEXT = """
The article argues that rigorous, ongoing, multi-method, and multi-faceted evaluation of LLMs is non-negotiable for safety, compliance, product quality, and development efficiency. Below is a structured, tone-preserving summary of its core content:

---

### **Why Bother With All This Testing? (More Than Job Security)**
Evaluation isn’t optional—it’s foundational. Key motivations:
- **Safety First**: Avoid “Digital Darwin Awards” by testing LLMs on risky edge cases (e.g., *“Don’t recommend swimming the Atlantic”*). Safety isn’t just ethical but increasingly legal.
- **Compliance Maze**: Regulations demand proof of safety and bias mitigation. Evaluation is your “decoder ring” for AI compliance.
- **Quality Control**: Prevent digital gremlins from causing PR nightmares (e.g., *“Recommending minefield sightseeing”*). A [2023 ArXiv paper](https://realharm.giskard.ai/) on *Real-World Language Model Failures* highlights reputational and legal risks.
- **Efficiency & Coffee Breaks**: Evaluation speeds development by enabling faster iteration, safer updates, and easier model comparisons (e.g., *“Avoid blaming cosmic rays for bugs”*).

Even “proof of concept” projects need evaluation to prove viability.

---

### **Key Evaluation Metrics and Methodologies**

#### **Automated Metrics: The Number Crunchers**
- **Accuracy**: Simple but risky for skewed datasets (*“Grading a student on one topic they studied”*).
- **F1-Score**: Balances precision and recall (*“Finding Waldo without false alarms”*).
- **BLEU/ROUGE**: Compare outputs to human references. BLEU = translation; ROUGE = summarization. Caveat: They miss meaning (*“Nonsensical but word-matching sentences”*).
- **METEOR**: Better for fluency and synonyms (*“Appeasing the humans”*).
- **Perplexity**: Measures prediction fluency but is dataset-dependent (*“Cat vs. dog surprises”*).
- **BERTScore**: Uses contextual embeddings to catch paraphrases (*“The cat is out of the bag ≠ literal feline escape”*).

#### **LLM-as-a-Judge**: AI Grades AI
- **Binary/Multi-Choice**: Test factual accuracy.
- **Pairwise/Ranking**: Compare outputs (*“Bake-off blue ribbons”*).
- **Direct Scoring/Critique Generation**: Evaluate traits like politeness or detect hallucinations (*“Full report card with comments”*).
- **Tool, Not Metric**: LLM judges require tailored prompts and are app-specific.

#### **Nuanced Capabilities**
- **Personalization/Sentiment**: Test if the LLM adapts to user profiles or detects sarcasm (*“Cockroach hotel reviews”*).
- **Planning/Sequencing**: Use **Recursive Thought Expansion (RTE)** and **Hierarchical Thought Decomposition (HTD)** to assess complex planning (*“Biking to the moon”*).
- **Refinement on Feedback**: Check if the model adjusts to user corrections (*“Budget travel ≠ private jet”*).

#### **Explainability**
- **Chain-of-Thought (CoT)** prompting reveals reasoning steps.
- **Input Attribution**: Identify if the model fixates on irrelevant keywords (*“Luxury yacht vs. budget travel”*).

#### **Human Evaluation**
- Captures creativity, coherence, and relevance. Essential for subtleties automated metrics miss (*“Focus groups vs. market data”*).

---

### **Intrinsic vs. Extrinsic Evaluation**
- **Intrinsic**: Judge output quality alone (e.g., *“Spaghetti texture”*).
- **Extrinsic**: Judge real-world impact (e.g., *“Does it nourish the eater?”*). Both are needed.

---

### **Beyond Core Metrics: Crucial Aspects**
- **Bias & Fairness**: Mitigate dataset-driven biases (*“Avoid Eurocentric travel plans”*).
- **Reliability**: Ensure consistency (*“No teenage mood swings”*).
- **Non-Determinism**: LLMs are unpredictable due to settings like **temperature** (*“Same question, wildly different answers”*). Testing must be repeated.
- **Efficiency/Cost**: Balance performance with latency, throughput, and financial/environmental costs.
- **Observability**: Monitor live behavior (*“Tiny cameras on digital antics”*).
- **User Satisfaction**: Surveys and feedback loops reveal if outputs are *actually* useful (*“Did it write a haiku or a travel plan?”*).

---

### **Conclusion**
No single metric suffices. Combine **automated**, **LLM-as-a-judge**, and **human evaluation** for a “clearer picture.” Build repeatable pipelines to avoid “tweaking knobs blindly.” Evaluation is the only way to ensure LLMs are powerful, safe, and maybe one day, will solve the *squirrel court* mystery.
"""

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

# --- 3. Define the LLM Interaction Function ---

def run_jury(article, summary, run_dir, jury_run_num):
    """Evaluates the summary with a panel of jurors and returns the average score."""
    print(f">> Convening the Jury for evaluation run {jury_run_num}...")
    all_scores = []
    all_verdicts = {}
    criteria = ["faithfulness", "coherence", "conciseness", "coverage"]
    for juror_model in JUROR_MODELS:
        print(f"   - Juror [{juror_model}] is deliberating...")
        raw_response_content = ""
        try:
            # Added a small random element to the system prompt to bypass potential caching
            system_prompt_variant = f"Request ID: {np.random.randint(1000, 9999)}"
            
            response = client.chat.completions.create(
                model=juror_model,
                messages=[
                    {"role": "system", "content": system_prompt_variant},
                    {"role": "user", "content": JURY_PROMPT.format(original_text=article, summary_to_evaluate=summary)}
                ],
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
            jury_output_path = run_dir / f"jury_run_{jury_run_num}_juror_{sanitized_juror_name}.json"
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

def log_final_analysis(log_file, all_scores):
    """Appends the final statistical analysis to the log file."""
    mean_score = np.mean(all_scores)
    std_dev = np.std(all_scores)
    min_score = np.min(all_scores)
    max_score = np.max(all_scores)
    
    with open(log_file, 'a', encoding="utf-8") as f:
        f.write("\n" + "="*50 + "\n")
        f.write("--- FINAL STATISTICAL ANALYSIS ---\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Total Jury Evaluation Runs: {len(all_scores)}\n\n")
        f.write(f"**Overall Mean Score: {mean_score:.2f} / 20**\n")
        f.write(f"**Standard Deviation: {std_dev:.2f}**\n")
        f.write(f"**Minimum Score Observed: {min_score:.2f}**\n")
        f.write(f"**Maximum Score Observed: {max_score:.2f}**\n\n")
        f.write("--- All Individual Scores ---\n")
        f.write(str(all_scores) + "\n")
        f.write("="*50 + "\n")

# --- 5. The Main Evaluation Loop ---

def main():
    """Main function to run the evaluation process."""
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / f"evaluation_{run_timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"--- All outputs will be saved in: {run_dir} ---")
    
    log_file = run_dir / "summary_evaluation_log.txt"

    with open(log_file, 'w', encoding="utf-8") as f:
        f.write(f"Summary Evaluation Log - Started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Evaluating the following summary:\n\n")
        f.write(WINNING_SUMMARY_TEXT + "\n\n")

    all_run_scores = []

    for i in range(1, NUM_JURY_RUNS + 1):
        score, _ = run_jury(SOURCE_ARTICLE, WINNING_SUMMARY_TEXT, run_dir, jury_run_num=i)
        
        if score > 0:
            all_run_scores.append(score)
        
        print(f"--> Average Score for Jury Run {i}: {score:.2f} / 20\n")
        time.sleep(5)

    if not all_run_scores:
        print("\n--- No scores were recorded. Cannot perform final analysis. ---")
        return

    mean_score = np.mean(all_run_scores)
    std_dev = np.std(all_run_scores)
    
    print("\n" + "="*25)
    print("--- EVALUATION COMPLETE ---")
    print("="*25)
    print(f"Total Jury Runs: {len(all_run_scores)}")
    print(f"\nOverall Mean Score: {mean_score:.2f} / 20")
    print(f"Standard Deviation: {std_dev:.2f}")
    print(f"\nDetailed logs and verdicts saved in: {run_dir}")

    log_final_analysis(log_file, all_run_scores)

if __name__ == "__main__":
    main()
