# optimize_summary_prompt.py

import json
import time
import os
from openai import OpenAI
from datetime import datetime

# ------------------------
# Configuration
# ------------------------
ARTICLE_PATH = "LLM evaluation fundamentals.md"
OUTPUT_HISTORY = "prompt_optimization_history.json"

# Set judges and generator models
JUDGES = ["mistral-tiny", "gemma-2-9b", "llama-3-8b", "phi-2"]
GENERATOR_MODEL = "llama-3-70b"  # You can use gpt-3.5 if preferred
IMPROVER_MODEL = "deepseek/deepseek-r1:free"

INITIAL_PROMPT = "Summarize this article clearly and concisely."
TARGET_SCORE = 18.5
MAX_ITERATIONS = 10

# ------------------------
# OpenRouter Client Setup
# ------------------------
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

# ------------------------
# Query utility
# ------------------------
def query_model(model, prompt):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"    ❌ Error querying {model}: {str(e)}")
        return "Error."

# ------------------------
# Load article
# ------------------------
def load_article():
    with open(ARTICLE_PATH, "r", encoding="utf-8") as f:
        return f.read()


# ------------------------
# Evaluate summary using judges
# ------------------------
def evaluate_summary(summary, judges):
    scores = []
    feedbacks = []

    print("🧠 Evaluating Summary with Judges...")
    for judge in judges:
        print(f"  ⚖️ Judge: {judge}")
        raw = query_model(judge, f"""
The following is a summary of an article on LLM evaluation.
Score it out of 20 on the following criteria:
- Accuracy: 5 points
- Conciseness: 5 points
- Completeness: 5 points
- Clarity: 5 points

Provide numeric score and qualitative feedback in JSON format:
{{
  "score": [numeric],
  "feedback": "[text]"
}}

Summary:
{summary}
""")
        try:
            data = json.loads(raw)
            scores.append(data["score"])
            feedbacks.append(data["feedback"])
        except:
            scores.append(0)
            feedbacks.append("Error parsing feedback.")

    avg = sum(scores) / len(scores) if scores else 0
    print(f"  📊 Average Score: {avg:.2f}\n")
    return avg, feedbacks


# ------------------------
# Improve prompt based on feedback
# ------------------------
def improve_prompt(current_prompt, feedbacks):
    print("🛠 Improving Prompt...")
    feedback_str = " | ".join(feedbacks)
    improved = query_model(IMPROVER_MODEL, f"""
You are improving a summarization prompt.

Original prompt: 
{current_prompt}

Feedback from judges: 
{feedback_str}

Suggest a revised prompt that is clearer and more effective for summarizing complex content accurately.
Return only the revised prompt.
""")
    return improved


# ------------------------
# Main Loop
# ------------------------
def main():
    article_text = load_article()
    current_prompt = INITIAL_PROMPT
    history = []

    print("🚀 Starting Prompt Optimization Loop")
    print("📝 Article Loaded:", ARTICLE_PATH)

    for i in range(MAX_ITERATIONS):
        print(f"\n🔁 Iteration {i + 1} | Prompt: {current_prompt}")

        # Generate summary
        summary_prompt = f"{current_prompt}\n\n{article_text}"
        summary = query_model(GENERATOR_MODEL, summary_prompt)
        print("📜 Generated Summary:", summary[:100] + "..." if len(summary) > 100 else summary)

        # Evaluate
        avg_score, feedbacks = evaluate_summary(summary, JUDGES)

        # Record history
        history.append({
            "iteration": i + 1,
            "prompt": current_prompt,
            "summary": summary,
            "avg_score": avg_score,
            "feedbacks": feedbacks,
            "timestamp": datetime.now().isoformat()
        })

        # Save to file every iteration
        with open(OUTPUT_HISTORY, "w") as f:
            json.dump(history, f, indent=2)
        print(f"💾 Backup Saved: {OUTPUT_HISTORY}")

        if avg_score >= TARGET_SCORE:
            print("🎯 Target Score Achieved! Stopping Early.")
            break

        # Improve prompt
        current_prompt = improve_prompt(current_prompt, feedbacks)
        time.sleep(2)

    print("\n🏁 Optimization Completed.")
    print("📜 Final Prompt: ", current_prompt)
    print("📊 Final History: ", OUTPUT_HISTORY)


# ------------------------
# Run Script
# ------------------------
if __name__ == "__main__":
    main()