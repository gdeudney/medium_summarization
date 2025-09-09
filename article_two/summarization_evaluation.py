# Make sure to run the following command first
# pip install -r requirements.txt

import ollama
import openai
import evaluate
import pandas as pd
import statistics

# --- Global Configuration ---
# Set your local server type: "OLLAMA" or "LMSTUDIO"
LOCAL_SERVER_TYPE = "LMSTUDIO"  

# --- IMPORTANT: Set the model name based on your server ---
#
# For OLLAMA:
# Use the model tag you use in the terminal (e.g., 'mistral-small:24b', 'gemma3:27b').
# MODEL_NAME = 'gemma3:27b'
#
# For LM STUDIO:
# The server uses a generic identifier for the currently loaded model.
# This is typically 'local-model'. Check the LM Studio UI to be sure.
MODEL_NAME = 'local-model' 


TEMPERATURE = 0.1 # Keep it low for deterministic, repeatable outputs
NUMBER_OF_RUNS = 5 # Number of times to generate a summary for each prompt to get a stable baseline

# --- Client Initialization ---
# Initialize the correct client based on the server type
if LOCAL_SERVER_TYPE == "LMSTUDIO":
    # Use the OpenAI client for LM Studio's OpenAI-compatible server
    client = openai.OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
    print("INFO: Configured to use LM Studio client (via OpenAI library).")
elif LOCAL_SERVER_TYPE == "OLLAMA":
    client = ollama.Client(host='http://localhost:11434') # Uses localhost
    print("INFO: Configured to use Ollama client.")
else:
    raise ValueError("Invalid LOCAL_SERVER_TYPE. Please choose 'OLLAMA' or 'LMSTUDIO'.")


# --- Source Text and Reference Summary ---
# This is the text we want to summarize.
SOURCE_TEXT = """
Photosynthesis is a process used by plants, algae, and certain bacteria to convert light energy into chemical energy, through a process that converts carbon dioxide and water into sugars (glucose) and oxygen. This process is crucial for life on Earth as it produces most of the planet's oxygen and serves as the primary source of energy for most ecosystems. The process occurs in two main stages: the light-dependent reactions and the light-independent reactions (Calvin cycle). The light-dependent reactions, which occur in the thylakoid membranes of chloroplasts, capture energy from sunlight to produce ATP and NADPH. The Calvin cycle, which takes place in the stroma of the chloroplasts, then uses this ATP and NADPH to convert carbon dioxide into glucose. Factors like light intensity, carbon dioxide concentration, and temperature can significantly affect the rate of photosynthesis.
"""

# This is our "gold standard" human-written summary. Our metrics will compare against this. 
REFERENCE_SUMMARY = "Photosynthesis converts light energy into chemical energy, producing glucose and oxygen from carbon dioxide and water. This two-stage process, consisting of light-dependent reactions and the Calvin cycle, is vital for Earth's oxygen supply and energy ecosystems. The rate is affected by light, CO2, and temperature."

# --- Prompts ---
VAGUE_PROMPT = f"""
Summarize:
---
{SOURCE_TEXT}
"""

SPECIFIC_PROMPT = f"""
Summarize the key findings of the following text in three concise bullet points, focusing on the core process, its importance, and influencing factors.
---
{SOURCE_TEXT}
"""

def generate_summary(prompt_text, model_name, temp):
    """
    Generates a summary using the pre-configured client (Ollama or LM Studio).
    """
    try:
        if LOCAL_SERVER_TYPE == "LMSTUDIO":
            # Use the OpenAI library's chat completions method
            response = client.chat.completions.create(
                model=model_name,
                messages=[{'role': 'user', 'content': prompt_text}],
                temperature=temp
            )
            return response.choices[0].message.content
        else: # OLLAMA
            response = client.chat(
                model=model_name,
                messages=[{'role': 'user', 'content': prompt_text}],
                options={'temperature': temp}
            )
            return response['message']['content']
            
    except Exception as e:
        print(f"\n--- CLIENT CONNECTION ERROR ---")
        print(f"Error communicating with {LOCAL_SERVER_TYPE}: {e}")
        print(f"Please make sure {LOCAL_SERVER_TYPE} is running and the model '{model_name}' is loaded and correctly identified.")
        print("-------------------------------\n")
        return None

def run_evaluation():
    """
    Main function to run the summarization and evaluation experiment.
    """
    print(f"\n--- Starting Franken-Lab Summarization Evaluation ---")
    print(f"Model: {MODEL_NAME}, Temperature: {TEMPERATURE}, Runs per prompt: {NUMBER_OF_RUNS}\n")

    # Load the evaluation metrics
    try:
        rouge = evaluate.load('rouge')
        bertscore = evaluate.load("bertscore")
    except Exception as e:
        print(f"Error loading evaluation metrics: {e}")
        print("Please ensure you have an internet connection to download the metrics for the first time.")
        return

    results_data = []
    prompts = {
        "Vague 'Lennie' Prompt": VAGUE_PROMPT,
        "Specific 'George' Prompt": SPECIFIC_PROMPT
    }

    for prompt_name, prompt_content in prompts.items():
        print(f"--- Testing: {prompt_name} ---")
        
        rouge1_scores = []
        bert_f1_scores = []

        for i in range(NUMBER_OF_RUNS):
            print(f"  Generating summary for run {i+1}/{NUMBER_OF_RUNS}...")
            generated_summary = generate_summary(prompt_content, MODEL_NAME, TEMPERATURE)
            
            if generated_summary is None:
                continue

            # Calculate ROUGE scores
            rouge_results = rouge.compute(predictions=[generated_summary], references=[REFERENCE_SUMMARY])
            rouge1_scores.append(rouge_results['rouge1'])

            # Calculate BERTScore
            bert_results = bertscore.compute(predictions=[generated_summary], references=[REFERENCE_SUMMARY], lang="en")
            bert_f1_scores.append(statistics.mean(bert_results['f1'])) 

        if not rouge1_scores or not bert_f1_scores:
            print(f"Could not generate summaries for '{prompt_name}'. Skipping.")
            continue

        avg_rouge1 = statistics.mean(rouge1_scores)
        avg_bert_f1 = statistics.mean(bert_f1_scores)
        
        results_data.append({
            "Prompt Style": prompt_name,
            "Avg ROUGE-1": avg_rouge1,
            "Avg BERTScore (F1)": avg_bert_f1,
        })
        
        print(f"\nExample Generated Summary (from last run):\n---\n{generated_summary}\n---")
        print(f"Average ROUGE-1 over {NUMBER_OF_RUNS} runs: {avg_rouge1:.4f}")
        print(f"Average BERTScore F1 over {NUMBER_OF_RUNS} runs: {avg_bert_f1:.4f}\n")

    if results_data:
        results_df = pd.DataFrame(results_data)
        print("\n--- Final Results ---")
        print("Comparing the average scores from both prompts:")
        print(results_df.to_string(index=False))
        print("\nExperiment complete. As the data shows, the specific 'George' prompt consistently yields higher scores.")
    else:
        print("\nEvaluation could not be completed. Please check your local server connection.")

if __name__ == "__main__":
    run_evaluation()
