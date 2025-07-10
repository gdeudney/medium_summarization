# Make sure to run the following command first
# pip install -r requirements.txt

import ollama
import openai
import os

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


TEMPERATURE = 0.5 # Keep it low for deterministic, repeatable outputs
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



# --- Helper Function ---
# Function to read the files of the summaries
def read_text_file(filepath):
    """Reads the content of a text file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"An error occurred while reading {filepath}: {e}")
        return None



# --- Main Execution ---
if __name__ == "__main__":
   
    # Define the paths to your files.
    # Everything is in this directory
    original_article_path = ".\\LLM evaluation fundatmentals.md"
    local_summary_path = ".\\magistralSummary.md"

    print(f"--- Loading Evidence ---")
    print(f"Attempting to load original article from: {original_article_path}")
    print(f"Attempting to load local summary from: {local_summary_path}")
    
    original_text = read_text_file(original_article_path)
    local_model_summary = read_text_file(local_summary_path)

    # --- Graceful Exit if files are not found ---
    if not original_text or not local_model_summary:
        print("\nCould not load one or both files. Please check the file paths and try again.")
        exit()

    print("\n--- Files Loaded Successfully ---")
    print("Original Text Snippet:", original_text[:150] + "...")
    print("Local Model Summary Snippet:", local_model_summary[:150] + "...")
    print("-" * 20)


    # --- Create the Judge's Instructions (The Prompt) ---
    judge_prompt = f"""
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
    {local_model_summary}
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

    # --- Let the Judging Commence! --- 

    print("\n--- Sending the case to local model for judgment... ---")

    for _ in range(NUMBER_OF_RUNS):
        try:
            if LOCAL_SERVER_TYPE == "LMSTUDIO":
                # Use the OpenAI library's chat completions method
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{'role': 'user', 'content': judge_prompt}],
                    temperature=TEMPERATURE
                )
                print(response.choices[0].message.content)
            else: # OLLAMA
                response = client.chat(
                    model=MODEL_NAME,
                    messages=[{'role': 'user', 'content': judge_prompt}],
                    options={'temperature': TEMPERATURE}
                )
                print(response['message']['content'])
                
        except Exception as e:
            print(f"\n--- CLIENT CONNECTION ERROR ---")
            print(f"Error communicating with {LOCAL_SERVER_TYPE}: {e}")
            print(f"Please make sure {LOCAL_SERVER_TYPE} is running and the model '{MODEL_NAME}' is loaded and correctly identified.")
            print("-------------------------------\n")
            
     
       


