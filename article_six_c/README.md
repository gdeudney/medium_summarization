# Generic Prompt-Optimization Engine (V2)

This project provides a robust, reusable engine for automatically optimizing LLM prompts. This version incorporates advanced features for validation, reliability, and cost control.

It operates in two phases:
1.  **Calibration:** Tests "juror" LLMs against a golden set to measure and correct for scoring biases.
2.  **Optimization:** Iteratively refines a prompt by testing it against an entire dataset, using the calibrated jury to score the results.

## Key Features

- **Robust Configuration:** Uses **Pydantic** to validate `config.yaml`, catching errors early.
- **Task-Agnostic:** Define any task (summarization, translation) in your config file. See the `examples/` directory.
- **Weighted Scoring:** Assign weights to evaluation criteria to prioritize what matters most.
- **Objective & Calibrated Jury:** Explicit calibration removes juror bias for more reliable scores.
- **Advanced Controls:**
    - **Convergence Checking:** Automatically stops if optimization stalls.
    - **Cost Management:** Set limits on total tokens and API calls to prevent runaway costs.
    - **Configurable Parameters:** Control temperatures, retries, and rate limits.
- **Extensible:** Clean, separated modules make it easy to add new components.
- **Structured Logging:** Uses Python's `logging` module for clear, leveled logs.

## How to Run

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Create your environment file:**
    Copy your LLM provider API key into a `.env` file.
    ```
    # For OpenRouter, OpenAI, and other compatible providers
    OPENAI_API_KEY="sk-..."
    ```

3.  **Prepare your data:**
    - Place your evaluation documents in a folder (e.g., `datasets/articles/`).
    - Review the example configurations in the `examples/` directory. Copy and modify one for your needs.

4.  **Run the engine:**
    ```bash
    python -m prompt_opt.engine examples/summarization_config.yaml
    ```

prompt-optimization-engine/
│
├── prompt_opt/
│   ├── __init__.py
│   ├── client.py
│   ├── config.py
│   ├── engine.py
│   ├── jury.py
│   ├── logger.py
│   ├── operator.py
│   └── optimizer.py
│
├── datasets/
│   └── articles/
│       ├── article_01.txt
│       └── article_02.txt
│
├── examples/
│   ├── summarization_config.yaml
│   └── translation_config.yaml
│
├── tests/
│   ├── __init__.py
│   └── test_engine_utils.py
│
├── .env
├── README.md
└── requirements.txt