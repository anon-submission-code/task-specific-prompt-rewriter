# TAPR: Enhancing LLM Performance with a Task‑Aware Prompt Rewriter

Large Language Models (LLMs) can achieve much stronger results when they receive well‑formed prompts, yet writing those prompts is a barrier for many users. This repository implements a **Task-Aware Prompt Rewriter** **(TAPR)** that automatically reformulates user queries and feeds the rewrites to a task LLM. The rewriter is trained with **Group Relative Policy Optimization (GRPO)** and rewarded with both classic text‑overlap metrics and **LLM‑as‑a‑Judge** scores. The work includes benchmarking on question answering, summarization, and arithmetic reasoning tasks.

---

## Setup

1. **Clone** the repository and create a virtual environment.
2. **Install** Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. **API keys (optional):**  
   If you evaluate with GPT or Llama Bedrock API models, add credentials to a `.env` file (see `helper_files/helpers.py` for variable names).

---

## Project Structure

```text
.
├── grpo_training.py             # Train the prompt rewriter with GRPO
├── ppo_training.py             # Train the prompt rewriter with PPO, this has not been succesful and requires using trl version 0.11
├── helper_files/
│   ├── data_helpers.py          # Dataset preprocessing utilities
│   ├── helpers.py               # Helper functions for both training and evaluation
│   └── utils.py                 # Metrics, CSV helpers
├── evaluation/
│   └── evaluation.py            # Core evaluation routines
│   ├── evaluation_with_sampling.py  # Test‑time sampling & selection of rewrites
│   └── evaluate_models.py         # Evaluate a trained rewriter
│   └── baseline_eval.py         # Evaluate baseline prompts
└── example.ipynb            # Example notebook for training and evaluation
```

---

## Main Scripts

| Script | Purpose |
|--------|---------|
| `grpo_training.py` | Fine‑tunes a base model (e.g. Llama‑3.2-3B-Instruct or Phi‑4-mini-instruct) as a Prompt Rewriter via GRPO. Key flags: `--rewriting_model_name`, `--response_model_name`, `--dataset`, `--batch_size`, `--lr`, `--epochs`, `--output_dir`. |
| `evaluate_models.py` | Rewrites each validation sample and measures downstream performance of the response model. Writes `final_results.csv`. All evaluations are recommended to be done with the vLLM library.|
| `evaluation_with_sampling.py` | Generates *n* candidate rewrites per sample, optionally asks an LLM to pick the best, and logs each candidate in `sampling.csv`. |
| `baseline_eval.py` | Runs a response model directly on the dataset with the initial prompt for baseline comparisons. |

---

## Quick Start

### 1. Train a Prompt Rewriter

```bash
python grpo_training.py \
  --rewriting_model_name unsloth/Phi-4-mini-instruct \
  --response_model_name BEDROCK \
  --dataset cnndm \
  --batch_size 4 \
  --lr 1e-5 \
  --epochs 2 \
  --max_new_tokens 256 \
  --output_dir ./grpo_phi_cnndm
```

### 2. Evaluate the Trained Model

```bash
python evaluate_models.py \
  --model_path ./grpo_phi_cnndm \
  --response_model_name gpt-4o-mini \
  --dataset_name cnndm \
  --batch_size 64 \
  --name "GRPO_Phi4_CNNDM"
```

### 3. Evaluate with Rewrite Sampling

```bash
python evaluation_with_sampling.py \
  --model_path ./grpo_phi_cnndm \
  --response_model_name gpt-4o-mini \
  --dataset_name natural_questions \
  --n 5 \
  --name "Phi4_NQ_sampling"
```

### 4. Baseline Prompt Evaluation

```bash
python baseline_eval.py \
  --response_model_name meta-llama/Llama-3.2-3B-Instruct \
  --dataset_name cnndm \
  --batch_size 128
```

---

## Outputs

All evaluation scripts append a line to a CSV file:

```csv
Name,Accuracy,F1,ROUGE,LLM-as-a-judge
GRPO_Phi4_CNNDM,0.46,0.55,0.44,0.60
```

* **Accuracy / F1** – token‑level comparison with target answers.
* **ROUGE‑L F1** – overlap with reference summaries (when applicable).
* **LLM‑as‑a‑Judge** – score returned by an evaluation prompt given to GPT‑4o-mini or another LLM.
