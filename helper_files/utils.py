import re
import string
from typing import List, Dict
from rouge_score import rouge_scorer
import numpy as np
import csv
import os
import json
from collections import Counter
from json_repair import repair_json

def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact(a_gold: str, a_pred: str) -> float:
    """Compute exact match score."""
    norm_gold = normalize_answer(a_gold)
    norm_pred = normalize_answer(a_pred)
    return float(norm_gold == norm_pred)


def compute_f1(a_gold: str, a_pred: str) -> float:
    """Compute token-level F1 score."""
    gold_toks = normalize_answer(a_gold).split()
    pred_toks = normalize_answer(a_pred).split()

    # If both are empty, perfect match
    if not gold_toks and not pred_toks:
        return 1.0
    # If either is empty, no match
    if not gold_toks or not pred_toks:
        return 0.0

    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


# Create a ROUGE scorer that computes the ROUGE-L F1 score.
rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def compute_rouge(final_answer: str, target_answer: str) -> float:
    """
    Computes the ROUGE-L F1 score between a generated answer and a target answer.
    """
    scores = rouge.score(target_answer, final_answer)
    # Use the F1 measure of ROUGE-L.
    return scores['rougeL'].fmeasure

def compute_metrics(predictions, references):
    """Compute average Exact Match and F1 scores."""
    total_em = 0
    total_f1 = 0
    if not isinstance(predictions, list):
        predictions = [predictions]
    if not isinstance(references, list):
        references = [references]
    for pred, ref_list in zip(predictions, references):
        em = 0
        f1 = 0
        if not isinstance(ref_list, list):
            ref_list = [ref_list]
        for ref in ref_list:
            em = max(em, compute_exact(pred, ref))
            f1 = max(f1, compute_f1(pred, ref))

        total_em += em
        total_f1 += f1
    avg_em = total_em / len(predictions) if predictions else 0
    avg_f1 = total_f1 / len(predictions) if predictions else 0
    return avg_em, avg_f1

def normalize_rewards(rewards, example, verbose=True):
    reward_mean = np.mean(rewards)
    if verbose:
        print(reward_mean)
    if verbose and (reward_mean == 0):
        print(example)
    reward_std = np.std(rewards) + 1e-6
    rewards = [(r - reward_mean) / reward_std for r in rewards]

    return rewards


def write_results_to_csv(name, avg_acc, avg_f1, avg_rouge, avg_llm, filename='final_results.csv'):
    """
    Appends the evaluation metrics to a CSV file.
    The first column will be the given 'name' value.
    """
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as csv_file:
        fieldnames = ['Name', 'Accuracy', 'F1', 'ROUGE', "LLM-as-a-judge"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'Name': name,
            'Accuracy': f"{avg_acc:.4f}",
            'F1': f"{avg_f1:.4f}",
            'ROUGE': f"{avg_rouge:.4f}",
            'LLM-as-a-judge': f"{avg_llm:.4f}",

        })
        
def write_samples_to_csv(original, rewrite, final, acc, llm, filename='samples.csv'):
    """
    Appends the evaluation metrics to a CSV file.
    The first column will be the given 'name' value.
    """
    parent = os.path.dirname(filename)
    if parent:
        os.makedirs(parent, exist_ok=True)
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as csv_file:
        fieldnames = ['Original','Rewrite', 'Final Answer','Accuracy','LLM-as-a-judge reward']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'Original': original,
            'Rewrite': rewrite,
            'Final Answer': final,
            'Accuracy': f"{acc:.2f}",
            'LLM-as-a-judge reward': f"{llm:.2f}",

        })



def try_parse_json(s: str) -> dict:
    """
    Try to extract a valid JSON object from `s`. We:
      1. Strip markdown fences,
      2. Attempt a simple json.loads,
      3. Fall back to taking everything up to the first closing brace,
      4. Finally hand off to json_repair.repair() if needed.
    """
    # 1) remove ```json``` fences
    s_clean = re.sub(r'```(?:json)?', '', s).strip()

    # 2) try a direct parse
    try:
        return json.loads(s_clean)
    except json.JSONDecodeError:
        pass

    # 3) naive cut‐at‐first closing brace
    end = s_clean.find('}')
    if end != -1:
        snippet = s_clean[: end + 1]
        if snippet.startswith('{'):
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                pass
        else:
            try:
                return json.loads('{' + snippet)
            except json.JSONDecodeError:
                pass

    # 4) last resort: auto‐repair
    repaired = repair_json(s_clean)
    return json.loads(repaired)

def extract_final_answer(response: str) -> str:
    if "####" in response:
        parts = response.split("####")
        final_part = parts[-1].strip()
    else:
        final_part = response.strip()
    # Find all numeric matches in the final part
    matches = re.findall(r"(-?\d*\.\d+|-?\d+)", final_part)
    if matches:
        # Return the last numeric match found
        return matches[-1]
    return ""

def evaluate_gsm8k_response(response: str, gold_answer: str, tolerance: float = 1e-3) -> float:
    """
    Evaluate a GSM8K response by:
      1. Extracting the final answer using extract_final_answer.
      2. Normalizing both the extracted answer and the gold answer.
      3. Converting them to floats and comparing within a given tolerance.
    Returns 1.0 if the answers match within tolerance, otherwise 0.0.
    """
    extracted = extract_final_answer(response)
    norm_extracted = normalize_answer(extracted)
    norm_gold = normalize_answer(gold_answer)
    try:
        extracted_num = float(norm_extracted)
        gold_num = float(norm_gold)
    except ValueError:
        return 0.0
    if abs(extracted_num - gold_num) <= tolerance:
        return 1.0
    else:
        return 0.0

def apply_chat_template(prompt: str, tokenizer) -> str:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)