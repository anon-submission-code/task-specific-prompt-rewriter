from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from utils import extract_final_answer, apply_chat_template

def create_meta_prompt(text: str, instruction: str = "Answer the question") -> str:
    """
    Create a meta prompt given a question.
    """
    if text == "":
        prompt = (
            "Rewrite the following instruction by rephrasing and/or adding specific requirements to ensure the best"
            "possible chance for another LLM to answer the query correctly. "
            "Use illustrative descriptions if needed. Don't add details that change the meaning of the instruction. Output your response as a JSON object with two keys: "
            "\"explanation\" and \"final_rewritten_query\". The \"explanation\" key should contain your reasoning for "
            "the changes you made, and the \"final_rewritten_query\" key should contain only the final rewritten instruction.\n\n"
            "Here's a demonstrative example:\n"
            "1. Original instruction: The old instruction\n"
            "New Instruction: {\"explanation\": \"Explanation of why this kind of rewriting was done\", "
            "\"final_rewritten_query\": \"The rewritten instruction.\"}\n\n"
            "Now write the new instruction. Respond only with valid JSON. Do not write an introduction or summary.\n"
            f"Original instruction: {instruction}\n\n"
            "New Instruction: {"
        )
        return prompt
    prompt = (
            "Rewrite the following instruction by rephrasing and/or adding specific requirements to ensure the best"
            "possible chance for another LLM to answer the query correctly."
            "Use illustrative descriptions if needed. Output your response as a JSON object with two keys: "
            "\"explanation\" and \"final_rewritten_query\". The \"explanation\" key should contain your reasoning for "
            "the changes you made, and the \"final_rewritten_query\" key should contain only the final rewritten instruction.\n\n"
            "Examples:\n"
            "1. Original instruction: The old instruction\n"
            "New Instruction: {\"explanation\": \"Explanation of why this kind of rewriting was done\", "
            "\"final_rewritten_query\": \"The rewritten instruction.\"}\n\n"
            "Now write the new instruction. Respond only with valid JSON. Do not write an introduction or summary.\n"
            f"Original instruction: {instruction}: \"{text}\"\n"
            "New Instruction: {"
    )
    return prompt

def extract_short_answers(annotations) -> list:
    """
    Extract short answers from the annotations dictionary.
    """
    answers = []
    short_answers = annotations.get("short_answers", [])
    if isinstance(short_answers, list):
        for ans in short_answers:
            texts = ans.get("text", [])
            if isinstance(texts, list):
                for text in texts:
                    text = text.strip()
                    if text:
                        answers.append(text)
            elif isinstance(texts, str):
                text = texts.strip()
                if text:
                    answers.append(text)
    return answers

def preprocess_nq(example: dict, full_prompt=False) -> dict:
    """
    Preprocess a Natural Questions example using shared logic.
    """
    question = example.get("question", {}).get("text", "").strip()
    if not question:
        return {"prompt": "", "target": [], "text": ""}
    instruction = "Answer the question"
    text_part = f"\nQuestion: {question}"
    if full_prompt:
        meta_prompt = create_meta_prompt(text_part, instruction)
    else:
        meta_prompt = create_meta_prompt("")
    short_answers = extract_short_answers(example.get("annotations", {}))
    return {"prompt": meta_prompt, "target": short_answers, "text": text_part, "original": instruction}

def preprocess_cnndm(example, full_prompt=False):
    """
    Preprocess a CNN/DM example.
    Extracts the article and highlights, creates a meta prompt for summarization,
    and returns the gold summary as a single-element list.
    """
    article = example.get("article", "").strip()
    highlights = example.get("highlights", "").strip()
    if not article or not highlights or len(article) > 5000:
        return {"prompt": "", "target": [], "text": ""}
    instruction = "Summarize the text"
    text_part = f"\nText: {article}"
    if full_prompt:
        meta_prompt = create_meta_prompt(text_part, instruction)
    else:
        meta_prompt = create_meta_prompt("", instruction)
    
    return {"prompt": meta_prompt, "target": [highlights], "text": text_part, "original": instruction}

def preprocess_gsm8k(example: dict, full_prompt=False) -> dict:
    """
    Preprocess a GSM8K example.
    The GSM8K dataset contains a 'question' and an 'answer' field.
    This function extracts the question and, if available, the final answer.
    A meta prompt is created with an instruction tailored for solving math problems.
    """
    question = example.get("question", "").strip()
    answer = example.get("answer", "").strip()
    if not question or not answer:
        return {"prompt": "", "target": [], "question": ""}
    final_answer = extract_final_answer(answer)
    instruction = "SOLUTION"
    text_part = f"\nQuestion: {question}"
    
    if full_prompt:
        meta_prompt = create_meta_prompt(text_part, instruction)
    else:
        meta_prompt = create_meta_prompt("", instruction)
    #meta_prompt = text_part
    return {"prompt": meta_prompt, "target": [final_answer], "text": text_part, "original": instruction}

def preprocess_hotpotqa(example: dict, full_prompt=False):
    question = example.get("question", "").strip()
    answer = example.get("answer", "").strip()
    all_sents = [sent for para in example['context']['sentences'] for sent in para]
    context = " ".join(all_sents)
    if not question or not answer or not context:
        return {"prompt": "", "target": [], "question": ""}
    
    instruction = "Answer the question based on the context"
    text_part = f"\nContext: {context}\nQuestion: {question}"
    
    if full_prompt:
        meta_prompt = create_meta_prompt(text_part, instruction)
    else:
        meta_prompt = create_meta_prompt("", instruction)
        
    return {"prompt": meta_prompt, "target": [answer], "text": text_part, "original": instruction}

def preprocess_scitldr(example: dict, full_prompt=False):
    source = example.get("source", "")
    if isinstance(source, list):
        # join all parts into one string, then strip whitespace
        article = " ".join(source).strip()
    else:
        article = source.strip()
    highlights = example.get("target", "")
    if isinstance(highlights, list):
        highlights = highlights[0]
    highlights = highlights.strip()
    if not article or not highlights or len(article) > 5000:
        return {"prompt": "", "target": [], "text": ""}
    instruction = "Summarize the text"
    text_part = f"\nText: {article}"
    if full_prompt:
        meta_prompt = create_meta_prompt(text_part, instruction)
    else:
        meta_prompt = create_meta_prompt("", instruction)
    
    return {"prompt": meta_prompt, "target": [highlights], "text": text_part, "original": instruction}

def load_and_preprocess_dataset(dataset_name: str, split: str, samples_amount: int):
    """
    Loads a dataset (either Natural Questions or CNN/DM) using the Hugging Face Datasets library,
    applies the corresponding preprocessing function, and returns a list of processed examples.
    """
    dataset_name = dataset_name.lower()
    if dataset_name in ["natural_questions", "nq"]:
        dataset = load_dataset("natural_questions", "default", split=split, streaming=True)
        if samples_amount > -1:
            print(samples_amount)
            dataset = list(dataset.take(samples_amount))
        processed = [preprocess_nq(ex) for ex in dataset][250:]
    elif dataset_name in ["hotpotqa", "hotpot"]:
        dataset = load_dataset("hotpotqa/hotpot_qa", "fullwiki", split=split, trust_remote_code=True)
        if samples_amount > -1:
            print(samples_amount)
            dataset = list(dataset.take(samples_amount))
        processed = [preprocess_hotpotqa(ex) for ex in dataset]#[2500:]
        #print(processed[:5])
    elif dataset_name in ["cnn_dailymail", "cnndm"]:
        if split == "validation":
            split = "test"
        dataset = load_dataset("cnn_dailymail", "3.0.0", split=split)
        if samples_amount > -1:
            dataset = list(dataset.take(samples_amount))
        processed = [preprocess_cnndm(ex) for ex in dataset][:5000]
    elif dataset_name in ["sci", "scitldr"]:
        if split == "validation":
            split = "test"
        dataset = load_dataset("allenai/scitldr", "Abstract", split=split)
        if samples_amount > -1:
            dataset = list(dataset.take(samples_amount))
        processed = [preprocess_scitldr(ex) for ex in dataset]#[1500:5000]
    elif dataset_name in ["gsm8k", "gsm8k"]:
        if split == "validation":
            split = "test"
        dataset = load_dataset("gsm8k", "main", split=split)
        if samples_amount > -1:
            dataset = list(dataset.take(samples_amount))
        processed = [preprocess_gsm8k(ex) for ex in dataset]#[1500:5000]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return processed

def train_test_split_dataset(processed_dataset, test_size=0.1, random_state=42):
    """
    Splits the processed dataset into training and testing sets.
    """
    train_data, test_data = train_test_split(processed_dataset, test_size=test_size, random_state=random_state)
    return train_data, test_data

def collate_fn(batch):
    """
    A simple collate function that converts a list of dicts into a dict of lists.
    """
    return {key: [d[key] for d in batch] for key in batch[0]}

def create_data(dataset_name: str, split: str, samples_amount: int, batch_size: int = 8, test_size: float = 0.1, random_state: int = 42, training_type: str="ppo", tokenizer=None, evaluate=False):
    """
    Loads and preprocesses the dataset, splits it into training and test sets,
    and returns PyTorch DataLoaders for each if needed.
    """
    processed = load_and_preprocess_dataset(dataset_name, split, samples_amount)

    if tokenizer is not None:
        try:
            for sample in processed:
                sample["prompt"] = apply_chat_template(sample["prompt"], tokenizer)
        except Exception as e:
            print(e)
    valid_samples = [sample for sample in processed if sample["target"]]
    if not valid_samples:
        raise ValueError("No valid samples with target answers available.")
    if dataset_name == "scitldr" and test_size == 0.999:
        test_size = 0.998
    train_data, test_data = train_test_split_dataset(valid_samples, test_size=test_size, random_state=random_state)
    if evaluate:
        test_data = test_data[:1000]
    if training_type == "ppo":
        train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        test_data = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_data, test_data