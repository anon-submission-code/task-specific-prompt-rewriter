import os
import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dotenv import load_dotenv
from accelerate import Accelerator
from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset, concatenate_datasets, Dataset as HFDataset  # Hugging Face Datasets

# Mapping function for rishavkundu/prompt_optimization_dataset
def map_prompt_opt(example):
    original = example.get("original_prompt", "").strip()
    analysis_text = example.get("analysis", "").strip()
    optimized = example.get("optimized_prompt", "").strip()
    meta_prompt = (
        "Rewrite the following instruction by rephrasing and/or adding specific requirements. "
        "Use illustrative descriptions if needed.\n"
        f"Original Instruction: {original}\n"
        "New Instruction:"
    )
    candidate = {"explanation": analysis_text, "final_rewritten_query": optimized}
    return {
        "original_query": meta_prompt,
        "candidate_prompt": json.dumps(candidate)
    }

# Mapping function for 10k Prompts Ranked dataset using the avg_rating column directly
def map_10k_prompts(example):
    raw_prompt = example.get("prompt", "").strip()
    avg_rating = example.get("avg_rating", "N/A")
    meta_prompt = (
        "Evaluate the following prompt by assigning a quality rating from 1 to 5 based on clarity and effectiveness.\n"
        f"Prompt: {raw_prompt}\n"
        "Quality Rating:"
    )
    candidate = {"explanation": "Aggregated human rating", "final_rating": str(avg_rating)}
    return {
        "original_query": meta_prompt,
        "candidate_prompt": json.dumps(candidate)
    }

def load_and_map_datasets():
    datasets_list = []
    
    # Load Prompt Optimization Dataset
    try:
        prompt_opt_ds = load_dataset("rishavkundu/prompt_optimization_dataset", split="train")
        prompt_opt_ds = prompt_opt_ds.map(map_prompt_opt)
        datasets_list.append(prompt_opt_ds)
    except Exception as e:
        print("Error loading rishavkundu/prompt_optimization_dataset:", e)
        raise e

    # Load 10k Prompts Ranked dataset
    try:
        ranking_ds = load_dataset("data-is-better-together/10k_prompts_ranked", split="train")
        ranking_ds = ranking_ds.map(map_10k_prompts)
    except Exception as e:
        print("Error loading 10k Prompts Ranked dataset:", e)
        raise e

    # Restrict ranking dataset to the same number of samples as in the prompt optimization dataset
    n_samples = len(prompt_opt_ds)
    ranking_ds = ranking_ds.select(range(min(n_samples, len(ranking_ds))))
    datasets_list.append(ranking_ds)
    
    # Concatenate the datasets
    combined_dataset = concatenate_datasets(datasets_list)
    print("Sample combined data:", combined_dataset[:5])
    return combined_dataset

# SFTDataset class remains the same
class SFTDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_length=1024):
        self.data = data_list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        original_query = sample.get("original_query", "").strip()
        candidate_prompt = sample.get("candidate_prompt", "").strip()
        
        prompt_tokens = self.tokenizer(original_query, add_special_tokens=False)["input_ids"]
        target_tokens = self.tokenizer(candidate_prompt, add_special_tokens=False)["input_ids"] + [self.tokenizer.eos_token_id]
        
        input_ids = prompt_tokens + target_tokens
        labels = ([-100] * len(prompt_tokens)) + target_tokens
        
        if len(input_ids) > self.max_length:
            available = self.max_length - len(prompt_tokens)
            target_tokens = target_tokens[:available]
            input_ids = prompt_tokens + target_tokens
            labels = ([-100] * len(prompt_tokens)) + target_tokens
        
        pad_len = self.max_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * pad_len
        labels += [-100] * pad_len
        attention_mask = [1] * (self.max_length - pad_len) + [0] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

def convert_to_hf_dataset(pt_dataset):
    data_list = list(pt_dataset)
    return HFDataset.from_dict({k: [d[k] for d in data_list] for k in data_list[0]})

def run_sft_training(
    batch_size=2,
    num_train_epochs=5,
    model_name="microsoft/Phi-4-mini-instruct",
    adapter_dir="",
    output_dir="./sft_model2"
):
    combined_dataset = load_and_map_datasets()
    data_list = list(combined_dataset)
    if not data_list:
        raise ValueError("Combined dataset is empty.")
    
    accelerator = Accelerator()
    device = accelerator.local_process_index

    # -------------- Configs ---------------
    lora_conf = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    bnb_conf = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

    # -------------- Base model -------------
    base = AutoModelForCausalLM.from_pretrained(
        "unsloth/Phi-4-mini-instruct",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_conf,
        device_map={"": device},
    )

    # ---------- Load or create adapter -----
    adapter_cfg_path = os.path.join(adapter_dir, "adapter_config.json")
    if os.path.exists(adapter_cfg_path):
        model = PeftModel.from_pretrained(base, adapter_dir)
        print(f"🔄 Loaded LoRA adapter from {adapter_dir}")
    else:
        model = get_peft_model(base, lora_conf)
        print("✨ Initialised new LoRA adapter")

    # -------------- Tokeniser --------------
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Phi-4-mini-instruct")
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model.train()

    # -------------- Dataset ---------------
    pt_ds = SFTDataset(HFDataset.from_list(data_list))
    train_ds = HFDataset.from_dict({k: [d[k] for d in pt_ds] for k in pt_ds[0]})

    # --------- Trainer arguments ----------
    sft_conf = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        learning_rate=5e-6,
        num_train_epochs=num_train_epochs,
        logging_steps=50,
        save_steps=len(train_ds) // 2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        gradient_accumulation_steps=1,
        warmup_ratio=0.2,
    )

    # -------------- Trainer ---------------
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        args=sft_conf,
        peft_config=lora_conf,
    )
    # -------- Resume logic (safe) ---------
    if os.path.isdir(adapter_dir):
        # *Do not* load optimizer/scheduler state to avoid parameter‑group clash.
        print(
            f"🚀 Resuming from weights in {adapter_dir} with a fresh optimiser (no state restore)."
        )
    else:
        adapter_dir = None  # safeguards
        print("🏁 Starting a fresh training run…")

    # ------ Train (weights already loaded) ----
    trainer.train(resume_from_checkpoint=None)  # we already loaded weights

    # -------------- Save -----------------
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("✅ SFT training complete. Adapter saved to:", output_dir)

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv("./.env")
    run_sft_training(
        batch_size=2,
        num_train_epochs=5,
        model_name="unsloth/Phi-4-mini-instruct",
        adapter_dir="./sft_phi_new/checkpoint-3265",
        output_dir="./sft_phi_last"
    )
