import argparse
import os
from functools import partial

import torch
from accelerate import Accelerator
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import GRPOTrainer, GRPOConfig

from helper_files.helpers import ClientFactory, RewardEvaluator
from helper_files.data_helpers import create_data
from evaluation.evaluation import save_model
# -----------------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------------
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from vllm import LLM
from peft import PeftModel
from huggingface_hub import snapshot_download

class RewardEarlyStoppingCallback(TrainerCallback):
    """
    Early stop when the moving average of a reward metric
    does not improve by `min_delta` over `patience` steps.

    Args:
        window_size (int): Number of recent steps to compute the moving average.
        patience (int): Number of moving-average values to wait for improvement.
        min_delta (float): Minimum required improvement in moving average.
        reward_key (str): Key in the `logs` dict for the per-step reward mean.
    """
    def __init__(
        self,
        window_size: int = 25,
        patience: int = 100,
        min_delta: float = 0.01,
        reward_key: str = "rewards/composite_reward_func/mean",
    ):
        self.window_size = window_size
        self.patience = patience
        self.min_delta = min_delta
        self.reward_key = reward_key
        self._metric_history: list[float] = []
        self._ma_history: list[float] = []

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, float],
        **kwargs
    ) -> TrainerControl:
        # Only proceed if our reward metric is in this log step
        if logs is None or self.reward_key not in logs:
            return control

        # Record the raw reward mean
        reward = logs[self.reward_key]
        self._metric_history.append(reward)

        # Once we have enough points, compute moving average
        if len(self._metric_history) >= self.window_size:
            window = self._metric_history[-self.window_size:]
            ma = sum(window) / self.window_size
            self._ma_history.append(ma)

            # If we have more than `patience` moving-avg values, check improvement
            if len(self._ma_history) > self.patience:
                prev_ma = self._ma_history[-(self.patience + 1)]
                if ma - prev_ma < self.min_delta:
                    control.should_training_stop = True
        return control



def named_partial(func, **kwargs):
    p = partial(func, **kwargs)
    p.__name__ = func.__name__
    return p


# -----------------------------------------------------------------------------
# Main training routine
# -----------------------------------------------------------------------------

def main(args):
    # ---------------- GRPO training config ----------------

    early_stopping = RewardEarlyStoppingCallback(
        window_size=25,
        patience=100,
        min_delta=0.01,
        reward_key="rewards/composite_reward_func/mean",
    )
    grpo_conf = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        adam_beta1=0.9,
        adam_beta2=0.95,
        bf16=True,
        logging_steps=1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.batch_size,
        num_generations=4,#args.n_gen,
        max_prompt_length=1024,
        max_completion_length=args.max_new_tokens,
        num_train_epochs=args.epochs,
        save_steps=50,
        max_grad_norm=1,
        log_on_each_node=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        loss_type="dr_grpo",
        disable_dropout=True,
        sync_ref_model=False,
        ref_model_sync_steps=512,
        ref_model_mixup_alpha=0.6,
        mask_truncated_completions=True,
        shuffle_dataset=True,
        temperature=1.0,
        top_k=40,
        top_p=0.95,
        repetition_penalty=1.0,
        num_iterations=2,
        epsilon=0.2,
        epsilon_high=0.28,
        beta=0.0,
        #delta=None,
        reward_weights=None,
        scale_rewards=False,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        #ent_coef=0.01,
        )

    # ---------------- Accelerator / device ----------------
    accelerator = Accelerator()
    device = accelerator.device
    current_device = accelerator.local_process_index

    # ---------------- Tokeniser ---------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.rewriting_model_name)
    tokenizer.padding_side = "left"
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------------- LoRA / 4‑bit configs ----------------
    lora_conf = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    bnb_conf = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

    # ---------------- Training / val data -----------------
    train_data, val_data = create_data(
        args.dataset,
        args.split,
        args.samples_amount,
        args.batch_size,
        training_type="grpo",
        tokenizer=tokenizer,
    )

    # ---------------- Rewriting model (LoRA) --------------
    base_model = AutoModelForCausalLM.from_pretrained(
        args.rewriting_model_name,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_conf,
        device_map={"": current_device},
        attn_implementation="eager",
    )

    adapter_cfg = os.path.join(args.adapter_dir, "adapter_config.json")
    if os.path.exists(adapter_cfg):
        rewriting_model = PeftModel.from_pretrained(base_model, args.adapter_dir)
        print(f"Loaded LoRA adapter from {args.adapter_dir}")
    else:
        rewriting_model = get_peft_model(base_model, lora_conf)
        print("Initialised new LoRA adapter")
    
    rewriting_model = rewriting_model.train()
    for n, p in rewriting_model.named_parameters():
        if "lora_" in n:
            p.requires_grad = True
        else:
            p.requires_grad = False

    # ---------------- Response model ----------------------
    bedrock_client = ClientFactory.create_bedrock_client()
    if args.response_model_name.upper() == "GPT":
        response_model = ClientFactory.create_api_client()
    elif args.response_model_name.upper() == "BEDROCK":
        response_model = bedrock_client
    else:
        config_dir = snapshot_download(
            repo_id=args.response_model_name,
            allow_patterns=["config.json"]
        )
        response_model = LLM(
            model=args.response_model_name,
            dtype="bfloat16",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.5,
            hf_config_path=config_dir,
            max_model_len=4096,
            enable_lora=True
        )

    # ---------------- Generation kwargs -------------------
    generation_kwargs = {
        "max_new_tokens": 512,
        "do_sample": False,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "temperature": 0.0,
    }

    # ---------------- Reward function ---------------------
    judge_client = ClientFactory.create_api_client()
    local_judge = None
    if args.local_judge_weight > 0:
        # reuse response_model if it is local; else load a small local judge
        if isinstance(response_model, AutoModelForCausalLM):
            local_judge = response_model
        else:
            local_judge = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Llama-3.2-3B-Instruct",
                quantization_config=bnb_conf,
                torch_dtype=torch.bfloat16,
                device_map={"": device},
            )

    reward_fn = named_partial(
        RewardEvaluator.composite_reward_func,
        response_model=response_model,
        gen_kwargs=generation_kwargs,
        tokenizer=tokenizer,
        device=device,
        client=judge_client,
        judge_model=local_judge,
        parsing_weight=args.json_weight,
        gpt_judge_weight=args.gpt_weight,
        local_judge_weight=args.local_judge_weight,
        rouge_weight=args.rouge_weight,
        perplexity_weight=args.perpl_weight,
        bedrock_weight=args.bedrock_weight,
        bedrock_client=bedrock_client,
        dataset_name=args.dataset,
    )

    # ---------------- GRPO Trainer ------------------------
    trainer = GRPOTrainer(
        args=grpo_conf,
        model=rewriting_model,
        reward_funcs=[reward_fn],
        train_dataset=train_data,
        eval_dataset=val_data,
        processing_class=tokenizer,
        callbacks=[early_stopping]
        #peft_config=lora_conf,
    )


    trainer.train()

    #Save the last model
    save_model(rewriting_model, tokenizer, args.rewriting_model_name + args.dataset + "_final", output_dir=args.output_dir)
    _, val_data = create_data(args.dataset_name, "validation", -1, args.batch_size, 0.999, tokenizer=tokenizer, evaluate=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train GRPO with PEFT‑safe resume")
    parser.add_argument("--rewriting_model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--response_model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--dataset", type=str, default="cnn_dailymail")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--samples_amount", type=int, default=2500)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--n_gen", type=int, default=4)
    parser.add_argument("--grad_norm", type=float, default=0.1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="./grpo_output")
    parser.add_argument("--adapter_dir", type=str, default="./grpo_adapter")
    parser.add_argument("--resume_checkpoint", type=str, default="")
    # reward weights
    parser.add_argument("--json_weight", type=float, default=0.0)
    parser.add_argument("--gpt_weight", type=float, default=0.25)
    parser.add_argument("--local_judge_weight", type=float, default=0.0)
    parser.add_argument("--rouge_weight", type=float, default=0.5)
    parser.add_argument("--perpl_weight", type=float, default=0.0)
    parser.add_argument("--bedrock_weight", type=float, default=0.25)

    args = parser.parse_args()
    main(args)
