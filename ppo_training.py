import argparse
import os
from functools import partial

import torch
from accelerate import Accelerator
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from tqdm import tqdm

from helper_files.helpers import ClientFactory, RewardEvaluator, Generator
from helper_files.utils import try_parse_json, write_samples_to_csv
from evaluation.evaluation import save_model
from helper_files.data_helpers import create_data

def named_partial(func, **kwargs):
    p = partial(func, **kwargs)
    p.__name__ = func.__name__
    return p


def attach_or_load_lora(base_model, lora_cfg: LoraConfig, adapter_dir: str):
    """Return `PeftModel` with exactly one adapter (load if present, else init)."""
    cfg_path = os.path.join(adapter_dir, "adapter_config.json")
    if isinstance(base_model, PeftModel):
        return base_model
    if os.path.exists(cfg_path):
        model = PeftModel.from_pretrained(base_model, adapter_dir)
        print(f"🔄 Loaded LoRA adapter from {adapter_dir}")
    else:
        model = get_peft_model(base_model, lora_cfg)
        print("✨ Initialised new LoRA adapter")
    return model

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(args):
    # ---------------- PPO config ------------------------
    ppo_cfg = PPOConfig(
        steps=2000,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        mini_batch_size=max(1, args.batch_size // 2),
        gradient_accumulation_steps=max(1, args.batch_size // 2),
        ppo_epochs=3,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=0.1,
        adap_kl_ctrl=True,
        init_kl_coef=0.1,
        kl_penalty="kl",
        target=1.0,
        horizon=1000,
        gamma=1.0,
        lam=0.95,
        whiten_rewards=True,
        max_grad_norm=1.0,
        ratio_threshold=5.0,
        early_stopping=False,
    )

    # Accelerator / device
    accelerator = Accelerator()
    device = accelerator.device

    # Tokeniser
    tokenizer = AutoTokenizer.from_pretrained(args.rewriting_model_name)
    if getattr(tokenizer, 'pad_token', None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA / 4-bit configs
    lora_conf = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)#, bnb_4bit_compute_dtype=torch.float16)

    # Data
    train_dl, val_dl = create_data(
        args.dataset, args.split, args.samples_amount, args.batch_size, tokenizer=tokenizer
    )

    # Rewriting model (with value head)
    rewriting_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.rewriting_model_name,
        quantization_config=bnb_cfg,
        device_map={"": device},
        peft_config=lora_conf,
        v_head_init_strategy="normal",
    )

    # Response model client
    gpt_client = ClientFactory.create_api_client()

    # Generation kwargs
    final_gen_kwargs = {
        "max_new_tokens": 512,
        "do_sample": False,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "temperature": 0.0,
    }
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": 0.7,
        "do_sample": True,
        "top_k": 50,
        "top_p": 0.95,
        #"no_repeat_ngram_size": 3,
    }

    # PPO Trainer
    ppo_trainer = PPOTrainer(ppo_cfg, model=rewriting_model, tokenizer=tokenizer)

    # Early stopping buffers
    from collections import deque
    reward_window = deque(maxlen=25)
    best_window_mean = -float("inf")
    steps_since_improve = 0
    stop = False
    instruction = "SOLUTION"

    # Training loop
    for epoch in range(args.epochs):
        if stop:
            print(f"🔶 Early stopping at epoch {epoch}")
            break
        for idx, batch in tqdm(enumerate(train_dl), total=len(train_dl)):
            try:
                prompts = batch["prompt"]
                targets = batch["target"]
                texts = batch["text"]
                if not targets:
                    continue

                # Generate rewrites & answers
                rewrittens, queries, responses = Generator.rewrite_prompts(
                    rewriting_model, tokenizer, prompts, generation_kwargs, device, use_vllm=False
                )
                for i, rewritten in enumerate(rewrittens):
                    try:
                        parsed = try_parse_json(rewritten)
                        rewrittens[i] = (parsed.get("final_rewritten_query") or rewritten) + texts[i]
                    except:
                        pass
                final_answers = Generator.generate_final_answers(
                    rewrittens, final_gen_kwargs, gpt_client, tokenizer, device
                )

                # Compute prompt & output rewards
                prompt_rewards = [
                    torch.tensor([
                        RewardEvaluator.compute_reward(
                            "llm_prompt", "","",prompt=instruction, rewritten=rewrite, client=gpt_client, gen_kwargs=final_gen_kwargs
                        )
                    ], device=device)
                    for rewrite in rewrittens
                ]
                output_rewards = [
                    torch.tensor([
                        RewardEvaluator.compute_reward(
                            args.reward, ans, tgt[0], prompt=q, rewritten=rw, client=gpt_client, gen_kwargs=final_gen_kwargs
                        )
                    ], device=device)
                    for ans, tgt, q, rw in zip(final_answers, targets, queries, rewrittens)
                ]

                # Combine and normalize
                combined_rewards = []
                for pr, or_ in zip(prompt_rewards, output_rewards):
                    p, o = pr.item(), or_.item()
                    if p >= 0 and o >= 0:
                        r = p/10.0 + o/2.0
                    elif p >= 0:
                        r = p/5.0
                    elif o >= 0:
                        r = o
                    else:
                        r = 0.0
                    combined_rewards.append(torch.tensor([r], device=device))

                # Log samples
                for orig, rew, final_ans, acc_t, llm_t in zip(
                    texts, rewrittens, final_answers, output_rewards, prompt_rewards
                ):
                    write_samples_to_csv(
                        original=orig,
                        rewrite=rew,
                        final=final_ans,
                        acc=acc_t.item(),
                        llm=llm_t.item(),
                        filename="samples_ppo_final.csv"
                    )

                # PPO step
                stats = ppo_trainer.step(queries, responses, combined_rewards)
                mean_r = stats['ppo/mean_scores']
                print(f"Mean scores={mean_r:.3f}  KL={stats['ppo/policy/policykl']:.3f}  loss={stats['ppo/loss/total']:.3f}")

                # Save checkpoint every 50 steps
                if (idx + 1) % 50 == 0:
                    save_model(rewriting_model, tokenizer, f"{args.name}_step_{epoch}_{idx+1}", args.output_dir)
                    print(f"✅ Saved model at step {idx+1}")

                # Early stopping logic
                reward_window.append(mean_r)
                if len(reward_window) == reward_window.maxlen:
                    window_mean = sum(reward_window) / len(reward_window)
                    if window_mean > best_window_mean + 0.01:
                        best_window_mean = window_mean
                        steps_since_improve = 0
                    else:
                        steps_since_improve += 1
                    if idx >= 200 and steps_since_improve >= 100:
                        stop = True
                        break

            except Exception as e:
                print(e)

    # Save final model
    save_model(rewriting_model, tokenizer, f"{args.name}_final", args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train PPO with PEFT-safe resume")
    parser.add_argument("--rewriting_model_name", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--response_model_name", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--dataset", default="cnn_dailymail")
    parser.add_argument("--split", default="train")
    parser.add_argument("--samples_amount", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-7)
    parser.add_argument("--kl_coef", type=float, default=0.5)
    parser.add_argument("--kl_penalty", default="full")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--reward", default="rouge")
    parser.add_argument("--name", default="PPO_Training")
    parser.add_argument("--output_dir", default="./ppo_output")
    parser.add_argument("--adapter_dir", default="./ppo_adapter")
    parser.add_argument("--full_prompt", type=bool, default=False)

    args = parser.parse_args()
    main(args)
