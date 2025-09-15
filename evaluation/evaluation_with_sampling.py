import os
import torch
from dotenv import load_dotenv
from helper_files.data_helpers import create_meta_prompt, create_data
from helper_files.utils import compute_metrics, write_results_to_csv, try_parse_json, evaluate_gsm8k_response
import argparse
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from helper_files.helpers import ClientFactory, Generator, RewardEvaluator
import re
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
from vllm import LLM
from huggingface_hub import snapshot_download
def extract_choice_index(choice: str, offset: int = 1) -> int:
    """
    Given an LLM output `choice`, return the zero-based index corresponding
    to its "final_selection".  e.g. final_selection=2 → returns 1.
    Returns -1 if nothing valid is found.
    """
    # 1) Try strict JSON parse first
    try:
        parsed = try_parse_json(choice)
        fs = parsed.get("final_selection", None)
        # if it's already an int
        if isinstance(fs, int):
            return fs - offset
        # if it's a string of digits
        if isinstance(fs, str):
            clean = fs.strip()
            if clean.isdigit():
                return int(clean) - offset
    except Exception:
        pass

    # 2) Fallback: find *all* digit substrings in the raw text, take the *last* one
    all_nums = re.findall(r"\d+", choice)
    if all_nums:
        return int(all_nums[-1]) - offset

    # 3) Nothing found
    return -1


def evaluate_model_with_sampling(
    rewriting_model,
    response_model,
    data,
    generation_kwargs,
    tokenizer,
    device,
    n: int = 5,
    json_format: bool = True,
    name: str = "GRPO_Rouge_3B_4o-mini_sampling",
    instruction: str = None,
    dataset_name: str = "nq",
    compute_stats: bool = True,
    compute_pick: bool = True,
    vllm_path=""
):
    # Prepare rewriting vs. evaluation generation kwargs
    rewrite_kwargs = generation_kwargs.copy()
    rewrite_kwargs.update({"do_sample": True, "temperature": 0.5})
    eval_kwargs = generation_kwargs.copy()
    eval_kwargs.update({"do_sample": False, "temperature": 0.0})

    judge_client = ClientFactory.create_api_client()

    # Helper: evaluate prompts_list (len = batch size) on response_model with eval_kwargs
    def eval_with_prompt_list(prompts_list):
        total = acc = f1s = rouges = llms = 0
        for batch in data:
            if instruction is not None:
                prompts = [prompts_list + " " + sample for sample in batch["text"]]
            else:
                prompts = prompts_list[: len(batch["text"])]
            #print(prompts)
            final_answers = Generator.generate_final_answers(
                prompts,
                eval_kwargs,
                response_model,
                tokenizer,
                device,
                use_vllm=True, vllm_path=vllm_path
            )
            for ans, targets, text, original in zip(
                final_answers, batch["target"], batch["text"], batch["original"]
            ):
                if not ans:
                    continue
                em, f1 = compute_metrics(ans, targets)
                if dataset_name == "gsm8k":
                    reward = evaluate_gsm8k_response(ans, targets[0])
                    r_val = llm_val = reward
                else:
                    prompt_for_reward = (instruction + text) if instruction else (original + text)
                    r_val = RewardEvaluator.compute_reward(
                        "rouge", ans, targets, prompt=prompt_for_reward, gen_kwargs=eval_kwargs
                    )
                    llm_val = RewardEvaluator.compute_reward(
                        "llm", ans, targets, prompt=prompt_for_reward, client=judge_client, gen_kwargs=eval_kwargs, dataset_name=dataset_name
                    )
                if r_val < 0 or llm_val < 0:
                    continue
                acc += em
                f1s += f1
                rouges += r_val
                llms += llm_val
                total += 1
        return {
            "accuracy": acc / total,
            "f1": f1s / total,
            "rouge": rouges / total,
            "llm": llms / total,
            "count": total,
        }

    # Instruction-level sampling
    if instruction is not None:
        meta_prompt = create_meta_prompt(text="", instruction=instruction)
        rewrites = Generator.rewrite_prompts(
            rewriting_model, tokenizer, [meta_prompt] * n, rewrite_kwargs, device, use_vllm=True, vllm_path=vllm_path
        )[0]
        metrics_by_idx = {}
        for idx, rw in enumerate(rewrites):
            try:
                parsed = try_parse_json(rw)
                fr = parsed.get("final_rewritten_query")
                if fr: rewrites[idx] = fr
            except:
                pass
            metrics = eval_with_prompt_list(rewrites[idx])
            print(f"Rewrite #{idx+1}: {rewrites[idx]}")
            print(metrics)
            write_results_to_csv(f"{name}_inst_{idx}_{rewrites[idx]}", metrics["accuracy"], metrics["f1"], metrics["rouge"], metrics["llm"], filename="sampling.csv")
            #rewrites.append(rw)
            metrics_by_idx[idx] = metrics
        if compute_pick:
            sel_meta_prompt = f"This was the initial instruction: {instruction}.\n Here are candidate rewrites of this instruction:\n"
            for j, instr_rw in enumerate(rewrites, 1): sel_meta_prompt += f"{j}. {instr_rw}\n"
            sel_meta_prompt += (
                f"Which rewrite will ensure the best possible chance for another LLM to answer the original query correctly? Make sure your selection does not change the meaning of the initial task."
                "Output your response as a JSON object with two keys: "
            "\"explanation\" and \"final_selection\". The \"explanation\" key should contain your reasoning about each"
            "choice and how you are making the selection between them, and the \"final_selection\" key should contain only the number of the selected rewrite, nothing else.\n\n{"
            )
            choice = Generator.generate_final_answers([sel_meta_prompt], eval_kwargs, rewriting_model, tokenizer, device, use_vllm=True, vllm_path=vllm_path)[0].strip()
            print(choice)
            pick_idx = extract_choice_index(choice)
            if pick_idx > -1 and pick_idx < n:
                print(pick_idx)
                pick_metrics = metrics_by_idx[pick_idx]
                write_results_to_csv(
                    f"{name}_picking_inst_{rewrites[pick_idx]}",
                    pick_metrics["accuracy"], pick_metrics["f1"], pick_metrics["rouge"], pick_metrics["llm"]
                )
        return  # done for instruction-only

    # Per-example sampling
    total_max = {k:0.0 for k in ["accuracy","f1","rouge","llm"]}
    total_mean = {k:0.0 for k in ["accuracy","f1","rouge","llm"]}
    total_choice = {k:0.0 for k in ["accuracy","f1","rouge","llm"]}
    example_count = 0
    selection_count = 0
    for batch in data:
        prompts = batch["prompt"]
        texts = batch["text"]; originals = batch["original"]; targets = batch["target"]
        bs = len(prompts)
        flat_rewrites = Generator.rewrite_prompts(
            rewriting_model, tokenizer, prompts * n, rewrite_kwargs, device
        )[0]
        #print(flat_rewrites)
        grouped = [[flat_rewrites[i + k * bs] for k in range(n) ] for i in range(bs)]
        for i in range(bs):
            orig, text, tgt = originals[i], texts[i], targets[i]
            rewrites = grouped[i]
            # If picking, evaluate pick metrics but do not alter rewrites used for stats
            pick_idx = -1
            for k in range(len(rewrites)):
                if json_format:
                    try:
                        parsed = try_parse_json(rewrites[k])
                        fr = parsed.get("final_rewritten_query")
                        if fr is not None:
                            rewrites[k] = fr
                    except:
                        pass
            if compute_pick:
                sel_p = f"This was the initial instruction: {orig} {text}.\n Here are candidate rewrites of this instruction:\n"
                for j, rw_opt in enumerate(rewrites,1): sel_p += f"{j}. {rw_opt}"
                sel_p += (
                    f"Which rewrite will ensure the best possible chance for another LLM to answer this query correctly?"
                    "Respond with the number of the best rewrite only, nothing else."
                )
                choice = Generator.generate_final_answers([sel_p], rewrite_kwargs, rewriting_model, tokenizer, device)[0].strip()
            pick_idx = extract_choice_index(choice)
            # compute stats on all rewrites
            ems, f1s, rouges, llms = [], [], [], []
            for rw in rewrites:
                full_prompt = (
                    f"This was the original query: {orig} {text}" +
                    f"Follow this new instruction, only use information from the original query if any important details are missing from the new instruction: {rw}"
                )
                #print(full_prompt)
                final = Generator.generate_final_answers([full_prompt], eval_kwargs, response_model, tokenizer, device)[0]
                #print(final)
                if not final:
                    print("not")
                    continue
                em, f1 = compute_metrics(final, tgt)
                if dataset_name.lower() in ["gsm8k","gsm"]:
                    r_val = evaluate_gsm8k_response(final, tgt[0])
                    llm_val = r_val
                else:
                    ctx = orig + text
                    r_val = RewardEvaluator.compute_reward("rouge", final, tgt, prompt=ctx, gen_kwargs=eval_kwargs)
                    llm_val = RewardEvaluator.compute_reward("llm_test", final, tgt, prompt=ctx, client=judge_client, gen_kwargs=eval_kwargs, dataset_name=dataset_name)
                if r_val < 0 or llm_val < 0:
                    continue
                ems.append(em)
                f1s.append(f1)
                rouges.append(r_val)
                llms.append(llm_val)
            if not ems:
                continue
            example_count += 1
            
            if compute_stats:
                total_max["accuracy"] += max(ems)
                total_max["f1"] += max(f1s)
                total_max["rouge"] += max(rouges)
                total_max["llm"] += max(llms)
                total_mean["accuracy"] += sum(ems) / len(ems)
                total_mean["f1"] += sum(f1s) / len(f1s)
                total_mean["rouge"] += sum(rouges) / len(rouges)
                total_mean["llm"] += sum(llms) / len(llms)
                if pick_idx > -1 and pick_idx < n:
                    selection_count += 1
                    total_choice["accuracy"] += ems[pick_idx]
                    total_choice["f1"] += f1s[pick_idx]
                    total_choice["rouge"] += rouges[pick_idx]
                    total_choice["llm"] += llms[pick_idx]
    print(example_count)
    print("Max metrics: ", total_max)
    print("Mean metrics:", total_mean)
    print("Choice metrics:", total_choice)
    if example_count>0:
        for k in total_max: total_max[k]/=example_count; total_mean[k]/=example_count; total_choice[k]/=selection_count
        print(f"{name} Sampling Evaluation:")
        print(example_count)
        print(selection_count)
        print("Max metrics: ", total_max)
        print("Mean metrics:", total_mean)
        print("Choice metrics:", total_choice)
        write_results_to_csv(f"{name}_max", total_max["accuracy"], total_max["f1"], total_max["rouge"], total_max["llm"])
        write_results_to_csv(f"{name}_mean", total_mean["accuracy"], total_mean["f1"], total_mean["rouge"], total_mean["llm"])
        write_results_to_csv(f"{name}_choice", total_choice["accuracy"], total_choice["f1"], total_choice["rouge"], total_choice["llm"])


def main(args):
    load_dotenv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bnb1=BitsAndBytesConfig(load_in_8bit=True)
    bnb2=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.float16)
    if args.vllm:
        if "qwen" in args.model_path:
            model_path = "Qwen/Qwen3-4B"
        elif "llama" in args.model_path:
            model_path = "meta-llama/Llama-3.2-3B-Instruct"
        else:
            model_path = "unsloth/Phi-4-mini-instruct"
        config_dir = snapshot_download(
            repo_id=model_path,
            allow_patterns=["config.json"]
        )
        rewriting_model = LLM(
            model=model_path,           # your HF‐style checkpoint or model ID
            dtype="bfloat16",                # to match your desired compute dtype
            #trust_remote_code=True,          # if your repo uses custom code
            tensor_parallel_size=1,           # shard weights
            gpu_memory_utilization=0.8,    # fraction of GPU RAM to reserve
            #load_format="bitsandbytes",
            hf_config_path=config_dir,# <-- instruct vLLM to load via BnB
            max_model_len=4096,
            enable_lora=True
        )
    else:
        rewriting_model = AutoModelForCausalLM.from_pretrained(args.model_path, quantization_config=bnb2, torch_dtype=torch.bfloat16, device_map={"":device})
        
    if args.response_model_name.upper()=="GPT": response_model=ClientFactory.create_api_client()
    elif args.response_model_name.upper()=="BEDROCK": response_model=ClientFactory.create_bedrock_client()
    else:
        response_model = LLM(
            model=args.response_model_name,           # your HF‐style checkpoint or model ID
            dtype="bfloat16",                # to match your desired compute dtype
            #trust_remote_code=True,          # if your repo uses custom code
            tensor_parallel_size=1,           # shard weights
            gpu_memory_utilization=0.43,    # fraction of GPU RAM to reserve
            #load_format="bitsandbytes",
            hf_config_path=config_dir,# <-- instruct vLLM to load via BnB
            max_model_len=2048,
            enable_lora=True
        )
    tokenizer=AutoTokenizer.from_pretrained(args.model_path); tokenizer.padding_side='left'; tokenizer.pad_token=tokenizer.pad_token or tokenizer.eos_token
    generation_kwargs={"max_new_tokens":512, "do_sample":False}
    batch_size = args.batch_size
   # if args.dataset_name.lower() in ["gsm8k","gsm"]:
    
    #batch_size = 2
    _, val_data=create_data(args.dataset_name,"validation",-1,batch_size,0.999,tokenizer=tokenizer, evaluate=True)
    print(len(val_data))
    instr=None
    dataset_name = args.dataset_name
    if args.instruction:
        instruction = args.instruction
    if not args.full_prompt and dataset_name in ["natural_questions", "nq"]:
        instruction = "Answer the question"
    elif not args.full_prompt and dataset_name in ["hotpot"]:
        instruction = "Answer the question based on the context"
    elif not args.full_prompt and dataset_name in ["cnn_dailymail","cnndm","sci","scitldr"]:
        instruction = "Summarize the text"
    elif not args.full_prompt and dataset_name in ["gsm8k", "gsm"]:
        instruction = "SOLUTION"
    else:
        instruction = None
    print(instr)
    model_path = args.model_path
    if model_path in ["meta-llama/Llama-3.2-3B-Instruct", "unsloth/Phi-4-mini-instruct", "Qwen/Qwen3-4B"]:
        model_path = ""
    evaluate_model_with_sampling(rewriting_model, response_model, val_data, generation_kwargs, tokenizer, device, n=args.n, json_format=args.json_format, name=args.name + args.model_path + args.dataset_name, instruction=instruction, dataset_name=args.dataset_name, compute_stats=args.compute_stats, compute_pick=args.compute_pick, vllm_path=model_path)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--response_model_name", default="BEDROCK")
    parser.add_argument("--dataset_name", default="cnndm")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--json_format", default=True)
    parser.add_argument("--full_prompt", default=False)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--name", default="GRPO_selection")
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--compute_stats", default=True)
    parser.add_argument("--compute_pick", default=True)
    parser.add_argument("--vllm", default=True)
    args = parser.parse_args()
    main(args)