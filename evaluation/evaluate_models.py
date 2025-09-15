import argparse
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from helper_files.data_helpers import create_data
from evaluation.evaluation import evaluate_model
from helper_files.helpers import ClientFactory
from vllm import LLM
from huggingface_hub import snapshot_download

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    bnb_config2 = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    
    if args.vllm:
        if "qwen" in args.model_path.lower():
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
            gpu_memory_utilization=0.43,    # fraction of GPU RAM to reserve
            #load_format="bitsandbytes",
            hf_config_path=config_dir,# <-- instruct vLLM to load via BnB
            max_model_len=2048,
            enable_lora=True
        )
    else:
        rewriting_model = AutoModelForCausalLM.from_pretrained(args.model_path, quantization_config=bnb_config2, torch_dtype=torch.bfloat16, device_map={"":device})

    if args.response_model_name.upper() == "GPT":
        print("Using GPT API client")
        response_model = ClientFactory.create_api_client()  # API client for GPT
    elif args.response_model_name.upper() == "BEDROCK":
        print("Using Bedrock client")
        response_model = ClientFactory.create_bedrock_client()
    else:
        config_dir = snapshot_download(
            repo_id=args.response_model_name,
            allow_patterns=["config.json"]
        )
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
        
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    """
    
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    """
    tokenizer.padding_side = "left"
    # Generation configuration for final answer generation
    generation_kwargs = {
        "max_new_tokens": 512,
        "do_sample": False,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    dataset_name = args.dataset_name
    data_amount = 0.025
    if dataset_name in ["gsm8k", "gsm"]:
        data_amount = 0.05
    if not args.full_prompt and dataset_name in ["natural_questions", "nq"]:
        instruction = "Answer the question"
    elif not args.full_prompt and dataset_name in ["hotpot"]:
        instruction = "Answer the question based on the context"
    elif not args.full_prompt and dataset_name in ["cnn_dailymail", "cnndm", "sci", "scitldr"]:
        instruction = "Summarize the text"
    elif not args.full_prompt and dataset_name in ["gsm8k", "gsm"]:
        instruction = "SOLUTION"
    else:
        instruction = None
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    _, val_data = create_data(args.dataset_name, "validation", -1, args.batch_size, 0.999, tokenizer=tokenizer, evaluate=True)
    # Call the evaluate_model function using the loaded models and data.
    model_path = args.model_path
    if model_path in ["meta-llama/Llama-3.2-3B-Instruct", "unsloth/Phi-4-mini-instruct", "Qwen/Qwen3-4B"]:
        model_path = ""
    evaluate_model(rewriting_model, response_model, val_data, generation_kwargs, tokenizer, device, name=args.name + args.model_path + args.dataset_name, json_format=args.json_format,
                    instruction=instruction, dataset_name=dataset_name, use_vllm=args.vllm, vllm_path=model_path)
    #evaluate_response_model(response_model, batches, tokenizer, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model using evaluate_model function.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--response_model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Reward evaluation model")
    parser.add_argument("--dataset_name", type=str, default="cnndm", help="Dataset name")
    parser.add_argument("--reward", type=str, default="llm_test", help="Evaluation type")
    parser.add_argument("--name", type=str, default="GRPO_GPT", help="Name for saving the result")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--json_format", type=bool, default=True, help="Are the rewritten prompts in JSON format")
    parser.add_argument("--full_prompt", type=bool, default=False, help="Are the rewritten prompts based on the entire query")
    parser.add_argument("--vllm", type=bool, default=True, help="Do we use vLLM")
    args = parser.parse_args()
    main(args)