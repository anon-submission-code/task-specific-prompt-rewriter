import argparse
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from datasets import load_dataset
from trl import AutoModelForCausalLMWithValueHead
from peft import LoraConfig
import warnings
from dotenv import load_dotenv
import openai
import os
from data_helpers import create_data
from evaluation import evaluate_response_model
from helpers import ClientFactory
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
from vllm import LLM
from huggingface_hub import snapshot_download


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    tokenizer = None
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
            gpu_memory_utilization=0.8,    # fraction of GPU RAM to reserve
            hf_config_path=config_dir,# <-- instruct vLLM to load via BnB
            max_model_len=4096,
            enable_lora=True
        )
        tokenizer = AutoTokenizer.from_pretrained(args.response_model_name)
        tokenizer.padding_side = 'left'
    # Generation configuration for final answer generation
    generation_kwargs = {
        "max_new_tokens": 512,
        "do_sample": False,
    }
    dataset_name = args.dataset_name
    if not args.full_prompt and dataset_name in ["natural_questions", "nq"]:
        instruction = "Answer the question"
    elif not args.full_prompt and dataset_name in ["hotpot"]:
        instruction = "Answer the question based on the context"
    elif not args.full_prompt and dataset_name in ["cnn_dailymail", "cnndm", "scitldr", "sci"]:
        instruction = "Summarize the text"
    elif not args.full_prompt and dataset_name in ["gsm8k", "gsm"]:
        instruction = "SOLUTION"
    else:
        instruction = None
    print(instruction)
    _, val_data = create_data(args.dataset_name, "validation",-1, args.batch_size, 0.999, evaluate=True) 
    evaluate_response_model(response_model, val_data, generation_kwargs, tokenizer, device, instruction, dataset_name=args.dataset_name, name=args.response_model_name + args.dataset_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model using evaluate_model function.")
    parser.add_argument("--response_model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Reward evaluation model")
    parser.add_argument("--dataset_name", type=str, default="cnndm", help="Dataset name")
    parser.add_argument("--reward", type=str, default="llm_test", help="Evaluation type")
    parser.add_argument("--name", type=str, default="GRPO_GPT", help="Name for saving the result")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--json_format", type=bool, default=True, help="Are the rewritten prompts in JSON format")
    parser.add_argument("--full_prompt", type=bool, default=False, help="Are the rewritten prompts based on the entire query")
    args = parser.parse_args()
    main(args)