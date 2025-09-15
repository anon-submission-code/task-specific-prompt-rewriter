import os
import json
from dotenv import load_dotenv
import openai
from helpers import RewardEvaluator, Generator, ClientFactory

def generate_candidate_prompts(original_query, reference_answer, num_candidates=3, gpt_client=None, gen_kwargs=None):
    """
    Uses the GPT client to generate candidate prompts.
    """
    base_prompt = f"""
    You are an AI assistant specialized in crafting highly effective prompts for other AI assistants. Given an original query and one or more reference answers, produce a prompt that will guide another AI assistant to generate a response matching at least one of the reference answers.

**Best Practices to Follow:**
1. **Meaning Preservation:** Do not reveal or restate the reference answer. Keep the original query’s intent unchanged.
2. **Clarity & Simplicity:** Write concise, straightforward instructions without ambiguous phrasing.
3. **Specificity & Context:** Specify desired response format, structure, or persona if helpful (e.g., "Respond as a domain expert," "Use JSON with fields ...").
4. **Positive Instruction Tone:** Use direct "Do X" instructions; avoid long lists of prohibitions.
5. **Examples & Reasoning Cues:** When appropriate, include examples and/or a step-by-step/chain-of-thought cue to help the responding AI reason through the query.

**Output Only the New Prompt:** 
Do not include any commentary, analysis, or mention of the reference answers.

[ORIGINAL QUERY]: {original_query}

[REFERENCE ANSWER]: {reference_answer}
    """
    candidate_prompts = []
    for _ in range(num_candidates):
        candidate = Generator.generate_api_response(gpt_client, base_prompt, **gen_kwargs)
        candidate_prompts.append(candidate.strip())
    #print(candidate_prompts)
    return candidate_prompts

def evaluate_candidate_prompt(candidate_prompt, original_query, reference_answer, gen_kwargs,
                              gpt_client, bedrock_client):
    """
    Generates final answers using both GPT and Bedrock clients and computes evaluation scores.
    Uses:
      - GPT client with model "gpt-4o-mini" for one response,
      - Bedrock client with model "meta.llama3-1-8b-instruct-v1:0" for the other response.
    Both responses are evaluated:
      - GPT evaluation is computed via compute_llm_reward.
      - Bedrock evaluation is computed via compute_bedrock_reward.
    Also computes ROUGE (for reporting only) using the GPT response.
    Returns:
      (gpt_score, bedrock_score, rouge_score, final_answer_gpt, final_answer_bedrock)
    """
    gen_kwargs2 = {"max_new_tokens": 512, "temperature": 0.0, "do_sample": False}
    # Generate final answer using GPT client (e.g., gpt-4o-mini)
    final_answer_gpt = Generator.generate_api_response(gpt_client, candidate_prompt, **gen_kwargs)
    # Generate final answer using Bedrock client (e.g., meta.llama3-1-8b-instruct-v1:0)
    final_answer_bedrock = Generator.generate_bedrock_response(
        bedrock_client, candidate_prompt, model_id="meta.llama3-1-8b-instruct-v1:0", **gen_kwargs
    )
    
    # Evaluate using GPT-based judging.
    gpt_score_for_gpt = RewardEvaluator.compute_llm_reward(
        gpt_client,
        original_query,          # original query
        candidate_prompt,        # candidate prompt used as rewritten query
        final_answer_gpt,
        [reference_answer],
        gen_kwargs2
    )
    gpt_score_for_llama = RewardEvaluator.compute_llm_reward(
        gpt_client,
        original_query,          # original query
        candidate_prompt,        # candidate prompt used as rewritten query
        final_answer_bedrock,
        [reference_answer],
        gen_kwargs2
    )
    # Evaluate using Bedrock-based judging.
    bedrock_score_for_llama = RewardEvaluator.compute_bedrock_reward(
        bedrock_client,
        original_query,          # original query
        candidate_prompt,        # candidate prompt used as rewritten query
        final_answer_bedrock,
        [reference_answer],
        gen_kwargs2
    )
    bedrock_score_for_gpt = RewardEvaluator.compute_bedrock_reward(
        bedrock_client,
        original_query,          # original query
        candidate_prompt,        # candidate prompt used as rewritten query
        final_answer_gpt,
        [reference_answer],
        gen_kwargs2
    )
    # Compute ROUGE score from the GPT final answer (for reporting only).
    rouge_score = RewardEvaluator.compute_rouge_reward(final_answer_gpt, [reference_answer])
    
    return gpt_score_for_gpt, gpt_score_for_llama, bedrock_score_for_llama, bedrock_score_for_gpt, rouge_score, final_answer_gpt, final_answer_bedrock

def run_sft_generation(samples, num_candidates=5, num_successful=100,
                       gen_kwargs=None, gpt_client=None, bedrock_client=None):
    """
    For each sample (with keys "original_query" and "reference_answer"),
    generate candidate prompts using the GPT client.
    For each candidate prompt, generate final answers with both GPT and Bedrock clients,
    and evaluate them with both judging functions.
    Only if both the GPT and Bedrock scores are at least 4, store the candidate prompt
    along with its responses and scores.
    """
    successful_prompts = []
    perfect_prompts = []
    for sample in samples:
        original_query = sample["original_query"]
        reference_answer = sample["reference_answer"]
        candidate_prompts = generate_candidate_prompts(original_query, reference_answer,
                                                       num_candidates=num_candidates,
                                                       gpt_client=gpt_client,
                                                       gen_kwargs=gen_kwargs)
        for candidate in candidate_prompts:
            gpt_score_for_gpt, gpt_score_for_llama, bedrock_score_for_llama, bedrock_score_for_gpt, rouge_score, final_answer_gpt, final_answer_bedrock = evaluate_candidate_prompt(
                candidate, original_query, reference_answer, gen_kwargs, gpt_client, bedrock_client
            )
            # Only store if both evaluation scores are at least 4.
            if gpt_score_for_gpt >= 4 and gpt_score_for_llama >= 4 and bedrock_score_for_llama >= 4 and bedrock_score_for_gpt >= 4:
                successful_prompts.append({
                    "original_query": original_query,
                    "reference_answer": reference_answer,
                    "candidate_prompt": candidate,
                    "final_answer_gpt": final_answer_gpt,
                    "final_answer_bedrock": final_answer_bedrock,
                    "scores": [gpt_score_for_gpt, gpt_score_for_llama, bedrock_score_for_llama, bedrock_score_for_gpt],
                    "rouge_score": rouge_score
                })
            if gpt_score_for_gpt >= 5 and gpt_score_for_llama >= 5 and bedrock_score_for_llama >= 5 and bedrock_score_for_gpt >= 5:
                perfect_prompts.append({
                    "original_query": original_query,
                    "reference_answer": reference_answer,
                    "candidate_prompt": candidate,
                    "final_answer_gpt": final_answer_gpt,
                    "final_answer_bedrock": final_answer_bedrock,
                    "scores": [gpt_score_for_gpt, gpt_score_for_llama, bedrock_score_for_llama, bedrock_score_for_gpt],
                    "rouge_score": rouge_score
                })
            if len(successful_prompts) >= num_successful:
                break
        if len(successful_prompts) >= num_successful:
            break
    return successful_prompts, perfect_prompts

if __name__ == "__main__":
    load_dotenv("./.env")
    # Load samples from a JSON file. Each sample must contain "original_query" and "reference_answer".
    with open("samples_hotpot.json", "r") as f:
        samples = json.load(f)

    # Generation parameters for API calls.
    gen_kwargs = {"max_new_tokens": 512, "temperature": 1, "top_p": 0.95}
    

    # Create the GPT API client (for prompt generation and GPT judging/response).
    gpt_client = openai.AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_API_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
        azure_deployment=os.getenv("AZURE_OPENAI_API_DEPLOYMENT")
    )

    # Create the Bedrock client (for Bedrock judging/response).
    bedrock_client = ClientFactory.create_bedrock_client()

    # Run SFT candidate generation and evaluation.
    successful_prompts, perfect_prompts = run_sft_generation(
        samples,
        num_candidates=4,
        num_successful=750,
        gen_kwargs=gen_kwargs,
        gpt_client=gpt_client,
        bedrock_client=bedrock_client
    )
    # Save successful candidate prompts to a JSON file.
    with open("successful_prompts_hotpot.json", "w") as f:
        json.dump(successful_prompts, f, indent=2)
    with open("perfect_prompts_hotpot.json", "w") as f:
        json.dump(perfect_prompts, f, indent=2)

    print(f"Collected {len(successful_prompts)} successful candidate prompts.")
