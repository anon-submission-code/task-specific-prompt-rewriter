import os
import json
import torch
import openai
from dotenv import load_dotenv
from boto3 import session
from botocore.config import Config
from transformers import AutoModelForCausalLM, pipeline, Phi3ForCausalLM, LlamaForCausalLM, Qwen3ForCausalLM
from rouge_score import rouge_scorer
from utils import normalize_rewards, try_parse_json, extract_final_answer, normalize_answer, evaluate_gsm8k_response, apply_chat_template, compute_exact, write_samples_to_csv
from trl import AutoModelForCausalLMWithValueHead
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
# Create a shared ROUGE scorer and evaluation prompt template.
rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

PROMPT_QUALITY_EVAL_TEMPLATE = """\
You are given an original user query **[QUERY]** and a proposed **[REWRITTEN QUERY]** that attempts to improve it. Your task is to **evaluate how well the rewritten prompt improves upon the original without altering its intended meaning.**

## Evaluation Criteria

1. **Meaning Preservation (Critical):**  
   - If the rewrite alters the original intent in any way—turning the task into something else, adding incorrect assumptions, or narrowing the scope—assign **score 0**. Explain exactly how the meaning shifted. Bear in mind that creativity or adding some requirements should not warrant a score of 0 if the overall meaning still stays the same.

2. **Clarity & Simplicity:**  
   - Does the rewrite remove ambiguity and unnecessary complexity? It should be concise and straightforward.

3. **Specificity & Context:**  
   - Does it add helpful details (format, length, style, role/persona) that guide the model, while staying true to the user’s intent?

4. **Positive Instruction Tone:**  
   - Does it use direct “Do X” instructions rather than long lists of “Don’t do Y”? Positive directives should communicate the desired outcome clearly.

5. **Examples & Reasoning Cues (Highly Encouraged):**  
   - Including a brief example (one-shot/few-shot) or a “think step-by-step” instruction often boosts performance. Rewrites that add an illustrative example or chain-of-thought cue should be rewarded, since “providing examples” and “CoT prompting” are proven best-practices.

## Scoring Guidelines

- **0** – **Meaning Changed:** The rewrite clearly asks for a different task.
- **1** – **Poor Improvement:** Same or more confusing, no real clarity or guidance added.  
- **2** – **Minimal Improvement:** Fixes trivial wording but still vague or under-specified.  
- **3** – **Moderate Improvement:** Noticeably clearer or more specific, but misses key opportunities.  
- **4** – **Strong Improvement:** Significantly clearer/specific, uses positive instructions and context; may lack examples.  
- **5** – **Excellent Improvement:** Fully faithful, clear, specific, uses role/context, positive tone **and** includes a helpful example or reasoning cue.

## Output Format

Return exactly one JSON object with keys:

{{
  "explanation": "…justification…",
  "score": X
}}

### Examples

1. **Zero-Score (Meaning Changed):**  
   [QUERY]: “List the top 5 tourist attractions in Paris.”  
   [REWRITTEN QUERY]: “Rewrite the prompt on listing the top 5 tourist attractions in Paris. ”  
   {{
     "explanation": "The rewrite turns a factual ranking task into a rewriting task, which is not supposed to happen",
     "score": 0
   }}

2. **Minimal Improvement (1):**  
   [QUERY]: “How do I bake a chocolate cake?”  
   [REWRITTEN QUERY]: “How do I bake a chocolate cake? Please respond.”  
   {{
     "explanation": "Simply adds 'Please respond' without clarifying any details or structure.",
     "score": 1
   }}

3. **Moderate Improvement (3):**  
   [QUERY]: “Explain quantum computing.”  
   [REWRITTEN QUERY]: “Explain the concept of quantum computing in simple terms.”  
   {{
     "explanation": "Adds clarity on complexity level but doesn't use any improvements according to the prompt engineering guidelines",
     "score": 3
   }}

4. **Strong Improvement (4):**
   [QUERY]: “What is the history of AI?”
   [REWRITTEN QUERY]: “Provide a brief overview of the history of artificial intelligence, highlighting key milestones in chronological order.”
   {{
     "explanation": "Clarifies scope (brief overview), expands acronym, specifies format (three milestones, chronological).",
     "score": 4
   }}

5. **Excellent Improvement (5) with Example:**  
   [QUERY]: “What are 3 Newton Laws?”  
   [REWRITTEN QUERY]:
   You are a physics expert. Explain what are the 3 Newton Laws. Name them, give a description, and give a real-life example about each law. Use a numbered list for this.
   Example:
   1. Name: Name of Law Description: This law is about ... Real-Life Example: Imagine a ... . 
   2. ...
   {{
     "explanation": "Retains intent, adds role, format (numbered list), and a clear example to guide style and length.",
     "score": 5
   }}
   
6. **Chain-of-Thought Enhancement (5) with Reasoning Cues:**  
   [QUERY]: "When I was 3 years old, my partner was 3 times my age. Now I'm 20 years old. How old is my partner?"  
   [REWRITTEN QUERY]: "Calculate step by step: When I was 3, my partner was 3 times my age. Now I'm 20. Show your reasoning and then give the partner's current age."  
   {{
     "explanation": "Preserves the original math query while adding a 'step by step' chain-of-thought cue to improve accuracy and transparency of the reasoning process.",
     "score": 5
   }}

Respond only with valid JSON. Do not write an introduction or summary. Now evaluate this case:
[QUERY]: "{query}"
[REWRITTEN QUERY]: "{rewritten_query}"
Evaluation:"""


EVAL_PROMPT_TEMPLATE = """\
You are an AI assistant specialized in judging whether a given response correctly answers the user’s query based on the reference answer(s). Your task is to analyze the user's [QUERY], the provided [RESPONSE], and the reference [REFERENCE], then determine if the response includes the correct answer as specified by the reference. 

Consider these criteria in your evaluation of [RESPONSE]:
1. Does the response contain the factual or expected information that matches at least one of the reference answers?
2. Ignore any extra or irrelevant details—focus only on whether the core answer is correct.
3. If the response correctly answers the query (even if it includes additional information), assign a score of 1. Otherwise, assign a score of 0.

After your analysis, create a JSON object with the keys "explanation" and "score". In your explanation, briefly detail why the response is correct or incorrect. The score must be either 1 (correct) or 0 (incorrect). The explanation must come before the score. 

Examples:
1. Correct Answer:
[QUERY]: "Question: Who wrote 'Pride and Prejudice'?"
[RESPONSE]: "Jane Austen wrote 'Pride and Prejudice' in 1813. She was an English novelist known for her keen observations."
[REFERENCE]: "['Jane Austen']"
Evaluation: {{"explanation": "The response correctly states that Jane Austen wrote 'Pride and Prejudice,' matching the reference.", "score": 1}}

2. Incorrect Answer (Missing Core Information):
[QUERY]: "Question: What is the chemical symbol for gold?"
[RESPONSE]: "Gold is a precious metal often used in jewelry and electronics."
[REFERENCE]: "['Au']"
Evaluation: {{"explanation": "The response describes gold but does not provide the chemical symbol 'Au' required by the reference.", "score": 0}}

3. Correct Answer with Extra Detail:
[QUERY]: "Question: What is the boiling point of water in Celsius?"
[RESPONSE]: "Water boils at 100°C under standard atmospheric pressure. This is equivalent to 212°F."
[REFERENCE]: "['100°C']"
Evaluation: {{"explanation": "The response correctly gives 100°C, which matches the reference, despite extra conversion details.", "score": 1}}

4. Incorrect Answer (Wrong Fact):
[QUERY]: "Question: Who was the first person to walk on the Moon?"
[RESPONSE]: "Buzz Aldrin was the first person to walk on the Moon in 1969."
[REFERENCE]: "['Neil Armstrong']"
Evaluation: {{"explanation": "The response incorrectly names Buzz Aldrin, but the reference specifies Neil Armstrong.", "score": 0}}

Respond only with valid JSON. Do not include any additional commentary. Now evaluate this case:
[QUERY]: "{query}"
[RESPONSE]: "{final_answer}"
[REFERENCE]: "{target_answers}"
Evaluation:"""

EVAL_PROMPT_TEMPLATE_SUMM = """You are an AI assistant specialized in judging whether a generated summary based on a text is correctly created based on the reference summary. You will be given [QUERY], [RESPONSE], and [REFERENCE] sections. Evaluate how well the summary ([RESPONSE]) captures the key information from the reference summary ([REFERENCE]), considering the original text ([QUERY]).

- Coverage: Does [RESPONSE] include all the main points from [REFERENCE]?
- Hallucination: Does [RESPONSE] add any incorrect or unsupported details not in [REFERENCE]?
- Relevance: Check for extra irrelevant content (brevity is not required, but excessive irrelevant detail should lower the score).

Scoring rubric:
- 5: Summary covers all key information with minimal extra content.
- 1: Summary misses essential information or includes incorrect/hallucinated content.
- (Scores 2–4 indicate intermediate cases of missing or extra information.)

Output:
Return a JSON object with keys "explanation" (your analysis) and "score" (an integer from 1 to 5).

Examples:

You are an AI assistant specialized in judging whether a generated summary based on a text is correctly created based on the reference summary. You will be given [QUERY], [RESPONSE], and [REFERENCE] sections. Evaluate how well the summary ([RESPONSE]) captures the key information from the reference summary ([REFERENCE]), considering the original text ([QUERY]).

- Coverage: Does [RESPONSE] include all the main points from [REFERENCE]?  
- Hallucination: Does [RESPONSE] add any incorrect or unsupported details not in [REFERENCE]?  
- Relevance: Check for extra irrelevant content (brevity is not required, but excessive irrelevant detail should lower the score).

Scoring rubric:  
- 5: Summary covers all key information with minimal extra content.  
- 1: Summary misses essential information or includes incorrect/hallucinated content.  
- (Scores 2–4 indicate intermediate cases of missing or extra information.)

Output:  
Return a JSON object with keys "explanation" (your analysis) and "score" (an integer from 1 to 5). Respond only with valid JSON. Do not include any additional commentary.

Examples:

1.  
[QUERY]  
“In reinforcement learning, it is common to let an agent interact with its environment for a fixed amount of time before resetting the environment and repeating the process in a series of episodes. The task that the agent has to learn can either be to maximize its performance over (i) that fixed amount of time, or (ii) an indefinite period where the time limit is only used during training. In this paper, we investigate theoretically how time limits could effectively be handled in each of the two cases. In the first one, we argue that the terminations due to time limits are in fact part of the environment, and propose to include a notion of the remaining time as part of the agent's input. In the second case, the time limits are not part of the environment and are only used to facilitate learning. We argue that such terminations should not be treated as environmental ones and propose a method, specific to value-based algorithms, that incorporates this insight by continuing to bootstrap at the end of each partial episode. To illustrate the significance of our proposals, we perform several experiments on a range of environments from simple few-state transition graphs to complex control tasks, including novel and standard benchmark domains. Our results show that the proposed methods improve the performance and stability of existing reinforcement learning algorithms.”  
[RESPONSE]  
“We study optimal policy learning under both fixed-time and unbounded episodes by incorporating remaining time as input to the agent.”  
[REFERENCE]  
“We consider the problem of learning optimal policies in time-limited and time-unlimited domains using time-limited interactions.”  
Evaluation:  
{{"explanation": "The response includes the dual setting (fixed-time and unbounded episodes), the core contribution (including remaining time as input), and the learning focus, matching the reference’s problem statement with equivalent detail.", "score": 5}}

2.  
[QUERY]  
“The field of Deep Reinforcement Learning (DRL) has recently seen a surge in the popularity of maximum entropy reinforcement learning algorithms. Their popularity stems from the intuitive interpretation of the maximum entropy objective and their superior sample efficiency on standard benchmarks. In this paper, we seek to understand the primary contribution of the entropy term to the performance of maximum entropy algorithms. For the Mujoco benchmark, we demonstrate that the entropy term in Soft Actor Critic (SAC) principally addresses the bounded nature of the action spaces. With this insight, we propose a simple normalization scheme which allows a streamlined algorithm without entropy maximization to match the performance of SAC. Our experimental results demonstrate a need to revisit the benefits of entropy regularization in DRL. We also propose a simple non-uniform sampling method for selecting transitions from the replay buffer during training. We further show that the streamlined algorithm with the simple non-uniform sampling scheme outperforms SAC and achieves state-of-the-art performance on challenging continuous control tasks.”  
[RESPONSE]  
“This work focuses on designing a curriculum learning strategy for exploration in DRL.”  
[REFERENCE]  
“We propose a new DRL off-policy algorithm achieving state-of-the-art performance.”  
Evaluation:  
{{"explanation": "The response entirely misses the proposed off-policy algorithm and its performance outcome, instead hallucinating a curriculum learning focus unrelated to the reference.", "score": 1}}

3.  
[QUERY]  
“Argentina coach Alejandro Sabella believes Lionel Messi’s habit of throwing up during games is because of nerves. The Barcelona star has vomited on the pitch during several games over the last few seasons and appeared to once again during Argentina’s last warm-up match against Slovenia on Saturday.”  
[RESPONSE]  
“Messi vomits during games because of nerves.”  
[REFERENCE]  
“Argentina coach Sabella believes Messi’s habit of being sick during games is down to nerves.”  
Evaluation:  
{{"explanation": "The response correctly captures the cause but omits that this is the coach’s belief, losing the attribution nuance from the reference.", "score": 4}}

4.  
[QUERY]  
“Beijing, China has blocked the popular video-sharing web site YouTube, but did not offer a reason for the ban. YouTube was blocked in China as of Wednesday.”  
[RESPONSE]  
“China blocked YouTube on Wednesday, leading to network timeouts for users.”  
[REFERENCE]  
“By early Wednesday, users inside China encountered: ‘network timeout.’”  
Evaluation:  
{{"explanation": "The response adds a causal interpretation (“leading to”) instead of the reference’s simple observation, introducing an unsupported inference.", "score": 2}}


Respond only with valid JSON. Do not include any additional commentary. Now evaluate this case:
[QUERY]: "{query}"
[RESPONSE]: "{final_answer}"
[REFERENCE]: "{target_answers}"
Evaluation:"""


# --- Client Creation Functions ---
class ClientFactory:
    @staticmethod
    def create_bedrock_client(env_path="./.env"):
        load_dotenv(env_path)
        s = session.Session(
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION"),
        )
        config = Config(read_timeout=1000)
        return s.client(service_name='bedrock-runtime', config=config)

    @staticmethod
    def create_api_client(env_path="./.env"):
        load_dotenv(env_path)
        api_key = os.getenv('AZURE_OPENAI_API_KEY')
        endpoint = os.getenv('AZURE_OPENAI_API_ENDPOINT')
        api_version = os.getenv('AZURE_OPENAI_API_VERSION')
        deployment_name = os.getenv('AZURE_OPENAI_API_DEPLOYMENT')
        return openai.AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
            azure_deployment=deployment_name
        )

# --- Generation Functions ---

# --- Generation Functions ---
class Generator:
    @staticmethod
    def rewrite_prompts(
        model,
        tokenizer,
        prompts: list[str],
        gen_kwargs: dict,
        device: torch.device,
        use_chat_template: bool = False,
        use_vllm: bool = True,
        vllm_path: str = "",
    ) -> tuple[list[str], list[torch.Tensor], list[torch.Tensor]]:
        """
        Rewrite prompts for both Llama and Phi models.
        If use_vllm is True, `model` should be a vllm.LLM instance;
        otherwise it’s a Hugging Face CausalLM.
        Returns (rewritten_texts, input_ids_list, completion_ids_list).
        """
        if use_chat_template:
            prompts = [apply_chat_template(p, tokenizer) for p in prompts]

        if use_vllm:
            # Build sampling parameters from gen_kwargs
            sampling_params = SamplingParams(
                temperature=gen_kwargs.get("temperature", 1.0),
                top_p=gen_kwargs.get("top_p", 0.95),
                top_k=gen_kwargs.get("top_k", 40),
                max_tokens=gen_kwargs.get("max_new_tokens", 512),
            )
            # vLLM generate takes raw prompt strings
            if vllm_path != "":
                outputs = model.generate(prompts, sampling_params, lora_request=LoRARequest(
                    "vllm_eval_gen",
                    1,
                    vllm_path
                ),)
            else:
                outputs = model.generate(prompts, sampling_params)
            rewritten, queries, responses = [], [], []
            for out in outputs:
                # out.prompt is the original string; out.outputs[0].text is the generated suffix
                gen_text = out.outputs[0].text.strip()
                # strip any leading prompt echo
                rewritten_text = gen_text[len(out.prompt):].strip() if gen_text.startswith(out.prompt) else gen_text
                rewritten.append(rewritten_text)
                queries.append(None)           # no HF input_ids here
                # we won't have token IDs easily here—store text-only or convert if needed
                responses.append(None)
            return rewritten, queries, responses

        else:
            # Hugging Face path
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)
            model.eval()
            with torch.no_grad():
                response_ids = model.generate(**inputs, **gen_kwargs)
            model.train()

            decoded = tokenizer.batch_decode(
                [response_ids[i][inputs["input_ids"].size(1):]
                 for i in range(response_ids.size(0))],
                skip_special_tokens=True
            )
            rewritten = [
                txt.replace(prompts[i], "").strip()
                for i, txt in enumerate(decoded)
            ]
            queries = list(inputs["input_ids"])
            responses = [
                response_ids[i][inputs["input_ids"].size(1):]
                for i in range(response_ids.size(0))
            ]
            return rewritten, queries, responses

    @staticmethod
    def generate_api_response(client, prompt, **gen_kwargs):
        # Extract known generation parameters for chat completions.
        max_tokens = gen_kwargs.get("max_tokens", 512)
        temperature = gen_kwargs.get("temperature", 0.0)
        top_p = gen_kwargs.get("top_p", 0)
        try:
            messages = [{"role": "user", "content": prompt}]
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(e)
            if "'jailbreak': {'filtered': True" in str(e):
                print(prompt)
            return ""

    @staticmethod
    def generate_bedrock_response(client, prompt, model_id="meta.llama3-1-8b-instruct-v1:0", **gen_kwargs):
        # Extract known generation parameters for Llama 3.1-8B-Instruct.
        max_new_tokens = gen_kwargs.get("max_new_tokens", 512)
        temperature = gen_kwargs.get("temperature", 0)
        top_p = gen_kwargs.get("top_p", 0)
        formatted_prompt = (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
            f"{prompt}\n"
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        )
        native_request = {
            "prompt": formatted_prompt,
            "max_gen_len": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p
        }
        try:
            response = client.invoke_model(modelId=model_id, body=json.dumps(native_request))
            response_body = json.loads(response["body"].read())
            return response_body.get("generation", "")
        except Exception as e:
            print(f"ERROR invoking model: {e}")
            return ""

    @staticmethod
    def generate_final_answers(
        rewritten_texts: list[str],
        gen_kwargs: dict,
        response_model,
        tokenizer=None,
        device: str = "cuda",
        use_chat_template: bool = False,
        use_vllm: bool = True,
        vllm_path: str = ""
    ) -> list[str]:
        """
        Given rewritten prompts, generate final answers either via:
         - Azure/GPT API
         - HuggingFace CausalLM (including value-head models)
         - Bedrock
         - vLLM (if use_vllm=True and response_model is vllm.LLM)

        Returns a list of generated answer strings (with any rewritten prefix stripped).
        """
        # Ensure input is a list
        if not isinstance(rewritten_texts, list):
            rewritten_texts = [rewritten_texts]

        # 1) API-based (Azure GPT)
        if isinstance(response_model, openai.AzureOpenAI):
            return [
                Generator.generate_api_response(
                    response_model,
                    prompt,
                    temperature=gen_kwargs.get("temperature", 0.0),
                    max_tokens=gen_kwargs.get("max_new_tokens", 256)
                )
                for prompt in rewritten_texts
            ]

        # 2) vLLM path
        if use_vllm and isinstance(response_model, LLM):
            sampling_params = SamplingParams(
                temperature=gen_kwargs.get("temperature", 1.0),
                top_p=gen_kwargs.get("top_p", 1.0),
                top_k=gen_kwargs.get("top_k", 40),
                max_tokens=gen_kwargs.get("max_new_tokens", 512),
            )
            if vllm_path != "":
                outputs = response_model.generate(rewritten_texts, sampling_params, lora_request=LoRARequest(
                    "vllm_eval_gen",
                    1,
                    vllm_path
                ),)
            else:
                outputs = response_model.generate(rewritten_texts, sampling_params)
            final = []
            for out in outputs:
                # vLLM returns `out.prompt` (string) and `out.outputs[0].text` (full text)
                text = out.outputs[0].text
                # strip prompt echo if present
                if text.startswith(out.prompt):
                    text = text[len(out.prompt):]
                final.append(text.strip())
            return final

        # 3) HuggingFace or local Llama/Phi models
        if (
            isinstance(response_model, LlamaForCausalLM)
            or isinstance(response_model, Phi3ForCausalLM)
            or isinstance(response_model, AutoModelForCausalLMWithValueHead)
        ):
            if use_chat_template:
                rewritten_texts = [apply_chat_template(t, tokenizer) for t in rewritten_texts]
            inputs = tokenizer(
                rewritten_texts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(device)
            output_ids = response_model.generate(**inputs, **gen_kwargs)
            decoded = tokenizer.batch_decode(
                [output_ids[i][inputs["input_ids"].size(1):] for i in range(len(output_ids))],
                skip_special_tokens=True
            )
            # strip the rewritten prefix if the model echoed it
            return [
                txt.replace(rt, "", 1).strip() if txt.startswith(rt) else txt.strip()
                for txt, rt in zip(decoded, rewritten_texts)
            ]

        # 4) Bedrock path
        return [
            Generator.generate_bedrock_response(
                response_model,
                prompt,
                temperature=gen_kwargs.get("temperature", 0.0),
                max_new_tokens=gen_kwargs.get("max_new_tokens", 256)
            )
            for prompt in rewritten_texts
        ]

# --- Reward Computation Functions ---
EVAL_PROMPT_TEMPLATE_LLM_TEST  = """\
You are an AI assistant specialized in judging whether a given response correctly answers the user’s query based on the reference answer(s). Your task is to analyze the user's [QUERY], the provided [RESPONSE], and the reference [REFERENCE], then determine if the response includes the correct answer as specified by the reference. 

Consider these criteria in your evaluation of [RESPONSE]:
1. Does the response contain the factual or expected information that matches at least one of the reference answers?
2. Ignore any extra or irrelevant details—focus only on whether the core answer is correct.
3. If the response correctly answers the query (even if it includes additional information), assign a score of 1. Otherwise, assign a score of 0.

After your analysis, create a JSON object with the keys "explanation" and "score". In your explanation, briefly detail why the response is correct or incorrect. The score must be either 1 (correct) or 0 (incorrect). The explanation must come before the score. 

Examples:
1. Correct Answer:
[QUERY]: "Question: Who wrote 'Pride and Prejudice'?"
[RESPONSE]: "Jane Austen wrote 'Pride and Prejudice' in 1813. She was an English novelist known for her keen observations."
[REFERENCE]: "['Jane Austen']"
Evaluation: {{"explanation": "The response correctly states that Jane Austen wrote 'Pride and Prejudice,' matching the reference.", "score": 1}}

2. Incorrect Answer (Missing Core Information):
[QUERY]: "Question: What is the chemical symbol for gold?"
[RESPONSE]: "Gold is a precious metal often used in jewelry and electronics."
[REFERENCE]: "['Au']"
Evaluation: {{"explanation": "The response describes gold but does not provide the chemical symbol 'Au' required by the reference.", "score": 0}}

3. Correct Answer with Extra Detail:
[QUERY]: "Question: What is the boiling point of water in Celsius?"
[RESPONSE]: "Water boils at 100°C under standard atmospheric pressure. This is equivalent to 212°F."
[REFERENCE]: "['100°C']"
Evaluation: {{"explanation": "The response correctly gives 100°C, which matches the reference, despite extra conversion details.", "score": 1}}

4. Incorrect Answer (Wrong Fact):
[QUERY]: "Question: Who was the first person to walk on the Moon?"
[RESPONSE]: "Buzz Aldrin was the first person to walk on the Moon in 1969."
[REFERENCE]: "['Neil Armstrong']"
Evaluation: {{"explanation": "The response incorrectly names Buzz Aldrin, but the reference specifies Neil Armstrong.", "score": 0}}

Respond only with valid JSON. Do not include any additional commentary. Now evaluate this case:
[QUERY]: "{query}"
[RESPONSE]: "{final_answer}"
[REFERENCE]: "{target_answers}"
Evaluation:"""


class RewardEvaluator:
    @staticmethod
    def compute_rouge_reward(final_answer, target_answers):
        max_score = 0.0
        for ans in target_answers:
            score = rouge.score(ans, final_answer)['rougeL'].fmeasure
            max_score = max(max_score, score)
        return float(max_score)

    @staticmethod
    def compute_prompt_llm_reward(client, query, rewritten_query, gen_kwargs):
        prompt = PROMPT_QUALITY_EVAL_TEMPLATE.format(
            query=query,
            rewritten_query=rewritten_query,
        )
        try:
            output = Generator.generate_api_response(client, prompt, **gen_kwargs)
            #print(output)
            result = json.loads(output)
            score = result.get("score")
            return float(score)
        except Exception as e:
            print("Error computing LLM reward:", e)
            return -1

    @staticmethod
    def compute_llm_reward(client, query, rewritten_query, final_answer, target_answers, gen_kwargs, dataset_name="nq"):
        if dataset_name in ["scitldr", "cnndm"]:
            prompt = EVAL_PROMPT_TEMPLATE_SUMM.format(
            query=query,
            final_answer=final_answer,
            target_answers=target_answers
            )
        else:    
            prompt = EVAL_PROMPT_TEMPLATE.format(
                query=query,
                final_answer=final_answer,
                target_answers=target_answers
            )
        try:
            output = Generator.generate_api_response(client, prompt, **gen_kwargs)
            result = json.loads(output)
            score = result.get("score")
            return float(score)
        except Exception as e:
            print("Error computing LLM reward:", e)
            return 0

    @staticmethod
    def compute_llm_reward_test(client, query, final_answer, target_answers, gen_kwargs):
        prompt = EVAL_PROMPT_TEMPLATE_LLM_TEST.format(
            query=query,
            final_answer=final_answer,
            target_answers=target_answers
        )
        try:
            output = Generator.generate_api_response(client, prompt, **gen_kwargs)
            result = json.loads(output)
            score = result.get("score")
            return float(score)
        except Exception as e:
            print("Error computing LLM test reward:", e)
            #print(prompt)
            return -1

    @staticmethod
    def compute_llm_reward_local(local_model, query, rewritten_query, final_answer, target_answers, gen_kwargs, tokenizer, device):
        prompt = EVAL_PROMPT_TEMPLATE.format(
            query=query,
            rewritten_query=rewritten_query,
            final_answer=final_answer,
            target_answers=target_answers
        ) + "{"
        try:
            output = Generator.generate_final_answers(prompt, gen_kwargs, local_model, tokenizer, device)[0]
            output = output.replace(prompt, "").strip()
            result = try_parse_json(output)
            score = result.get("score")
            if score in [0, 5]:
                print(output)
            return float(score)
        except Exception as e:
            print("Error computing local LLM reward:", e)
            return -1

    @staticmethod
    def compute_bedrock_reward(bedrock_client, query, rewritten_query, final_answer, target_answers,
                               gen_kwargs, model_id="meta.llama3-1-8b-instruct-v1:0", prompt_eval=False, dataset_name="nq"):
        """
        Uses the Bedrock client to evaluate the final answer using the same evaluation prompt.
        """
        if dataset_name in ["scitldr", "cnndm"]:
            prompt = EVAL_PROMPT_TEMPLATE_SUMM.format(
            query=query,
            final_answer=final_answer,
            target_answers=target_answers
            ) + "{"
        else:
            prompt = EVAL_PROMPT_TEMPLATE.format(
                query=query,
                final_answer=final_answer,
                target_answers=target_answers
            ) + "{"
        if prompt_eval:
            prompt = PROMPT_QUALITY_EVAL_TEMPLATE.format(
                query=query,
                rewritten_query=rewritten_query,
            ) + "{"
            
        try:
            output = Generator.generate_bedrock_response(
                bedrock_client, prompt, **gen_kwargs, model_id=model_id
            )
            #print(output)
            output = output.replace("'score'", '"score"')
            #result = json.loads(output)
            result = try_parse_json(output)
            score = result.get("score")
            return float(score)
        except Exception as e:
            print("Error computing Bedrock reward:", e)
            return -1

    @staticmethod
    def reward_rouge_func(response_model, prompts, completions, gen_kwargs, tokenizer=None, device=None, **kwargs):
        targets = kwargs.get("target")
        texts = kwargs.get("text")
        if targets is None:
            raise ValueError("Provide gold target answers via kwargs with key 'target'.")
        rewards = []
        for _, comp, tgt, text in zip(prompts, completions, targets, texts):
            try:
                parsed = json.loads("{" + comp)
                final_query = parsed.get("final_rewritten_query") + text
                final_answer = Generator.generate_final_answers(final_query, gen_kwargs, response_model, tokenizer, device)[0]
                reward = RewardEvaluator.compute_rouge_reward(final_answer, tgt)
            except Exception as e:
                print(e)
                reward = 0
            rewards.append(reward)
        return normalize_rewards(rewards, completions[0])

    @staticmethod
    def reward_llm_func(response_model, prompts, completions, gen_kwargs, tokenizer=None, device=None, client=None, **kwargs):
        targets = kwargs.get("target")
        texts = kwargs.get("text")
        originals = kwargs.get("original")
        if targets is None:
            raise ValueError("Provide gold target answers via kwargs with key 'target'.")
        rewards = []
        for prompt, comp, tgt, text, original in zip(prompts, completions, targets, texts, originals):
            try:
                parsed = json.loads("{" + comp)
                final_query = parsed.get("final_rewritten_query") + text
                final_answer = Generator.generate_final_answers(final_query, gen_kwargs, response_model, tokenizer, device)[0]
                reward = RewardEvaluator.compute_llm_reward(client, original + text, final_query, final_answer, tgt, gen_kwargs)
            except Exception as e:
                print(e)
                reward = 0
            rewards.append(reward)
        return normalize_rewards(rewards, completions[0])

    @staticmethod
    def compute_gsm8k_reward(final_answer, gold_answer, tolerance: float = 1e-3):
        """
        Compute the reward for a GSM8K example.
        Uses helper functions to extract and normalize the final answer and then
        compares it with the gold answer.
        Returns 1.0 if the numeric answer matches within tolerance, else 0.0.
        """
        # Here, final_answer is the raw model output.
        # gold_answer is expected to be a string containing the correct numeric answer.
        return evaluate_gsm8k_response(final_answer, gold_answer, tolerance)

    @staticmethod
    def composite_reward_func(response_model, prompts, completions, gen_kwargs,
                                tokenizer=None, device=None, client=None, judge_model=None,
                                parsing_weight=0.2, rouge_weight=0.3, gpt_judge_weight=0.5,
                                local_judge_weight=0, perplexity_weight=0, bedrock_weight=0,
                                dataset_name=None, output_file=None,
                                **kwargs):
        """
        Computes a weighted composite reward from multiple components.
        If dataset_name is "gsm8k", then only the GSM8K reward is computed for each sample.
        Otherwise, computes the composite reward as a weighted sum of:
          JSON parsing success, ROUGE score, LLM evaluations (GPT and local), perplexity, and Bedrock evaluation.
        Additional kwargs must include:
          - "text": list of extra text to append per sample,
          - "original": list of original prompt texts,
          - "target": list of target answers.
        Optionally, include a "bedrock_client" in kwargs if bedrock_weight > 0.
        """
        extra_texts = kwargs.get("text")
        originals = kwargs.get("original")
        targets = kwargs.get("target")
        if targets is None or extra_texts is None or originals is None:
            raise ValueError("Must provide 'target', 'text', and 'original' lists in kwargs.")
        composite_rewards = []
        bedrock_client = kwargs.get("bedrock_client")
        def effective_reward(reward, weight, rescale_factor=None):
            if reward == -1:
                return 0.0, 0.0
            if rescale_factor is not None and rescale_factor != 0:
                reward /= rescale_factor
            return min(max(reward, 0.0), 1.0), weight
        final_answers = []
        final_queries = []
        if isinstance(response_model, LLM):
            for prompt, comp, extra, original, tgt in zip(prompts, completions, extra_texts, originals, targets):    
                try:
                    comp = comp.replace("'final_rewritten_query'", '"final_rewritten_query"')
                    parsed = try_parse_json(comp)
                    final_query = (parsed.get("final_rewritten_query") or comp) + extra
                    json_success = 1.0 if parsed.get("final_rewritten_query") is not None else 0.0
                except Exception as e:
                    print("Composite JSON parse error:", e)
                    json_success = 0.0
                    final_query = comp + extra
                final_queries.append(final_query)
            final_answers = Generator.generate_final_answers(final_queries, gen_kwargs, response_model, tokenizer, device)#[0]
        i = -1
        for prompt, comp, extra, original, tgt in zip(prompts, completions, extra_texts, originals, targets):
            i += 1
            if not isinstance(response_model, LLM):
                try:
                    # Fixes for more robust JSON parsing
                    comp = comp.replace("'final_rewritten_query'", '"final_rewritten_query"')
                    parsed = try_parse_json(comp)
                    final_query = (parsed.get("final_rewritten_query") or comp) + extra
                    json_success = 1.0 if parsed.get("final_rewritten_query") is not None else 0.0
                except Exception as e:
                    print("Composite JSON parse error:", e)
                    json_success = 0.0
                    final_query = comp + extra
                
                final_answer = Generator.generate_final_answers(final_query, gen_kwargs, response_model, tokenizer, device)[0]
            else:
                final_query = final_queries[i]
                final_answer = final_answers[i]

            # If dataset_name is gsm8k, use only the GSM8K reward:
            if dataset_name == "gsm8k":
                gsm_reward = RewardEvaluator.compute_gsm8k_reward(final_answer, tgt[0])
                prompt_reward_gpt = RewardEvaluator.compute_prompt_llm_reward(client, original + extra, final_query, gen_kwargs)
                prompt_reward_llama = RewardEvaluator.compute_bedrock_reward(bedrock_client, original + extra, final_query, "","", gen_kwargs, prompt_eval=True)
                if rouge_weight > 0 and prompt_reward_gpt != -1 and prompt_reward_llama != 1:
                    composite = (gsm_reward / 2) + (prompt_reward_gpt / 20) + (prompt_reward_llama / 20)
                elif rouge_weight > 0 and prompt_reward_gpt != -1:
                    composite = (gsm_reward / 2) + (prompt_reward_gpt / 10)
                elif rouge_weight > 0 and prompt_reward_llama != -1:
                    composite = (gsm_reward / 2) + (prompt_reward_llama / 10)
                else:
                    composite = gsm_reward
                write_samples_to_csv(original + extra, final_query, final_answer, gsm_reward, (prompt_reward_gpt + prompt_reward_llama) / 2, filename="samples_gsm_final_base.csv")
                composite_rewards.append(composite)
                continue
            # Otherwise, compute the full composite reward.
            reward_json = json_success
            reward_rouge = (RewardEvaluator.compute_prompt_llm_reward(client, original + extra, final_query, gen_kwargs) + RewardEvaluator.compute_bedrock_reward(bedrock_client, original + extra, final_query, "","", gen_kwargs, prompt_eval=True, dataset_name=dataset_name)) / 2
            reward_llm_gpt = (RewardEvaluator.compute_llm_reward(client, original + extra, final_query, final_answer, tgt, gen_kwargs, dataset_name=dataset_name)
                              if gpt_judge_weight > 0 and client is not None else -1)
            reward_llm_local = (RewardEvaluator.compute_llm_reward_local(judge_model, original + extra, final_query, final_answer, tgt, gen_kwargs, tokenizer, device)
                                if local_judge_weight > 0 else -1)
            reward_perplexity = -1
            if perplexity_weight > 0 and isinstance(response_model, AutoModelForCausalLM):
                try:
                    if isinstance(comp, torch.Tensor):
                        preds = torch.argmax(comp, dim=-1)
                        loss = torch.nn.functional.cross_entropy(comp, preds, reduction='mean')
                        perplexity = torch.exp(loss)
                        reward_perplexity = 1.0 / perplexity.item()
                except Exception as e:
                    print("Perplexity reward error:", e)
            reward_bedrock = -1
            if bedrock_weight > 0:
                bedrock_client = kwargs.get("bedrock_client")
                if bedrock_client is not None:
                    try:
                        reward_bedrock = RewardEvaluator.compute_bedrock_reward(
                            bedrock_client, original + extra, final_query, final_answer, tgt, gen_kwargs, dataset_name=dataset_name)
                    except Exception as e:
                        print("Bedrock reward error:", e)
            rescale_factor = None
            if dataset_name in ["scitldr", "cnndm"]:
                rescale_factor = 5
            eff_json, w_json = effective_reward(reward_json, parsing_weight)
            eff_rouge, w_rouge = effective_reward(reward_rouge, rouge_weight, rescale_factor=5)
            eff_llm_gpt, w_llm_gpt = effective_reward(reward_llm_gpt, gpt_judge_weight, rescale_factor=rescale_factor)
            eff_llm_local, w_llm_local = effective_reward(reward_llm_local, local_judge_weight, rescale_factor=rescale_factor)
            eff_perp, w_perp = effective_reward(reward_perplexity, perplexity_weight)
            eff_bedrock, w_bedrock = effective_reward(reward_bedrock, bedrock_weight, rescale_factor=rescale_factor) if bedrock_weight > 0 else (0.0, 0.0)

            total_weight = w_json + w_rouge + w_llm_gpt + w_llm_local + w_perp + w_bedrock
            composite = (w_json * eff_json + w_rouge * eff_rouge + w_llm_gpt * eff_llm_gpt +
                         w_llm_local * eff_llm_local + w_perp * eff_perp + w_bedrock * eff_bedrock) / total_weight if total_weight else 0.0
            write_samples_to_csv(original + extra, final_query, final_answer, w_llm_gpt * eff_llm_gpt + w_perp * eff_perp + w_bedrock * eff_bedrock, w_rouge * eff_rouge, filename="samples_sum2.csv")
            if composite > 0.99:
                print(final_query)
                print(final_answer)
            composite_rewards.append(composite)
        return composite_rewards

    @staticmethod
    def compute_reward(reward_type, final_answer, target_answers, prompt="", rewritten="", client=None, gen_kwargs=None, dataset_name="nq"):
        if reward_type == "rouge":
            return RewardEvaluator.compute_rouge_reward(final_answer, target_answers)
        elif reward_type == "em":
            return compute_exact(final_answer, target_answers)
        elif reward_type == "gsm8k":
            return RewardEvaluator.compute_gsm8k_reward(final_answer, target_answers)
        elif reward_type == "llm_prompt":
            return RewardEvaluator.compute_prompt_llm_reward(client, prompt, rewritten, gen_kwargs)
        elif reward_type == "llm":
            return RewardEvaluator.compute_llm_reward(client, prompt, rewritten, final_answer, target_answers, gen_kwargs, dataset_name=dataset_name)
        elif reward_type == "llm_test":
            return RewardEvaluator.compute_llm_reward_test(client, prompt, final_answer, target_answers, gen_kwargs)
        else:
            return None