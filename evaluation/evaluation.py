import os
from helper_files.data_helpers import create_meta_prompt
from helper_files.utils import compute_metrics, write_results_to_csv, try_parse_json, evaluate_gsm8k_response, write_samples_to_csv

# Import new helper classes for API creation, generation, and reward evaluation.
from helper_files.helpers import ClientFactory, Generator, RewardEvaluator

def evaluate_gt(response_model, data, generation_kwargs, tokenizer, device, instruction, name="GRPO_Rouge_3B_4o-mini"):
    """
    Evaluates the response model using ground truth.
    """
    total_acc, total_f1, total_rouge, total_llm, total = 0, 0, 0, 0, 0
    generation_kwargs['do_sample'] = False
    generation_kwargs['temperature'] = 0.0
    judge_client = ClientFactory.create_api_client()
    for batch in data:
        target_answers = batch["target"]
        prompts = [instruction + " " + sample for sample in batch["text"]]
        final_answers = Generator.generate_final_answers(prompts, generation_kwargs, response_model, tokenizer, device)
        for final_answer, targets, prompt in zip(final_answers, target_answers, prompts):
            if final_answer == "":
                continue
            em, f1 = compute_metrics(final_answer, targets)
            rouge_val = RewardEvaluator.compute_reward("rouge", final_answer, targets, prompt=prompt, gen_kwargs=generation_kwargs)
            llm_val = RewardEvaluator.compute_reward("llm_test", final_answer, targets, prompt=prompt, client=judge_client, gen_kwargs=generation_kwargs)
            if rouge_val == -1 or llm_val == -1:
                print("error")
                continue
            total_acc += em
            total_f1 += f1
            total_rouge += rouge_val
            total_llm += llm_val
            total += 1
    if total > 0:
        avg_acc = total_acc / total
        avg_f1 = total_f1 / total
        avg_rouge = total_rouge / total
        avg_llm = total_llm / total
        print(total)
        print(f"Evaluation - Accuracy: {avg_acc:.4f}, F1: {avg_f1:.4f}, ROUGE: {avg_rouge:.4f}, LLM: {avg_llm:.4f}")
        write_results_to_csv(name, avg_acc, avg_f1, avg_rouge, avg_llm)
    else:
        print("No samples evaluated.")

def evaluate_response_model(response_model, data, generation_kwargs, tokenizer, device, instruction, name="GRPO_Rouge_3B_4o-mini", dataset_name="nq"):
    """
    Evaluates the response model.
    """
    total_acc, total_f1, total_rouge, total_llm, total = 0, 0, 0, 0, 0
    generation_kwargs['do_sample'] = False
    generation_kwargs['temperature'] = 0.0
    judge_client = ClientFactory.create_api_client()
    instruction = "Provide a detailed explanation of the process you used to answer the question, including any relevant examples or illustrations."
    for batch in data:
        target_answers = batch["target"]
        prompts = [instruction + " " + sample for sample in batch["text"]]
        final_answers = Generator.generate_final_answers(prompts, generation_kwargs, response_model, tokenizer, device)
        for final_answer, targets, prompt in zip(final_answers, target_answers, prompts):
            if final_answer == "":
                continue
            em, f1 = compute_metrics(final_answer, targets)
            if dataset_name == "gsm8k":
                gsm_reward = evaluate_gsm8k_response(final_answer, targets[0])
                rouge_val = gsm_reward
                llm_val = gsm_reward
            else:
                rouge_val = RewardEvaluator.compute_reward("rouge", final_answer, targets, prompt=prompt, gen_kwargs=generation_kwargs)
                llm_val = RewardEvaluator.compute_reward("llm", final_answer, targets, prompt=prompt, client=judge_client, gen_kwargs=generation_kwargs, dataset_name=dataset_name)
            if rouge_val == -1 or llm_val == -1:
                print("error")
                continue
            total_acc += em
            total_f1 += f1
            total_rouge += rouge_val
            total_llm += llm_val
            total += 1
    if total > 0:
        avg_acc = total_acc / total
        avg_f1 = total_f1 / total
        avg_rouge = total_rouge / total
        avg_llm = total_llm / total
        print(total)
        print(f"Evaluation - Accuracy: {avg_acc:.4f}, F1: {avg_f1:.4f}, ROUGE: {avg_rouge:.4f}, LLM: {avg_llm:.4f}")
        write_results_to_csv(name, avg_acc, avg_f1, avg_rouge, avg_llm)
    else:
        print("No samples evaluated.")

def evaluate_model(rewriting_model, response_model, data, generation_kwargs, tokenizer, device, json_format=True, name="GRPO_Rouge_3B_4o-mini", instruction=None, dataset_name="nq", use_vllm=True, vllm_path=""):
    """
    Evaluates the response model by rewriting prompts (if needed) and then generating final answers.
    """
    total_acc, total_f1, total_rouge, total_llm, total, total_prompts, correct_json_count = 0, 0, 0, 0, 0, 0, 0
    generation_kwargs['do_sample'] = False
    generation_kwargs['temperature'] = 0.0
    judge_client = ClientFactory.create_api_client()
    new_prompt = ""
    print(instruction)
    if instruction is not None:
        prompt = create_meta_prompt(text="", instruction=instruction)
        new_prompt = Generator.rewrite_prompts(rewriting_model, tokenizer, [prompt], generation_kwargs, device, use_vllm=True, vllm_path=vllm_path)[0][0]
        if json_format:
            try:
                parsed = try_parse_json(new_prompt)
                final_rewritten = parsed["final_rewritten_query"]
                if final_rewritten is not None:
                    new_prompt = final_rewritten
            except Exception as e:
                print("Batch JSON parse error:", e)
        print(new_prompt)

    for batch in data:
        if instruction is None:
            prompts = []
            # Rewrite the prompts from the batch
            rewritten_texts, _, _ = Generator.rewrite_prompts(rewriting_model, tokenizer, batch["prompt"], generation_kwargs, device, use_vllm=True, vllm_path=vllm_path)
            if json_format:
                for original, rewritten_text in zip(batch["text"], rewritten_texts):
                    total_prompts += 1
                    try:
                        # Try to parse the rewritten text assuming it's JSON formatted.
                        parsed = try_parse_json(rewritten_text)
                        #print(parsed)
                        final_rewritten = parsed.get("final_rewritten_query")
                        if final_rewritten is not None:
                            prompts.append(
                                "This was the original query: " + original +
                                "\nFollow this new instruction, only use information from the original query if any important details are missing from the new instruction: " + final_rewritten
                            )
                            correct_json_count += 1
                        else:
                            # In case the JSON does not return a final rewritten query, use the rewritten text directly.
                            prompts.append(rewritten_text)
                    except Exception as e:
                        print("Batch JSON parse error:", e)
                        print("Rewritten text:", rewritten_text)
                        prompts.append(
                            "This was the original query: " + original +
                            "\nFollow this new instruction, only use information from the original query if any important details are missing from the new instruction: " + rewritten_text
                        )
        else:
            prompts = [new_prompt + " " + sample for sample in batch["text"]]
            #print(prompts)
        target_answers = batch["target"]
        texts = batch["text"]
        originals = batch["original"]
        final_answers = Generator.generate_final_answers(prompts, generation_kwargs, response_model, tokenizer, device, use_vllm=True, vllm_path=vllm_path)
        for final_answer, targets, text, original in zip(final_answers, target_answers, texts, originals):
            if final_answer == "":
                continue
            em, f1 = compute_metrics(final_answer, targets)
            if dataset_name == "gsm8k":
                gsm_reward = evaluate_gsm8k_response(final_answer, targets[0])
                rouge_val = gsm_reward
                llm_val = gsm_reward
            elif instruction is not None:
                rouge_val = RewardEvaluator.compute_reward("rouge", final_answer, targets, prompt=instruction + text, gen_kwargs=generation_kwargs)
                llm_val = RewardEvaluator.compute_reward("llm", final_answer, targets, prompt=instruction + text, client=judge_client, gen_kwargs=generation_kwargs, dataset_name=dataset_name)
            else:
                rouge_val = RewardEvaluator.compute_reward("rouge", final_answer, targets, prompt=original + text, gen_kwargs=generation_kwargs)
                llm_val = RewardEvaluator.compute_reward("llm", final_answer, targets, prompt=original + text, client=judge_client, gen_kwargs=generation_kwargs, dataset_name=dataset_name)
            if rouge_val == -1 or llm_val == -1:
                print("Error in reward calculations")
                continue
            total_acc += em
            total_f1 += f1
            total_rouge += rouge_val
            total_llm += llm_val
            try:
                write_samples_to_csv(original, new_prompt + text, final_answer, rouge_val, llm_val, filename="eval_samples_" + name + ".csv")
            except Exception as e:
                print(e)
            total += 1
    if total > 0:
        avg_acc = total_acc / total
        avg_f1 = total_f1 / total
        avg_rouge = total_rouge / total
        avg_llm = total_llm / total
        print(total)
        print(f"Evaluation - Accuracy: {avg_acc:.4f}, F1: {avg_f1:.4f}, ROUGE: {avg_rouge:.4f}, LLM: {avg_llm:.4f}")
        print(total_prompts)
        print(correct_json_count)
        write_results_to_csv(name + ": " + new_prompt, avg_acc, avg_f1, avg_rouge, avg_llm)
    else:
        print("No samples evaluated.")

def save_model(model, tokenizer, name, output_dir="./ppo_output"):
    """
    Saves the model and tokenizer to the specified output directory.
    """
    os.makedirs(f"{output_dir}/{name}", exist_ok=True)
    model.save_pretrained(f"{output_dir}/{name}")
    tokenizer.save_pretrained(f"{output_dir}/{name}")
    print(f"Model and tokenizer saved to {output_dir}/{name}")
