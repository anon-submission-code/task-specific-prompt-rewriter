import argparse
import pandas as pd
import random
import json
from helper_files.helpers import ClientFactory, Generator

# Pairwise evaluation prompt template with full examples
EVAL_PROMPT_TEMPLATE_PAIR = """
You are an AI assistant specialized in judging which of two candidate summaries better captures the key information from a reference summary. You will be given:

[ORIGINAL]: The original text to be summarized.
[OPTION 1]: The first candidate summary.
[OPTION 2]: The second candidate summary.
[REFERENCE]: The ground truth summary.

Criteria:
- Coverage: which option includes more of the main points from the reference?
- Hallucination: which option adds fewer unsupported details?
- Relevance: which option avoids irrelevant content?

Output:
Return a JSON object with keys:
- "selection": 1 or 2 (1 if OPTION 1 is better, 2 if OPTION 2 is better)
- "explanation": a brief analysis of your choice.

Examples:

1.
[ORIGINAL]
“In reinforcement learning, it is common to let an agent interact with its environment for a fixed amount of time before resetting the environment… Our results show that the proposed methods improve the performance and stability of existing reinforcement learning algorithms.”
[REFERENCE]
“We consider the problem of learning optimal policies in time-limited and time-unlimited domains using time-limited interactions.”
[OPTION 1]
“We study optimal policy learning under both fixed-time and unbounded episodes by incorporating remaining time as input to the agent.”
[OPTION 2]
“This paper studies the effect of time limits on agent resets and proposes a reward-shaping technique to stabilize performance over fixed and indefinite episodes.”
Evaluation:
{"selection": 1, "explanation": "Option 1 accurately restates the dual setting and use of remaining time, matching the reference. Option 2 introduces a reward-shaping technique not mentioned in the reference."}

2.
[ORIGINAL]
“The field of Deep Reinforcement Learning (DRL) has recently seen a surge in the popularity of maximum entropy reinforcement learning algorithms… We further show that the streamlined algorithm with the simple non-uniform sampling scheme outperforms SAC and achieves state-of-the-art performance on challenging continuous control tasks.”
[REFERENCE]
“We propose a new DRL off-policy algorithm achieving state-of-the-art performance.”
[OPTION 1]
“We propose a new DRL off-policy algorithm achieving state-of-the-art performance.”
[OPTION 2]
“This work focuses on designing a curriculum learning strategy for exploration in DRL.”
Evaluation:
{"selection": 1, "explanation": "Option 1 directly matches the reference’s contribution and outcome. Option 2 misses the off-policy algorithm and invents a curriculum learning focus."}

3.
[ORIGINAL]
“Argentina coach Alejandro Sabella believes Lionel Messi’s habit of throwing up during games is because of nerves. The Barcelona star has vomited on the pitch during several games…”
[REFERENCE]
“Argentina coach Sabella believes Messi’s habit of being sick during games is down to nerves.”
[OPTION 1]
“Messi vomits during games because of nerves.”
[OPTION 2]
“Sabella says Messi’s sickness during matches is caused by pre-game anxiety and stress.”
Evaluation:
{"selection": 2, "explanation": "Option 2 retains the coach’s attribution and aligns with ‘nerves’ as anxiety and stress. Option 1 omits the attribution nuance."}

4.
[ORIGINAL]
“Beijing, China has blocked the popular video-sharing web site YouTube, but did not offer a reason for the ban. YouTube was blocked in China as of Wednesday.”
[REFERENCE]
“By early Wednesday, users inside China encountered: ‘network timeout.’”
[OPTION 1]
“China blocked YouTube on Wednesday, leading to network timeouts for users.”
[OPTION 2]
“Users in China experienced network timeouts on Wednesday when YouTube was blocked.”
Evaluation:
{"selection": 2, "explanation": "Option 2 mirrors the reference’s observational focus on user experience. Option 1 adds a causal framing (‘leading to’) not present in the reference."}
"""

class Evaluator:
    @staticmethod
    def compute_selection(client, original, reference, opt1, opt2, gen_kwargs):
        # Randomize order to avoid bias
        if random.random() < 0.5:
            mapping = {1: 1, 2: 2}
            opts = {"OPTION 1": opt1, "OPTION 2": opt2}
        else:
            mapping = {1: 2, 2: 1}
            opts = {"OPTION 1": opt2, "OPTION 2": opt1}
        prompt = EVAL_PROMPT_TEMPLATE_PAIR.format(
            ORIGINAL=original,
            REFERENCE=reference,
            **opts
        )
        output = Generator.generate_api_response(client, prompt, **gen_kwargs)
        result = json.loads(output)
        sel = int(result["selection"])
        return mapping[sel], result.get("explanation", "")

def compute_win_rate(selections):
    wins2 = sum(1 for sel, _ in selections if sel == 2)
    return wins2 / len(selections) if selections else 0.0

def main():
    parser = argparse.ArgumentParser(description="Pairwise summary evaluation between two CSVs")
    parser.add_argument("--file_a", type=str, required=True,
                        help="CSV A with columns ['Original','Final Answer']")
    parser.add_argument("--file_b", type=str, required=True,
                        help="CSV B with columns ['Original','Final Answer']")
    parser.add_argument("--response_model", type=str, default="GPT",
                        help="Response model: GPT or BEDROCK or HF model ID")
    args = parser.parse_args()

    # Initialize client
    if args.response_model.upper() == "GPT":
        client = ClientFactory.create_api_client()
    elif args.response_model.upper() == "BEDROCK":
        client = ClientFactory.create_bedrock_client()

    gen_kwargs = {"max_new_tokens": 512, "do_sample": False}

    # Load CSVs
    df_a = pd.read_csv(args.file_a).rename(columns={"Final Answer": "Option1"})
    df_b = pd.read_csv(args.file_b).rename(columns={"Final Answer": "Option2"})
    # Merge on 'Original'
    df = pd.merge(df_a[["Original", "Option1"]], df_b[["Original", "Option2"]], on="Original")

    selections = []
    for _, row in df.iterrows():
        original = row["Original"]
        opt1 = row["Option1"]
        opt2 = row["Option2"]
        # Ground truth reference must be included in each merged row (if provided as column)
        # Here we assume reference is also in df_a or df_b if available; otherwise adjust accordingly.
        reference = row.get("Reference", "")  # adapt if your CSVs include reference
        sel, explanation = Evaluator.compute_selection(
            client, original, reference, opt1, opt2, gen_kwargs
        )
        selections.append((sel, explanation))
        print(json.dumps({
            "Original": original,
            "selection": sel,
            "explanation": explanation
        }, ensure_ascii=False))

    win_rate = compute_win_rate(selections)
    print(f"\nOption 2 win rate: {win_rate*100:.2f}%")

if __name__ == "__main__":
    main()