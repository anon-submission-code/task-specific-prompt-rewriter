import json
from dotenv import load_dotenv
from data_helpers import load_and_preprocess_dataset  # Uses the functions from your data_helpers file

def create_samples_json(dataset_name="cnn_dailymail", split="train", samples_amount=1000, output_file="samples.json"):
    """
    Loads and preprocesses a dataset (using logic from data_helpers) and creates a samples.json file.
    Each sample in the JSON will contain:
      - "original_query": The original instruction (e.g. "Summarize the text" or "Answer the question")
      - "reference_answer": The gold answer (e.g. the summary or short answer)
    
    Only samples with non-empty targets are saved.
    """
    # Load and preprocess the dataset using your existing data_helpers logic.
    processed = load_and_preprocess_dataset(dataset_name, split, samples_amount)
    
    # Filter and convert samples to the desired format.
    samples = []
    for sample in processed:
        # Expecting sample to have keys: "original", "target", and possibly others ("prompt", "text")
        target_list = sample.get("target", [])
        original = sample.get("original", "").strip()
        text = sample.get("text", "").strip()
        if target_list and original and text:
            # Use the first target as the reference answer.
            reference_answer = target_list[0].strip()
            if reference_answer:
                samples.append({
                    "prompt": sample.get("prompt", "").strip(),
                    "original": original,
                    "text": text,
                    "original_query": original + text,
                    "reference_answer": reference_answer
                })
    
    # Save the resulting samples to a JSON file.
    with open(output_file, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"Saved {len(samples)} samples to {output_file}")

if __name__ == "__main__":
    load_dotenv("./.env")
    # Adjust dataset_name, split, and samples_amount as needed.
    create_samples_json(dataset_name="hotpot", split="train", samples_amount=5000, output_file="samples_hotpot.json")
