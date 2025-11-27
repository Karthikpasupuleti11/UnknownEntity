import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import load_dataset
from src.config import Config
import pickle
import os


def evaluate_system():
    print(">>> Loading System for Evaluation...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Model
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(Config.OUTPUT_DIR)
    model.to(device)
    model.eval()

    # Load Stats
    with open(os.path.join(Config.STATS_DIR, "stats.pkl"), "rb") as f:
        stats = pickle.load(f)

    # Load TEST Data (Which contains the hidden ORGs)
    print(">>> Loading Test Data...")
    dataset = load_dataset("lhoestq/conll2003", split="test")

    # Counters
    metrics = {
        "known_total": 0, "known_correct": 0,  # For PER, LOC, MISC
        "org_total": 0, "org_caught": 0  # For ORG (The Hidden Class)
    }

    print(">>> Running Evaluation on Test Set...")
    # We will look at the first 1000 sentences to save time
    for i in tqdm(range(1000)):
        example = dataset[i]
        text = " ".join(example['tokens'])
        original_tags = example['ner_tags']  # These are the REAL answers

        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        predictions = torch.argmax(outputs.logits, dim=2)
        last_hidden_state = outputs.hidden_states[-1]

        # align tokens
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        word_ids = inputs.word_ids()

        for idx, word_id in enumerate(word_ids):
            if word_id is None: continue  # Skip special tokens

            # Get the True Label from the dataset
            if word_id < len(original_tags):
                true_label_id = original_tags[word_id]
                true_label = Config.ID2LABEL.get(true_label_id, "O")
            else:
                continue

            # Get Model Prediction
            pred_id = predictions[0][idx].item()

            # --- LOGIC ---

            # 1. Check if it's an Organization (The "Unknown" Test)
            if "ORG" in true_label:
                metrics["org_total"] += 1

                # Did we catch it?
                # Caught if:
                # A) Model predicted 'O' and we flagged it as Suspicious (Capitalized)
                # B) Model predicted PER/LOC but Distance > Threshold

                caught = False
                token_str = tokens[idx]

                if pred_id == 0:  # Model said 'O'
                    if token_str[0].isupper() and not token_str.startswith("##"):
                        caught = True  # We suspected it!
                elif pred_id in stats:  # Model guessed PER/LOC/MISC
                    vector = last_hidden_state[0][idx].cpu().numpy().reshape(1, -1)
                    cov = stats[pred_id]["covariance_model"]
                    dist = np.sqrt(cov.mahalanobis(vector)[0])
                    if dist > Config.MAHALANOBIS_THRESHOLD:
                        caught = True  # We rejected it based on distance!

                if caught:
                    metrics["org_caught"] += 1

            # 2. Check Known Classes (PER, LOC, MISC)
            elif true_label != "O":
                metrics["known_total"] += 1
                if pred_id == true_label_id:
                    # Check if we accidentally rejected it?
                    if pred_id in stats:
                        vector = last_hidden_state[0][idx].cpu().numpy().reshape(1, -1)
                        cov = stats[pred_id]["covariance_model"]
                        dist = np.sqrt(cov.mahalanobis(vector)[0])
                        if dist <= Config.MAHALANOBIS_THRESHOLD:
                            metrics["known_correct"] += 1

    # --- PRINT REPORT ---
    print("\n" + "=" * 40)
    print("FINAL EVALUATION REPORT")
    print("=" * 40)

    known_acc = (metrics['known_correct'] / metrics['known_total']) * 100
    print(f"1. Known Entity Accuracy (PER/LOC/MISC): {known_acc:.2f}%")
    print(f"   (Ideally high: We shouldn't break normal detection)")

    if metrics['org_total'] > 0:
        ood_rate = (metrics['org_caught'] / metrics['org_total']) * 100
        print(f"2. Unknown Detection Rate (Hidden ORGs): {ood_rate:.2f}%")
        print(f"   (This is your 'Success Rate' for the project)")
    else:
        print("2. No Organizations found in the sample.")


if __name__ == "__main__":
    evaluate_system()