import torch
import pickle
import os
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification
from src.config import Config


class UnknownEntityDetector:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        self.model = AutoModelForTokenClassification.from_pretrained(Config.OUTPUT_DIR)
        self.model.to(self.device)
        self.model.eval()

        stats_path = os.path.join(Config.STATS_DIR, "stats.pkl")
        if not os.path.exists(stats_path):
            raise FileNotFoundError("Stats file not found. Run --stats first.")
        with open(stats_path, "rb") as f:
            self.stats = pickle.load(f)

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        predictions = torch.argmax(outputs.logits, dim=2)
        last_hidden_state = outputs.hidden_states[-1]
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        results = []

        print(f"\nScanning: '{text}'")
        print(f"{'TOKEN':<12} | {'MODEL PREDICTION':<18} | {'VERDICT'}")
        print("-" * 60)

        for i, token in enumerate(tokens):
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue

            pred_id = predictions[0][i].item()
            pred_label = Config.ID2LABEL[pred_id]

            # --- LOGIC BRANCHING ---

            # BRANCH A: Model thinks it is 'O' (Nothing)
            if pred_id == 0:
                # Heuristic: If it's capitalized (and not a subword ##), it might be a hidden entity!
                if token[0].isupper() and not token.startswith("##"):
                    print(f"{token:<12} | {pred_label:<18} | UNKNOWN (Suspected - Capitalized 'O')")
                else:
                    # It's truly nothing (like 'is', 'the', 'founded')
                    pass
                continue

            # BRANCH B: Model predicts an Entity, but we have no stats for it
            if pred_id not in self.stats:
                print(f"{token:<12} | {pred_label:<18} | SKIPPED (No stats for this label)")
                continue

            # BRANCH C: Model predicts Entity -> We check Mahalanobis Distance
            vector = last_hidden_state[0][i].cpu().numpy().reshape(1, -1)
            cov_model = self.stats[pred_id]["covariance_model"]
            dist = np.sqrt(cov_model.mahalanobis(vector)[0])

            verdict = "ACCEPTED"
            if dist > Config.MAHALANOBIS_THRESHOLD:
                verdict = "UNKNOWN / OOD"

            print(f"{token:<12} | {pred_label:<18} | {verdict} (Dist: {dist:.1f})")

        return results