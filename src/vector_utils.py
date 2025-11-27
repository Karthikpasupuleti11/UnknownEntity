import torch
import numpy as np
import os
import pickle
from tqdm import tqdm
from sklearn.covariance import EmpiricalCovariance
from transformers import AutoModelForTokenClassification, AutoTokenizer
from src.dataset_setup import get_lobotomized_dataset
from src.config import Config


def calculate_and_save_stats():
    print("Loading fine-tuned model...")
    model = AutoModelForTokenClassification.from_pretrained(Config.OUTPUT_DIR)
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    model.eval()

    dataset = get_lobotomized_dataset()
    train_data = dataset['train']

    # FIX 1: Strict check. Only exclude if the label is EXACTLY "O"
    embeddings_by_class = {label_id: [] for label_id in Config.ID2LABEL if Config.ID2LABEL[label_id] != "O"}

    print("Extracting embeddings from training data...")
    subset_size = 10000

    with torch.no_grad():
        for i in tqdm(range(min(len(train_data), subset_size))):
            example = train_data[i]
            input_ids = torch.tensor([example['input_ids']])
            attention_mask = torch.tensor([example['attention_mask']])
            labels = example['labels']

            outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1].squeeze(0)

            for token_idx, label_id in enumerate(labels):
                # FIX 2: Ensure we don't crash on padding (-100) or O (0)
                if label_id != -100 and label_id != 0:
                    if label_id in embeddings_by_class:
                        vector = last_hidden_state[token_idx].numpy()
                        embeddings_by_class[label_id].append(vector)

    stats = {}
    print("Fitting Covariance Models...")
    for label_id, vectors in embeddings_by_class.items():
        if len(vectors) < 5:
            print(f"Skipping class {Config.ID2LABEL[label_id]} - Not enough data.")
            continue
        matrix = np.stack(vectors)
        cov_model = EmpiricalCovariance().fit(matrix)
        stats[label_id] = {"mean": cov_model.location_, "covariance_model": cov_model}
        print(f"Class {Config.ID2LABEL[label_id]}: Fitted on {len(vectors)} examples.")

    os.makedirs(Config.STATS_DIR, exist_ok=True)
    with open(os.path.join(Config.STATS_DIR, "stats.pkl"), "wb") as f:
        pickle.dump(stats, f)
    print("Stats saved successfully.")