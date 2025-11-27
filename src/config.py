import os


class Config:
    # Paths
    MODEL_NAME = "bert-base-cased"
    OUTPUT_DIR = "saved_models/bert_finetuned"
    STATS_DIR = "saved_models/ood_stats"

    # Speed Settings
    BATCH_SIZE = 8  # Smaller batch size for CPU
    EPOCHS = 1  # Only train once (Fast mode)
    LEARNING_RATE = 2e-5
    MAX_LEN = 64  # Shorter sentences (Drastically faster)

    # Labels
    LABEL_LIST = [
        "O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC",
    ]

    ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}
    LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}

    MAHALANOBIS_THRESHOLD = 3500.0