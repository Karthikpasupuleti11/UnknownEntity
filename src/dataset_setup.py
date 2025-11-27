from datasets import load_dataset
from transformers import AutoTokenizer
from src.config import Config


def filter_orgs(example):
    """
    Turns ORG labels (3 and 4) into O (0) for the Training set.
    This effectively 'blinds' the model to Organizations.
    """
    new_tags = []
    for tag in example['ner_tags']:
        if tag == 3 or tag == 4:  # B-ORG or I-ORG
            new_tags.append(0)  # Become O
        else:
            new_tags.append(tag)
    example['ner_tags'] = new_tags
    return example


def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=Config.MAX_LEN
    )

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Ignore special tokens
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])  # Label for first token of word
            else:
                label_ids.append(-100)  # Ignore subsequent subwords
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def get_lobotomized_dataset():
    print("Downloading CoNLL-2003 (Parquet version)...")
    # FIX: Use 'lhoestq/conll2003' to avoid the 'Dataset scripts not supported' error
    dataset = load_dataset("lhoestq/conll2003")

    # Apply the Lobotomy (Filter ORGs from Train only)
    print("Filtering Organizations from Training Set...")
    dataset['train'] = dataset['train'].map(filter_orgs)

    # Tokenization
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    tokenized_datasets = dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        batched=True
    )

    return tokenized_datasets