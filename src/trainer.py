from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, \
    DataCollatorForTokenClassification, AutoTokenizer
from src.config import Config


def train_model(tokenized_dataset):
    # 1. Load Tokenizer (FIX: We need this for the Data Collator)
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # 2. Load Model
    model = AutoModelForTokenClassification.from_pretrained(
        Config.MODEL_NAME,
        num_labels=len(Config.LABEL_LIST),
        id2label=Config.ID2LABEL,
        label2id=Config.LABEL2ID
    )

    # 3. Setup Training Arguments
    args = TrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        eval_strategy="epoch",
        learning_rate=Config.LEARNING_RATE,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        num_train_epochs=Config.EPOCHS,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_dir='./logs',
    )

    # 4. Setup Data Collator (FIX: Passed tokenizer here)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
    )

    print("Starting Training...")
    trainer.train()

    print(f"Saving model to {Config.OUTPUT_DIR}")
    trainer.save_model(Config.OUTPUT_DIR)
    tokenizer.save_pretrained(Config.OUTPUT_DIR)  # Save tokenizer too for safety