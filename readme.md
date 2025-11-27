# 🕵️ Unknown Entity Detection using BERT & Mahalanobis Distance

> A Post‑Hoc Out‑of‑Distribution (OOD) Detection System for Named Entity Recognition (NER).

This repository implements an uncertainty‑aware NER pipeline that detects "Unknown" or "Novel" entities (entities not seen during training) by combining a fine‑tuned BERT token classifier with a statistical Mahalanobis distance check over token embeddings.

---

## 🚀 Project Overview

Standard NER models are closed‑set: they only recognize classes seen at training time (e.g., PER, LOC). When presented with a novel entity type (for example, ORG when the model was never trained on it), the model tends to mislabel or ignore it.

This project implements a post‑hoc guard that:
- Extracts contextual token embeddings from a fine‑tuned BERT model,
- Builds per‑class statistical fingerprints (centroid + covariance),
- Uses Mahalanobis distance at inference to flag tokens whose embeddings lie far outside the class "safe zone" as OOD / Unknown,
- Adds lightweight heuristics (capitalized proper‑noun detection) to catch suspicious tokens predicted as background (`O`).

---

## 🧠 The Core Experiment: "The Lobotomy"

1. The "Lobotomy": We remove all Organization (ORG) labels from the CoNLL‑2003 training set.
2. Training: Fine‑tune `bert-base-cased` for token classification on the restricted labels: PER, LOC, MISC (ORG is unseen).
3. Test: Evaluate on data that still contains ORG tokens (e.g., "SpaceX", "Google").
4. Goal: The system should flag these ORG tokens as UNKNOWN / Out‑of‑Distribution instead of incorrectly assigning them to a known class or leaving them as background.

---

## 🛠️ Architecture & Flow

1. Base model: `bert-base-cased` fine‑tuned for token classification on PER, LOC, MISC.
2. Vector extraction: For each token at inference, extract the last hidden layer embedding (768 dims).
3. Offline statistical profiling:
   - Compute mean (centroid) and covariance matrix per known class (PER, LOC, MISC).
   - Store these fingerprints (e.g., as .pkl files under saved_models/).
4. Inference Guard:
   - Mahalanobis distance of token embedding to predicted class centroid. If distance > threshold → label as UNKNOWN (OOD).
   - Heuristic: If predicted as `O` but token is a capitalized proper noun → mark as Suspected Unknown.

---

## 📂 Directory Structure

```text
UnknownEntityProject/
│
├── src/
│   ├── dataset_setup.py   # Downloads CoNLL-2003 & removes 'ORG' tags (The Lobotomy)
│   ├── trainer.py         # Fine-tunes BERT on the restricted dataset
│   ├── vector_utils.py    # Extracts embeddings & computes Class Centroids/Covariance
│   ├── detector.py        # The Guard: Logic for Distance & Heuristic checks
│   ├── evaluate.py        # Runs accuracy metrics on the Test Set
│   └── config.py          # Global settings (Thresholds, Hyperparameters)
│
├── saved_models/          # Trained BERT model & statistical fingerprints (.pkl)
├── notebooks/             # Jupyter notebooks for visualization and analysis
├── requirements.txt       # Project dependencies
└── main.py                # Main CLI entry point
```

---

## 💻 Installation & Setup

Clone the repository:

```bash
git clone https://github.com/Karthikpasupuleti11/UnknownEntity.git
cd UnknownEntity
```

(Recommended) Create a virtual environment:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Notes:
- For full training and faster runs, a GPU is highly recommended.
- The project uses Hugging Face datasets & transformers to download CoNLL‑2003 and BERT.

---

## ⚡ Usage Guide

Phase 1: Train the model (downloads data, removes ORG, fine‑tunes BERT):

```bash
python main.py --train
```

Phase 2: Compute fingerprints (centroids & covariances for each known class):

```bash
python main.py --stats
# or
python -m src.vector_utils    # depending on the CLI wrappers provided
```

Phase 3: Real‑time prediction / demo:

```bash
python main.py --predict "Elon Musk founded SpaceX in California."
```

Expected output (example):

```
Elon Musk: PER (Accepted)
California: LOC (Accepted)
SpaceX: UNKNOWN (Suspected)  # Successfully detected as OOD
```

Phase 4: Full evaluation (run the evaluation script against CoNLL‑2003 test set):

```bash
python -m src.evaluate
```

---

## 📊 Example Results (1 Epoch, CPU)

- Known Entity Accuracy: 62.58%  
  (How accurate the model is on PER, LOC, MISC labels)
- Unknown Detection Rate: 36.92%  
  (Percent of hidden ORG tokens successfully flagged as Unknown)

These are baseline results intended to demonstrate the pipeline. Expect much better performance after multiple epochs and GPU training.

---

## 🔍 Key Implementation Files (high level)

- src/dataset_setup.py — download CoNLL‑2003, remove ORG labels from training set, produce restricted dataset.
- src/trainer.py — BERT fine‑tuning code (token classification head).
- src/vector_utils.py — embedding extraction, centroid and covariance computation, serialization to disk.
- src/detector.py — Mahalanobis distance check + heuristic rules applied at inference time.
- src/evaluate.py — accuracy, OOD detection metrics, confusion matrices.
- src/config.py — hyperparameters, thresholds, file paths.

---

## 🔮 Future Work

- Contrastive learning (Supervised Contrastive Loss) to better separate classes in embedding space.
- Energy‑based scoring as an alternative to Mahalanobis distance.
- Active learning loop: present flagged Unknowns to annotators, expand label set iteratively.
- Better heuristics and calibrated thresholds per class; use validation set for threshold selection.

---

## 📜 Acknowledgments

- Dataset: CoNLL‑2003 via Hugging Face.
- Model: BERT (Google).
- Technique: Mahalanobis Distance for OOD detection.
- Inspired by prior work on post‑hoc OOD detection for classification and token classification tasks.

---

## Contributing

Contributions welcome — open issues or PRs. If you make changes to the pipeline, please:
1. Add or update unit/integration tests where appropriate.
2. Update notebooks and eval scripts to reflect new metrics.
3. Document new CLI flags in main.py.

---

## License

This project is provided for research and educational purposes. Add an explicit license file (e.g., MIT) to clarify usage.

---



