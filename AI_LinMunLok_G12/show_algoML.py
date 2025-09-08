import os
import json
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- Config ----------
MODEL_DIR = "./intent_results/model"
CSV_PATH = "CustomerSupportTraining.csv"
TEXT_COLUMN = "instruction"
LABEL_COLUMN = "intent"
MAX_LENGTH = 128

# ---------- NLTK setup ----------
try:
    _ = nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# ---------- Load model & tokenizer ----------
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

with open(os.path.join(MODEL_DIR, "label_map.json"), "r", encoding="utf-8") as f:
    maps = json.load(f)
id2label = {int(k): v for k, v in maps["id2label"].items()}
label2id = maps["label2id"]

with open(os.path.join(MODEL_DIR, "intent_to_response.json"), "r", encoding="utf-8") as f:
    intent_to_response = json.load(f)

# ---------- Load dataset ----------
df = pd.read_csv(CSV_PATH, sep=None, engine="python")
df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN]).reset_index(drop=True)
df[LABEL_COLUMN] = df[LABEL_COLUMN].astype("category")
labels = list(df[LABEL_COLUMN].cat.categories)

X_test = df[TEXT_COLUMN].astype(str).tolist()
y_test = df[LABEL_COLUMN].cat.codes.values

# ---------- Prediction function ----------
def predict(texts, batch_size=16):  # you can adjust batch_size
    preds = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        enc = tokenizer(batch_texts, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            outputs = model(**enc)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
        preds.extend(np.argmax(probs, axis=1))
    return np.array(preds)


y_pred = predict(X_test)

# ---------- Precision, Recall, F1 ----------
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=labels, zero_division=0))

# ---------- Confusion Matrix Plot ----------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(max(8, len(labels) * 0.5), max(6, len(labels) * 0.5)))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Intent")
plt.ylabel("True Intent")
plt.title("Confusion Matrix")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# ---------- ROUGE & BLEU ----------
scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
smooth = SmoothingFunction().method4

rouge1_f, rougel_f, bleu_scores = [], [], []

for i in range(len(y_test)):
    gold_intent = id2label[y_test[i]]
    pred_intent = id2label[y_pred[i]]

    gold_resp = intent_to_response.get(gold_intent, "")
    pred_resp = intent_to_response.get(pred_intent, "")

    # ROUGE
    r = scorer.score(gold_resp, pred_resp)
    rouge1_f.append(r["rouge1"].fmeasure)
    rougel_f.append(r["rougeL"].fmeasure)

    # BLEU
    ref_tokens = [gold_resp.split()]
    cand_tokens = pred_resp.split()
    bleu = sentence_bleu(ref_tokens, cand_tokens, smoothing_function=smooth) if cand_tokens else 0.0
    bleu_scores.append(bleu)

print("\n=== Response Quality ===")
print(f"ROUGE-1 F1 (avg): {np.mean(rouge1_f):.4f}")
print(f"ROUGE-L F1 (avg): {np.mean(rougel_f):.4f}")
print(f"BLEU (avg):       {np.mean(bleu_scores):.4f}")

# ---------- ROUGE & BLEU Boxplot ----------
plt.figure(figsize=(10, 5))
plt.boxplot([rouge1_f, rougel_f, bleu_scores], tick_labels=["ROUGE-1", "ROUGE-L", "BLEU"])
plt.title("Response Quality Distribution")
plt.ylabel("Score")
plt.show()

# ---------- Optional: Demo predictions ----------
sample_texts = [
    "I want to cancel order 12345", 
    "How do I return a product?",
    "Can I get a refund for my purchase?"
]

for t in sample_texts:
    enc = tokenizer(t, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        outputs = model(**enc)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    top_idx = probs.argsort()[::-1][:3]
    results = [{"intent": id2label[idx], "score": float(probs[idx]), "response": intent_to_response.get(id2label[idx], "")} for idx in top_idx]
    print(f"\nInput: {t}")
    print("Predictions:", results)
