import json
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


TRAIN_FILE = "data/azerg/t4_train_clean.jsonl"
TEST_FILE = "data/azerg/t4_test_clean.jsonl"


def load_jsonl(path):
    data = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    return data


def make_input(record):
    return (
        f"SOURCE: {record['source']} "
        f"TARGET: {record['target']} "
        f"TEXT: {record['text']}"
    )


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

train_data = load_jsonl(TRAIN_FILE)
test_data = load_jsonl(TEST_FILE)

X_train_text = [make_input(x) for x in train_data]
X_test_text = [make_input(x) for x in test_data]

y_train = [x["relation"] for x in train_data]
y_test = [x["relation"] for x in test_data]


# ---------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------

print("Fitting TF-IDF...")

word_vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_features=50000,
    sublinear_tf=True
)

char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=2,
    max_features=50000,
    sublinear_tf=True
)

vectorizer = FeatureUnion([
    ("word", word_vectorizer),
    ("char", char_vectorizer)
])

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

print("Train matrix:", X_train.shape)
print("Test matrix:", X_test.shape)


# ---------------------------------------------------------
# Linear SVM
# ---------------------------------------------------------

print("\nTraining Linear SVM...")

model = LinearSVC(
    class_weight="balanced",
    C=1.0
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
micro_f1 = f1_score(y_test, y_pred, average="micro")
macro_f1 = f1_score(y_test, y_pred, average="macro")
weighted_f1 = f1_score(y_test, y_pred, average="weighted")


print("\n" + "=" * 70)
print("TF-IDF + LINEAR SVM RESULTS")
print("=" * 70)

print(f"Accuracy:   {accuracy:.4f}")
print(f"Micro-F1:    {micro_f1:.4f}")
print(f"Macro-F1:    {macro_f1:.4f}")
print(f"Weighted-F1: {weighted_f1:.4f}")


# ---------------------------------------------------------
# Per-class results
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("PER-CLASS RESULTS")
print("=" * 70)

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ---------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------

labels = sorted(set(y_test))

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print("Labels:")
print(labels)

print("\nMatrix:")
print(cm)