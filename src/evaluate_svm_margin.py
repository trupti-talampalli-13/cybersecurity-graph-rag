import json
import numpy as np

from pathlib import Path
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)


# =========================================================
# Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_FILE = (
    BASE_DIR
    / "data"
    / "azerg"
    / "t4_train_clean.jsonl"
)

TEST_FILE = (
    BASE_DIR
    / "data"
    / "azerg"
    / "t4_test_clean.jsonl"
)


# =========================================================
# Load JSONL
# =========================================================

def load_jsonl(path):

    data = []

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            if line.strip():
                data.append(json.loads(line))

    return data


# =========================================================
# Same input format as our original SVM experiment
# =========================================================

def make_input(record):

    return (
        f"SOURCE: {record['source']} "
        f"TARGET: {record['target']} "
        f"TEXT: {record['text']}"
    )


# =========================================================
# Load data
# =========================================================

print("=" * 70)
print("SVM DECISION-MARGIN THRESHOLD ANALYSIS")
print("=" * 70)

train_data = load_jsonl(TRAIN_FILE)
test_data = load_jsonl(TEST_FILE)

X_train_text = [
    make_input(x)
    for x in train_data
]

X_test_text = [
    make_input(x)
    for x in test_data
]

y_train = [
    x["relation"]
    for x in train_data
]

y_test = [
    x["relation"]
    for x in test_data
]


print(f"\nTrain examples: {len(train_data)}")
print(f"Test examples : {len(test_data)}")


# =========================================================
# TF-IDF
# EXACT SAME CONFIGURATION
# =========================================================

print("\nFitting TF-IDF...")

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
print("Test matrix :", X_test.shape)


# =========================================================
# Train SVM
# =========================================================

print("\nTraining Linear SVM...")

model = LinearSVC(
    class_weight="balanced",
    C=1.0
)

model.fit(X_train, y_train)


# =========================================================
# Predictions
# =========================================================

y_pred = model.predict(X_test)

decision_scores = model.decision_function(X_test)


# =========================================================
# Calculate decision margins
# =========================================================

margins = []

for i in range(len(y_test)):

    scores = decision_scores[i]

    sorted_scores = np.sort(scores)[::-1]

    best_score = sorted_scores[0]
    second_best_score = sorted_scores[1]

    margin = (
        best_score
        - second_best_score
    )

    margins.append(margin)

margins = np.array(margins)


# =========================================================
# Baseline performance
# =========================================================

print("\n" + "=" * 70)
print("BASELINE SVM PERFORMANCE")
print("=" * 70)

print(
    f"Accuracy   : "
    f"{accuracy_score(y_test, y_pred):.4f}"
)

print(
    f"Macro-F1   : "
    f"{f1_score(y_test, y_pred, average='macro'):.4f}"
)

print(
    f"Weighted-F1: "
    f"{f1_score(y_test, y_pred, average='weighted'):.4f}"
)


# =========================================================
# Margin statistics
# =========================================================

print("\n" + "=" * 70)
print("MARGIN DISTRIBUTION")
print("=" * 70)

percentiles = [
    10,
    25,
    50,
    75,
    90,
    95
]

for p in percentiles:

    print(
        f"P{p:<2}: "
        f"{np.percentile(margins, p):.4f}"
    )

print(
    f"Mean  : {np.mean(margins):.4f}"
)

print(
    f"Min   : {np.min(margins):.4f}"
)

print(
    f"Max   : {np.max(margins):.4f}"
)


# =========================================================
# Threshold experiment
#
# IMPORTANT:
#
# We are NOT pretending rejected examples are incorrect.
#
# We are measuring precision among retained predictions
# and how much of the data would be retained.
# =========================================================

thresholds = [
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.60,
    0.70,
    0.80,
    1.00
]


print("\n" + "=" * 70)
print("MARGIN THRESHOLD ANALYSIS")
print("=" * 70)

print(
    f"{'Threshold':>10} "
    f"{'Retained':>10} "
    f"{'Coverage':>10} "
    f"{'Accuracy':>10} "
    f"{'Macro-F1':>10} "
    f"{'Weighted-F1':>12}"
)

print("-" * 70)


threshold_results = []


for threshold in thresholds:

    keep = margins >= threshold

    retained = np.sum(keep)

    coverage = (
        retained / len(y_test)
    )

    if retained == 0:

        continue

    y_true_kept = np.array(y_test)[keep]
    y_pred_kept = np.array(y_pred)[keep]

    accuracy = accuracy_score(
        y_true_kept,
        y_pred_kept
    )

    macro_f1 = f1_score(
        y_true_kept,
        y_pred_kept,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_true_kept,
        y_pred_kept,
        average="weighted",
        zero_division=0
    )

    result = {
        "threshold": threshold,
        "retained": int(retained),
        "coverage": float(coverage),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1)
    }

    threshold_results.append(result)

    print(
        f"{threshold:10.2f} "
        f"{retained:10d} "
        f"{coverage:10.3f} "
        f"{accuracy:10.3f} "
        f"{macro_f1:10.3f} "
        f"{weighted_f1:12.3f}"
    )


# =========================================================
# Find useful operating points
# =========================================================

print("\n" + "=" * 70)
print("BEST THRESHOLDS")
print("=" * 70)


best_macro = max(
    threshold_results,
    key=lambda x: x["macro_f1"]
)

best_accuracy = max(
    threshold_results,
    key=lambda x: x["accuracy"]
)


print("\nBest retained Macro-F1:")

print(
    f"Threshold : {best_macro['threshold']:.2f}"
)

print(
    f"Coverage  : {best_macro['coverage']:.3f}"
)

print(
    f"Macro-F1  : {best_macro['macro_f1']:.4f}"
)


print("\nBest retained Accuracy:")

print(
    f"Threshold : {best_accuracy['threshold']:.2f}"
)

print(
    f"Coverage  : {best_accuracy['coverage']:.3f}"
)

print(
    f"Accuracy  : {best_accuracy['accuracy']:.4f}"
)


# =========================================================
# Prediction correctness vs margin
# =========================================================

correct = (
    np.array(y_pred)
    == np.array(y_test)
)


print("\n" + "=" * 70)
print("CORRECTNESS BY MARGIN RANGE")
print("=" * 70)

ranges = [
    (0.00, 0.10),
    (0.10, 0.20),
    (0.20, 0.30),
    (0.30, 0.40),
    (0.40, 0.50),
    (0.50, 0.75),
    (0.75, 1.00),
    (1.00, float("inf"))
]


for low, high in ranges:

    mask = (
        (margins >= low)
        & (margins < high)
    )

    count = np.sum(mask)

    if count == 0:
        continue

    accuracy = np.mean(
        correct[mask]
    )

    print(
        f"{low:.2f} - {high:.2f} : "
        f"{count:4d} examples | "
        f"accuracy = {accuracy:.3f}"
    )


print("\nDone.")