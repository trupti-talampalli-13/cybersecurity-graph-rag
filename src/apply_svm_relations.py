import json
import numpy as np

from pathlib import Path
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC


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

CANDIDATE_FILE = (
    BASE_DIR
    / "data"
    / "annoctr_relation_candidates_filtered.jsonl"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "annoctr_relation_predictions.jsonl"
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
# AZERG input format
# =========================================================

def make_azerg_input(record):

    return (
        f"SOURCE: {record['source']} "
        f"TARGET: {record['target']} "
        f"TEXT: {record['text']}"
    )


# =========================================================
# AnnoCTR input format
#
# IMPORTANT:
# We keep the same SOURCE/TARGET/TEXT structure used
# during SVM training.
# =========================================================

def make_annocr_input(record):

    return (
        f"SOURCE: {record['source_mention']} "
        f"TARGET: {record['target_mention']} "
        f"TEXT: {record['text']}"
    )


# =========================================================
# Load training data
# =========================================================

print("=" * 70)
print("AZERG TRAINING → AnnoCTR RELATION PREDICTION")
print("=" * 70)

print("\nLoading AZERG training data...")

train_data = load_jsonl(TRAIN_FILE)

print(f"Training examples: {len(train_data)}")


# =========================================================
# Prepare AZERG training data
# =========================================================

X_train_text = [
    make_azerg_input(record)
    for record in train_data
]

y_train = [
    record["relation"]
    for record in train_data
]


print(
    f"Relation classes: "
    f"{len(set(y_train))}"
)

print("\nTraining relation distribution:")

for relation, count in Counter(y_train).most_common():

    print(
        f"{relation:25s} : {count}"
    )


# =========================================================
# Load AnnoCTR candidates
# =========================================================

print("\nLoading AnnoCTR candidates...")

candidates = load_jsonl(CANDIDATE_FILE)

print(
    f"AnnoCTR candidate pairs: "
    f"{len(candidates)}"
)


# =========================================================
# Prepare AnnoCTR inputs
# =========================================================

X_candidate_text = [
    make_annocr_input(record)
    for record in candidates
]


# =========================================================
# TF-IDF
#
# EXACTLY the same configuration as the experiment
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


X_train = vectorizer.fit_transform(
    X_train_text
)

X_candidates = vectorizer.transform(
    X_candidate_text
)


print(
    "Train matrix:",
    X_train.shape
)

print(
    "Candidate matrix:",
    X_candidates.shape
)


# =========================================================
# Train Linear SVM
#
# EXACT SAME CONFIGURATION
# =========================================================

print("\nTraining Linear SVM...")

model = LinearSVC(
    class_weight="balanced",
    C=1.0
)

model.fit(
    X_train,
    y_train
)

print("Training complete.")


# =========================================================
# Predict relations
# =========================================================

print("\nPredicting AnnoCTR relations...")

predictions = model.predict(
    X_candidates
)

decision_scores = model.decision_function(
    X_candidates
)


# =========================================================
# Convert decision scores
#
# LinearSVC returns:
#
#   multiclass:
#       [score_class_1, score_class_2, ...]
#
# We store:
#
#   predicted_relation
#   decision_score
#   second_best_score
#   margin
#
# These are NOT probabilities.
# =========================================================

classes = model.classes_

results = []

for i, record in enumerate(candidates):

    scores = decision_scores[i]

    # Handle binary case just in case
    if scores.ndim == 0:

        best_idx = 0
        sorted_indices = [0]

    else:

        sorted_indices = np.argsort(scores)[::-1]
        best_idx = sorted_indices[0]

    predicted_relation = classes[best_idx]

    best_score = float(
        scores[best_idx]
    )

    if len(sorted_indices) > 1:

        second_idx = sorted_indices[1]

        second_score = float(
            scores[second_idx]
        )

        margin = (
            best_score - second_score
        )

    else:

        second_score = None
        margin = None

    result = {
        "document": record["document"],
        "segment": record["segment"],

        "text": record["text"],

        "source_mention": record[
            "source_mention"
        ],

        "source_type": record[
            "source_type"
        ],

        "target_mention": record[
            "target_mention"
        ],

        "target_type": record[
            "target_type"
        ],

        "predicted_relation": predicted_relation,

        "decision_score": best_score,

        "second_best_score": second_score,

        "margin": margin
    }

    results.append(result)


# =========================================================
# Save predictions
# =========================================================

print("\nSaving predictions...")

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    for result in results:

        f.write(
            json.dumps(
                result,
                ensure_ascii=False
            ) + "\n"
        )


# =========================================================
# Prediction distribution
# =========================================================

distribution = Counter(
    result["predicted_relation"]
    for result in results
)


print("\n" + "=" * 70)
print("PREDICTED RELATION DISTRIBUTION")
print("=" * 70)

for relation, count in distribution.most_common():

    percentage = (
        count / len(results) * 100
    )

    print(
        f"{relation:25s} : "
        f"{count:5d} "
        f"({percentage:6.2f}%)"
    )


# =========================================================
# Score statistics
# =========================================================

margins = [
    r["margin"]
    for r in results
    if r["margin"] is not None
]

if margins:

    print("\n" + "=" * 70)
    print("DECISION MARGIN STATISTICS")
    print("=" * 70)

    print(
        f"Minimum margin : {min(margins):.4f}"
    )

    print(
        f"Mean margin    : {np.mean(margins):.4f}"
    )

    print(
        f"Median margin  : {np.median(margins):.4f}"
    )

    print(
        f"Maximum margin  : {max(margins):.4f}"
    )


# =========================================================
# Show examples
# =========================================================

print("\n" + "=" * 70)
print("SAMPLE PREDICTIONS")
print("=" * 70)

for result in results[:20]:

    print("\n"
          f"[{result['source_type']}] "
          f"{result['source_mention']} "
          f"--{result['predicted_relation']}--> "
          f"[{result['target_type']}] "
          f"{result['target_mention']}")

    print(
        f"Score: {result['decision_score']:.4f} "
        f"| Margin: {result['margin']:.4f}"
    )

    print(
        f"Document: {result['document']}"
    )

    print(
        f"Segment: {result['segment']}"
    )


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print("\nSaved:")
print(OUTPUT_FILE)