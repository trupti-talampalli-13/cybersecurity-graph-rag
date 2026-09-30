import json
from pathlib import Path

from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "azerg"


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    return records


def make_input(record):
    return (
        f"SOURCE: {record['source']} "
        f"TARGET: {record['target']} "
        f"TEXT: {record['text']}"
    )


train = load_jsonl(DATA_DIR / "t4_train_clean.jsonl")
test = load_jsonl(DATA_DIR / "t4_test_clean.jsonl")


X_train = [make_input(r) for r in train]
y_train = [r["relation"] for r in train]

X_test = [make_input(r) for r in test]
y_test = [r["relation"] for r in test]


# Word-level TF-IDF
word_vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_features=50000,
    sublinear_tf=True,
)


# Character-level TF-IDF
char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=2,
    max_features=50000,
    sublinear_tf=True,
)


vectorizer = FeatureUnion([
    ("word", word_vectorizer),
    ("char", char_vectorizer),
])


print("Fitting TF-IDF...")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("Train matrix:", X_train_vec.shape)
print("Test matrix:", X_test_vec.shape)


model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
)


print("\nTraining Logistic Regression...")
model.fit(X_train_vec, y_train)


predictions = model.predict(X_test_vec)


accuracy = accuracy_score(y_test, predictions)
micro_f1 = f1_score(y_test, predictions, average="micro")
macro_f1 = f1_score(y_test, predictions, average="macro")
weighted_f1 = f1_score(y_test, predictions, average="weighted")


print("\n" + "=" * 70)
print("TF-IDF + LOGISTIC REGRESSION RESULTS")
print("=" * 70)

print(f"Accuracy:   {accuracy:.4f}")
print(f"Micro-F1:    {micro_f1:.4f}")
print(f"Macro-F1:    {macro_f1:.4f}")
print(f"Weighted-F1: {weighted_f1:.4f}")


print("\n" + "=" * 70)
print("PER-CLASS RESULTS")
print("=" * 70)

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0,
    )
)


print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

labels = sorted(set(y_test))

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=labels,
)

print("Labels:")
print(labels)

print("\nMatrix:")
print(matrix)