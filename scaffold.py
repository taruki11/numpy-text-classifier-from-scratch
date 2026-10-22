"""
NumPy Text Classifier from Scratch scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""End-to-end demo: NumPy TF-IDF + L2 logistic spam/sentiment classifier."""
import numpy as np


def main() -> None:
    np.random.seed(0)

    texts = [
        "Win a free prize now click here",
        "Meeting scheduled for Monday morning",
        "Congratulations you won the lottery jackpot",
        "Please review the attached project report",
        "Cheap meds online limited offer today",
        "Lunch with the team at noon tomorrow",
        "Urgent claim your reward immediately",
        "Can we reschedule the client call",
        "Free money guaranteed no risk act now",
        "Notes from yesterday standup are ready",
        "Exclusive deal buy one get one free",
        "The quarterly budget looks healthy",
        "You have been selected for a gift card",
        "Draft agenda for the all hands meeting",
        "Click to unsubscribe and win cash",
        "Ship the release candidate this Friday",
    ]
    labels = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=float)

    tokenized = tokenize_corpus(texts)
    n = len(texts)
    train_idx, val_idx, test_idx = split_train_val_test_indices(
        n, val_fraction=0.25, test_fraction=0.25, seed=0
    )

    train_tok = [tokenized[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    test_texts = [texts[i] for i in test_idx]
    y_train = labels[train_idx]
    y_val = labels[val_idx]
    y_test = labels[test_idx]

    word_counts = count_word_frequencies(train_tok)
    vocab = build_vocabulary(word_counts, max_size=40)
    bow_train = corpus_to_bow_matrix(train_tok, vocab)
    idf = fit_tfidf(bow_train)
    X_train = transform_tfidf(bow_train, idf)

    w, b, losses = train_logistic_regression(
        X_train, y_train, lr=0.5, l2_lambda=0.01, n_epochs=80
    )
    print("vocab_size", len(vocab))
    print("final_train_loss", round(float(losses[-1]), 4))

    X_val = vectorize_texts(val_texts, vocab, idf)
    val_proba = logistic_predict_proba(X_val, w, b)
    best_t, best_f1 = tune_decision_threshold(y_val, val_proba)
    print("best_threshold", round(float(best_t), 3), "val_f1", round(float(best_f1), 3))

    X_test = vectorize_texts(test_texts, vocab, idf)
    test_proba = logistic_predict_proba(X_test, w, b)
    y_pred = predict_labels(test_proba, threshold=best_t)
    report = evaluate_predictions(y_test, y_pred)
    print("test_metrics", {k: round(v, 3) if isinstance(v, float) else v for k, v in report.items()})

    sample = "Free prize click here to claim now"
    pred = predict_text(sample, vocab, idf, w, b, threshold=best_t)
    print("sample_text", sample)
    print("sample_pred", int(pred))

    errors = collect_prediction_errors(test_texts, y_test, y_pred)
    print("n_false_positives", len(errors["false_positives"]))
    print("n_false_negatives", len(errors["false_negatives"]))


if __name__ == "__main__":
    main()
