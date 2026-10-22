# NumPy Text Classifier from Scratch

Build a complete spam/sentiment text classifier in pure NumPy: clean and tokenize text, extract fixed-vocabulary TF-IDF features, train L2-regularized logistic regression with batch gradient descent, tune the decision threshold on validation data, and deliver a predict(text) API with false-positive/false-negative error analysis.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** clean_text
- [x] **2.** tokenize
- [x] **3.** tokenize_corpus
- [x] **4.** split_train_val_test_indices
- [x] **5.** count_word_frequencies
- [x] **6.** build_vocabulary
- [x] **7.** tokens_to_bow
- [x] **8.** corpus_to_bow_matrix
- [x] **9.** compute_document_frequencies
- [x] **10.** compute_idf
- [ ] **11.** transform_tfidf
- [ ] **12.** fit_tfidf
- [ ] **13.** sigmoid
- [ ] **14.** logistic_predict_proba
- [ ] **15.** binary_cross_entropy
- [ ] **16.** logistic_gradients
- [ ] **17.** initialize_logistic_params
- [ ] **18.** gradient_descent_step
- [ ] **19.** train_logistic_regression
- [ ] **20.** predict_labels
- [ ] **21.** confusion_counts
- [ ] **22.** metrics_from_counts
- [ ] **23.** tune_decision_threshold
- [ ] **24.** evaluate_predictions
- [ ] **25.** vectorize_texts
- [ ] **26.** predict_text
- [ ] **27.** collect_prediction_errors

---

Built on Deep-ML.
