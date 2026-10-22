"""
NumPy Text Classifier from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - clean_text
import numpy as np


# ============================================================
# 1-3. 文本清洗与分词
# ============================================================

def clean_text(text: str) -> str:
    """
    转成小写，并把每个非字母字符替换为空格。
    注意：保留连续空格以及开头、结尾产生的空格。
    """
    text = text.lower()
    return "".join(ch if ch.isalpha() else " " for ch in text)


def tokenize(text: str) -> list:
    """按任意空白字符切分，并自动丢弃空 token。"""
    return text.split()


def tokenize_corpus(texts: list) -> list:
    """清洗并分词整个文本列表。"""
    return [tokenize(clean_text(text)) for text in texts]


# ============================================================
# 4-8. 数据划分、词频、词表与 Bag-of-Words
# ============================================================

def split_train_val_test_indices(
    n_samples: int,
    val_fraction: float,
    test_fraction: float,
    seed: int = 0
) -> tuple:
    """
    生成互不重叠的 train/validation/test 索引。
    按 train、validation、test 的顺序切分打乱后的索引。
    """
    np.random.seed(seed)

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    n_val = int(n_samples * val_fraction)
    n_test = int(n_samples * test_fraction)
    n_train = n_samples - n_val - n_test

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return train_idx, val_idx, test_idx


def count_word_frequencies(tokenized_docs: list) -> dict:
    """统计每个 token 在整个语料中出现的总次数。"""
    counts = {}

    for document in tokenized_docs:
        for token in document:
            counts[token] = counts.get(token, 0) + 1

    return counts


def build_vocabulary(word_counts: dict, max_size: int) -> dict:
    """
    选出词频最高的 max_size 个词。
    词频相同时，按字母顺序排列。
    """
    max_size = max(0, int(max_size))

    ranked_words = sorted(
        word_counts.items(),
        key=lambda item: (-item[1], item[0])
    )

    return {
        word: index
        for index, (word, _) in enumerate(ranked_words[:max_size])
    }


def tokens_to_bow(tokens: list, vocab: dict) -> np.ndarray:
    """将一篇文档转成词袋计数向量，未知词忽略。"""
    vector = np.zeros(len(vocab), dtype=float)

    for token in tokens:
        if token in vocab:
            vector[vocab[token]] += 1.0

    return vector


def corpus_to_bow_matrix(tokenized_docs: list, vocab: dict) -> np.ndarray:
    """将多个文档转成形状为 (文档数, 词表大小) 的词袋矩阵。"""
    matrix = np.zeros(
        (len(tokenized_docs), len(vocab)),
        dtype=float
    )

    for row, tokens in enumerate(tokenized_docs):
        matrix[row] = tokens_to_bow(tokens, vocab)

    return matrix


# ============================================================
# 9-12. 文档频率与 TF-IDF
# ============================================================

def compute_document_frequencies(bow_matrix: np.ndarray) -> np.ndarray:
    """
    计算每个词出现于多少篇文档。
    每篇文档中，一个词出现一次或多次，都只计为出现过。
    """
    bow_matrix = np.asarray(bow_matrix)
    return np.sum(bow_matrix > 0, axis=0).astype(int)


def compute_idf(df: np.ndarray, n_docs: int) -> np.ndarray:
    """计算平滑 IDF，使用自然对数。"""
    df = np.asarray(df, dtype=float)
    return np.log((n_docs + 1) / (df + 1)) + 1.0


def transform_tfidf(
    bow_matrix: np.ndarray,
    idf: np.ndarray
) -> np.ndarray:
    """
    每列的词袋计数乘对应 IDF。
    支持输入一个向量 (V,) 或矩阵 (N, V)。
    """
    bow_matrix = np.asarray(bow_matrix, dtype=float)
    idf = np.asarray(idf, dtype=float)
    return bow_matrix * idf


def fit_tfidf(bow_train: np.ndarray) -> np.ndarray:
    """只根据训练集词袋矩阵拟合 IDF。"""
    bow_train = np.asarray(bow_train, dtype=float)

    df = compute_document_frequencies(bow_train)
    n_docs = bow_train.shape[0]

    return compute_idf(df, n_docs)


# ============================================================
# 13-16. 逻辑回归核心：Sigmoid、概率、损失与梯度
# ============================================================

def sigmoid(z: np.ndarray) -> np.ndarray:
    """数值稳定的 sigmoid，返回与输入形状相同的数组。"""
    z = np.asarray(z, dtype=float)
    flat = z.reshape(-1)
    result = np.empty_like(flat)

    # 正数和负数分开计算，避免 exp 溢出
    positive = flat >= 0
    negative = ~positive

    result[positive] = 1.0 / (1.0 + np.exp(-flat[positive]))

    exp_z = np.exp(flat[negative])
    result[negative] = exp_z / (1.0 + exp_z)

    return result.reshape(z.shape)


def logistic_predict_proba(
    X: np.ndarray,
    w: np.ndarray,
    b: float
) -> np.ndarray:
    """计算每一行属于类别 1 的概率。"""
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)

    return sigmoid(X @ w + b)


def binary_cross_entropy(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    w: np.ndarray,
    l2_lambda: float
) -> float:
    """
    平均二元交叉熵 + L2 权重惩罚。
    bias 不在这里传入，所以不会被正则化。
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    w = np.asarray(w, dtype=float)

    # 避免 log(0)
    y_prob = np.clip(y_prob, 1e-15, 1.0 - 1e-15)

    if y_true.size == 0:
        mean_bce = 0.0
    else:
        mean_bce = -np.mean(
            y_true * np.log(y_prob)
            + (1.0 - y_true) * np.log(1.0 - y_prob)
        )

    l2_penalty = l2_lambda * np.sum(w ** 2) / 2.0

    return float(mean_bce + l2_penalty)


def logistic_gradients(
    X: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    w: np.ndarray,
    l2_lambda: float
) -> tuple:
    """计算平均 BCE + L2 对权重和 bias 的梯度。"""
    X = np.asarray(X, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    w = np.asarray(w, dtype=float)

    n_samples = X.shape[0]

    if n_samples == 0:
        return l2_lambda * w, 0.0

    residual = y_prob - y_true

    # dL/dw：数据损失梯度 + L2 梯度
    dw = (X.T @ residual) / n_samples + l2_lambda * w

    # bias 不加 L2 惩罚
    db = float(np.mean(residual))

    return dw, db


# ============================================================
# 17-20. 初始化、梯度下降训练与阈值预测
# ============================================================

def initialize_logistic_params(n_features: int) -> tuple:
    """权重和 bias 都从 0 开始。"""
    w = np.zeros(n_features, dtype=float)
    b = 0.0

    return w, b


def gradient_descent_step(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    b: float,
    lr: float,
    l2_lambda: float
) -> tuple:
    """
    做一次全批量梯度下降。
    返回的 loss 是更新参数之前计算的损失。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)

    probabilities = logistic_predict_proba(X, w, b)
    loss = binary_cross_entropy(y, probabilities, w, l2_lambda)

    dw, db = logistic_gradients(
        X, y, probabilities, w, l2_lambda
    )

    w_new = w - lr * dw
    b_new = float(b - lr * db)

    return w_new, b_new, loss


def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    lr: float,
    l2_lambda: float,
    n_epochs: int
) -> tuple:
    """从零开始训练逻辑回归，并记录每轮更新前的 loss。"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    w, b = initialize_logistic_params(X.shape[1])
    losses = []

    for _ in range(n_epochs):
        w, b, loss = gradient_descent_step(
            X, y, w, b, lr, l2_lambda
        )
        losses.append(float(loss))

    return w, float(b), losses


def predict_labels(
    proba: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """概率大于等于阈值时预测为 1，否则预测为 0。"""
    proba = np.asarray(proba, dtype=float)
    return (proba >= threshold).astype(int)


# ============================================================
# 21-24. 混淆矩阵计数、分类指标与阈值调优
# ============================================================

def confusion_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> tuple:
    """返回 (TP, FP, TN, FN)，均为 Python int。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    return tp, fp, tn, fn


def metrics_from_counts(
    tp: int,
    fp: int,
    tn: int,
    fn: int
) -> dict:
    """根据混淆矩阵计数计算 precision、recall、F1 和 accuracy。"""
    precision_denominator = tp + fp
    recall_denominator = tp + fn
    f1_denominator = 2 * tp + fp + fn
    total = tp + fp + tn + fn

    precision = (
        tp / precision_denominator
        if precision_denominator > 0 else 0.0
    )
    recall = (
        tp / recall_denominator
        if recall_denominator > 0 else 0.0
    )
    f1 = (
        2 * tp / f1_denominator
        if f1_denominator > 0 else 0.0
    )
    accuracy = (
        (tp + tn) / total
        if total > 0 else 0.0
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


def tune_decision_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    thresholds: np.ndarray = None
) -> tuple:
    """
    在候选阈值中选 F1 最高的那个。
    F1 并列时保留 thresholds 中最先出现的阈值。
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)

    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)
    else:
        thresholds = np.asarray(thresholds, dtype=float)

    if len(thresholds) == 0:
        return 0.0, 0.0

    best_threshold = float(thresholds[0])
    best_f1 = -1.0

    for threshold in thresholds:
        y_pred = predict_labels(proba, threshold)
        tp, fp, tn, fn = confusion_counts(y_true, y_pred)
        f1 = metrics_from_counts(tp, fp, tn, fn)["f1"]

        # 使用严格大于，保证并列时留下先遇到的阈值
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return float(best_threshold), float(best_f1)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """返回混淆矩阵计数和分类指标的汇总字典。"""
    tp, fp, tn, fn = confusion_counts(y_true, y_pred)
    metrics = metrics_from_counts(tp, fp, tn, fn)

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "accuracy": metrics["accuracy"],
    }


# ============================================================
# 25-27. 新文本向量化、单条预测与错误分析
# ============================================================

def vectorize_texts(
    texts: list,
    vocab: dict,
    idf: np.ndarray
) -> np.ndarray:
    """把原始文本转换成训练时同样格式的 TF-IDF 矩阵。"""
    tokenized_docs = tokenize_corpus(texts)
    bow_matrix = corpus_to_bow_matrix(tokenized_docs, vocab)

    return transform_tfidf(bow_matrix, idf)


def predict_text(
    text: str,
    vocab: dict,
    idf: np.ndarray,
    w: np.ndarray,
    b: float,
    threshold: float = 0.5
) -> int:
    """使用训练好的参数预测一条原始文本。"""
    X = vectorize_texts([text], vocab, idf)
    proba = logistic_predict_proba(X, w, b)
    prediction = predict_labels(proba, threshold)

    return int(prediction[0])


def collect_prediction_errors(
    texts: list,
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """收集假阳性和假阴性的原始文本，保持输入顺序。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    false_positives = []
    false_negatives = []

    for text, true_label, predicted_label in zip(texts, y_true, y_pred):
        if true_label == 0 and predicted_label == 1:
            false_positives.append(text)
        elif true_label == 1 and predicted_label == 0:
            false_negatives.append(text)

    return {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }

# Step 2 - tokenize
import numpy as np


# ============================================================
# 1-3. 文本清洗与分词
# ============================================================

def clean_text(text: str) -> str:
    """
    转成小写，并把每个非字母字符替换为空格。
    注意：保留连续空格以及开头、结尾产生的空格。
    """
    text = text.lower()
    return "".join(ch if ch.isalpha() else " " for ch in text)


def tokenize(text: str) -> list:
    """按任意空白字符切分，并自动丢弃空 token。"""
    return text.split()


def tokenize_corpus(texts: list) -> list:
    """清洗并分词整个文本列表。"""
    return [tokenize(clean_text(text)) for text in texts]


# ============================================================
# 4-8. 数据划分、词频、词表与 Bag-of-Words
# ============================================================

def split_train_val_test_indices(
    n_samples: int,
    val_fraction: float,
    test_fraction: float,
    seed: int = 0
) -> tuple:
    """
    生成互不重叠的 train/validation/test 索引。
    按 train、validation、test 的顺序切分打乱后的索引。
    """
    np.random.seed(seed)

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    n_val = int(n_samples * val_fraction)
    n_test = int(n_samples * test_fraction)
    n_train = n_samples - n_val - n_test

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return train_idx, val_idx, test_idx


def count_word_frequencies(tokenized_docs: list) -> dict:
    """统计每个 token 在整个语料中出现的总次数。"""
    counts = {}

    for document in tokenized_docs:
        for token in document:
            counts[token] = counts.get(token, 0) + 1

    return counts


def build_vocabulary(word_counts: dict, max_size: int) -> dict:
    """
    选出词频最高的 max_size 个词。
    词频相同时，按字母顺序排列。
    """
    max_size = max(0, int(max_size))

    ranked_words = sorted(
        word_counts.items(),
        key=lambda item: (-item[1], item[0])
    )

    return {
        word: index
        for index, (word, _) in enumerate(ranked_words[:max_size])
    }


def tokens_to_bow(tokens: list, vocab: dict) -> np.ndarray:
    """将一篇文档转成词袋计数向量，未知词忽略。"""
    vector = np.zeros(len(vocab), dtype=float)

    for token in tokens:
        if token in vocab:
            vector[vocab[token]] += 1.0

    return vector


def corpus_to_bow_matrix(tokenized_docs: list, vocab: dict) -> np.ndarray:
    """将多个文档转成形状为 (文档数, 词表大小) 的词袋矩阵。"""
    matrix = np.zeros(
        (len(tokenized_docs), len(vocab)),
        dtype=float
    )

    for row, tokens in enumerate(tokenized_docs):
        matrix[row] = tokens_to_bow(tokens, vocab)

    return matrix


# ============================================================
# 9-12. 文档频率与 TF-IDF
# ============================================================

def compute_document_frequencies(bow_matrix: np.ndarray) -> np.ndarray:
    """
    计算每个词出现于多少篇文档。
    每篇文档中，一个词出现一次或多次，都只计为出现过。
    """
    bow_matrix = np.asarray(bow_matrix)
    return np.sum(bow_matrix > 0, axis=0).astype(int)


def compute_idf(df: np.ndarray, n_docs: int) -> np.ndarray:
    """计算平滑 IDF，使用自然对数。"""
    df = np.asarray(df, dtype=float)
    return np.log((n_docs + 1) / (df + 1)) + 1.0


def transform_tfidf(
    bow_matrix: np.ndarray,
    idf: np.ndarray
) -> np.ndarray:
    """
    每列的词袋计数乘对应 IDF。
    支持输入一个向量 (V,) 或矩阵 (N, V)。
    """
    bow_matrix = np.asarray(bow_matrix, dtype=float)
    idf = np.asarray(idf, dtype=float)
    return bow_matrix * idf


def fit_tfidf(bow_train: np.ndarray) -> np.ndarray:
    """只根据训练集词袋矩阵拟合 IDF。"""
    bow_train = np.asarray(bow_train, dtype=float)

    df = compute_document_frequencies(bow_train)
    n_docs = bow_train.shape[0]

    return compute_idf(df, n_docs)


# ============================================================
# 13-16. 逻辑回归核心：Sigmoid、概率、损失与梯度
# ============================================================

def sigmoid(z: np.ndarray) -> np.ndarray:
    """数值稳定的 sigmoid，返回与输入形状相同的数组。"""
    z = np.asarray(z, dtype=float)
    flat = z.reshape(-1)
    result = np.empty_like(flat)

    # 正数和负数分开计算，避免 exp 溢出
    positive = flat >= 0
    negative = ~positive

    result[positive] = 1.0 / (1.0 + np.exp(-flat[positive]))

    exp_z = np.exp(flat[negative])
    result[negative] = exp_z / (1.0 + exp_z)

    return result.reshape(z.shape)


def logistic_predict_proba(
    X: np.ndarray,
    w: np.ndarray,
    b: float
) -> np.ndarray:
    """计算每一行属于类别 1 的概率。"""
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)

    return sigmoid(X @ w + b)


def binary_cross_entropy(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    w: np.ndarray,
    l2_lambda: float
) -> float:
    """
    平均二元交叉熵 + L2 权重惩罚。
    bias 不在这里传入，所以不会被正则化。
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    w = np.asarray(w, dtype=float)

    # 避免 log(0)
    y_prob = np.clip(y_prob, 1e-15, 1.0 - 1e-15)

    if y_true.size == 0:
        mean_bce = 0.0
    else:
        mean_bce = -np.mean(
            y_true * np.log(y_prob)
            + (1.0 - y_true) * np.log(1.0 - y_prob)
        )

    l2_penalty = l2_lambda * np.sum(w ** 2) / 2.0

    return float(mean_bce + l2_penalty)


def logistic_gradients(
    X: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    w: np.ndarray,
    l2_lambda: float
) -> tuple:
    """计算平均 BCE + L2 对权重和 bias 的梯度。"""
    X = np.asarray(X, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    w = np.asarray(w, dtype=float)

    n_samples = X.shape[0]

    if n_samples == 0:
        return l2_lambda * w, 0.0

    residual = y_prob - y_true

    # dL/dw：数据损失梯度 + L2 梯度
    dw = (X.T @ residual) / n_samples + l2_lambda * w

    # bias 不加 L2 惩罚
    db = float(np.mean(residual))

    return dw, db


# ============================================================
# 17-20. 初始化、梯度下降训练与阈值预测
# ============================================================

def initialize_logistic_params(n_features: int) -> tuple:
    """权重和 bias 都从 0 开始。"""
    w = np.zeros(n_features, dtype=float)
    b = 0.0

    return w, b


def gradient_descent_step(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    b: float,
    lr: float,
    l2_lambda: float
) -> tuple:
    """
    做一次全批量梯度下降。
    返回的 loss 是更新参数之前计算的损失。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)

    probabilities = logistic_predict_proba(X, w, b)
    loss = binary_cross_entropy(y, probabilities, w, l2_lambda)

    dw, db = logistic_gradients(
        X, y, probabilities, w, l2_lambda
    )

    w_new = w - lr * dw
    b_new = float(b - lr * db)

    return w_new, b_new, loss


def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    lr: float,
    l2_lambda: float,
    n_epochs: int
) -> tuple:
    """从零开始训练逻辑回归，并记录每轮更新前的 loss。"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    w, b = initialize_logistic_params(X.shape[1])
    losses = []

    for _ in range(n_epochs):
        w, b, loss = gradient_descent_step(
            X, y, w, b, lr, l2_lambda
        )
        losses.append(float(loss))

    return w, float(b), losses


def predict_labels(
    proba: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """概率大于等于阈值时预测为 1，否则预测为 0。"""
    proba = np.asarray(proba, dtype=float)
    return (proba >= threshold).astype(int)


# ============================================================
# 21-24. 混淆矩阵计数、分类指标与阈值调优
# ============================================================

def confusion_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> tuple:
    """返回 (TP, FP, TN, FN)，均为 Python int。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    return tp, fp, tn, fn


def metrics_from_counts(
    tp: int,
    fp: int,
    tn: int,
    fn: int
) -> dict:
    """根据混淆矩阵计数计算 precision、recall、F1 和 accuracy。"""
    precision_denominator = tp + fp
    recall_denominator = tp + fn
    f1_denominator = 2 * tp + fp + fn
    total = tp + fp + tn + fn

    precision = (
        tp / precision_denominator
        if precision_denominator > 0 else 0.0
    )
    recall = (
        tp / recall_denominator
        if recall_denominator > 0 else 0.0
    )
    f1 = (
        2 * tp / f1_denominator
        if f1_denominator > 0 else 0.0
    )
    accuracy = (
        (tp + tn) / total
        if total > 0 else 0.0
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


def tune_decision_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    thresholds: np.ndarray = None
) -> tuple:
    """
    在候选阈值中选 F1 最高的那个。
    F1 并列时保留 thresholds 中最先出现的阈值。
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)

    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)
    else:
        thresholds = np.asarray(thresholds, dtype=float)

    if len(thresholds) == 0:
        return 0.0, 0.0

    best_threshold = float(thresholds[0])
    best_f1 = -1.0

    for threshold in thresholds:
        y_pred = predict_labels(proba, threshold)
        tp, fp, tn, fn = confusion_counts(y_true, y_pred)
        f1 = metrics_from_counts(tp, fp, tn, fn)["f1"]

        # 使用严格大于，保证并列时留下先遇到的阈值
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return float(best_threshold), float(best_f1)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """返回混淆矩阵计数和分类指标的汇总字典。"""
    tp, fp, tn, fn = confusion_counts(y_true, y_pred)
    metrics = metrics_from_counts(tp, fp, tn, fn)

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "accuracy": metrics["accuracy"],
    }


# ============================================================
# 25-27. 新文本向量化、单条预测与错误分析
# ============================================================

def vectorize_texts(
    texts: list,
    vocab: dict,
    idf: np.ndarray
) -> np.ndarray:
    """把原始文本转换成训练时同样格式的 TF-IDF 矩阵。"""
    tokenized_docs = tokenize_corpus(texts)
    bow_matrix = corpus_to_bow_matrix(tokenized_docs, vocab)

    return transform_tfidf(bow_matrix, idf)


def predict_text(
    text: str,
    vocab: dict,
    idf: np.ndarray,
    w: np.ndarray,
    b: float,
    threshold: float = 0.5
) -> int:
    """使用训练好的参数预测一条原始文本。"""
    X = vectorize_texts([text], vocab, idf)
    proba = logistic_predict_proba(X, w, b)
    prediction = predict_labels(proba, threshold)

    return int(prediction[0])


def collect_prediction_errors(
    texts: list,
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """收集假阳性和假阴性的原始文本，保持输入顺序。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    false_positives = []
    false_negatives = []

    for text, true_label, predicted_label in zip(texts, y_true, y_pred):
        if true_label == 0 and predicted_label == 1:
            false_positives.append(text)
        elif true_label == 1 and predicted_label == 0:
            false_negatives.append(text)

    return {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }

# Step 3 - tokenize_corpus
import numpy as np


# ============================================================
# 1-3. 文本清洗与分词
# ============================================================

def clean_text(text: str) -> str:
    """
    转成小写，并把每个非字母字符替换为空格。
    注意：保留连续空格以及开头、结尾产生的空格。
    """
    text = text.lower()
    return "".join(ch if ch.isalpha() else " " for ch in text)


def tokenize(text: str) -> list:
    """按任意空白字符切分，并自动丢弃空 token。"""
    return text.split()


def tokenize_corpus(texts: list) -> list:
    """清洗并分词整个文本列表。"""
    return [tokenize(clean_text(text)) for text in texts]


# ============================================================
# 4-8. 数据划分、词频、词表与 Bag-of-Words
# ============================================================

def split_train_val_test_indices(
    n_samples: int,
    val_fraction: float,
    test_fraction: float,
    seed: int = 0
) -> tuple:
    """
    生成互不重叠的 train/validation/test 索引。
    按 train、validation、test 的顺序切分打乱后的索引。
    """
    np.random.seed(seed)

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    n_val = int(n_samples * val_fraction)
    n_test = int(n_samples * test_fraction)
    n_train = n_samples - n_val - n_test

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return train_idx, val_idx, test_idx


def count_word_frequencies(tokenized_docs: list) -> dict:
    """统计每个 token 在整个语料中出现的总次数。"""
    counts = {}

    for document in tokenized_docs:
        for token in document:
            counts[token] = counts.get(token, 0) + 1

    return counts


def build_vocabulary(word_counts: dict, max_size: int) -> dict:
    """
    选出词频最高的 max_size 个词。
    词频相同时，按字母顺序排列。
    """
    max_size = max(0, int(max_size))

    ranked_words = sorted(
        word_counts.items(),
        key=lambda item: (-item[1], item[0])
    )

    return {
        word: index
        for index, (word, _) in enumerate(ranked_words[:max_size])
    }


def tokens_to_bow(tokens: list, vocab: dict) -> np.ndarray:
    """将一篇文档转成词袋计数向量，未知词忽略。"""
    vector = np.zeros(len(vocab), dtype=float)

    for token in tokens:
        if token in vocab:
            vector[vocab[token]] += 1.0

    return vector


def corpus_to_bow_matrix(tokenized_docs: list, vocab: dict) -> np.ndarray:
    """将多个文档转成形状为 (文档数, 词表大小) 的词袋矩阵。"""
    matrix = np.zeros(
        (len(tokenized_docs), len(vocab)),
        dtype=float
    )

    for row, tokens in enumerate(tokenized_docs):
        matrix[row] = tokens_to_bow(tokens, vocab)

    return matrix


# ============================================================
# 9-12. 文档频率与 TF-IDF
# ============================================================

def compute_document_frequencies(bow_matrix: np.ndarray) -> np.ndarray:
    """
    计算每个词出现于多少篇文档。
    每篇文档中，一个词出现一次或多次，都只计为出现过。
    """
    bow_matrix = np.asarray(bow_matrix)
    return np.sum(bow_matrix > 0, axis=0).astype(int)


def compute_idf(df: np.ndarray, n_docs: int) -> np.ndarray:
    """计算平滑 IDF，使用自然对数。"""
    df = np.asarray(df, dtype=float)
    return np.log((n_docs + 1) / (df + 1)) + 1.0


def transform_tfidf(
    bow_matrix: np.ndarray,
    idf: np.ndarray
) -> np.ndarray:
    """
    每列的词袋计数乘对应 IDF。
    支持输入一个向量 (V,) 或矩阵 (N, V)。
    """
    bow_matrix = np.asarray(bow_matrix, dtype=float)
    idf = np.asarray(idf, dtype=float)
    return bow_matrix * idf


def fit_tfidf(bow_train: np.ndarray) -> np.ndarray:
    """只根据训练集词袋矩阵拟合 IDF。"""
    bow_train = np.asarray(bow_train, dtype=float)

    df = compute_document_frequencies(bow_train)
    n_docs = bow_train.shape[0]

    return compute_idf(df, n_docs)


# ============================================================
# 13-16. 逻辑回归核心：Sigmoid、概率、损失与梯度
# ============================================================

def sigmoid(z: np.ndarray) -> np.ndarray:
    """数值稳定的 sigmoid，返回与输入形状相同的数组。"""
    z = np.asarray(z, dtype=float)
    flat = z.reshape(-1)
    result = np.empty_like(flat)

    # 正数和负数分开计算，避免 exp 溢出
    positive = flat >= 0
    negative = ~positive

    result[positive] = 1.0 / (1.0 + np.exp(-flat[positive]))

    exp_z = np.exp(flat[negative])
    result[negative] = exp_z / (1.0 + exp_z)

    return result.reshape(z.shape)


def logistic_predict_proba(
    X: np.ndarray,
    w: np.ndarray,
    b: float
) -> np.ndarray:
    """计算每一行属于类别 1 的概率。"""
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)

    return sigmoid(X @ w + b)


def binary_cross_entropy(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    w: np.ndarray,
    l2_lambda: float
) -> float:
    """
    平均二元交叉熵 + L2 权重惩罚。
    bias 不在这里传入，所以不会被正则化。
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    w = np.asarray(w, dtype=float)

    # 避免 log(0)
    y_prob = np.clip(y_prob, 1e-15, 1.0 - 1e-15)

    if y_true.size == 0:
        mean_bce = 0.0
    else:
        mean_bce = -np.mean(
            y_true * np.log(y_prob)
            + (1.0 - y_true) * np.log(1.0 - y_prob)
        )

    l2_penalty = l2_lambda * np.sum(w ** 2) / 2.0

    return float(mean_bce + l2_penalty)


def logistic_gradients(
    X: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    w: np.ndarray,
    l2_lambda: float
) -> tuple:
    """计算平均 BCE + L2 对权重和 bias 的梯度。"""
    X = np.asarray(X, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    w = np.asarray(w, dtype=float)

    n_samples = X.shape[0]

    if n_samples == 0:
        return l2_lambda * w, 0.0

    residual = y_prob - y_true

    # dL/dw：数据损失梯度 + L2 梯度
    dw = (X.T @ residual) / n_samples + l2_lambda * w

    # bias 不加 L2 惩罚
    db = float(np.mean(residual))

    return dw, db


# ============================================================
# 17-20. 初始化、梯度下降训练与阈值预测
# ============================================================

def initialize_logistic_params(n_features: int) -> tuple:
    """权重和 bias 都从 0 开始。"""
    w = np.zeros(n_features, dtype=float)
    b = 0.0

    return w, b


def gradient_descent_step(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    b: float,
    lr: float,
    l2_lambda: float
) -> tuple:
    """
    做一次全批量梯度下降。
    返回的 loss 是更新参数之前计算的损失。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)

    probabilities = logistic_predict_proba(X, w, b)
    loss = binary_cross_entropy(y, probabilities, w, l2_lambda)

    dw, db = logistic_gradients(
        X, y, probabilities, w, l2_lambda
    )

    w_new = w - lr * dw
    b_new = float(b - lr * db)

    return w_new, b_new, loss


def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    lr: float,
    l2_lambda: float,
    n_epochs: int
) -> tuple:
    """从零开始训练逻辑回归，并记录每轮更新前的 loss。"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    w, b = initialize_logistic_params(X.shape[1])
    losses = []

    for _ in range(n_epochs):
        w, b, loss = gradient_descent_step(
            X, y, w, b, lr, l2_lambda
        )
        losses.append(float(loss))

    return w, float(b), losses


def predict_labels(
    proba: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """概率大于等于阈值时预测为 1，否则预测为 0。"""
    proba = np.asarray(proba, dtype=float)
    return (proba >= threshold).astype(int)


# ============================================================
# 21-24. 混淆矩阵计数、分类指标与阈值调优
# ============================================================

def confusion_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> tuple:
    """返回 (TP, FP, TN, FN)，均为 Python int。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    return tp, fp, tn, fn


def metrics_from_counts(
    tp: int,
    fp: int,
    tn: int,
    fn: int
) -> dict:
    """根据混淆矩阵计数计算 precision、recall、F1 和 accuracy。"""
    precision_denominator = tp + fp
    recall_denominator = tp + fn
    f1_denominator = 2 * tp + fp + fn
    total = tp + fp + tn + fn

    precision = (
        tp / precision_denominator
        if precision_denominator > 0 else 0.0
    )
    recall = (
        tp / recall_denominator
        if recall_denominator > 0 else 0.0
    )
    f1 = (
        2 * tp / f1_denominator
        if f1_denominator > 0 else 0.0
    )
    accuracy = (
        (tp + tn) / total
        if total > 0 else 0.0
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


def tune_decision_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    thresholds: np.ndarray = None
) -> tuple:
    """
    在候选阈值中选 F1 最高的那个。
    F1 并列时保留 thresholds 中最先出现的阈值。
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)

    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)
    else:
        thresholds = np.asarray(thresholds, dtype=float)

    if len(thresholds) == 0:
        return 0.0, 0.0

    best_threshold = float(thresholds[0])
    best_f1 = -1.0

    for threshold in thresholds:
        y_pred = predict_labels(proba, threshold)
        tp, fp, tn, fn = confusion_counts(y_true, y_pred)
        f1 = metrics_from_counts(tp, fp, tn, fn)["f1"]

        # 使用严格大于，保证并列时留下先遇到的阈值
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return float(best_threshold), float(best_f1)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """返回混淆矩阵计数和分类指标的汇总字典。"""
    tp, fp, tn, fn = confusion_counts(y_true, y_pred)
    metrics = metrics_from_counts(tp, fp, tn, fn)

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "accuracy": metrics["accuracy"],
    }


# ============================================================
# 25-27. 新文本向量化、单条预测与错误分析
# ============================================================

def vectorize_texts(
    texts: list,
    vocab: dict,
    idf: np.ndarray
) -> np.ndarray:
    """把原始文本转换成训练时同样格式的 TF-IDF 矩阵。"""
    tokenized_docs = tokenize_corpus(texts)
    bow_matrix = corpus_to_bow_matrix(tokenized_docs, vocab)

    return transform_tfidf(bow_matrix, idf)


def predict_text(
    text: str,
    vocab: dict,
    idf: np.ndarray,
    w: np.ndarray,
    b: float,
    threshold: float = 0.5
) -> int:
    """使用训练好的参数预测一条原始文本。"""
    X = vectorize_texts([text], vocab, idf)
    proba = logistic_predict_proba(X, w, b)
    prediction = predict_labels(proba, threshold)

    return int(prediction[0])


def collect_prediction_errors(
    texts: list,
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """收集假阳性和假阴性的原始文本，保持输入顺序。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    false_positives = []
    false_negatives = []

    for text, true_label, predicted_label in zip(texts, y_true, y_pred):
        if true_label == 0 and predicted_label == 1:
            false_positives.append(text)
        elif true_label == 1 and predicted_label == 0:
            false_negatives.append(text)

    return {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }

# Step 4 - split_train_val_test_indices
import numpy as np


# ============================================================
# 1-3. 文本清洗与分词
# ============================================================

def clean_text(text: str) -> str:
    """
    转成小写，并把每个非字母字符替换为空格。
    注意：保留连续空格以及开头、结尾产生的空格。
    """
    text = text.lower()
    return "".join(ch if ch.isalpha() else " " for ch in text)


def tokenize(text: str) -> list:
    """按任意空白字符切分，并自动丢弃空 token。"""
    return text.split()


def tokenize_corpus(texts: list) -> list:
    """清洗并分词整个文本列表。"""
    return [tokenize(clean_text(text)) for text in texts]


# ============================================================
# 4-8. 数据划分、词频、词表与 Bag-of-Words
# ============================================================

def split_train_val_test_indices(
    n_samples: int,
    val_fraction: float,
    test_fraction: float,
    seed: int = 0
) -> tuple:
    """
    生成互不重叠的 train/validation/test 索引。
    按 train、validation、test 的顺序切分打乱后的索引。
    """
    np.random.seed(seed)

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    n_val = int(n_samples * val_fraction)
    n_test = int(n_samples * test_fraction)
    n_train = n_samples - n_val - n_test

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return train_idx, val_idx, test_idx


def count_word_frequencies(tokenized_docs: list) -> dict:
    """统计每个 token 在整个语料中出现的总次数。"""
    counts = {}

    for document in tokenized_docs:
        for token in document:
            counts[token] = counts.get(token, 0) + 1

    return counts


def build_vocabulary(word_counts: dict, max_size: int) -> dict:
    """
    选出词频最高的 max_size 个词。
    词频相同时，按字母顺序排列。
    """
    max_size = max(0, int(max_size))

    ranked_words = sorted(
        word_counts.items(),
        key=lambda item: (-item[1], item[0])
    )

    return {
        word: index
        for index, (word, _) in enumerate(ranked_words[:max_size])
    }


def tokens_to_bow(tokens: list, vocab: dict) -> np.ndarray:
    """将一篇文档转成词袋计数向量，未知词忽略。"""
    vector = np.zeros(len(vocab), dtype=float)

    for token in tokens:
        if token in vocab:
            vector[vocab[token]] += 1.0

    return vector


def corpus_to_bow_matrix(tokenized_docs: list, vocab: dict) -> np.ndarray:
    """将多个文档转成形状为 (文档数, 词表大小) 的词袋矩阵。"""
    matrix = np.zeros(
        (len(tokenized_docs), len(vocab)),
        dtype=float
    )

    for row, tokens in enumerate(tokenized_docs):
        matrix[row] = tokens_to_bow(tokens, vocab)

    return matrix


# ============================================================
# 9-12. 文档频率与 TF-IDF
# ============================================================

def compute_document_frequencies(bow_matrix: np.ndarray) -> np.ndarray:
    """
    计算每个词出现于多少篇文档。
    每篇文档中，一个词出现一次或多次，都只计为出现过。
    """
    bow_matrix = np.asarray(bow_matrix)
    return np.sum(bow_matrix > 0, axis=0).astype(int)


def compute_idf(df: np.ndarray, n_docs: int) -> np.ndarray:
    """计算平滑 IDF，使用自然对数。"""
    df = np.asarray(df, dtype=float)
    return np.log((n_docs + 1) / (df + 1)) + 1.0


def transform_tfidf(
    bow_matrix: np.ndarray,
    idf: np.ndarray
) -> np.ndarray:
    """
    每列的词袋计数乘对应 IDF。
    支持输入一个向量 (V,) 或矩阵 (N, V)。
    """
    bow_matrix = np.asarray(bow_matrix, dtype=float)
    idf = np.asarray(idf, dtype=float)
    return bow_matrix * idf


def fit_tfidf(bow_train: np.ndarray) -> np.ndarray:
    """只根据训练集词袋矩阵拟合 IDF。"""
    bow_train = np.asarray(bow_train, dtype=float)

    df = compute_document_frequencies(bow_train)
    n_docs = bow_train.shape[0]

    return compute_idf(df, n_docs)


# ============================================================
# 13-16. 逻辑回归核心：Sigmoid、概率、损失与梯度
# ============================================================

def sigmoid(z: np.ndarray) -> np.ndarray:
    """数值稳定的 sigmoid，返回与输入形状相同的数组。"""
    z = np.asarray(z, dtype=float)
    flat = z.reshape(-1)
    result = np.empty_like(flat)

    # 正数和负数分开计算，避免 exp 溢出
    positive = flat >= 0
    negative = ~positive

    result[positive] = 1.0 / (1.0 + np.exp(-flat[positive]))

    exp_z = np.exp(flat[negative])
    result[negative] = exp_z / (1.0 + exp_z)

    return result.reshape(z.shape)


def logistic_predict_proba(
    X: np.ndarray,
    w: np.ndarray,
    b: float
) -> np.ndarray:
    """计算每一行属于类别 1 的概率。"""
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)

    return sigmoid(X @ w + b)


def binary_cross_entropy(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    w: np.ndarray,
    l2_lambda: float
) -> float:
    """
    平均二元交叉熵 + L2 权重惩罚。
    bias 不在这里传入，所以不会被正则化。
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    w = np.asarray(w, dtype=float)

    # 避免 log(0)
    y_prob = np.clip(y_prob, 1e-15, 1.0 - 1e-15)

    if y_true.size == 0:
        mean_bce = 0.0
    else:
        mean_bce = -np.mean(
            y_true * np.log(y_prob)
            + (1.0 - y_true) * np.log(1.0 - y_prob)
        )

    l2_penalty = l2_lambda * np.sum(w ** 2) / 2.0

    return float(mean_bce + l2_penalty)


def logistic_gradients(
    X: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    w: np.ndarray,
    l2_lambda: float
) -> tuple:
    """计算平均 BCE + L2 对权重和 bias 的梯度。"""
    X = np.asarray(X, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    w = np.asarray(w, dtype=float)

    n_samples = X.shape[0]

    if n_samples == 0:
        return l2_lambda * w, 0.0

    residual = y_prob - y_true

    # dL/dw：数据损失梯度 + L2 梯度
    dw = (X.T @ residual) / n_samples + l2_lambda * w

    # bias 不加 L2 惩罚
    db = float(np.mean(residual))

    return dw, db


# ============================================================
# 17-20. 初始化、梯度下降训练与阈值预测
# ============================================================

def initialize_logistic_params(n_features: int) -> tuple:
    """权重和 bias 都从 0 开始。"""
    w = np.zeros(n_features, dtype=float)
    b = 0.0

    return w, b


def gradient_descent_step(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    b: float,
    lr: float,
    l2_lambda: float
) -> tuple:
    """
    做一次全批量梯度下降。
    返回的 loss 是更新参数之前计算的损失。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)

    probabilities = logistic_predict_proba(X, w, b)
    loss = binary_cross_entropy(y, probabilities, w, l2_lambda)

    dw, db = logistic_gradients(
        X, y, probabilities, w, l2_lambda
    )

    w_new = w - lr * dw
    b_new = float(b - lr * db)

    return w_new, b_new, loss


def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    lr: float,
    l2_lambda: float,
    n_epochs: int
) -> tuple:
    """从零开始训练逻辑回归，并记录每轮更新前的 loss。"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    w, b = initialize_logistic_params(X.shape[1])
    losses = []

    for _ in range(n_epochs):
        w, b, loss = gradient_descent_step(
            X, y, w, b, lr, l2_lambda
        )
        losses.append(float(loss))

    return w, float(b), losses


def predict_labels(
    proba: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """概率大于等于阈值时预测为 1，否则预测为 0。"""
    proba = np.asarray(proba, dtype=float)
    return (proba >= threshold).astype(int)


# ============================================================
# 21-24. 混淆矩阵计数、分类指标与阈值调优
# ============================================================

def confusion_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> tuple:
    """返回 (TP, FP, TN, FN)，均为 Python int。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    return tp, fp, tn, fn


def metrics_from_counts(
    tp: int,
    fp: int,
    tn: int,
    fn: int
) -> dict:
    """根据混淆矩阵计数计算 precision、recall、F1 和 accuracy。"""
    precision_denominator = tp + fp
    recall_denominator = tp + fn
    f1_denominator = 2 * tp + fp + fn
    total = tp + fp + tn + fn

    precision = (
        tp / precision_denominator
        if precision_denominator > 0 else 0.0
    )
    recall = (
        tp / recall_denominator
        if recall_denominator > 0 else 0.0
    )
    f1 = (
        2 * tp / f1_denominator
        if f1_denominator > 0 else 0.0
    )
    accuracy = (
        (tp + tn) / total
        if total > 0 else 0.0
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


def tune_decision_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    thresholds: np.ndarray = None
) -> tuple:
    """
    在候选阈值中选 F1 最高的那个。
    F1 并列时保留 thresholds 中最先出现的阈值。
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)

    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)
    else:
        thresholds = np.asarray(thresholds, dtype=float)

    if len(thresholds) == 0:
        return 0.0, 0.0

    best_threshold = float(thresholds[0])
    best_f1 = -1.0

    for threshold in thresholds:
        y_pred = predict_labels(proba, threshold)
        tp, fp, tn, fn = confusion_counts(y_true, y_pred)
        f1 = metrics_from_counts(tp, fp, tn, fn)["f1"]

        # 使用严格大于，保证并列时留下先遇到的阈值
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return float(best_threshold), float(best_f1)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """返回混淆矩阵计数和分类指标的汇总字典。"""
    tp, fp, tn, fn = confusion_counts(y_true, y_pred)
    metrics = metrics_from_counts(tp, fp, tn, fn)

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "accuracy": metrics["accuracy"],
    }


# ============================================================
# 25-27. 新文本向量化、单条预测与错误分析
# ============================================================

def vectorize_texts(
    texts: list,
    vocab: dict,
    idf: np.ndarray
) -> np.ndarray:
    """把原始文本转换成训练时同样格式的 TF-IDF 矩阵。"""
    tokenized_docs = tokenize_corpus(texts)
    bow_matrix = corpus_to_bow_matrix(tokenized_docs, vocab)

    return transform_tfidf(bow_matrix, idf)


def predict_text(
    text: str,
    vocab: dict,
    idf: np.ndarray,
    w: np.ndarray,
    b: float,
    threshold: float = 0.5
) -> int:
    """使用训练好的参数预测一条原始文本。"""
    X = vectorize_texts([text], vocab, idf)
    proba = logistic_predict_proba(X, w, b)
    prediction = predict_labels(proba, threshold)

    return int(prediction[0])


def collect_prediction_errors(
    texts: list,
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """收集假阳性和假阴性的原始文本，保持输入顺序。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    false_positives = []
    false_negatives = []

    for text, true_label, predicted_label in zip(texts, y_true, y_pred):
        if true_label == 0 and predicted_label == 1:
            false_positives.append(text)
        elif true_label == 1 and predicted_label == 0:
            false_negatives.append(text)

    return {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }

# Step 5 - count_word_frequencies
import numpy as np


# ============================================================
# 1-3. 文本清洗与分词
# ============================================================

def clean_text(text: str) -> str:
    """
    转成小写，并把每个非字母字符替换为空格。
    注意：保留连续空格以及开头、结尾产生的空格。
    """
    text = text.lower()
    return "".join(ch if ch.isalpha() else " " for ch in text)


def tokenize(text: str) -> list:
    """按任意空白字符切分，并自动丢弃空 token。"""
    return text.split()


def tokenize_corpus(texts: list) -> list:
    """清洗并分词整个文本列表。"""
    return [tokenize(clean_text(text)) for text in texts]


# ============================================================
# 4-8. 数据划分、词频、词表与 Bag-of-Words
# ============================================================

def split_train_val_test_indices(
    n_samples: int,
    val_fraction: float,
    test_fraction: float,
    seed: int = 0
) -> tuple:
    """
    生成互不重叠的 train/validation/test 索引。
    按 train、validation、test 的顺序切分打乱后的索引。
    """
    np.random.seed(seed)

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    n_val = int(n_samples * val_fraction)
    n_test = int(n_samples * test_fraction)
    n_train = n_samples - n_val - n_test

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return train_idx, val_idx, test_idx


def count_word_frequencies(tokenized_docs: list) -> dict:
    """统计每个 token 在整个语料中出现的总次数。"""
    counts = {}

    for document in tokenized_docs:
        for token in document:
            counts[token] = counts.get(token, 0) + 1

    return counts


def build_vocabulary(word_counts: dict, max_size: int) -> dict:
    """
    选出词频最高的 max_size 个词。
    词频相同时，按字母顺序排列。
    """
    max_size = max(0, int(max_size))

    ranked_words = sorted(
        word_counts.items(),
        key=lambda item: (-item[1], item[0])
    )

    return {
        word: index
        for index, (word, _) in enumerate(ranked_words[:max_size])
    }


def tokens_to_bow(tokens: list, vocab: dict) -> np.ndarray:
    """将一篇文档转成词袋计数向量，未知词忽略。"""
    vector = np.zeros(len(vocab), dtype=float)

    for token in tokens:
        if token in vocab:
            vector[vocab[token]] += 1.0

    return vector


def corpus_to_bow_matrix(tokenized_docs: list, vocab: dict) -> np.ndarray:
    """将多个文档转成形状为 (文档数, 词表大小) 的词袋矩阵。"""
    matrix = np.zeros(
        (len(tokenized_docs), len(vocab)),
        dtype=float
    )

    for row, tokens in enumerate(tokenized_docs):
        matrix[row] = tokens_to_bow(tokens, vocab)

    return matrix


# ============================================================
# 9-12. 文档频率与 TF-IDF
# ============================================================

def compute_document_frequencies(bow_matrix: np.ndarray) -> np.ndarray:
    """
    计算每个词出现于多少篇文档。
    每篇文档中，一个词出现一次或多次，都只计为出现过。
    """
    bow_matrix = np.asarray(bow_matrix)
    return np.sum(bow_matrix > 0, axis=0).astype(int)


def compute_idf(df: np.ndarray, n_docs: int) -> np.ndarray:
    """计算平滑 IDF，使用自然对数。"""
    df = np.asarray(df, dtype=float)
    return np.log((n_docs + 1) / (df + 1)) + 1.0


def transform_tfidf(
    bow_matrix: np.ndarray,
    idf: np.ndarray
) -> np.ndarray:
    """
    每列的词袋计数乘对应 IDF。
    支持输入一个向量 (V,) 或矩阵 (N, V)。
    """
    bow_matrix = np.asarray(bow_matrix, dtype=float)
    idf = np.asarray(idf, dtype=float)
    return bow_matrix * idf


def fit_tfidf(bow_train: np.ndarray) -> np.ndarray:
    """只根据训练集词袋矩阵拟合 IDF。"""
    bow_train = np.asarray(bow_train, dtype=float)

    df = compute_document_frequencies(bow_train)
    n_docs = bow_train.shape[0]

    return compute_idf(df, n_docs)


# ============================================================
# 13-16. 逻辑回归核心：Sigmoid、概率、损失与梯度
# ============================================================

def sigmoid(z: np.ndarray) -> np.ndarray:
    """数值稳定的 sigmoid，返回与输入形状相同的数组。"""
    z = np.asarray(z, dtype=float)
    flat = z.reshape(-1)
    result = np.empty_like(flat)

    # 正数和负数分开计算，避免 exp 溢出
    positive = flat >= 0
    negative = ~positive

    result[positive] = 1.0 / (1.0 + np.exp(-flat[positive]))

    exp_z = np.exp(flat[negative])
    result[negative] = exp_z / (1.0 + exp_z)

    return result.reshape(z.shape)


def logistic_predict_proba(
    X: np.ndarray,
    w: np.ndarray,
    b: float
) -> np.ndarray:
    """计算每一行属于类别 1 的概率。"""
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)

    return sigmoid(X @ w + b)


def binary_cross_entropy(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    w: np.ndarray,
    l2_lambda: float
) -> float:
    """
    平均二元交叉熵 + L2 权重惩罚。
    bias 不在这里传入，所以不会被正则化。
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    w = np.asarray(w, dtype=float)

    # 避免 log(0)
    y_prob = np.clip(y_prob, 1e-15, 1.0 - 1e-15)

    if y_true.size == 0:
        mean_bce = 0.0
    else:
        mean_bce = -np.mean(
            y_true * np.log(y_prob)
            + (1.0 - y_true) * np.log(1.0 - y_prob)
        )

    l2_penalty = l2_lambda * np.sum(w ** 2) / 2.0

    return float(mean_bce + l2_penalty)


def logistic_gradients(
    X: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    w: np.ndarray,
    l2_lambda: float
) -> tuple:
    """计算平均 BCE + L2 对权重和 bias 的梯度。"""
    X = np.asarray(X, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    w = np.asarray(w, dtype=float)

    n_samples = X.shape[0]

    if n_samples == 0:
        return l2_lambda * w, 0.0

    residual = y_prob - y_true

    # dL/dw：数据损失梯度 + L2 梯度
    dw = (X.T @ residual) / n_samples + l2_lambda * w

    # bias 不加 L2 惩罚
    db = float(np.mean(residual))

    return dw, db


# ============================================================
# 17-20. 初始化、梯度下降训练与阈值预测
# ============================================================

def initialize_logistic_params(n_features: int) -> tuple:
    """权重和 bias 都从 0 开始。"""
    w = np.zeros(n_features, dtype=float)
    b = 0.0

    return w, b


def gradient_descent_step(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    b: float,
    lr: float,
    l2_lambda: float
) -> tuple:
    """
    做一次全批量梯度下降。
    返回的 loss 是更新参数之前计算的损失。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)

    probabilities = logistic_predict_proba(X, w, b)
    loss = binary_cross_entropy(y, probabilities, w, l2_lambda)

    dw, db = logistic_gradients(
        X, y, probabilities, w, l2_lambda
    )

    w_new = w - lr * dw
    b_new = float(b - lr * db)

    return w_new, b_new, loss


def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    lr: float,
    l2_lambda: float,
    n_epochs: int
) -> tuple:
    """从零开始训练逻辑回归，并记录每轮更新前的 loss。"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    w, b = initialize_logistic_params(X.shape[1])
    losses = []

    for _ in range(n_epochs):
        w, b, loss = gradient_descent_step(
            X, y, w, b, lr, l2_lambda
        )
        losses.append(float(loss))

    return w, float(b), losses


def predict_labels(
    proba: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """概率大于等于阈值时预测为 1，否则预测为 0。"""
    proba = np.asarray(proba, dtype=float)
    return (proba >= threshold).astype(int)


# ============================================================
# 21-24. 混淆矩阵计数、分类指标与阈值调优
# ============================================================

def confusion_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> tuple:
    """返回 (TP, FP, TN, FN)，均为 Python int。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    return tp, fp, tn, fn


def metrics_from_counts(
    tp: int,
    fp: int,
    tn: int,
    fn: int
) -> dict:
    """根据混淆矩阵计数计算 precision、recall、F1 和 accuracy。"""
    precision_denominator = tp + fp
    recall_denominator = tp + fn
    f1_denominator = 2 * tp + fp + fn
    total = tp + fp + tn + fn

    precision = (
        tp / precision_denominator
        if precision_denominator > 0 else 0.0
    )
    recall = (
        tp / recall_denominator
        if recall_denominator > 0 else 0.0
    )
    f1 = (
        2 * tp / f1_denominator
        if f1_denominator > 0 else 0.0
    )
    accuracy = (
        (tp + tn) / total
        if total > 0 else 0.0
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


def tune_decision_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    thresholds: np.ndarray = None
) -> tuple:
    """
    在候选阈值中选 F1 最高的那个。
    F1 并列时保留 thresholds 中最先出现的阈值。
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)

    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)
    else:
        thresholds = np.asarray(thresholds, dtype=float)

    if len(thresholds) == 0:
        return 0.0, 0.0

    best_threshold = float(thresholds[0])
    best_f1 = -1.0

    for threshold in thresholds:
        y_pred = predict_labels(proba, threshold)
        tp, fp, tn, fn = confusion_counts(y_true, y_pred)
        f1 = metrics_from_counts(tp, fp, tn, fn)["f1"]

        # 使用严格大于，保证并列时留下先遇到的阈值
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return float(best_threshold), float(best_f1)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """返回混淆矩阵计数和分类指标的汇总字典。"""
    tp, fp, tn, fn = confusion_counts(y_true, y_pred)
    metrics = metrics_from_counts(tp, fp, tn, fn)

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "accuracy": metrics["accuracy"],
    }


# ============================================================
# 25-27. 新文本向量化、单条预测与错误分析
# ============================================================

def vectorize_texts(
    texts: list,
    vocab: dict,
    idf: np.ndarray
) -> np.ndarray:
    """把原始文本转换成训练时同样格式的 TF-IDF 矩阵。"""
    tokenized_docs = tokenize_corpus(texts)
    bow_matrix = corpus_to_bow_matrix(tokenized_docs, vocab)

    return transform_tfidf(bow_matrix, idf)


def predict_text(
    text: str,
    vocab: dict,
    idf: np.ndarray,
    w: np.ndarray,
    b: float,
    threshold: float = 0.5
) -> int:
    """使用训练好的参数预测一条原始文本。"""
    X = vectorize_texts([text], vocab, idf)
    proba = logistic_predict_proba(X, w, b)
    prediction = predict_labels(proba, threshold)

    return int(prediction[0])


def collect_prediction_errors(
    texts: list,
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """收集假阳性和假阴性的原始文本，保持输入顺序。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    false_positives = []
    false_negatives = []

    for text, true_label, predicted_label in zip(texts, y_true, y_pred):
        if true_label == 0 and predicted_label == 1:
            false_positives.append(text)
        elif true_label == 1 and predicted_label == 0:
            false_negatives.append(text)

    return {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }

# Step 6 - build_vocabulary
import numpy as np


# ============================================================
# 1-3. 文本清洗与分词
# ============================================================

def clean_text(text: str) -> str:
    """
    转成小写，并把每个非字母字符替换为空格。
    注意：保留连续空格以及开头、结尾产生的空格。
    """
    text = text.lower()
    return "".join(ch if ch.isalpha() else " " for ch in text)


def tokenize(text: str) -> list:
    """按任意空白字符切分，并自动丢弃空 token。"""
    return text.split()


def tokenize_corpus(texts: list) -> list:
    """清洗并分词整个文本列表。"""
    return [tokenize(clean_text(text)) for text in texts]


# ============================================================
# 4-8. 数据划分、词频、词表与 Bag-of-Words
# ============================================================

def split_train_val_test_indices(
    n_samples: int,
    val_fraction: float,
    test_fraction: float,
    seed: int = 0
) -> tuple:
    """
    生成互不重叠的 train/validation/test 索引。
    按 train、validation、test 的顺序切分打乱后的索引。
    """
    np.random.seed(seed)

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    n_val = int(n_samples * val_fraction)
    n_test = int(n_samples * test_fraction)
    n_train = n_samples - n_val - n_test

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return train_idx, val_idx, test_idx


def count_word_frequencies(tokenized_docs: list) -> dict:
    """统计每个 token 在整个语料中出现的总次数。"""
    counts = {}

    for document in tokenized_docs:
        for token in document:
            counts[token] = counts.get(token, 0) + 1

    return counts


def build_vocabulary(word_counts: dict, max_size: int) -> dict:
    """
    选出词频最高的 max_size 个词。
    词频相同时，按字母顺序排列。
    """
    max_size = max(0, int(max_size))

    ranked_words = sorted(
        word_counts.items(),
        key=lambda item: (-item[1], item[0])
    )

    return {
        word: index
        for index, (word, _) in enumerate(ranked_words[:max_size])
    }


def tokens_to_bow(tokens: list, vocab: dict) -> np.ndarray:
    """将一篇文档转成词袋计数向量，未知词忽略。"""
    vector = np.zeros(len(vocab), dtype=float)

    for token in tokens:
        if token in vocab:
            vector[vocab[token]] += 1.0

    return vector


def corpus_to_bow_matrix(tokenized_docs: list, vocab: dict) -> np.ndarray:
    """将多个文档转成形状为 (文档数, 词表大小) 的词袋矩阵。"""
    matrix = np.zeros(
        (len(tokenized_docs), len(vocab)),
        dtype=float
    )

    for row, tokens in enumerate(tokenized_docs):
        matrix[row] = tokens_to_bow(tokens, vocab)

    return matrix


# ============================================================
# 9-12. 文档频率与 TF-IDF
# ============================================================

def compute_document_frequencies(bow_matrix: np.ndarray) -> np.ndarray:
    """
    计算每个词出现于多少篇文档。
    每篇文档中，一个词出现一次或多次，都只计为出现过。
    """
    bow_matrix = np.asarray(bow_matrix)
    return np.sum(bow_matrix > 0, axis=0).astype(int)


def compute_idf(df: np.ndarray, n_docs: int) -> np.ndarray:
    """计算平滑 IDF，使用自然对数。"""
    df = np.asarray(df, dtype=float)
    return np.log((n_docs + 1) / (df + 1)) + 1.0


def transform_tfidf(
    bow_matrix: np.ndarray,
    idf: np.ndarray
) -> np.ndarray:
    """
    每列的词袋计数乘对应 IDF。
    支持输入一个向量 (V,) 或矩阵 (N, V)。
    """
    bow_matrix = np.asarray(bow_matrix, dtype=float)
    idf = np.asarray(idf, dtype=float)
    return bow_matrix * idf


def fit_tfidf(bow_train: np.ndarray) -> np.ndarray:
    """只根据训练集词袋矩阵拟合 IDF。"""
    bow_train = np.asarray(bow_train, dtype=float)

    df = compute_document_frequencies(bow_train)
    n_docs = bow_train.shape[0]

    return compute_idf(df, n_docs)


# ============================================================
# 13-16. 逻辑回归核心：Sigmoid、概率、损失与梯度
# ============================================================

def sigmoid(z: np.ndarray) -> np.ndarray:
    """数值稳定的 sigmoid，返回与输入形状相同的数组。"""
    z = np.asarray(z, dtype=float)
    flat = z.reshape(-1)
    result = np.empty_like(flat)

    # 正数和负数分开计算，避免 exp 溢出
    positive = flat >= 0
    negative = ~positive

    result[positive] = 1.0 / (1.0 + np.exp(-flat[positive]))

    exp_z = np.exp(flat[negative])
    result[negative] = exp_z / (1.0 + exp_z)

    return result.reshape(z.shape)


def logistic_predict_proba(
    X: np.ndarray,
    w: np.ndarray,
    b: float
) -> np.ndarray:
    """计算每一行属于类别 1 的概率。"""
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)

    return sigmoid(X @ w + b)


def binary_cross_entropy(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    w: np.ndarray,
    l2_lambda: float
) -> float:
    """
    平均二元交叉熵 + L2 权重惩罚。
    bias 不在这里传入，所以不会被正则化。
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    w = np.asarray(w, dtype=float)

    # 避免 log(0)
    y_prob = np.clip(y_prob, 1e-15, 1.0 - 1e-15)

    if y_true.size == 0:
        mean_bce = 0.0
    else:
        mean_bce = -np.mean(
            y_true * np.log(y_prob)
            + (1.0 - y_true) * np.log(1.0 - y_prob)
        )

    l2_penalty = l2_lambda * np.sum(w ** 2) / 2.0

    return float(mean_bce + l2_penalty)


def logistic_gradients(
    X: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    w: np.ndarray,
    l2_lambda: float
) -> tuple:
    """计算平均 BCE + L2 对权重和 bias 的梯度。"""
    X = np.asarray(X, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    w = np.asarray(w, dtype=float)

    n_samples = X.shape[0]

    if n_samples == 0:
        return l2_lambda * w, 0.0

    residual = y_prob - y_true

    # dL/dw：数据损失梯度 + L2 梯度
    dw = (X.T @ residual) / n_samples + l2_lambda * w

    # bias 不加 L2 惩罚
    db = float(np.mean(residual))

    return dw, db


# ============================================================
# 17-20. 初始化、梯度下降训练与阈值预测
# ============================================================

def initialize_logistic_params(n_features: int) -> tuple:
    """权重和 bias 都从 0 开始。"""
    w = np.zeros(n_features, dtype=float)
    b = 0.0

    return w, b


def gradient_descent_step(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    b: float,
    lr: float,
    l2_lambda: float
) -> tuple:
    """
    做一次全批量梯度下降。
    返回的 loss 是更新参数之前计算的损失。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)

    probabilities = logistic_predict_proba(X, w, b)
    loss = binary_cross_entropy(y, probabilities, w, l2_lambda)

    dw, db = logistic_gradients(
        X, y, probabilities, w, l2_lambda
    )

    w_new = w - lr * dw
    b_new = float(b - lr * db)

    return w_new, b_new, loss


def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    lr: float,
    l2_lambda: float,
    n_epochs: int
) -> tuple:
    """从零开始训练逻辑回归，并记录每轮更新前的 loss。"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    w, b = initialize_logistic_params(X.shape[1])
    losses = []

    for _ in range(n_epochs):
        w, b, loss = gradient_descent_step(
            X, y, w, b, lr, l2_lambda
        )
        losses.append(float(loss))

    return w, float(b), losses


def predict_labels(
    proba: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """概率大于等于阈值时预测为 1，否则预测为 0。"""
    proba = np.asarray(proba, dtype=float)
    return (proba >= threshold).astype(int)


# ============================================================
# 21-24. 混淆矩阵计数、分类指标与阈值调优
# ============================================================

def confusion_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> tuple:
    """返回 (TP, FP, TN, FN)，均为 Python int。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    return tp, fp, tn, fn


def metrics_from_counts(
    tp: int,
    fp: int,
    tn: int,
    fn: int
) -> dict:
    """根据混淆矩阵计数计算 precision、recall、F1 和 accuracy。"""
    precision_denominator = tp + fp
    recall_denominator = tp + fn
    f1_denominator = 2 * tp + fp + fn
    total = tp + fp + tn + fn

    precision = (
        tp / precision_denominator
        if precision_denominator > 0 else 0.0
    )
    recall = (
        tp / recall_denominator
        if recall_denominator > 0 else 0.0
    )
    f1 = (
        2 * tp / f1_denominator
        if f1_denominator > 0 else 0.0
    )
    accuracy = (
        (tp + tn) / total
        if total > 0 else 0.0
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


def tune_decision_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    thresholds: np.ndarray = None
) -> tuple:
    """
    在候选阈值中选 F1 最高的那个。
    F1 并列时保留 thresholds 中最先出现的阈值。
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)

    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)
    else:
        thresholds = np.asarray(thresholds, dtype=float)

    if len(thresholds) == 0:
        return 0.0, 0.0

    best_threshold = float(thresholds[0])
    best_f1 = -1.0

    for threshold in thresholds:
        y_pred = predict_labels(proba, threshold)
        tp, fp, tn, fn = confusion_counts(y_true, y_pred)
        f1 = metrics_from_counts(tp, fp, tn, fn)["f1"]

        # 使用严格大于，保证并列时留下先遇到的阈值
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return float(best_threshold), float(best_f1)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """返回混淆矩阵计数和分类指标的汇总字典。"""
    tp, fp, tn, fn = confusion_counts(y_true, y_pred)
    metrics = metrics_from_counts(tp, fp, tn, fn)

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "accuracy": metrics["accuracy"],
    }


# ============================================================
# 25-27. 新文本向量化、单条预测与错误分析
# ============================================================

def vectorize_texts(
    texts: list,
    vocab: dict,
    idf: np.ndarray
) -> np.ndarray:
    """把原始文本转换成训练时同样格式的 TF-IDF 矩阵。"""
    tokenized_docs = tokenize_corpus(texts)
    bow_matrix = corpus_to_bow_matrix(tokenized_docs, vocab)

    return transform_tfidf(bow_matrix, idf)


def predict_text(
    text: str,
    vocab: dict,
    idf: np.ndarray,
    w: np.ndarray,
    b: float,
    threshold: float = 0.5
) -> int:
    """使用训练好的参数预测一条原始文本。"""
    X = vectorize_texts([text], vocab, idf)
    proba = logistic_predict_proba(X, w, b)
    prediction = predict_labels(proba, threshold)

    return int(prediction[0])


def collect_prediction_errors(
    texts: list,
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """收集假阳性和假阴性的原始文本，保持输入顺序。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    false_positives = []
    false_negatives = []

    for text, true_label, predicted_label in zip(texts, y_true, y_pred):
        if true_label == 0 and predicted_label == 1:
            false_positives.append(text)
        elif true_label == 1 and predicted_label == 0:
            false_negatives.append(text)

    return {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }

# Step 7 - tokens_to_bow
import numpy as np


# ============================================================
# 1-3. 文本清洗与分词
# ============================================================

def clean_text(text: str) -> str:
    """
    转成小写，并把每个非字母字符替换为空格。
    注意：保留连续空格以及开头、结尾产生的空格。
    """
    text = text.lower()
    return "".join(ch if ch.isalpha() else " " for ch in text)


def tokenize(text: str) -> list:
    """按任意空白字符切分，并自动丢弃空 token。"""
    return text.split()


def tokenize_corpus(texts: list) -> list:
    """清洗并分词整个文本列表。"""
    return [tokenize(clean_text(text)) for text in texts]


# ============================================================
# 4-8. 数据划分、词频、词表与 Bag-of-Words
# ============================================================

def split_train_val_test_indices(
    n_samples: int,
    val_fraction: float,
    test_fraction: float,
    seed: int = 0
) -> tuple:
    """
    生成互不重叠的 train/validation/test 索引。
    按 train、validation、test 的顺序切分打乱后的索引。
    """
    np.random.seed(seed)

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    n_val = int(n_samples * val_fraction)
    n_test = int(n_samples * test_fraction)
    n_train = n_samples - n_val - n_test

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return train_idx, val_idx, test_idx


def count_word_frequencies(tokenized_docs: list) -> dict:
    """统计每个 token 在整个语料中出现的总次数。"""
    counts = {}

    for document in tokenized_docs:
        for token in document:
            counts[token] = counts.get(token, 0) + 1

    return counts


def build_vocabulary(word_counts: dict, max_size: int) -> dict:
    """
    选出词频最高的 max_size 个词。
    词频相同时，按字母顺序排列。
    """
    max_size = max(0, int(max_size))

    ranked_words = sorted(
        word_counts.items(),
        key=lambda item: (-item[1], item[0])
    )

    return {
        word: index
        for index, (word, _) in enumerate(ranked_words[:max_size])
    }


def tokens_to_bow(tokens: list, vocab: dict) -> np.ndarray:
    """将一篇文档转成词袋计数向量，未知词忽略。"""
    vector = np.zeros(len(vocab), dtype=float)

    for token in tokens:
        if token in vocab:
            vector[vocab[token]] += 1.0

    return vector


def corpus_to_bow_matrix(tokenized_docs: list, vocab: dict) -> np.ndarray:
    """将多个文档转成形状为 (文档数, 词表大小) 的词袋矩阵。"""
    matrix = np.zeros(
        (len(tokenized_docs), len(vocab)),
        dtype=float
    )

    for row, tokens in enumerate(tokenized_docs):
        matrix[row] = tokens_to_bow(tokens, vocab)

    return matrix


# ============================================================
# 9-12. 文档频率与 TF-IDF
# ============================================================

def compute_document_frequencies(bow_matrix: np.ndarray) -> np.ndarray:
    """
    计算每个词出现于多少篇文档。
    每篇文档中，一个词出现一次或多次，都只计为出现过。
    """
    bow_matrix = np.asarray(bow_matrix)
    return np.sum(bow_matrix > 0, axis=0).astype(int)


def compute_idf(df: np.ndarray, n_docs: int) -> np.ndarray:
    """计算平滑 IDF，使用自然对数。"""
    df = np.asarray(df, dtype=float)
    return np.log((n_docs + 1) / (df + 1)) + 1.0


def transform_tfidf(
    bow_matrix: np.ndarray,
    idf: np.ndarray
) -> np.ndarray:
    """
    每列的词袋计数乘对应 IDF。
    支持输入一个向量 (V,) 或矩阵 (N, V)。
    """
    bow_matrix = np.asarray(bow_matrix, dtype=float)
    idf = np.asarray(idf, dtype=float)
    return bow_matrix * idf


def fit_tfidf(bow_train: np.ndarray) -> np.ndarray:
    """只根据训练集词袋矩阵拟合 IDF。"""
    bow_train = np.asarray(bow_train, dtype=float)

    df = compute_document_frequencies(bow_train)
    n_docs = bow_train.shape[0]

    return compute_idf(df, n_docs)


# ============================================================
# 13-16. 逻辑回归核心：Sigmoid、概率、损失与梯度
# ============================================================

def sigmoid(z: np.ndarray) -> np.ndarray:
    """数值稳定的 sigmoid，返回与输入形状相同的数组。"""
    z = np.asarray(z, dtype=float)
    flat = z.reshape(-1)
    result = np.empty_like(flat)

    # 正数和负数分开计算，避免 exp 溢出
    positive = flat >= 0
    negative = ~positive

    result[positive] = 1.0 / (1.0 + np.exp(-flat[positive]))

    exp_z = np.exp(flat[negative])
    result[negative] = exp_z / (1.0 + exp_z)

    return result.reshape(z.shape)


def logistic_predict_proba(
    X: np.ndarray,
    w: np.ndarray,
    b: float
) -> np.ndarray:
    """计算每一行属于类别 1 的概率。"""
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)

    return sigmoid(X @ w + b)


def binary_cross_entropy(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    w: np.ndarray,
    l2_lambda: float
) -> float:
    """
    平均二元交叉熵 + L2 权重惩罚。
    bias 不在这里传入，所以不会被正则化。
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    w = np.asarray(w, dtype=float)

    # 避免 log(0)
    y_prob = np.clip(y_prob, 1e-15, 1.0 - 1e-15)

    if y_true.size == 0:
        mean_bce = 0.0
    else:
        mean_bce = -np.mean(
            y_true * np.log(y_prob)
            + (1.0 - y_true) * np.log(1.0 - y_prob)
        )

    l2_penalty = l2_lambda * np.sum(w ** 2) / 2.0

    return float(mean_bce + l2_penalty)


def logistic_gradients(
    X: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    w: np.ndarray,
    l2_lambda: float
) -> tuple:
    """计算平均 BCE + L2 对权重和 bias 的梯度。"""
    X = np.asarray(X, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    w = np.asarray(w, dtype=float)

    n_samples = X.shape[0]

    if n_samples == 0:
        return l2_lambda * w, 0.0

    residual = y_prob - y_true

    # dL/dw：数据损失梯度 + L2 梯度
    dw = (X.T @ residual) / n_samples + l2_lambda * w

    # bias 不加 L2 惩罚
    db = float(np.mean(residual))

    return dw, db


# ============================================================
# 17-20. 初始化、梯度下降训练与阈值预测
# ============================================================

def initialize_logistic_params(n_features: int) -> tuple:
    """权重和 bias 都从 0 开始。"""
    w = np.zeros(n_features, dtype=float)
    b = 0.0

    return w, b


def gradient_descent_step(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    b: float,
    lr: float,
    l2_lambda: float
) -> tuple:
    """
    做一次全批量梯度下降。
    返回的 loss 是更新参数之前计算的损失。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)

    probabilities = logistic_predict_proba(X, w, b)
    loss = binary_cross_entropy(y, probabilities, w, l2_lambda)

    dw, db = logistic_gradients(
        X, y, probabilities, w, l2_lambda
    )

    w_new = w - lr * dw
    b_new = float(b - lr * db)

    return w_new, b_new, loss


def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    lr: float,
    l2_lambda: float,
    n_epochs: int
) -> tuple:
    """从零开始训练逻辑回归，并记录每轮更新前的 loss。"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    w, b = initialize_logistic_params(X.shape[1])
    losses = []

    for _ in range(n_epochs):
        w, b, loss = gradient_descent_step(
            X, y, w, b, lr, l2_lambda
        )
        losses.append(float(loss))

    return w, float(b), losses


def predict_labels(
    proba: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """概率大于等于阈值时预测为 1，否则预测为 0。"""
    proba = np.asarray(proba, dtype=float)
    return (proba >= threshold).astype(int)


# ============================================================
# 21-24. 混淆矩阵计数、分类指标与阈值调优
# ============================================================

def confusion_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> tuple:
    """返回 (TP, FP, TN, FN)，均为 Python int。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    return tp, fp, tn, fn


def metrics_from_counts(
    tp: int,
    fp: int,
    tn: int,
    fn: int
) -> dict:
    """根据混淆矩阵计数计算 precision、recall、F1 和 accuracy。"""
    precision_denominator = tp + fp
    recall_denominator = tp + fn
    f1_denominator = 2 * tp + fp + fn
    total = tp + fp + tn + fn

    precision = (
        tp / precision_denominator
        if precision_denominator > 0 else 0.0
    )
    recall = (
        tp / recall_denominator
        if recall_denominator > 0 else 0.0
    )
    f1 = (
        2 * tp / f1_denominator
        if f1_denominator > 0 else 0.0
    )
    accuracy = (
        (tp + tn) / total
        if total > 0 else 0.0
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


def tune_decision_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    thresholds: np.ndarray = None
) -> tuple:
    """
    在候选阈值中选 F1 最高的那个。
    F1 并列时保留 thresholds 中最先出现的阈值。
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)

    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)
    else:
        thresholds = np.asarray(thresholds, dtype=float)

    if len(thresholds) == 0:
        return 0.0, 0.0

    best_threshold = float(thresholds[0])
    best_f1 = -1.0

    for threshold in thresholds:
        y_pred = predict_labels(proba, threshold)
        tp, fp, tn, fn = confusion_counts(y_true, y_pred)
        f1 = metrics_from_counts(tp, fp, tn, fn)["f1"]

        # 使用严格大于，保证并列时留下先遇到的阈值
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return float(best_threshold), float(best_f1)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """返回混淆矩阵计数和分类指标的汇总字典。"""
    tp, fp, tn, fn = confusion_counts(y_true, y_pred)
    metrics = metrics_from_counts(tp, fp, tn, fn)

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "accuracy": metrics["accuracy"],
    }


# ============================================================
# 25-27. 新文本向量化、单条预测与错误分析
# ============================================================

def vectorize_texts(
    texts: list,
    vocab: dict,
    idf: np.ndarray
) -> np.ndarray:
    """把原始文本转换成训练时同样格式的 TF-IDF 矩阵。"""
    tokenized_docs = tokenize_corpus(texts)
    bow_matrix = corpus_to_bow_matrix(tokenized_docs, vocab)

    return transform_tfidf(bow_matrix, idf)


def predict_text(
    text: str,
    vocab: dict,
    idf: np.ndarray,
    w: np.ndarray,
    b: float,
    threshold: float = 0.5
) -> int:
    """使用训练好的参数预测一条原始文本。"""
    X = vectorize_texts([text], vocab, idf)
    proba = logistic_predict_proba(X, w, b)
    prediction = predict_labels(proba, threshold)

    return int(prediction[0])


def collect_prediction_errors(
    texts: list,
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """收集假阳性和假阴性的原始文本，保持输入顺序。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    false_positives = []
    false_negatives = []

    for text, true_label, predicted_label in zip(texts, y_true, y_pred):
        if true_label == 0 and predicted_label == 1:
            false_positives.append(text)
        elif true_label == 1 and predicted_label == 0:
            false_negatives.append(text)

    return {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }

# Step 8 - corpus_to_bow_matrix
import numpy as np


# ============================================================
# 1-3. 文本清洗与分词
# ============================================================

def clean_text(text: str) -> str:
    """
    转成小写，并把每个非字母字符替换为空格。
    注意：保留连续空格以及开头、结尾产生的空格。
    """
    text = text.lower()
    return "".join(ch if ch.isalpha() else " " for ch in text)


def tokenize(text: str) -> list:
    """按任意空白字符切分，并自动丢弃空 token。"""
    return text.split()


def tokenize_corpus(texts: list) -> list:
    """清洗并分词整个文本列表。"""
    return [tokenize(clean_text(text)) for text in texts]


# ============================================================
# 4-8. 数据划分、词频、词表与 Bag-of-Words
# ============================================================

def split_train_val_test_indices(
    n_samples: int,
    val_fraction: float,
    test_fraction: float,
    seed: int = 0
) -> tuple:
    """
    生成互不重叠的 train/validation/test 索引。
    按 train、validation、test 的顺序切分打乱后的索引。
    """
    np.random.seed(seed)

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    n_val = int(n_samples * val_fraction)
    n_test = int(n_samples * test_fraction)
    n_train = n_samples - n_val - n_test

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return train_idx, val_idx, test_idx


def count_word_frequencies(tokenized_docs: list) -> dict:
    """统计每个 token 在整个语料中出现的总次数。"""
    counts = {}

    for document in tokenized_docs:
        for token in document:
            counts[token] = counts.get(token, 0) + 1

    return counts


def build_vocabulary(word_counts: dict, max_size: int) -> dict:
    """
    选出词频最高的 max_size 个词。
    词频相同时，按字母顺序排列。
    """
    max_size = max(0, int(max_size))

    ranked_words = sorted(
        word_counts.items(),
        key=lambda item: (-item[1], item[0])
    )

    return {
        word: index
        for index, (word, _) in enumerate(ranked_words[:max_size])
    }


def tokens_to_bow(tokens: list, vocab: dict) -> np.ndarray:
    """将一篇文档转成词袋计数向量，未知词忽略。"""
    vector = np.zeros(len(vocab), dtype=float)

    for token in tokens:
        if token in vocab:
            vector[vocab[token]] += 1.0

    return vector


def corpus_to_bow_matrix(tokenized_docs: list, vocab: dict) -> np.ndarray:
    """将多个文档转成形状为 (文档数, 词表大小) 的词袋矩阵。"""
    matrix = np.zeros(
        (len(tokenized_docs), len(vocab)),
        dtype=float
    )

    for row, tokens in enumerate(tokenized_docs):
        matrix[row] = tokens_to_bow(tokens, vocab)

    return matrix


# ============================================================
# 9-12. 文档频率与 TF-IDF
# ============================================================

def compute_document_frequencies(bow_matrix: np.ndarray) -> np.ndarray:
    """
    计算每个词出现于多少篇文档。
    每篇文档中，一个词出现一次或多次，都只计为出现过。
    """
    bow_matrix = np.asarray(bow_matrix)
    return np.sum(bow_matrix > 0, axis=0).astype(int)


def compute_idf(df: np.ndarray, n_docs: int) -> np.ndarray:
    """计算平滑 IDF，使用自然对数。"""
    df = np.asarray(df, dtype=float)
    return np.log((n_docs + 1) / (df + 1)) + 1.0


def transform_tfidf(
    bow_matrix: np.ndarray,
    idf: np.ndarray
) -> np.ndarray:
    """
    每列的词袋计数乘对应 IDF。
    支持输入一个向量 (V,) 或矩阵 (N, V)。
    """
    bow_matrix = np.asarray(bow_matrix, dtype=float)
    idf = np.asarray(idf, dtype=float)
    return bow_matrix * idf


def fit_tfidf(bow_train: np.ndarray) -> np.ndarray:
    """只根据训练集词袋矩阵拟合 IDF。"""
    bow_train = np.asarray(bow_train, dtype=float)

    df = compute_document_frequencies(bow_train)
    n_docs = bow_train.shape[0]

    return compute_idf(df, n_docs)


# ============================================================
# 13-16. 逻辑回归核心：Sigmoid、概率、损失与梯度
# ============================================================

def sigmoid(z: np.ndarray) -> np.ndarray:
    """数值稳定的 sigmoid，返回与输入形状相同的数组。"""
    z = np.asarray(z, dtype=float)
    flat = z.reshape(-1)
    result = np.empty_like(flat)

    # 正数和负数分开计算，避免 exp 溢出
    positive = flat >= 0
    negative = ~positive

    result[positive] = 1.0 / (1.0 + np.exp(-flat[positive]))

    exp_z = np.exp(flat[negative])
    result[negative] = exp_z / (1.0 + exp_z)

    return result.reshape(z.shape)


def logistic_predict_proba(
    X: np.ndarray,
    w: np.ndarray,
    b: float
) -> np.ndarray:
    """计算每一行属于类别 1 的概率。"""
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)

    return sigmoid(X @ w + b)


def binary_cross_entropy(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    w: np.ndarray,
    l2_lambda: float
) -> float:
    """
    平均二元交叉熵 + L2 权重惩罚。
    bias 不在这里传入，所以不会被正则化。
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    w = np.asarray(w, dtype=float)

    # 避免 log(0)
    y_prob = np.clip(y_prob, 1e-15, 1.0 - 1e-15)

    if y_true.size == 0:
        mean_bce = 0.0
    else:
        mean_bce = -np.mean(
            y_true * np.log(y_prob)
            + (1.0 - y_true) * np.log(1.0 - y_prob)
        )

    l2_penalty = l2_lambda * np.sum(w ** 2) / 2.0

    return float(mean_bce + l2_penalty)


def logistic_gradients(
    X: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    w: np.ndarray,
    l2_lambda: float
) -> tuple:
    """计算平均 BCE + L2 对权重和 bias 的梯度。"""
    X = np.asarray(X, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    w = np.asarray(w, dtype=float)

    n_samples = X.shape[0]

    if n_samples == 0:
        return l2_lambda * w, 0.0

    residual = y_prob - y_true

    # dL/dw：数据损失梯度 + L2 梯度
    dw = (X.T @ residual) / n_samples + l2_lambda * w

    # bias 不加 L2 惩罚
    db = float(np.mean(residual))

    return dw, db


# ============================================================
# 17-20. 初始化、梯度下降训练与阈值预测
# ============================================================

def initialize_logistic_params(n_features: int) -> tuple:
    """权重和 bias 都从 0 开始。"""
    w = np.zeros(n_features, dtype=float)
    b = 0.0

    return w, b


def gradient_descent_step(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    b: float,
    lr: float,
    l2_lambda: float
) -> tuple:
    """
    做一次全批量梯度下降。
    返回的 loss 是更新参数之前计算的损失。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)

    probabilities = logistic_predict_proba(X, w, b)
    loss = binary_cross_entropy(y, probabilities, w, l2_lambda)

    dw, db = logistic_gradients(
        X, y, probabilities, w, l2_lambda
    )

    w_new = w - lr * dw
    b_new = float(b - lr * db)

    return w_new, b_new, loss


def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    lr: float,
    l2_lambda: float,
    n_epochs: int
) -> tuple:
    """从零开始训练逻辑回归，并记录每轮更新前的 loss。"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    w, b = initialize_logistic_params(X.shape[1])
    losses = []

    for _ in range(n_epochs):
        w, b, loss = gradient_descent_step(
            X, y, w, b, lr, l2_lambda
        )
        losses.append(float(loss))

    return w, float(b), losses


def predict_labels(
    proba: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """概率大于等于阈值时预测为 1，否则预测为 0。"""
    proba = np.asarray(proba, dtype=float)
    return (proba >= threshold).astype(int)


# ============================================================
# 21-24. 混淆矩阵计数、分类指标与阈值调优
# ============================================================

def confusion_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> tuple:
    """返回 (TP, FP, TN, FN)，均为 Python int。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    return tp, fp, tn, fn


def metrics_from_counts(
    tp: int,
    fp: int,
    tn: int,
    fn: int
) -> dict:
    """根据混淆矩阵计数计算 precision、recall、F1 和 accuracy。"""
    precision_denominator = tp + fp
    recall_denominator = tp + fn
    f1_denominator = 2 * tp + fp + fn
    total = tp + fp + tn + fn

    precision = (
        tp / precision_denominator
        if precision_denominator > 0 else 0.0
    )
    recall = (
        tp / recall_denominator
        if recall_denominator > 0 else 0.0
    )
    f1 = (
        2 * tp / f1_denominator
        if f1_denominator > 0 else 0.0
    )
    accuracy = (
        (tp + tn) / total
        if total > 0 else 0.0
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


def tune_decision_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    thresholds: np.ndarray = None
) -> tuple:
    """
    在候选阈值中选 F1 最高的那个。
    F1 并列时保留 thresholds 中最先出现的阈值。
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)

    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)
    else:
        thresholds = np.asarray(thresholds, dtype=float)

    if len(thresholds) == 0:
        return 0.0, 0.0

    best_threshold = float(thresholds[0])
    best_f1 = -1.0

    for threshold in thresholds:
        y_pred = predict_labels(proba, threshold)
        tp, fp, tn, fn = confusion_counts(y_true, y_pred)
        f1 = metrics_from_counts(tp, fp, tn, fn)["f1"]

        # 使用严格大于，保证并列时留下先遇到的阈值
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return float(best_threshold), float(best_f1)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """返回混淆矩阵计数和分类指标的汇总字典。"""
    tp, fp, tn, fn = confusion_counts(y_true, y_pred)
    metrics = metrics_from_counts(tp, fp, tn, fn)

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "accuracy": metrics["accuracy"],
    }


# ============================================================
# 25-27. 新文本向量化、单条预测与错误分析
# ============================================================

def vectorize_texts(
    texts: list,
    vocab: dict,
    idf: np.ndarray
) -> np.ndarray:
    """把原始文本转换成训练时同样格式的 TF-IDF 矩阵。"""
    tokenized_docs = tokenize_corpus(texts)
    bow_matrix = corpus_to_bow_matrix(tokenized_docs, vocab)

    return transform_tfidf(bow_matrix, idf)


def predict_text(
    text: str,
    vocab: dict,
    idf: np.ndarray,
    w: np.ndarray,
    b: float,
    threshold: float = 0.5
) -> int:
    """使用训练好的参数预测一条原始文本。"""
    X = vectorize_texts([text], vocab, idf)
    proba = logistic_predict_proba(X, w, b)
    prediction = predict_labels(proba, threshold)

    return int(prediction[0])


def collect_prediction_errors(
    texts: list,
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """收集假阳性和假阴性的原始文本，保持输入顺序。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    false_positives = []
    false_negatives = []

    for text, true_label, predicted_label in zip(texts, y_true, y_pred):
        if true_label == 0 and predicted_label == 1:
            false_positives.append(text)
        elif true_label == 1 and predicted_label == 0:
            false_negatives.append(text)

    return {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }

# Step 9 - compute_document_frequencies
import numpy as np


# ============================================================
# 1-3. 文本清洗与分词
# ============================================================

def clean_text(text: str) -> str:
    """
    转成小写，并把每个非字母字符替换为空格。
    注意：保留连续空格以及开头、结尾产生的空格。
    """
    text = text.lower()
    return "".join(ch if ch.isalpha() else " " for ch in text)


def tokenize(text: str) -> list:
    """按任意空白字符切分，并自动丢弃空 token。"""
    return text.split()


def tokenize_corpus(texts: list) -> list:
    """清洗并分词整个文本列表。"""
    return [tokenize(clean_text(text)) for text in texts]


# ============================================================
# 4-8. 数据划分、词频、词表与 Bag-of-Words
# ============================================================

def split_train_val_test_indices(
    n_samples: int,
    val_fraction: float,
    test_fraction: float,
    seed: int = 0
) -> tuple:
    """
    生成互不重叠的 train/validation/test 索引。
    按 train、validation、test 的顺序切分打乱后的索引。
    """
    np.random.seed(seed)

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    n_val = int(n_samples * val_fraction)
    n_test = int(n_samples * test_fraction)
    n_train = n_samples - n_val - n_test

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return train_idx, val_idx, test_idx


def count_word_frequencies(tokenized_docs: list) -> dict:
    """统计每个 token 在整个语料中出现的总次数。"""
    counts = {}

    for document in tokenized_docs:
        for token in document:
            counts[token] = counts.get(token, 0) + 1

    return counts


def build_vocabulary(word_counts: dict, max_size: int) -> dict:
    """
    选出词频最高的 max_size 个词。
    词频相同时，按字母顺序排列。
    """
    max_size = max(0, int(max_size))

    ranked_words = sorted(
        word_counts.items(),
        key=lambda item: (-item[1], item[0])
    )

    return {
        word: index
        for index, (word, _) in enumerate(ranked_words[:max_size])
    }


def tokens_to_bow(tokens: list, vocab: dict) -> np.ndarray:
    """将一篇文档转成词袋计数向量，未知词忽略。"""
    vector = np.zeros(len(vocab), dtype=float)

    for token in tokens:
        if token in vocab:
            vector[vocab[token]] += 1.0

    return vector


def corpus_to_bow_matrix(tokenized_docs: list, vocab: dict) -> np.ndarray:
    """将多个文档转成形状为 (文档数, 词表大小) 的词袋矩阵。"""
    matrix = np.zeros(
        (len(tokenized_docs), len(vocab)),
        dtype=float
    )

    for row, tokens in enumerate(tokenized_docs):
        matrix[row] = tokens_to_bow(tokens, vocab)

    return matrix


# ============================================================
# 9-12. 文档频率与 TF-IDF
# ============================================================

def compute_document_frequencies(bow_matrix: np.ndarray) -> np.ndarray:
    """
    计算每个词出现于多少篇文档。
    每篇文档中，一个词出现一次或多次，都只计为出现过。
    """
    bow_matrix = np.asarray(bow_matrix)
    return np.sum(bow_matrix > 0, axis=0).astype(int)


def compute_idf(df: np.ndarray, n_docs: int) -> np.ndarray:
    """计算平滑 IDF，使用自然对数。"""
    df = np.asarray(df, dtype=float)
    return np.log((n_docs + 1) / (df + 1)) + 1.0


def transform_tfidf(
    bow_matrix: np.ndarray,
    idf: np.ndarray
) -> np.ndarray:
    """
    每列的词袋计数乘对应 IDF。
    支持输入一个向量 (V,) 或矩阵 (N, V)。
    """
    bow_matrix = np.asarray(bow_matrix, dtype=float)
    idf = np.asarray(idf, dtype=float)
    return bow_matrix * idf


def fit_tfidf(bow_train: np.ndarray) -> np.ndarray:
    """只根据训练集词袋矩阵拟合 IDF。"""
    bow_train = np.asarray(bow_train, dtype=float)

    df = compute_document_frequencies(bow_train)
    n_docs = bow_train.shape[0]

    return compute_idf(df, n_docs)


# ============================================================
# 13-16. 逻辑回归核心：Sigmoid、概率、损失与梯度
# ============================================================

def sigmoid(z: np.ndarray) -> np.ndarray:
    """数值稳定的 sigmoid，返回与输入形状相同的数组。"""
    z = np.asarray(z, dtype=float)
    flat = z.reshape(-1)
    result = np.empty_like(flat)

    # 正数和负数分开计算，避免 exp 溢出
    positive = flat >= 0
    negative = ~positive

    result[positive] = 1.0 / (1.0 + np.exp(-flat[positive]))

    exp_z = np.exp(flat[negative])
    result[negative] = exp_z / (1.0 + exp_z)

    return result.reshape(z.shape)


def logistic_predict_proba(
    X: np.ndarray,
    w: np.ndarray,
    b: float
) -> np.ndarray:
    """计算每一行属于类别 1 的概率。"""
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)

    return sigmoid(X @ w + b)


def binary_cross_entropy(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    w: np.ndarray,
    l2_lambda: float
) -> float:
    """
    平均二元交叉熵 + L2 权重惩罚。
    bias 不在这里传入，所以不会被正则化。
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    w = np.asarray(w, dtype=float)

    # 避免 log(0)
    y_prob = np.clip(y_prob, 1e-15, 1.0 - 1e-15)

    if y_true.size == 0:
        mean_bce = 0.0
    else:
        mean_bce = -np.mean(
            y_true * np.log(y_prob)
            + (1.0 - y_true) * np.log(1.0 - y_prob)
        )

    l2_penalty = l2_lambda * np.sum(w ** 2) / 2.0

    return float(mean_bce + l2_penalty)


def logistic_gradients(
    X: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    w: np.ndarray,
    l2_lambda: float
) -> tuple:
    """计算平均 BCE + L2 对权重和 bias 的梯度。"""
    X = np.asarray(X, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    w = np.asarray(w, dtype=float)

    n_samples = X.shape[0]

    if n_samples == 0:
        return l2_lambda * w, 0.0

    residual = y_prob - y_true

    # dL/dw：数据损失梯度 + L2 梯度
    dw = (X.T @ residual) / n_samples + l2_lambda * w

    # bias 不加 L2 惩罚
    db = float(np.mean(residual))

    return dw, db


# ============================================================
# 17-20. 初始化、梯度下降训练与阈值预测
# ============================================================

def initialize_logistic_params(n_features: int) -> tuple:
    """权重和 bias 都从 0 开始。"""
    w = np.zeros(n_features, dtype=float)
    b = 0.0

    return w, b


def gradient_descent_step(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    b: float,
    lr: float,
    l2_lambda: float
) -> tuple:
    """
    做一次全批量梯度下降。
    返回的 loss 是更新参数之前计算的损失。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)

    probabilities = logistic_predict_proba(X, w, b)
    loss = binary_cross_entropy(y, probabilities, w, l2_lambda)

    dw, db = logistic_gradients(
        X, y, probabilities, w, l2_lambda
    )

    w_new = w - lr * dw
    b_new = float(b - lr * db)

    return w_new, b_new, loss


def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    lr: float,
    l2_lambda: float,
    n_epochs: int
) -> tuple:
    """从零开始训练逻辑回归，并记录每轮更新前的 loss。"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    w, b = initialize_logistic_params(X.shape[1])
    losses = []

    for _ in range(n_epochs):
        w, b, loss = gradient_descent_step(
            X, y, w, b, lr, l2_lambda
        )
        losses.append(float(loss))

    return w, float(b), losses


def predict_labels(
    proba: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """概率大于等于阈值时预测为 1，否则预测为 0。"""
    proba = np.asarray(proba, dtype=float)
    return (proba >= threshold).astype(int)


# ============================================================
# 21-24. 混淆矩阵计数、分类指标与阈值调优
# ============================================================

def confusion_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> tuple:
    """返回 (TP, FP, TN, FN)，均为 Python int。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    return tp, fp, tn, fn


def metrics_from_counts(
    tp: int,
    fp: int,
    tn: int,
    fn: int
) -> dict:
    """根据混淆矩阵计数计算 precision、recall、F1 和 accuracy。"""
    precision_denominator = tp + fp
    recall_denominator = tp + fn
    f1_denominator = 2 * tp + fp + fn
    total = tp + fp + tn + fn

    precision = (
        tp / precision_denominator
        if precision_denominator > 0 else 0.0
    )
    recall = (
        tp / recall_denominator
        if recall_denominator > 0 else 0.0
    )
    f1 = (
        2 * tp / f1_denominator
        if f1_denominator > 0 else 0.0
    )
    accuracy = (
        (tp + tn) / total
        if total > 0 else 0.0
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


def tune_decision_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    thresholds: np.ndarray = None
) -> tuple:
    """
    在候选阈值中选 F1 最高的那个。
    F1 并列时保留 thresholds 中最先出现的阈值。
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)

    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)
    else:
        thresholds = np.asarray(thresholds, dtype=float)

    if len(thresholds) == 0:
        return 0.0, 0.0

    best_threshold = float(thresholds[0])
    best_f1 = -1.0

    for threshold in thresholds:
        y_pred = predict_labels(proba, threshold)
        tp, fp, tn, fn = confusion_counts(y_true, y_pred)
        f1 = metrics_from_counts(tp, fp, tn, fn)["f1"]

        # 使用严格大于，保证并列时留下先遇到的阈值
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return float(best_threshold), float(best_f1)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """返回混淆矩阵计数和分类指标的汇总字典。"""
    tp, fp, tn, fn = confusion_counts(y_true, y_pred)
    metrics = metrics_from_counts(tp, fp, tn, fn)

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "accuracy": metrics["accuracy"],
    }


# ============================================================
# 25-27. 新文本向量化、单条预测与错误分析
# ============================================================

def vectorize_texts(
    texts: list,
    vocab: dict,
    idf: np.ndarray
) -> np.ndarray:
    """把原始文本转换成训练时同样格式的 TF-IDF 矩阵。"""
    tokenized_docs = tokenize_corpus(texts)
    bow_matrix = corpus_to_bow_matrix(tokenized_docs, vocab)

    return transform_tfidf(bow_matrix, idf)


def predict_text(
    text: str,
    vocab: dict,
    idf: np.ndarray,
    w: np.ndarray,
    b: float,
    threshold: float = 0.5
) -> int:
    """使用训练好的参数预测一条原始文本。"""
    X = vectorize_texts([text], vocab, idf)
    proba = logistic_predict_proba(X, w, b)
    prediction = predict_labels(proba, threshold)

    return int(prediction[0])


def collect_prediction_errors(
    texts: list,
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """收集假阳性和假阴性的原始文本，保持输入顺序。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    false_positives = []
    false_negatives = []

    for text, true_label, predicted_label in zip(texts, y_true, y_pred):
        if true_label == 0 and predicted_label == 1:
            false_positives.append(text)
        elif true_label == 1 and predicted_label == 0:
            false_negatives.append(text)

    return {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }

# Step 10 - compute_idf
import numpy as np


# ============================================================
# 1-3. 文本清洗与分词
# ============================================================

def clean_text(text: str) -> str:
    """
    转成小写，并把每个非字母字符替换为空格。
    注意：保留连续空格以及开头、结尾产生的空格。
    """
    text = text.lower()
    return "".join(ch if ch.isalpha() else " " for ch in text)


def tokenize(text: str) -> list:
    """按任意空白字符切分，并自动丢弃空 token。"""
    return text.split()


def tokenize_corpus(texts: list) -> list:
    """清洗并分词整个文本列表。"""
    return [tokenize(clean_text(text)) for text in texts]


# ============================================================
# 4-8. 数据划分、词频、词表与 Bag-of-Words
# ============================================================

def split_train_val_test_indices(
    n_samples: int,
    val_fraction: float,
    test_fraction: float,
    seed: int = 0
) -> tuple:
    """
    生成互不重叠的 train/validation/test 索引。
    按 train、validation、test 的顺序切分打乱后的索引。
    """
    np.random.seed(seed)

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    n_val = int(n_samples * val_fraction)
    n_test = int(n_samples * test_fraction)
    n_train = n_samples - n_val - n_test

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return train_idx, val_idx, test_idx


def count_word_frequencies(tokenized_docs: list) -> dict:
    """统计每个 token 在整个语料中出现的总次数。"""
    counts = {}

    for document in tokenized_docs:
        for token in document:
            counts[token] = counts.get(token, 0) + 1

    return counts


def build_vocabulary(word_counts: dict, max_size: int) -> dict:
    """
    选出词频最高的 max_size 个词。
    词频相同时，按字母顺序排列。
    """
    max_size = max(0, int(max_size))

    ranked_words = sorted(
        word_counts.items(),
        key=lambda item: (-item[1], item[0])
    )

    return {
        word: index
        for index, (word, _) in enumerate(ranked_words[:max_size])
    }


def tokens_to_bow(tokens: list, vocab: dict) -> np.ndarray:
    """将一篇文档转成词袋计数向量，未知词忽略。"""
    vector = np.zeros(len(vocab), dtype=float)

    for token in tokens:
        if token in vocab:
            vector[vocab[token]] += 1.0

    return vector


def corpus_to_bow_matrix(tokenized_docs: list, vocab: dict) -> np.ndarray:
    """将多个文档转成形状为 (文档数, 词表大小) 的词袋矩阵。"""
    matrix = np.zeros(
        (len(tokenized_docs), len(vocab)),
        dtype=float
    )

    for row, tokens in enumerate(tokenized_docs):
        matrix[row] = tokens_to_bow(tokens, vocab)

    return matrix


# ============================================================
# 9-12. 文档频率与 TF-IDF
# ============================================================

def compute_document_frequencies(bow_matrix: np.ndarray) -> np.ndarray:
    """
    计算每个词出现于多少篇文档。
    每篇文档中，一个词出现一次或多次，都只计为出现过。
    """
    bow_matrix = np.asarray(bow_matrix)
    return np.sum(bow_matrix > 0, axis=0).astype(int)


def compute_idf(df: np.ndarray, n_docs: int) -> np.ndarray:
    """计算平滑 IDF，使用自然对数。"""
    df = np.asarray(df, dtype=float)
    return np.log((n_docs + 1) / (df + 1)) + 1.0


def transform_tfidf(
    bow_matrix: np.ndarray,
    idf: np.ndarray
) -> np.ndarray:
    """
    每列的词袋计数乘对应 IDF。
    支持输入一个向量 (V,) 或矩阵 (N, V)。
    """
    bow_matrix = np.asarray(bow_matrix, dtype=float)
    idf = np.asarray(idf, dtype=float)
    return bow_matrix * idf


def fit_tfidf(bow_train: np.ndarray) -> np.ndarray:
    """只根据训练集词袋矩阵拟合 IDF。"""
    bow_train = np.asarray(bow_train, dtype=float)

    df = compute_document_frequencies(bow_train)
    n_docs = bow_train.shape[0]

    return compute_idf(df, n_docs)


# ============================================================
# 13-16. 逻辑回归核心：Sigmoid、概率、损失与梯度
# ============================================================

def sigmoid(z: np.ndarray) -> np.ndarray:
    """数值稳定的 sigmoid，返回与输入形状相同的数组。"""
    z = np.asarray(z, dtype=float)
    flat = z.reshape(-1)
    result = np.empty_like(flat)

    # 正数和负数分开计算，避免 exp 溢出
    positive = flat >= 0
    negative = ~positive

    result[positive] = 1.0 / (1.0 + np.exp(-flat[positive]))

    exp_z = np.exp(flat[negative])
    result[negative] = exp_z / (1.0 + exp_z)

    return result.reshape(z.shape)


def logistic_predict_proba(
    X: np.ndarray,
    w: np.ndarray,
    b: float
) -> np.ndarray:
    """计算每一行属于类别 1 的概率。"""
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)

    return sigmoid(X @ w + b)


def binary_cross_entropy(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    w: np.ndarray,
    l2_lambda: float
) -> float:
    """
    平均二元交叉熵 + L2 权重惩罚。
    bias 不在这里传入，所以不会被正则化。
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    w = np.asarray(w, dtype=float)

    # 避免 log(0)
    y_prob = np.clip(y_prob, 1e-15, 1.0 - 1e-15)

    if y_true.size == 0:
        mean_bce = 0.0
    else:
        mean_bce = -np.mean(
            y_true * np.log(y_prob)
            + (1.0 - y_true) * np.log(1.0 - y_prob)
        )

    l2_penalty = l2_lambda * np.sum(w ** 2) / 2.0

    return float(mean_bce + l2_penalty)


def logistic_gradients(
    X: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    w: np.ndarray,
    l2_lambda: float
) -> tuple:
    """计算平均 BCE + L2 对权重和 bias 的梯度。"""
    X = np.asarray(X, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    w = np.asarray(w, dtype=float)

    n_samples = X.shape[0]

    if n_samples == 0:
        return l2_lambda * w, 0.0

    residual = y_prob - y_true

    # dL/dw：数据损失梯度 + L2 梯度
    dw = (X.T @ residual) / n_samples + l2_lambda * w

    # bias 不加 L2 惩罚
    db = float(np.mean(residual))

    return dw, db


# ============================================================
# 17-20. 初始化、梯度下降训练与阈值预测
# ============================================================

def initialize_logistic_params(n_features: int) -> tuple:
    """权重和 bias 都从 0 开始。"""
    w = np.zeros(n_features, dtype=float)
    b = 0.0

    return w, b


def gradient_descent_step(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    b: float,
    lr: float,
    l2_lambda: float
) -> tuple:
    """
    做一次全批量梯度下降。
    返回的 loss 是更新参数之前计算的损失。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)

    probabilities = logistic_predict_proba(X, w, b)
    loss = binary_cross_entropy(y, probabilities, w, l2_lambda)

    dw, db = logistic_gradients(
        X, y, probabilities, w, l2_lambda
    )

    w_new = w - lr * dw
    b_new = float(b - lr * db)

    return w_new, b_new, loss


def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    lr: float,
    l2_lambda: float,
    n_epochs: int
) -> tuple:
    """从零开始训练逻辑回归，并记录每轮更新前的 loss。"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    w, b = initialize_logistic_params(X.shape[1])
    losses = []

    for _ in range(n_epochs):
        w, b, loss = gradient_descent_step(
            X, y, w, b, lr, l2_lambda
        )
        losses.append(float(loss))

    return w, float(b), losses


def predict_labels(
    proba: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """概率大于等于阈值时预测为 1，否则预测为 0。"""
    proba = np.asarray(proba, dtype=float)
    return (proba >= threshold).astype(int)


# ============================================================
# 21-24. 混淆矩阵计数、分类指标与阈值调优
# ============================================================

def confusion_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> tuple:
    """返回 (TP, FP, TN, FN)，均为 Python int。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    return tp, fp, tn, fn


def metrics_from_counts(
    tp: int,
    fp: int,
    tn: int,
    fn: int
) -> dict:
    """根据混淆矩阵计数计算 precision、recall、F1 和 accuracy。"""
    precision_denominator = tp + fp
    recall_denominator = tp + fn
    f1_denominator = 2 * tp + fp + fn
    total = tp + fp + tn + fn

    precision = (
        tp / precision_denominator
        if precision_denominator > 0 else 0.0
    )
    recall = (
        tp / recall_denominator
        if recall_denominator > 0 else 0.0
    )
    f1 = (
        2 * tp / f1_denominator
        if f1_denominator > 0 else 0.0
    )
    accuracy = (
        (tp + tn) / total
        if total > 0 else 0.0
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


def tune_decision_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    thresholds: np.ndarray = None
) -> tuple:
    """
    在候选阈值中选 F1 最高的那个。
    F1 并列时保留 thresholds 中最先出现的阈值。
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)

    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)
    else:
        thresholds = np.asarray(thresholds, dtype=float)

    if len(thresholds) == 0:
        return 0.0, 0.0

    best_threshold = float(thresholds[0])
    best_f1 = -1.0

    for threshold in thresholds:
        y_pred = predict_labels(proba, threshold)
        tp, fp, tn, fn = confusion_counts(y_true, y_pred)
        f1 = metrics_from_counts(tp, fp, tn, fn)["f1"]

        # 使用严格大于，保证并列时留下先遇到的阈值
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return float(best_threshold), float(best_f1)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """返回混淆矩阵计数和分类指标的汇总字典。"""
    tp, fp, tn, fn = confusion_counts(y_true, y_pred)
    metrics = metrics_from_counts(tp, fp, tn, fn)

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "accuracy": metrics["accuracy"],
    }


# ============================================================
# 25-27. 新文本向量化、单条预测与错误分析
# ============================================================

def vectorize_texts(
    texts: list,
    vocab: dict,
    idf: np.ndarray
) -> np.ndarray:
    """把原始文本转换成训练时同样格式的 TF-IDF 矩阵。"""
    tokenized_docs = tokenize_corpus(texts)
    bow_matrix = corpus_to_bow_matrix(tokenized_docs, vocab)

    return transform_tfidf(bow_matrix, idf)


def predict_text(
    text: str,
    vocab: dict,
    idf: np.ndarray,
    w: np.ndarray,
    b: float,
    threshold: float = 0.5
) -> int:
    """使用训练好的参数预测一条原始文本。"""
    X = vectorize_texts([text], vocab, idf)
    proba = logistic_predict_proba(X, w, b)
    prediction = predict_labels(proba, threshold)

    return int(prediction[0])


def collect_prediction_errors(
    texts: list,
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """收集假阳性和假阴性的原始文本，保持输入顺序。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    false_positives = []
    false_negatives = []

    for text, true_label, predicted_label in zip(texts, y_true, y_pred):
        if true_label == 0 and predicted_label == 1:
            false_positives.append(text)
        elif true_label == 1 and predicted_label == 0:
            false_negatives.append(text)

    return {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }

# Step 11 - transform_tfidf
import numpy as np


# ============================================================
# 1-3. 文本清洗与分词
# ============================================================

def clean_text(text: str) -> str:
    """
    转成小写，并把每个非字母字符替换为空格。
    注意：保留连续空格以及开头、结尾产生的空格。
    """
    text = text.lower()
    return "".join(ch if ch.isalpha() else " " for ch in text)


def tokenize(text: str) -> list:
    """按任意空白字符切分，并自动丢弃空 token。"""
    return text.split()


def tokenize_corpus(texts: list) -> list:
    """清洗并分词整个文本列表。"""
    return [tokenize(clean_text(text)) for text in texts]


# ============================================================
# 4-8. 数据划分、词频、词表与 Bag-of-Words
# ============================================================

def split_train_val_test_indices(
    n_samples: int,
    val_fraction: float,
    test_fraction: float,
    seed: int = 0
) -> tuple:
    """
    生成互不重叠的 train/validation/test 索引。
    按 train、validation、test 的顺序切分打乱后的索引。
    """
    np.random.seed(seed)

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    n_val = int(n_samples * val_fraction)
    n_test = int(n_samples * test_fraction)
    n_train = n_samples - n_val - n_test

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return train_idx, val_idx, test_idx


def count_word_frequencies(tokenized_docs: list) -> dict:
    """统计每个 token 在整个语料中出现的总次数。"""
    counts = {}

    for document in tokenized_docs:
        for token in document:
            counts[token] = counts.get(token, 0) + 1

    return counts


def build_vocabulary(word_counts: dict, max_size: int) -> dict:
    """
    选出词频最高的 max_size 个词。
    词频相同时，按字母顺序排列。
    """
    max_size = max(0, int(max_size))

    ranked_words = sorted(
        word_counts.items(),
        key=lambda item: (-item[1], item[0])
    )

    return {
        word: index
        for index, (word, _) in enumerate(ranked_words[:max_size])
    }


def tokens_to_bow(tokens: list, vocab: dict) -> np.ndarray:
    """将一篇文档转成词袋计数向量，未知词忽略。"""
    vector = np.zeros(len(vocab), dtype=float)

    for token in tokens:
        if token in vocab:
            vector[vocab[token]] += 1.0

    return vector


def corpus_to_bow_matrix(tokenized_docs: list, vocab: dict) -> np.ndarray:
    """将多个文档转成形状为 (文档数, 词表大小) 的词袋矩阵。"""
    matrix = np.zeros(
        (len(tokenized_docs), len(vocab)),
        dtype=float
    )

    for row, tokens in enumerate(tokenized_docs):
        matrix[row] = tokens_to_bow(tokens, vocab)

    return matrix


# ============================================================
# 9-12. 文档频率与 TF-IDF
# ============================================================

def compute_document_frequencies(bow_matrix: np.ndarray) -> np.ndarray:
    """
    计算每个词出现于多少篇文档。
    每篇文档中，一个词出现一次或多次，都只计为出现过。
    """
    bow_matrix = np.asarray(bow_matrix)
    return np.sum(bow_matrix > 0, axis=0).astype(int)


def compute_idf(df: np.ndarray, n_docs: int) -> np.ndarray:
    """计算平滑 IDF，使用自然对数。"""
    df = np.asarray(df, dtype=float)
    return np.log((n_docs + 1) / (df + 1)) + 1.0


def transform_tfidf(
    bow_matrix: np.ndarray,
    idf: np.ndarray
) -> np.ndarray:
    """
    每列的词袋计数乘对应 IDF。
    支持输入一个向量 (V,) 或矩阵 (N, V)。
    """
    bow_matrix = np.asarray(bow_matrix, dtype=float)
    idf = np.asarray(idf, dtype=float)
    return bow_matrix * idf


def fit_tfidf(bow_train: np.ndarray) -> np.ndarray:
    """只根据训练集词袋矩阵拟合 IDF。"""
    bow_train = np.asarray(bow_train, dtype=float)

    df = compute_document_frequencies(bow_train)
    n_docs = bow_train.shape[0]

    return compute_idf(df, n_docs)


# ============================================================
# 13-16. 逻辑回归核心：Sigmoid、概率、损失与梯度
# ============================================================

def sigmoid(z: np.ndarray) -> np.ndarray:
    """数值稳定的 sigmoid，返回与输入形状相同的数组。"""
    z = np.asarray(z, dtype=float)
    flat = z.reshape(-1)
    result = np.empty_like(flat)

    # 正数和负数分开计算，避免 exp 溢出
    positive = flat >= 0
    negative = ~positive

    result[positive] = 1.0 / (1.0 + np.exp(-flat[positive]))

    exp_z = np.exp(flat[negative])
    result[negative] = exp_z / (1.0 + exp_z)

    return result.reshape(z.shape)


def logistic_predict_proba(
    X: np.ndarray,
    w: np.ndarray,
    b: float
) -> np.ndarray:
    """计算每一行属于类别 1 的概率。"""
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)

    return sigmoid(X @ w + b)


def binary_cross_entropy(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    w: np.ndarray,
    l2_lambda: float
) -> float:
    """
    平均二元交叉熵 + L2 权重惩罚。
    bias 不在这里传入，所以不会被正则化。
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    w = np.asarray(w, dtype=float)

    # 避免 log(0)
    y_prob = np.clip(y_prob, 1e-15, 1.0 - 1e-15)

    if y_true.size == 0:
        mean_bce = 0.0
    else:
        mean_bce = -np.mean(
            y_true * np.log(y_prob)
            + (1.0 - y_true) * np.log(1.0 - y_prob)
        )

    l2_penalty = l2_lambda * np.sum(w ** 2) / 2.0

    return float(mean_bce + l2_penalty)


def logistic_gradients(
    X: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    w: np.ndarray,
    l2_lambda: float
) -> tuple:
    """计算平均 BCE + L2 对权重和 bias 的梯度。"""
    X = np.asarray(X, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    w = np.asarray(w, dtype=float)

    n_samples = X.shape[0]

    if n_samples == 0:
        return l2_lambda * w, 0.0

    residual = y_prob - y_true

    # dL/dw：数据损失梯度 + L2 梯度
    dw = (X.T @ residual) / n_samples + l2_lambda * w

    # bias 不加 L2 惩罚
    db = float(np.mean(residual))

    return dw, db


# ============================================================
# 17-20. 初始化、梯度下降训练与阈值预测
# ============================================================

def initialize_logistic_params(n_features: int) -> tuple:
    """权重和 bias 都从 0 开始。"""
    w = np.zeros(n_features, dtype=float)
    b = 0.0

    return w, b


def gradient_descent_step(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    b: float,
    lr: float,
    l2_lambda: float
) -> tuple:
    """
    做一次全批量梯度下降。
    返回的 loss 是更新参数之前计算的损失。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)

    probabilities = logistic_predict_proba(X, w, b)
    loss = binary_cross_entropy(y, probabilities, w, l2_lambda)

    dw, db = logistic_gradients(
        X, y, probabilities, w, l2_lambda
    )

    w_new = w - lr * dw
    b_new = float(b - lr * db)

    return w_new, b_new, loss


def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    lr: float,
    l2_lambda: float,
    n_epochs: int
) -> tuple:
    """从零开始训练逻辑回归，并记录每轮更新前的 loss。"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    w, b = initialize_logistic_params(X.shape[1])
    losses = []

    for _ in range(n_epochs):
        w, b, loss = gradient_descent_step(
            X, y, w, b, lr, l2_lambda
        )
        losses.append(float(loss))

    return w, float(b), losses


def predict_labels(
    proba: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """概率大于等于阈值时预测为 1，否则预测为 0。"""
    proba = np.asarray(proba, dtype=float)
    return (proba >= threshold).astype(int)


# ============================================================
# 21-24. 混淆矩阵计数、分类指标与阈值调优
# ============================================================

def confusion_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> tuple:
    """返回 (TP, FP, TN, FN)，均为 Python int。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    return tp, fp, tn, fn


def metrics_from_counts(
    tp: int,
    fp: int,
    tn: int,
    fn: int
) -> dict:
    """根据混淆矩阵计数计算 precision、recall、F1 和 accuracy。"""
    precision_denominator = tp + fp
    recall_denominator = tp + fn
    f1_denominator = 2 * tp + fp + fn
    total = tp + fp + tn + fn

    precision = (
        tp / precision_denominator
        if precision_denominator > 0 else 0.0
    )
    recall = (
        tp / recall_denominator
        if recall_denominator > 0 else 0.0
    )
    f1 = (
        2 * tp / f1_denominator
        if f1_denominator > 0 else 0.0
    )
    accuracy = (
        (tp + tn) / total
        if total > 0 else 0.0
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


def tune_decision_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    thresholds: np.ndarray = None
) -> tuple:
    """
    在候选阈值中选 F1 最高的那个。
    F1 并列时保留 thresholds 中最先出现的阈值。
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)

    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)
    else:
        thresholds = np.asarray(thresholds, dtype=float)

    if len(thresholds) == 0:
        return 0.0, 0.0

    best_threshold = float(thresholds[0])
    best_f1 = -1.0

    for threshold in thresholds:
        y_pred = predict_labels(proba, threshold)
        tp, fp, tn, fn = confusion_counts(y_true, y_pred)
        f1 = metrics_from_counts(tp, fp, tn, fn)["f1"]

        # 使用严格大于，保证并列时留下先遇到的阈值
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return float(best_threshold), float(best_f1)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """返回混淆矩阵计数和分类指标的汇总字典。"""
    tp, fp, tn, fn = confusion_counts(y_true, y_pred)
    metrics = metrics_from_counts(tp, fp, tn, fn)

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "accuracy": metrics["accuracy"],
    }


# ============================================================
# 25-27. 新文本向量化、单条预测与错误分析
# ============================================================

def vectorize_texts(
    texts: list,
    vocab: dict,
    idf: np.ndarray
) -> np.ndarray:
    """把原始文本转换成训练时同样格式的 TF-IDF 矩阵。"""
    tokenized_docs = tokenize_corpus(texts)
    bow_matrix = corpus_to_bow_matrix(tokenized_docs, vocab)

    return transform_tfidf(bow_matrix, idf)


def predict_text(
    text: str,
    vocab: dict,
    idf: np.ndarray,
    w: np.ndarray,
    b: float,
    threshold: float = 0.5
) -> int:
    """使用训练好的参数预测一条原始文本。"""
    X = vectorize_texts([text], vocab, idf)
    proba = logistic_predict_proba(X, w, b)
    prediction = predict_labels(proba, threshold)

    return int(prediction[0])


def collect_prediction_errors(
    texts: list,
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """收集假阳性和假阴性的原始文本，保持输入顺序。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    false_positives = []
    false_negatives = []

    for text, true_label, predicted_label in zip(texts, y_true, y_pred):
        if true_label == 0 and predicted_label == 1:
            false_positives.append(text)
        elif true_label == 1 and predicted_label == 0:
            false_negatives.append(text)

    return {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }

# Step 12 - fit_tfidf
import numpy as np


# ============================================================
# 1-3. 文本清洗与分词
# ============================================================

def clean_text(text: str) -> str:
    """
    转成小写，并把每个非字母字符替换为空格。
    注意：保留连续空格以及开头、结尾产生的空格。
    """
    text = text.lower()
    return "".join(ch if ch.isalpha() else " " for ch in text)


def tokenize(text: str) -> list:
    """按任意空白字符切分，并自动丢弃空 token。"""
    return text.split()


def tokenize_corpus(texts: list) -> list:
    """清洗并分词整个文本列表。"""
    return [tokenize(clean_text(text)) for text in texts]


# ============================================================
# 4-8. 数据划分、词频、词表与 Bag-of-Words
# ============================================================

def split_train_val_test_indices(
    n_samples: int,
    val_fraction: float,
    test_fraction: float,
    seed: int = 0
) -> tuple:
    """
    生成互不重叠的 train/validation/test 索引。
    按 train、validation、test 的顺序切分打乱后的索引。
    """
    np.random.seed(seed)

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    n_val = int(n_samples * val_fraction)
    n_test = int(n_samples * test_fraction)
    n_train = n_samples - n_val - n_test

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return train_idx, val_idx, test_idx


def count_word_frequencies(tokenized_docs: list) -> dict:
    """统计每个 token 在整个语料中出现的总次数。"""
    counts = {}

    for document in tokenized_docs:
        for token in document:
            counts[token] = counts.get(token, 0) + 1

    return counts


def build_vocabulary(word_counts: dict, max_size: int) -> dict:
    """
    选出词频最高的 max_size 个词。
    词频相同时，按字母顺序排列。
    """
    max_size = max(0, int(max_size))

    ranked_words = sorted(
        word_counts.items(),
        key=lambda item: (-item[1], item[0])
    )

    return {
        word: index
        for index, (word, _) in enumerate(ranked_words[:max_size])
    }


def tokens_to_bow(tokens: list, vocab: dict) -> np.ndarray:
    """将一篇文档转成词袋计数向量，未知词忽略。"""
    vector = np.zeros(len(vocab), dtype=float)

    for token in tokens:
        if token in vocab:
            vector[vocab[token]] += 1.0

    return vector


def corpus_to_bow_matrix(tokenized_docs: list, vocab: dict) -> np.ndarray:
    """将多个文档转成形状为 (文档数, 词表大小) 的词袋矩阵。"""
    matrix = np.zeros(
        (len(tokenized_docs), len(vocab)),
        dtype=float
    )

    for row, tokens in enumerate(tokenized_docs):
        matrix[row] = tokens_to_bow(tokens, vocab)

    return matrix


# ============================================================
# 9-12. 文档频率与 TF-IDF
# ============================================================

def compute_document_frequencies(bow_matrix: np.ndarray) -> np.ndarray:
    """
    计算每个词出现于多少篇文档。
    每篇文档中，一个词出现一次或多次，都只计为出现过。
    """
    bow_matrix = np.asarray(bow_matrix)
    return np.sum(bow_matrix > 0, axis=0).astype(int)


def compute_idf(df: np.ndarray, n_docs: int) -> np.ndarray:
    """计算平滑 IDF，使用自然对数。"""
    df = np.asarray(df, dtype=float)
    return np.log((n_docs + 1) / (df + 1)) + 1.0


def transform_tfidf(
    bow_matrix: np.ndarray,
    idf: np.ndarray
) -> np.ndarray:
    """
    每列的词袋计数乘对应 IDF。
    支持输入一个向量 (V,) 或矩阵 (N, V)。
    """
    bow_matrix = np.asarray(bow_matrix, dtype=float)
    idf = np.asarray(idf, dtype=float)
    return bow_matrix * idf


def fit_tfidf(bow_train: np.ndarray) -> np.ndarray:
    """只根据训练集词袋矩阵拟合 IDF。"""
    bow_train = np.asarray(bow_train, dtype=float)

    df = compute_document_frequencies(bow_train)
    n_docs = bow_train.shape[0]

    return compute_idf(df, n_docs)


# ============================================================
# 13-16. 逻辑回归核心：Sigmoid、概率、损失与梯度
# ============================================================

def sigmoid(z: np.ndarray) -> np.ndarray:
    """数值稳定的 sigmoid，返回与输入形状相同的数组。"""
    z = np.asarray(z, dtype=float)
    flat = z.reshape(-1)
    result = np.empty_like(flat)

    # 正数和负数分开计算，避免 exp 溢出
    positive = flat >= 0
    negative = ~positive

    result[positive] = 1.0 / (1.0 + np.exp(-flat[positive]))

    exp_z = np.exp(flat[negative])
    result[negative] = exp_z / (1.0 + exp_z)

    return result.reshape(z.shape)


def logistic_predict_proba(
    X: np.ndarray,
    w: np.ndarray,
    b: float
) -> np.ndarray:
    """计算每一行属于类别 1 的概率。"""
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)

    return sigmoid(X @ w + b)


def binary_cross_entropy(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    w: np.ndarray,
    l2_lambda: float
) -> float:
    """
    平均二元交叉熵 + L2 权重惩罚。
    bias 不在这里传入，所以不会被正则化。
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    w = np.asarray(w, dtype=float)

    # 避免 log(0)
    y_prob = np.clip(y_prob, 1e-15, 1.0 - 1e-15)

    if y_true.size == 0:
        mean_bce = 0.0
    else:
        mean_bce = -np.mean(
            y_true * np.log(y_prob)
            + (1.0 - y_true) * np.log(1.0 - y_prob)
        )

    l2_penalty = l2_lambda * np.sum(w ** 2) / 2.0

    return float(mean_bce + l2_penalty)


def logistic_gradients(
    X: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    w: np.ndarray,
    l2_lambda: float
) -> tuple:
    """计算平均 BCE + L2 对权重和 bias 的梯度。"""
    X = np.asarray(X, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    w = np.asarray(w, dtype=float)

    n_samples = X.shape[0]

    if n_samples == 0:
        return l2_lambda * w, 0.0

    residual = y_prob - y_true

    # dL/dw：数据损失梯度 + L2 梯度
    dw = (X.T @ residual) / n_samples + l2_lambda * w

    # bias 不加 L2 惩罚
    db = float(np.mean(residual))

    return dw, db


# ============================================================
# 17-20. 初始化、梯度下降训练与阈值预测
# ============================================================

def initialize_logistic_params(n_features: int) -> tuple:
    """权重和 bias 都从 0 开始。"""
    w = np.zeros(n_features, dtype=float)
    b = 0.0

    return w, b


def gradient_descent_step(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    b: float,
    lr: float,
    l2_lambda: float
) -> tuple:
    """
    做一次全批量梯度下降。
    返回的 loss 是更新参数之前计算的损失。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)

    probabilities = logistic_predict_proba(X, w, b)
    loss = binary_cross_entropy(y, probabilities, w, l2_lambda)

    dw, db = logistic_gradients(
        X, y, probabilities, w, l2_lambda
    )

    w_new = w - lr * dw
    b_new = float(b - lr * db)

    return w_new, b_new, loss


def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    lr: float,
    l2_lambda: float,
    n_epochs: int
) -> tuple:
    """从零开始训练逻辑回归，并记录每轮更新前的 loss。"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    w, b = initialize_logistic_params(X.shape[1])
    losses = []

    for _ in range(n_epochs):
        w, b, loss = gradient_descent_step(
            X, y, w, b, lr, l2_lambda
        )
        losses.append(float(loss))

    return w, float(b), losses


def predict_labels(
    proba: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """概率大于等于阈值时预测为 1，否则预测为 0。"""
    proba = np.asarray(proba, dtype=float)
    return (proba >= threshold).astype(int)


# ============================================================
# 21-24. 混淆矩阵计数、分类指标与阈值调优
# ============================================================

def confusion_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> tuple:
    """返回 (TP, FP, TN, FN)，均为 Python int。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    return tp, fp, tn, fn


def metrics_from_counts(
    tp: int,
    fp: int,
    tn: int,
    fn: int
) -> dict:
    """根据混淆矩阵计数计算 precision、recall、F1 和 accuracy。"""
    precision_denominator = tp + fp
    recall_denominator = tp + fn
    f1_denominator = 2 * tp + fp + fn
    total = tp + fp + tn + fn

    precision = (
        tp / precision_denominator
        if precision_denominator > 0 else 0.0
    )
    recall = (
        tp / recall_denominator
        if recall_denominator > 0 else 0.0
    )
    f1 = (
        2 * tp / f1_denominator
        if f1_denominator > 0 else 0.0
    )
    accuracy = (
        (tp + tn) / total
        if total > 0 else 0.0
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


def tune_decision_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    thresholds: np.ndarray = None
) -> tuple:
    """
    在候选阈值中选 F1 最高的那个。
    F1 并列时保留 thresholds 中最先出现的阈值。
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)

    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)
    else:
        thresholds = np.asarray(thresholds, dtype=float)

    if len(thresholds) == 0:
        return 0.0, 0.0

    best_threshold = float(thresholds[0])
    best_f1 = -1.0

    for threshold in thresholds:
        y_pred = predict_labels(proba, threshold)
        tp, fp, tn, fn = confusion_counts(y_true, y_pred)
        f1 = metrics_from_counts(tp, fp, tn, fn)["f1"]

        # 使用严格大于，保证并列时留下先遇到的阈值
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return float(best_threshold), float(best_f1)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """返回混淆矩阵计数和分类指标的汇总字典。"""
    tp, fp, tn, fn = confusion_counts(y_true, y_pred)
    metrics = metrics_from_counts(tp, fp, tn, fn)

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "accuracy": metrics["accuracy"],
    }


# ============================================================
# 25-27. 新文本向量化、单条预测与错误分析
# ============================================================

def vectorize_texts(
    texts: list,
    vocab: dict,
    idf: np.ndarray
) -> np.ndarray:
    """把原始文本转换成训练时同样格式的 TF-IDF 矩阵。"""
    tokenized_docs = tokenize_corpus(texts)
    bow_matrix = corpus_to_bow_matrix(tokenized_docs, vocab)

    return transform_tfidf(bow_matrix, idf)


def predict_text(
    text: str,
    vocab: dict,
    idf: np.ndarray,
    w: np.ndarray,
    b: float,
    threshold: float = 0.5
) -> int:
    """使用训练好的参数预测一条原始文本。"""
    X = vectorize_texts([text], vocab, idf)
    proba = logistic_predict_proba(X, w, b)
    prediction = predict_labels(proba, threshold)

    return int(prediction[0])


def collect_prediction_errors(
    texts: list,
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """收集假阳性和假阴性的原始文本，保持输入顺序。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    false_positives = []
    false_negatives = []

    for text, true_label, predicted_label in zip(texts, y_true, y_pred):
        if true_label == 0 and predicted_label == 1:
            false_positives.append(text)
        elif true_label == 1 and predicted_label == 0:
            false_negatives.append(text)

    return {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }

# Step 13 - sigmoid
import numpy as np


# ============================================================
# 1-3. 文本清洗与分词
# ============================================================

def clean_text(text: str) -> str:
    """
    转成小写，并把每个非字母字符替换为空格。
    注意：保留连续空格以及开头、结尾产生的空格。
    """
    text = text.lower()
    return "".join(ch if ch.isalpha() else " " for ch in text)


def tokenize(text: str) -> list:
    """按任意空白字符切分，并自动丢弃空 token。"""
    return text.split()


def tokenize_corpus(texts: list) -> list:
    """清洗并分词整个文本列表。"""
    return [tokenize(clean_text(text)) for text in texts]


# ============================================================
# 4-8. 数据划分、词频、词表与 Bag-of-Words
# ============================================================

def split_train_val_test_indices(
    n_samples: int,
    val_fraction: float,
    test_fraction: float,
    seed: int = 0
) -> tuple:
    """
    生成互不重叠的 train/validation/test 索引。
    按 train、validation、test 的顺序切分打乱后的索引。
    """
    np.random.seed(seed)

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    n_val = int(n_samples * val_fraction)
    n_test = int(n_samples * test_fraction)
    n_train = n_samples - n_val - n_test

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return train_idx, val_idx, test_idx


def count_word_frequencies(tokenized_docs: list) -> dict:
    """统计每个 token 在整个语料中出现的总次数。"""
    counts = {}

    for document in tokenized_docs:
        for token in document:
            counts[token] = counts.get(token, 0) + 1

    return counts


def build_vocabulary(word_counts: dict, max_size: int) -> dict:
    """
    选出词频最高的 max_size 个词。
    词频相同时，按字母顺序排列。
    """
    max_size = max(0, int(max_size))

    ranked_words = sorted(
        word_counts.items(),
        key=lambda item: (-item[1], item[0])
    )

    return {
        word: index
        for index, (word, _) in enumerate(ranked_words[:max_size])
    }


def tokens_to_bow(tokens: list, vocab: dict) -> np.ndarray:
    """将一篇文档转成词袋计数向量，未知词忽略。"""
    vector = np.zeros(len(vocab), dtype=float)

    for token in tokens:
        if token in vocab:
            vector[vocab[token]] += 1.0

    return vector


def corpus_to_bow_matrix(tokenized_docs: list, vocab: dict) -> np.ndarray:
    """将多个文档转成形状为 (文档数, 词表大小) 的词袋矩阵。"""
    matrix = np.zeros(
        (len(tokenized_docs), len(vocab)),
        dtype=float
    )

    for row, tokens in enumerate(tokenized_docs):
        matrix[row] = tokens_to_bow(tokens, vocab)

    return matrix


# ============================================================
# 9-12. 文档频率与 TF-IDF
# ============================================================

def compute_document_frequencies(bow_matrix: np.ndarray) -> np.ndarray:
    """
    计算每个词出现于多少篇文档。
    每篇文档中，一个词出现一次或多次，都只计为出现过。
    """
    bow_matrix = np.asarray(bow_matrix)
    return np.sum(bow_matrix > 0, axis=0).astype(int)


def compute_idf(df: np.ndarray, n_docs: int) -> np.ndarray:
    """计算平滑 IDF，使用自然对数。"""
    df = np.asarray(df, dtype=float)
    return np.log((n_docs + 1) / (df + 1)) + 1.0


def transform_tfidf(
    bow_matrix: np.ndarray,
    idf: np.ndarray
) -> np.ndarray:
    """
    每列的词袋计数乘对应 IDF。
    支持输入一个向量 (V,) 或矩阵 (N, V)。
    """
    bow_matrix = np.asarray(bow_matrix, dtype=float)
    idf = np.asarray(idf, dtype=float)
    return bow_matrix * idf


def fit_tfidf(bow_train: np.ndarray) -> np.ndarray:
    """只根据训练集词袋矩阵拟合 IDF。"""
    bow_train = np.asarray(bow_train, dtype=float)

    df = compute_document_frequencies(bow_train)
    n_docs = bow_train.shape[0]

    return compute_idf(df, n_docs)


# ============================================================
# 13-16. 逻辑回归核心：Sigmoid、概率、损失与梯度
# ============================================================

def sigmoid(z: np.ndarray) -> np.ndarray:
    """数值稳定的 sigmoid，返回与输入形状相同的数组。"""
    z = np.asarray(z, dtype=float)
    flat = z.reshape(-1)
    result = np.empty_like(flat)

    # 正数和负数分开计算，避免 exp 溢出
    positive = flat >= 0
    negative = ~positive

    result[positive] = 1.0 / (1.0 + np.exp(-flat[positive]))

    exp_z = np.exp(flat[negative])
    result[negative] = exp_z / (1.0 + exp_z)

    return result.reshape(z.shape)


def logistic_predict_proba(
    X: np.ndarray,
    w: np.ndarray,
    b: float
) -> np.ndarray:
    """计算每一行属于类别 1 的概率。"""
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)

    return sigmoid(X @ w + b)


def binary_cross_entropy(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    w: np.ndarray,
    l2_lambda: float
) -> float:
    """
    平均二元交叉熵 + L2 权重惩罚。
    bias 不在这里传入，所以不会被正则化。
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    w = np.asarray(w, dtype=float)

    # 避免 log(0)
    y_prob = np.clip(y_prob, 1e-15, 1.0 - 1e-15)

    if y_true.size == 0:
        mean_bce = 0.0
    else:
        mean_bce = -np.mean(
            y_true * np.log(y_prob)
            + (1.0 - y_true) * np.log(1.0 - y_prob)
        )

    l2_penalty = l2_lambda * np.sum(w ** 2) / 2.0

    return float(mean_bce + l2_penalty)


def logistic_gradients(
    X: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    w: np.ndarray,
    l2_lambda: float
) -> tuple:
    """计算平均 BCE + L2 对权重和 bias 的梯度。"""
    X = np.asarray(X, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    w = np.asarray(w, dtype=float)

    n_samples = X.shape[0]

    if n_samples == 0:
        return l2_lambda * w, 0.0

    residual = y_prob - y_true

    # dL/dw：数据损失梯度 + L2 梯度
    dw = (X.T @ residual) / n_samples + l2_lambda * w

    # bias 不加 L2 惩罚
    db = float(np.mean(residual))

    return dw, db


# ============================================================
# 17-20. 初始化、梯度下降训练与阈值预测
# ============================================================

def initialize_logistic_params(n_features: int) -> tuple:
    """权重和 bias 都从 0 开始。"""
    w = np.zeros(n_features, dtype=float)
    b = 0.0

    return w, b


def gradient_descent_step(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    b: float,
    lr: float,
    l2_lambda: float
) -> tuple:
    """
    做一次全批量梯度下降。
    返回的 loss 是更新参数之前计算的损失。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)

    probabilities = logistic_predict_proba(X, w, b)
    loss = binary_cross_entropy(y, probabilities, w, l2_lambda)

    dw, db = logistic_gradients(
        X, y, probabilities, w, l2_lambda
    )

    w_new = w - lr * dw
    b_new = float(b - lr * db)

    return w_new, b_new, loss


def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    lr: float,
    l2_lambda: float,
    n_epochs: int
) -> tuple:
    """从零开始训练逻辑回归，并记录每轮更新前的 loss。"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    w, b = initialize_logistic_params(X.shape[1])
    losses = []

    for _ in range(n_epochs):
        w, b, loss = gradient_descent_step(
            X, y, w, b, lr, l2_lambda
        )
        losses.append(float(loss))

    return w, float(b), losses


def predict_labels(
    proba: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """概率大于等于阈值时预测为 1，否则预测为 0。"""
    proba = np.asarray(proba, dtype=float)
    return (proba >= threshold).astype(int)


# ============================================================
# 21-24. 混淆矩阵计数、分类指标与阈值调优
# ============================================================

def confusion_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> tuple:
    """返回 (TP, FP, TN, FN)，均为 Python int。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    return tp, fp, tn, fn


def metrics_from_counts(
    tp: int,
    fp: int,
    tn: int,
    fn: int
) -> dict:
    """根据混淆矩阵计数计算 precision、recall、F1 和 accuracy。"""
    precision_denominator = tp + fp
    recall_denominator = tp + fn
    f1_denominator = 2 * tp + fp + fn
    total = tp + fp + tn + fn

    precision = (
        tp / precision_denominator
        if precision_denominator > 0 else 0.0
    )
    recall = (
        tp / recall_denominator
        if recall_denominator > 0 else 0.0
    )
    f1 = (
        2 * tp / f1_denominator
        if f1_denominator > 0 else 0.0
    )
    accuracy = (
        (tp + tn) / total
        if total > 0 else 0.0
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


def tune_decision_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    thresholds: np.ndarray = None
) -> tuple:
    """
    在候选阈值中选 F1 最高的那个。
    F1 并列时保留 thresholds 中最先出现的阈值。
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)

    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)
    else:
        thresholds = np.asarray(thresholds, dtype=float)

    if len(thresholds) == 0:
        return 0.0, 0.0

    best_threshold = float(thresholds[0])
    best_f1 = -1.0

    for threshold in thresholds:
        y_pred = predict_labels(proba, threshold)
        tp, fp, tn, fn = confusion_counts(y_true, y_pred)
        f1 = metrics_from_counts(tp, fp, tn, fn)["f1"]

        # 使用严格大于，保证并列时留下先遇到的阈值
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return float(best_threshold), float(best_f1)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """返回混淆矩阵计数和分类指标的汇总字典。"""
    tp, fp, tn, fn = confusion_counts(y_true, y_pred)
    metrics = metrics_from_counts(tp, fp, tn, fn)

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "accuracy": metrics["accuracy"],
    }


# ============================================================
# 25-27. 新文本向量化、单条预测与错误分析
# ============================================================

def vectorize_texts(
    texts: list,
    vocab: dict,
    idf: np.ndarray
) -> np.ndarray:
    """把原始文本转换成训练时同样格式的 TF-IDF 矩阵。"""
    tokenized_docs = tokenize_corpus(texts)
    bow_matrix = corpus_to_bow_matrix(tokenized_docs, vocab)

    return transform_tfidf(bow_matrix, idf)


def predict_text(
    text: str,
    vocab: dict,
    idf: np.ndarray,
    w: np.ndarray,
    b: float,
    threshold: float = 0.5
) -> int:
    """使用训练好的参数预测一条原始文本。"""
    X = vectorize_texts([text], vocab, idf)
    proba = logistic_predict_proba(X, w, b)
    prediction = predict_labels(proba, threshold)

    return int(prediction[0])


def collect_prediction_errors(
    texts: list,
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """收集假阳性和假阴性的原始文本，保持输入顺序。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    false_positives = []
    false_negatives = []

    for text, true_label, predicted_label in zip(texts, y_true, y_pred):
        if true_label == 0 and predicted_label == 1:
            false_positives.append(text)
        elif true_label == 1 and predicted_label == 0:
            false_negatives.append(text)

    return {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }

# Step 14 - logistic_predict_proba
import numpy as np


# ============================================================
# 1-3. 文本清洗与分词
# ============================================================

def clean_text(text: str) -> str:
    """
    转成小写，并把每个非字母字符替换为空格。
    注意：保留连续空格以及开头、结尾产生的空格。
    """
    text = text.lower()
    return "".join(ch if ch.isalpha() else " " for ch in text)


def tokenize(text: str) -> list:
    """按任意空白字符切分，并自动丢弃空 token。"""
    return text.split()


def tokenize_corpus(texts: list) -> list:
    """清洗并分词整个文本列表。"""
    return [tokenize(clean_text(text)) for text in texts]


# ============================================================
# 4-8. 数据划分、词频、词表与 Bag-of-Words
# ============================================================

def split_train_val_test_indices(
    n_samples: int,
    val_fraction: float,
    test_fraction: float,
    seed: int = 0
) -> tuple:
    """
    生成互不重叠的 train/validation/test 索引。
    按 train、validation、test 的顺序切分打乱后的索引。
    """
    np.random.seed(seed)

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    n_val = int(n_samples * val_fraction)
    n_test = int(n_samples * test_fraction)
    n_train = n_samples - n_val - n_test

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return train_idx, val_idx, test_idx


def count_word_frequencies(tokenized_docs: list) -> dict:
    """统计每个 token 在整个语料中出现的总次数。"""
    counts = {}

    for document in tokenized_docs:
        for token in document:
            counts[token] = counts.get(token, 0) + 1

    return counts


def build_vocabulary(word_counts: dict, max_size: int) -> dict:
    """
    选出词频最高的 max_size 个词。
    词频相同时，按字母顺序排列。
    """
    max_size = max(0, int(max_size))

    ranked_words = sorted(
        word_counts.items(),
        key=lambda item: (-item[1], item[0])
    )

    return {
        word: index
        for index, (word, _) in enumerate(ranked_words[:max_size])
    }


def tokens_to_bow(tokens: list, vocab: dict) -> np.ndarray:
    """将一篇文档转成词袋计数向量，未知词忽略。"""
    vector = np.zeros(len(vocab), dtype=float)

    for token in tokens:
        if token in vocab:
            vector[vocab[token]] += 1.0

    return vector


def corpus_to_bow_matrix(tokenized_docs: list, vocab: dict) -> np.ndarray:
    """将多个文档转成形状为 (文档数, 词表大小) 的词袋矩阵。"""
    matrix = np.zeros(
        (len(tokenized_docs), len(vocab)),
        dtype=float
    )

    for row, tokens in enumerate(tokenized_docs):
        matrix[row] = tokens_to_bow(tokens, vocab)

    return matrix


# ============================================================
# 9-12. 文档频率与 TF-IDF
# ============================================================

def compute_document_frequencies(bow_matrix: np.ndarray) -> np.ndarray:
    """
    计算每个词出现于多少篇文档。
    每篇文档中，一个词出现一次或多次，都只计为出现过。
    """
    bow_matrix = np.asarray(bow_matrix)
    return np.sum(bow_matrix > 0, axis=0).astype(int)


def compute_idf(df: np.ndarray, n_docs: int) -> np.ndarray:
    """计算平滑 IDF，使用自然对数。"""
    df = np.asarray(df, dtype=float)
    return np.log((n_docs + 1) / (df + 1)) + 1.0


def transform_tfidf(
    bow_matrix: np.ndarray,
    idf: np.ndarray
) -> np.ndarray:
    """
    每列的词袋计数乘对应 IDF。
    支持输入一个向量 (V,) 或矩阵 (N, V)。
    """
    bow_matrix = np.asarray(bow_matrix, dtype=float)
    idf = np.asarray(idf, dtype=float)
    return bow_matrix * idf


def fit_tfidf(bow_train: np.ndarray) -> np.ndarray:
    """只根据训练集词袋矩阵拟合 IDF。"""
    bow_train = np.asarray(bow_train, dtype=float)

    df = compute_document_frequencies(bow_train)
    n_docs = bow_train.shape[0]

    return compute_idf(df, n_docs)


# ============================================================
# 13-16. 逻辑回归核心：Sigmoid、概率、损失与梯度
# ============================================================

def sigmoid(z: np.ndarray) -> np.ndarray:
    """数值稳定的 sigmoid，返回与输入形状相同的数组。"""
    z = np.asarray(z, dtype=float)
    flat = z.reshape(-1)
    result = np.empty_like(flat)

    # 正数和负数分开计算，避免 exp 溢出
    positive = flat >= 0
    negative = ~positive

    result[positive] = 1.0 / (1.0 + np.exp(-flat[positive]))

    exp_z = np.exp(flat[negative])
    result[negative] = exp_z / (1.0 + exp_z)

    return result.reshape(z.shape)


def logistic_predict_proba(
    X: np.ndarray,
    w: np.ndarray,
    b: float
) -> np.ndarray:
    """计算每一行属于类别 1 的概率。"""
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)

    return sigmoid(X @ w + b)


def binary_cross_entropy(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    w: np.ndarray,
    l2_lambda: float
) -> float:
    """
    平均二元交叉熵 + L2 权重惩罚。
    bias 不在这里传入，所以不会被正则化。
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    w = np.asarray(w, dtype=float)

    # 避免 log(0)
    y_prob = np.clip(y_prob, 1e-15, 1.0 - 1e-15)

    if y_true.size == 0:
        mean_bce = 0.0
    else:
        mean_bce = -np.mean(
            y_true * np.log(y_prob)
            + (1.0 - y_true) * np.log(1.0 - y_prob)
        )

    l2_penalty = l2_lambda * np.sum(w ** 2) / 2.0

    return float(mean_bce + l2_penalty)


def logistic_gradients(
    X: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    w: np.ndarray,
    l2_lambda: float
) -> tuple:
    """计算平均 BCE + L2 对权重和 bias 的梯度。"""
    X = np.asarray(X, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    w = np.asarray(w, dtype=float)

    n_samples = X.shape[0]

    if n_samples == 0:
        return l2_lambda * w, 0.0

    residual = y_prob - y_true

    # dL/dw：数据损失梯度 + L2 梯度
    dw = (X.T @ residual) / n_samples + l2_lambda * w

    # bias 不加 L2 惩罚
    db = float(np.mean(residual))

    return dw, db


# ============================================================
# 17-20. 初始化、梯度下降训练与阈值预测
# ============================================================

def initialize_logistic_params(n_features: int) -> tuple:
    """权重和 bias 都从 0 开始。"""
    w = np.zeros(n_features, dtype=float)
    b = 0.0

    return w, b


def gradient_descent_step(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    b: float,
    lr: float,
    l2_lambda: float
) -> tuple:
    """
    做一次全批量梯度下降。
    返回的 loss 是更新参数之前计算的损失。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)

    probabilities = logistic_predict_proba(X, w, b)
    loss = binary_cross_entropy(y, probabilities, w, l2_lambda)

    dw, db = logistic_gradients(
        X, y, probabilities, w, l2_lambda
    )

    w_new = w - lr * dw
    b_new = float(b - lr * db)

    return w_new, b_new, loss


def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    lr: float,
    l2_lambda: float,
    n_epochs: int
) -> tuple:
    """从零开始训练逻辑回归，并记录每轮更新前的 loss。"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    w, b = initialize_logistic_params(X.shape[1])
    losses = []

    for _ in range(n_epochs):
        w, b, loss = gradient_descent_step(
            X, y, w, b, lr, l2_lambda
        )
        losses.append(float(loss))

    return w, float(b), losses


def predict_labels(
    proba: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """概率大于等于阈值时预测为 1，否则预测为 0。"""
    proba = np.asarray(proba, dtype=float)
    return (proba >= threshold).astype(int)


# ============================================================
# 21-24. 混淆矩阵计数、分类指标与阈值调优
# ============================================================

def confusion_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> tuple:
    """返回 (TP, FP, TN, FN)，均为 Python int。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    return tp, fp, tn, fn


def metrics_from_counts(
    tp: int,
    fp: int,
    tn: int,
    fn: int
) -> dict:
    """根据混淆矩阵计数计算 precision、recall、F1 和 accuracy。"""
    precision_denominator = tp + fp
    recall_denominator = tp + fn
    f1_denominator = 2 * tp + fp + fn
    total = tp + fp + tn + fn

    precision = (
        tp / precision_denominator
        if precision_denominator > 0 else 0.0
    )
    recall = (
        tp / recall_denominator
        if recall_denominator > 0 else 0.0
    )
    f1 = (
        2 * tp / f1_denominator
        if f1_denominator > 0 else 0.0
    )
    accuracy = (
        (tp + tn) / total
        if total > 0 else 0.0
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


def tune_decision_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    thresholds: np.ndarray = None
) -> tuple:
    """
    在候选阈值中选 F1 最高的那个。
    F1 并列时保留 thresholds 中最先出现的阈值。
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)

    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)
    else:
        thresholds = np.asarray(thresholds, dtype=float)

    if len(thresholds) == 0:
        return 0.0, 0.0

    best_threshold = float(thresholds[0])
    best_f1 = -1.0

    for threshold in thresholds:
        y_pred = predict_labels(proba, threshold)
        tp, fp, tn, fn = confusion_counts(y_true, y_pred)
        f1 = metrics_from_counts(tp, fp, tn, fn)["f1"]

        # 使用严格大于，保证并列时留下先遇到的阈值
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return float(best_threshold), float(best_f1)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """返回混淆矩阵计数和分类指标的汇总字典。"""
    tp, fp, tn, fn = confusion_counts(y_true, y_pred)
    metrics = metrics_from_counts(tp, fp, tn, fn)

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "accuracy": metrics["accuracy"],
    }


# ============================================================
# 25-27. 新文本向量化、单条预测与错误分析
# ============================================================

def vectorize_texts(
    texts: list,
    vocab: dict,
    idf: np.ndarray
) -> np.ndarray:
    """把原始文本转换成训练时同样格式的 TF-IDF 矩阵。"""
    tokenized_docs = tokenize_corpus(texts)
    bow_matrix = corpus_to_bow_matrix(tokenized_docs, vocab)

    return transform_tfidf(bow_matrix, idf)


def predict_text(
    text: str,
    vocab: dict,
    idf: np.ndarray,
    w: np.ndarray,
    b: float,
    threshold: float = 0.5
) -> int:
    """使用训练好的参数预测一条原始文本。"""
    X = vectorize_texts([text], vocab, idf)
    proba = logistic_predict_proba(X, w, b)
    prediction = predict_labels(proba, threshold)

    return int(prediction[0])


def collect_prediction_errors(
    texts: list,
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """收集假阳性和假阴性的原始文本，保持输入顺序。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    false_positives = []
    false_negatives = []

    for text, true_label, predicted_label in zip(texts, y_true, y_pred):
        if true_label == 0 and predicted_label == 1:
            false_positives.append(text)
        elif true_label == 1 and predicted_label == 0:
            false_negatives.append(text)

    return {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }

# Step 15 - binary_cross_entropy
import numpy as np


# ============================================================
# 1-3. 文本清洗与分词
# ============================================================

def clean_text(text: str) -> str:
    """
    转成小写，并把每个非字母字符替换为空格。
    注意：保留连续空格以及开头、结尾产生的空格。
    """
    text = text.lower()
    return "".join(ch if ch.isalpha() else " " for ch in text)


def tokenize(text: str) -> list:
    """按任意空白字符切分，并自动丢弃空 token。"""
    return text.split()


def tokenize_corpus(texts: list) -> list:
    """清洗并分词整个文本列表。"""
    return [tokenize(clean_text(text)) for text in texts]


# ============================================================
# 4-8. 数据划分、词频、词表与 Bag-of-Words
# ============================================================

def split_train_val_test_indices(
    n_samples: int,
    val_fraction: float,
    test_fraction: float,
    seed: int = 0
) -> tuple:
    """
    生成互不重叠的 train/validation/test 索引。
    按 train、validation、test 的顺序切分打乱后的索引。
    """
    np.random.seed(seed)

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    n_val = int(n_samples * val_fraction)
    n_test = int(n_samples * test_fraction)
    n_train = n_samples - n_val - n_test

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return train_idx, val_idx, test_idx


def count_word_frequencies(tokenized_docs: list) -> dict:
    """统计每个 token 在整个语料中出现的总次数。"""
    counts = {}

    for document in tokenized_docs:
        for token in document:
            counts[token] = counts.get(token, 0) + 1

    return counts


def build_vocabulary(word_counts: dict, max_size: int) -> dict:
    """
    选出词频最高的 max_size 个词。
    词频相同时，按字母顺序排列。
    """
    max_size = max(0, int(max_size))

    ranked_words = sorted(
        word_counts.items(),
        key=lambda item: (-item[1], item[0])
    )

    return {
        word: index
        for index, (word, _) in enumerate(ranked_words[:max_size])
    }


def tokens_to_bow(tokens: list, vocab: dict) -> np.ndarray:
    """将一篇文档转成词袋计数向量，未知词忽略。"""
    vector = np.zeros(len(vocab), dtype=float)

    for token in tokens:
        if token in vocab:
            vector[vocab[token]] += 1.0

    return vector


def corpus_to_bow_matrix(tokenized_docs: list, vocab: dict) -> np.ndarray:
    """将多个文档转成形状为 (文档数, 词表大小) 的词袋矩阵。"""
    matrix = np.zeros(
        (len(tokenized_docs), len(vocab)),
        dtype=float
    )

    for row, tokens in enumerate(tokenized_docs):
        matrix[row] = tokens_to_bow(tokens, vocab)

    return matrix


# ============================================================
# 9-12. 文档频率与 TF-IDF
# ============================================================

def compute_document_frequencies(bow_matrix: np.ndarray) -> np.ndarray:
    """
    计算每个词出现于多少篇文档。
    每篇文档中，一个词出现一次或多次，都只计为出现过。
    """
    bow_matrix = np.asarray(bow_matrix)
    return np.sum(bow_matrix > 0, axis=0).astype(int)


def compute_idf(df: np.ndarray, n_docs: int) -> np.ndarray:
    """计算平滑 IDF，使用自然对数。"""
    df = np.asarray(df, dtype=float)
    return np.log((n_docs + 1) / (df + 1)) + 1.0


def transform_tfidf(
    bow_matrix: np.ndarray,
    idf: np.ndarray
) -> np.ndarray:
    """
    每列的词袋计数乘对应 IDF。
    支持输入一个向量 (V,) 或矩阵 (N, V)。
    """
    bow_matrix = np.asarray(bow_matrix, dtype=float)
    idf = np.asarray(idf, dtype=float)
    return bow_matrix * idf


def fit_tfidf(bow_train: np.ndarray) -> np.ndarray:
    """只根据训练集词袋矩阵拟合 IDF。"""
    bow_train = np.asarray(bow_train, dtype=float)

    df = compute_document_frequencies(bow_train)
    n_docs = bow_train.shape[0]

    return compute_idf(df, n_docs)


# ============================================================
# 13-16. 逻辑回归核心：Sigmoid、概率、损失与梯度
# ============================================================

def sigmoid(z: np.ndarray) -> np.ndarray:
    """数值稳定的 sigmoid，返回与输入形状相同的数组。"""
    z = np.asarray(z, dtype=float)
    flat = z.reshape(-1)
    result = np.empty_like(flat)

    # 正数和负数分开计算，避免 exp 溢出
    positive = flat >= 0
    negative = ~positive

    result[positive] = 1.0 / (1.0 + np.exp(-flat[positive]))

    exp_z = np.exp(flat[negative])
    result[negative] = exp_z / (1.0 + exp_z)

    return result.reshape(z.shape)


def logistic_predict_proba(
    X: np.ndarray,
    w: np.ndarray,
    b: float
) -> np.ndarray:
    """计算每一行属于类别 1 的概率。"""
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)

    return sigmoid(X @ w + b)


def binary_cross_entropy(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    w: np.ndarray,
    l2_lambda: float
) -> float:
    """
    平均二元交叉熵 + L2 权重惩罚。
    bias 不在这里传入，所以不会被正则化。
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    w = np.asarray(w, dtype=float)

    # 避免 log(0)
    y_prob = np.clip(y_prob, 1e-15, 1.0 - 1e-15)

    if y_true.size == 0:
        mean_bce = 0.0
    else:
        mean_bce = -np.mean(
            y_true * np.log(y_prob)
            + (1.0 - y_true) * np.log(1.0 - y_prob)
        )

    l2_penalty = l2_lambda * np.sum(w ** 2) / 2.0

    return float(mean_bce + l2_penalty)


def logistic_gradients(
    X: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    w: np.ndarray,
    l2_lambda: float
) -> tuple:
    """计算平均 BCE + L2 对权重和 bias 的梯度。"""
    X = np.asarray(X, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    w = np.asarray(w, dtype=float)

    n_samples = X.shape[0]

    if n_samples == 0:
        return l2_lambda * w, 0.0

    residual = y_prob - y_true

    # dL/dw：数据损失梯度 + L2 梯度
    dw = (X.T @ residual) / n_samples + l2_lambda * w

    # bias 不加 L2 惩罚
    db = float(np.mean(residual))

    return dw, db


# ============================================================
# 17-20. 初始化、梯度下降训练与阈值预测
# ============================================================

def initialize_logistic_params(n_features: int) -> tuple:
    """权重和 bias 都从 0 开始。"""
    w = np.zeros(n_features, dtype=float)
    b = 0.0

    return w, b


def gradient_descent_step(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    b: float,
    lr: float,
    l2_lambda: float
) -> tuple:
    """
    做一次全批量梯度下降。
    返回的 loss 是更新参数之前计算的损失。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)

    probabilities = logistic_predict_proba(X, w, b)
    loss = binary_cross_entropy(y, probabilities, w, l2_lambda)

    dw, db = logistic_gradients(
        X, y, probabilities, w, l2_lambda
    )

    w_new = w - lr * dw
    b_new = float(b - lr * db)

    return w_new, b_new, loss


def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    lr: float,
    l2_lambda: float,
    n_epochs: int
) -> tuple:
    """从零开始训练逻辑回归，并记录每轮更新前的 loss。"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    w, b = initialize_logistic_params(X.shape[1])
    losses = []

    for _ in range(n_epochs):
        w, b, loss = gradient_descent_step(
            X, y, w, b, lr, l2_lambda
        )
        losses.append(float(loss))

    return w, float(b), losses


def predict_labels(
    proba: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """概率大于等于阈值时预测为 1，否则预测为 0。"""
    proba = np.asarray(proba, dtype=float)
    return (proba >= threshold).astype(int)


# ============================================================
# 21-24. 混淆矩阵计数、分类指标与阈值调优
# ============================================================

def confusion_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> tuple:
    """返回 (TP, FP, TN, FN)，均为 Python int。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    return tp, fp, tn, fn


def metrics_from_counts(
    tp: int,
    fp: int,
    tn: int,
    fn: int
) -> dict:
    """根据混淆矩阵计数计算 precision、recall、F1 和 accuracy。"""
    precision_denominator = tp + fp
    recall_denominator = tp + fn
    f1_denominator = 2 * tp + fp + fn
    total = tp + fp + tn + fn

    precision = (
        tp / precision_denominator
        if precision_denominator > 0 else 0.0
    )
    recall = (
        tp / recall_denominator
        if recall_denominator > 0 else 0.0
    )
    f1 = (
        2 * tp / f1_denominator
        if f1_denominator > 0 else 0.0
    )
    accuracy = (
        (tp + tn) / total
        if total > 0 else 0.0
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


def tune_decision_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    thresholds: np.ndarray = None
) -> tuple:
    """
    在候选阈值中选 F1 最高的那个。
    F1 并列时保留 thresholds 中最先出现的阈值。
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)

    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)
    else:
        thresholds = np.asarray(thresholds, dtype=float)

    if len(thresholds) == 0:
        return 0.0, 0.0

    best_threshold = float(thresholds[0])
    best_f1 = -1.0

    for threshold in thresholds:
        y_pred = predict_labels(proba, threshold)
        tp, fp, tn, fn = confusion_counts(y_true, y_pred)
        f1 = metrics_from_counts(tp, fp, tn, fn)["f1"]

        # 使用严格大于，保证并列时留下先遇到的阈值
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return float(best_threshold), float(best_f1)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """返回混淆矩阵计数和分类指标的汇总字典。"""
    tp, fp, tn, fn = confusion_counts(y_true, y_pred)
    metrics = metrics_from_counts(tp, fp, tn, fn)

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "accuracy": metrics["accuracy"],
    }


# ============================================================
# 25-27. 新文本向量化、单条预测与错误分析
# ============================================================

def vectorize_texts(
    texts: list,
    vocab: dict,
    idf: np.ndarray
) -> np.ndarray:
    """把原始文本转换成训练时同样格式的 TF-IDF 矩阵。"""
    tokenized_docs = tokenize_corpus(texts)
    bow_matrix = corpus_to_bow_matrix(tokenized_docs, vocab)

    return transform_tfidf(bow_matrix, idf)


def predict_text(
    text: str,
    vocab: dict,
    idf: np.ndarray,
    w: np.ndarray,
    b: float,
    threshold: float = 0.5
) -> int:
    """使用训练好的参数预测一条原始文本。"""
    X = vectorize_texts([text], vocab, idf)
    proba = logistic_predict_proba(X, w, b)
    prediction = predict_labels(proba, threshold)

    return int(prediction[0])


def collect_prediction_errors(
    texts: list,
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """收集假阳性和假阴性的原始文本，保持输入顺序。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    false_positives = []
    false_negatives = []

    for text, true_label, predicted_label in zip(texts, y_true, y_pred):
        if true_label == 0 and predicted_label == 1:
            false_positives.append(text)
        elif true_label == 1 and predicted_label == 0:
            false_negatives.append(text)

    return {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }

# Step 16 - logistic_gradients
import numpy as np


# ============================================================
# 1-3. 文本清洗与分词
# ============================================================

def clean_text(text: str) -> str:
    """
    转成小写，并把每个非字母字符替换为空格。
    注意：保留连续空格以及开头、结尾产生的空格。
    """
    text = text.lower()
    return "".join(ch if ch.isalpha() else " " for ch in text)


def tokenize(text: str) -> list:
    """按任意空白字符切分，并自动丢弃空 token。"""
    return text.split()


def tokenize_corpus(texts: list) -> list:
    """清洗并分词整个文本列表。"""
    return [tokenize(clean_text(text)) for text in texts]


# ============================================================
# 4-8. 数据划分、词频、词表与 Bag-of-Words
# ============================================================

def split_train_val_test_indices(
    n_samples: int,
    val_fraction: float,
    test_fraction: float,
    seed: int = 0
) -> tuple:
    """
    生成互不重叠的 train/validation/test 索引。
    按 train、validation、test 的顺序切分打乱后的索引。
    """
    np.random.seed(seed)

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    n_val = int(n_samples * val_fraction)
    n_test = int(n_samples * test_fraction)
    n_train = n_samples - n_val - n_test

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return train_idx, val_idx, test_idx


def count_word_frequencies(tokenized_docs: list) -> dict:
    """统计每个 token 在整个语料中出现的总次数。"""
    counts = {}

    for document in tokenized_docs:
        for token in document:
            counts[token] = counts.get(token, 0) + 1

    return counts


def build_vocabulary(word_counts: dict, max_size: int) -> dict:
    """
    选出词频最高的 max_size 个词。
    词频相同时，按字母顺序排列。
    """
    max_size = max(0, int(max_size))

    ranked_words = sorted(
        word_counts.items(),
        key=lambda item: (-item[1], item[0])
    )

    return {
        word: index
        for index, (word, _) in enumerate(ranked_words[:max_size])
    }


def tokens_to_bow(tokens: list, vocab: dict) -> np.ndarray:
    """将一篇文档转成词袋计数向量，未知词忽略。"""
    vector = np.zeros(len(vocab), dtype=float)

    for token in tokens:
        if token in vocab:
            vector[vocab[token]] += 1.0

    return vector


def corpus_to_bow_matrix(tokenized_docs: list, vocab: dict) -> np.ndarray:
    """将多个文档转成形状为 (文档数, 词表大小) 的词袋矩阵。"""
    matrix = np.zeros(
        (len(tokenized_docs), len(vocab)),
        dtype=float
    )

    for row, tokens in enumerate(tokenized_docs):
        matrix[row] = tokens_to_bow(tokens, vocab)

    return matrix


# ============================================================
# 9-12. 文档频率与 TF-IDF
# ============================================================

def compute_document_frequencies(bow_matrix: np.ndarray) -> np.ndarray:
    """
    计算每个词出现于多少篇文档。
    每篇文档中，一个词出现一次或多次，都只计为出现过。
    """
    bow_matrix = np.asarray(bow_matrix)
    return np.sum(bow_matrix > 0, axis=0).astype(int)


def compute_idf(df: np.ndarray, n_docs: int) -> np.ndarray:
    """计算平滑 IDF，使用自然对数。"""
    df = np.asarray(df, dtype=float)
    return np.log((n_docs + 1) / (df + 1)) + 1.0


def transform_tfidf(
    bow_matrix: np.ndarray,
    idf: np.ndarray
) -> np.ndarray:
    """
    每列的词袋计数乘对应 IDF。
    支持输入一个向量 (V,) 或矩阵 (N, V)。
    """
    bow_matrix = np.asarray(bow_matrix, dtype=float)
    idf = np.asarray(idf, dtype=float)
    return bow_matrix * idf


def fit_tfidf(bow_train: np.ndarray) -> np.ndarray:
    """只根据训练集词袋矩阵拟合 IDF。"""
    bow_train = np.asarray(bow_train, dtype=float)

    df = compute_document_frequencies(bow_train)
    n_docs = bow_train.shape[0]

    return compute_idf(df, n_docs)


# ============================================================
# 13-16. 逻辑回归核心：Sigmoid、概率、损失与梯度
# ============================================================

def sigmoid(z: np.ndarray) -> np.ndarray:
    """数值稳定的 sigmoid，返回与输入形状相同的数组。"""
    z = np.asarray(z, dtype=float)
    flat = z.reshape(-1)
    result = np.empty_like(flat)

    # 正数和负数分开计算，避免 exp 溢出
    positive = flat >= 0
    negative = ~positive

    result[positive] = 1.0 / (1.0 + np.exp(-flat[positive]))

    exp_z = np.exp(flat[negative])
    result[negative] = exp_z / (1.0 + exp_z)

    return result.reshape(z.shape)


def logistic_predict_proba(
    X: np.ndarray,
    w: np.ndarray,
    b: float
) -> np.ndarray:
    """计算每一行属于类别 1 的概率。"""
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)

    return sigmoid(X @ w + b)


def binary_cross_entropy(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    w: np.ndarray,
    l2_lambda: float
) -> float:
    """
    平均二元交叉熵 + L2 权重惩罚。
    bias 不在这里传入，所以不会被正则化。
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    w = np.asarray(w, dtype=float)

    # 避免 log(0)
    y_prob = np.clip(y_prob, 1e-15, 1.0 - 1e-15)

    if y_true.size == 0:
        mean_bce = 0.0
    else:
        mean_bce = -np.mean(
            y_true * np.log(y_prob)
            + (1.0 - y_true) * np.log(1.0 - y_prob)
        )

    l2_penalty = l2_lambda * np.sum(w ** 2) / 2.0

    return float(mean_bce + l2_penalty)


def logistic_gradients(
    X: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    w: np.ndarray,
    l2_lambda: float
) -> tuple:
    """计算平均 BCE + L2 对权重和 bias 的梯度。"""
    X = np.asarray(X, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    w = np.asarray(w, dtype=float)

    n_samples = X.shape[0]

    if n_samples == 0:
        return l2_lambda * w, 0.0

    residual = y_prob - y_true

    # dL/dw：数据损失梯度 + L2 梯度
    dw = (X.T @ residual) / n_samples + l2_lambda * w

    # bias 不加 L2 惩罚
    db = float(np.mean(residual))

    return dw, db


# ============================================================
# 17-20. 初始化、梯度下降训练与阈值预测
# ============================================================

def initialize_logistic_params(n_features: int) -> tuple:
    """权重和 bias 都从 0 开始。"""
    w = np.zeros(n_features, dtype=float)
    b = 0.0

    return w, b


def gradient_descent_step(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    b: float,
    lr: float,
    l2_lambda: float
) -> tuple:
    """
    做一次全批量梯度下降。
    返回的 loss 是更新参数之前计算的损失。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)

    probabilities = logistic_predict_proba(X, w, b)
    loss = binary_cross_entropy(y, probabilities, w, l2_lambda)

    dw, db = logistic_gradients(
        X, y, probabilities, w, l2_lambda
    )

    w_new = w - lr * dw
    b_new = float(b - lr * db)

    return w_new, b_new, loss


def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    lr: float,
    l2_lambda: float,
    n_epochs: int
) -> tuple:
    """从零开始训练逻辑回归，并记录每轮更新前的 loss。"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    w, b = initialize_logistic_params(X.shape[1])
    losses = []

    for _ in range(n_epochs):
        w, b, loss = gradient_descent_step(
            X, y, w, b, lr, l2_lambda
        )
        losses.append(float(loss))

    return w, float(b), losses


def predict_labels(
    proba: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """概率大于等于阈值时预测为 1，否则预测为 0。"""
    proba = np.asarray(proba, dtype=float)
    return (proba >= threshold).astype(int)


# ============================================================
# 21-24. 混淆矩阵计数、分类指标与阈值调优
# ============================================================

def confusion_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> tuple:
    """返回 (TP, FP, TN, FN)，均为 Python int。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    return tp, fp, tn, fn


def metrics_from_counts(
    tp: int,
    fp: int,
    tn: int,
    fn: int
) -> dict:
    """根据混淆矩阵计数计算 precision、recall、F1 和 accuracy。"""
    precision_denominator = tp + fp
    recall_denominator = tp + fn
    f1_denominator = 2 * tp + fp + fn
    total = tp + fp + tn + fn

    precision = (
        tp / precision_denominator
        if precision_denominator > 0 else 0.0
    )
    recall = (
        tp / recall_denominator
        if recall_denominator > 0 else 0.0
    )
    f1 = (
        2 * tp / f1_denominator
        if f1_denominator > 0 else 0.0
    )
    accuracy = (
        (tp + tn) / total
        if total > 0 else 0.0
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


def tune_decision_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    thresholds: np.ndarray = None
) -> tuple:
    """
    在候选阈值中选 F1 最高的那个。
    F1 并列时保留 thresholds 中最先出现的阈值。
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)

    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)
    else:
        thresholds = np.asarray(thresholds, dtype=float)

    if len(thresholds) == 0:
        return 0.0, 0.0

    best_threshold = float(thresholds[0])
    best_f1 = -1.0

    for threshold in thresholds:
        y_pred = predict_labels(proba, threshold)
        tp, fp, tn, fn = confusion_counts(y_true, y_pred)
        f1 = metrics_from_counts(tp, fp, tn, fn)["f1"]

        # 使用严格大于，保证并列时留下先遇到的阈值
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return float(best_threshold), float(best_f1)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """返回混淆矩阵计数和分类指标的汇总字典。"""
    tp, fp, tn, fn = confusion_counts(y_true, y_pred)
    metrics = metrics_from_counts(tp, fp, tn, fn)

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "accuracy": metrics["accuracy"],
    }


# ============================================================
# 25-27. 新文本向量化、单条预测与错误分析
# ============================================================

def vectorize_texts(
    texts: list,
    vocab: dict,
    idf: np.ndarray
) -> np.ndarray:
    """把原始文本转换成训练时同样格式的 TF-IDF 矩阵。"""
    tokenized_docs = tokenize_corpus(texts)
    bow_matrix = corpus_to_bow_matrix(tokenized_docs, vocab)

    return transform_tfidf(bow_matrix, idf)


def predict_text(
    text: str,
    vocab: dict,
    idf: np.ndarray,
    w: np.ndarray,
    b: float,
    threshold: float = 0.5
) -> int:
    """使用训练好的参数预测一条原始文本。"""
    X = vectorize_texts([text], vocab, idf)
    proba = logistic_predict_proba(X, w, b)
    prediction = predict_labels(proba, threshold)

    return int(prediction[0])


def collect_prediction_errors(
    texts: list,
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """收集假阳性和假阴性的原始文本，保持输入顺序。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    false_positives = []
    false_negatives = []

    for text, true_label, predicted_label in zip(texts, y_true, y_pred):
        if true_label == 0 and predicted_label == 1:
            false_positives.append(text)
        elif true_label == 1 and predicted_label == 0:
            false_negatives.append(text)

    return {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }

# Step 17 - initialize_logistic_params
import numpy as np


# ============================================================
# 1-3. 文本清洗与分词
# ============================================================

def clean_text(text: str) -> str:
    """
    转成小写，并把每个非字母字符替换为空格。
    注意：保留连续空格以及开头、结尾产生的空格。
    """
    text = text.lower()
    return "".join(ch if ch.isalpha() else " " for ch in text)


def tokenize(text: str) -> list:
    """按任意空白字符切分，并自动丢弃空 token。"""
    return text.split()


def tokenize_corpus(texts: list) -> list:
    """清洗并分词整个文本列表。"""
    return [tokenize(clean_text(text)) for text in texts]


# ============================================================
# 4-8. 数据划分、词频、词表与 Bag-of-Words
# ============================================================

def split_train_val_test_indices(
    n_samples: int,
    val_fraction: float,
    test_fraction: float,
    seed: int = 0
) -> tuple:
    """
    生成互不重叠的 train/validation/test 索引。
    按 train、validation、test 的顺序切分打乱后的索引。
    """
    np.random.seed(seed)

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    n_val = int(n_samples * val_fraction)
    n_test = int(n_samples * test_fraction)
    n_train = n_samples - n_val - n_test

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return train_idx, val_idx, test_idx


def count_word_frequencies(tokenized_docs: list) -> dict:
    """统计每个 token 在整个语料中出现的总次数。"""
    counts = {}

    for document in tokenized_docs:
        for token in document:
            counts[token] = counts.get(token, 0) + 1

    return counts


def build_vocabulary(word_counts: dict, max_size: int) -> dict:
    """
    选出词频最高的 max_size 个词。
    词频相同时，按字母顺序排列。
    """
    max_size = max(0, int(max_size))

    ranked_words = sorted(
        word_counts.items(),
        key=lambda item: (-item[1], item[0])
    )

    return {
        word: index
        for index, (word, _) in enumerate(ranked_words[:max_size])
    }


def tokens_to_bow(tokens: list, vocab: dict) -> np.ndarray:
    """将一篇文档转成词袋计数向量，未知词忽略。"""
    vector = np.zeros(len(vocab), dtype=float)

    for token in tokens:
        if token in vocab:
            vector[vocab[token]] += 1.0

    return vector


def corpus_to_bow_matrix(tokenized_docs: list, vocab: dict) -> np.ndarray:
    """将多个文档转成形状为 (文档数, 词表大小) 的词袋矩阵。"""
    matrix = np.zeros(
        (len(tokenized_docs), len(vocab)),
        dtype=float
    )

    for row, tokens in enumerate(tokenized_docs):
        matrix[row] = tokens_to_bow(tokens, vocab)

    return matrix


# ============================================================
# 9-12. 文档频率与 TF-IDF
# ============================================================

def compute_document_frequencies(bow_matrix: np.ndarray) -> np.ndarray:
    """
    计算每个词出现于多少篇文档。
    每篇文档中，一个词出现一次或多次，都只计为出现过。
    """
    bow_matrix = np.asarray(bow_matrix)
    return np.sum(bow_matrix > 0, axis=0).astype(int)


def compute_idf(df: np.ndarray, n_docs: int) -> np.ndarray:
    """计算平滑 IDF，使用自然对数。"""
    df = np.asarray(df, dtype=float)
    return np.log((n_docs + 1) / (df + 1)) + 1.0


def transform_tfidf(
    bow_matrix: np.ndarray,
    idf: np.ndarray
) -> np.ndarray:
    """
    每列的词袋计数乘对应 IDF。
    支持输入一个向量 (V,) 或矩阵 (N, V)。
    """
    bow_matrix = np.asarray(bow_matrix, dtype=float)
    idf = np.asarray(idf, dtype=float)
    return bow_matrix * idf


def fit_tfidf(bow_train: np.ndarray) -> np.ndarray:
    """只根据训练集词袋矩阵拟合 IDF。"""
    bow_train = np.asarray(bow_train, dtype=float)

    df = compute_document_frequencies(bow_train)
    n_docs = bow_train.shape[0]

    return compute_idf(df, n_docs)


# ============================================================
# 13-16. 逻辑回归核心：Sigmoid、概率、损失与梯度
# ============================================================

def sigmoid(z: np.ndarray) -> np.ndarray:
    """数值稳定的 sigmoid，返回与输入形状相同的数组。"""
    z = np.asarray(z, dtype=float)
    flat = z.reshape(-1)
    result = np.empty_like(flat)

    # 正数和负数分开计算，避免 exp 溢出
    positive = flat >= 0
    negative = ~positive

    result[positive] = 1.0 / (1.0 + np.exp(-flat[positive]))

    exp_z = np.exp(flat[negative])
    result[negative] = exp_z / (1.0 + exp_z)

    return result.reshape(z.shape)


def logistic_predict_proba(
    X: np.ndarray,
    w: np.ndarray,
    b: float
) -> np.ndarray:
    """计算每一行属于类别 1 的概率。"""
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)

    return sigmoid(X @ w + b)


def binary_cross_entropy(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    w: np.ndarray,
    l2_lambda: float
) -> float:
    """
    平均二元交叉熵 + L2 权重惩罚。
    bias 不在这里传入，所以不会被正则化。
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    w = np.asarray(w, dtype=float)

    # 避免 log(0)
    y_prob = np.clip(y_prob, 1e-15, 1.0 - 1e-15)

    if y_true.size == 0:
        mean_bce = 0.0
    else:
        mean_bce = -np.mean(
            y_true * np.log(y_prob)
            + (1.0 - y_true) * np.log(1.0 - y_prob)
        )

    l2_penalty = l2_lambda * np.sum(w ** 2) / 2.0

    return float(mean_bce + l2_penalty)


def logistic_gradients(
    X: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    w: np.ndarray,
    l2_lambda: float
) -> tuple:
    """计算平均 BCE + L2 对权重和 bias 的梯度。"""
    X = np.asarray(X, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    w = np.asarray(w, dtype=float)

    n_samples = X.shape[0]

    if n_samples == 0:
        return l2_lambda * w, 0.0

    residual = y_prob - y_true

    # dL/dw：数据损失梯度 + L2 梯度
    dw = (X.T @ residual) / n_samples + l2_lambda * w

    # bias 不加 L2 惩罚
    db = float(np.mean(residual))

    return dw, db


# ============================================================
# 17-20. 初始化、梯度下降训练与阈值预测
# ============================================================

def initialize_logistic_params(n_features: int) -> tuple:
    """权重和 bias 都从 0 开始。"""
    w = np.zeros(n_features, dtype=float)
    b = 0.0

    return w, b


def gradient_descent_step(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    b: float,
    lr: float,
    l2_lambda: float
) -> tuple:
    """
    做一次全批量梯度下降。
    返回的 loss 是更新参数之前计算的损失。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)

    probabilities = logistic_predict_proba(X, w, b)
    loss = binary_cross_entropy(y, probabilities, w, l2_lambda)

    dw, db = logistic_gradients(
        X, y, probabilities, w, l2_lambda
    )

    w_new = w - lr * dw
    b_new = float(b - lr * db)

    return w_new, b_new, loss


def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    lr: float,
    l2_lambda: float,
    n_epochs: int
) -> tuple:
    """从零开始训练逻辑回归，并记录每轮更新前的 loss。"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    w, b = initialize_logistic_params(X.shape[1])
    losses = []

    for _ in range(n_epochs):
        w, b, loss = gradient_descent_step(
            X, y, w, b, lr, l2_lambda
        )
        losses.append(float(loss))

    return w, float(b), losses


def predict_labels(
    proba: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """概率大于等于阈值时预测为 1，否则预测为 0。"""
    proba = np.asarray(proba, dtype=float)
    return (proba >= threshold).astype(int)


# ============================================================
# 21-24. 混淆矩阵计数、分类指标与阈值调优
# ============================================================

def confusion_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> tuple:
    """返回 (TP, FP, TN, FN)，均为 Python int。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    return tp, fp, tn, fn


def metrics_from_counts(
    tp: int,
    fp: int,
    tn: int,
    fn: int
) -> dict:
    """根据混淆矩阵计数计算 precision、recall、F1 和 accuracy。"""
    precision_denominator = tp + fp
    recall_denominator = tp + fn
    f1_denominator = 2 * tp + fp + fn
    total = tp + fp + tn + fn

    precision = (
        tp / precision_denominator
        if precision_denominator > 0 else 0.0
    )
    recall = (
        tp / recall_denominator
        if recall_denominator > 0 else 0.0
    )
    f1 = (
        2 * tp / f1_denominator
        if f1_denominator > 0 else 0.0
    )
    accuracy = (
        (tp + tn) / total
        if total > 0 else 0.0
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


def tune_decision_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    thresholds: np.ndarray = None
) -> tuple:
    """
    在候选阈值中选 F1 最高的那个。
    F1 并列时保留 thresholds 中最先出现的阈值。
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)

    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)
    else:
        thresholds = np.asarray(thresholds, dtype=float)

    if len(thresholds) == 0:
        return 0.0, 0.0

    best_threshold = float(thresholds[0])
    best_f1 = -1.0

    for threshold in thresholds:
        y_pred = predict_labels(proba, threshold)
        tp, fp, tn, fn = confusion_counts(y_true, y_pred)
        f1 = metrics_from_counts(tp, fp, tn, fn)["f1"]

        # 使用严格大于，保证并列时留下先遇到的阈值
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return float(best_threshold), float(best_f1)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """返回混淆矩阵计数和分类指标的汇总字典。"""
    tp, fp, tn, fn = confusion_counts(y_true, y_pred)
    metrics = metrics_from_counts(tp, fp, tn, fn)

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "accuracy": metrics["accuracy"],
    }


# ============================================================
# 25-27. 新文本向量化、单条预测与错误分析
# ============================================================

def vectorize_texts(
    texts: list,
    vocab: dict,
    idf: np.ndarray
) -> np.ndarray:
    """把原始文本转换成训练时同样格式的 TF-IDF 矩阵。"""
    tokenized_docs = tokenize_corpus(texts)
    bow_matrix = corpus_to_bow_matrix(tokenized_docs, vocab)

    return transform_tfidf(bow_matrix, idf)


def predict_text(
    text: str,
    vocab: dict,
    idf: np.ndarray,
    w: np.ndarray,
    b: float,
    threshold: float = 0.5
) -> int:
    """使用训练好的参数预测一条原始文本。"""
    X = vectorize_texts([text], vocab, idf)
    proba = logistic_predict_proba(X, w, b)
    prediction = predict_labels(proba, threshold)

    return int(prediction[0])


def collect_prediction_errors(
    texts: list,
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """收集假阳性和假阴性的原始文本，保持输入顺序。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    false_positives = []
    false_negatives = []

    for text, true_label, predicted_label in zip(texts, y_true, y_pred):
        if true_label == 0 and predicted_label == 1:
            false_positives.append(text)
        elif true_label == 1 and predicted_label == 0:
            false_negatives.append(text)

    return {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }

# Step 18 - gradient_descent_step
import numpy as np


# ============================================================
# 1-3. 文本清洗与分词
# ============================================================

def clean_text(text: str) -> str:
    """
    转成小写，并把每个非字母字符替换为空格。
    注意：保留连续空格以及开头、结尾产生的空格。
    """
    text = text.lower()
    return "".join(ch if ch.isalpha() else " " for ch in text)


def tokenize(text: str) -> list:
    """按任意空白字符切分，并自动丢弃空 token。"""
    return text.split()


def tokenize_corpus(texts: list) -> list:
    """清洗并分词整个文本列表。"""
    return [tokenize(clean_text(text)) for text in texts]


# ============================================================
# 4-8. 数据划分、词频、词表与 Bag-of-Words
# ============================================================

def split_train_val_test_indices(
    n_samples: int,
    val_fraction: float,
    test_fraction: float,
    seed: int = 0
) -> tuple:
    """
    生成互不重叠的 train/validation/test 索引。
    按 train、validation、test 的顺序切分打乱后的索引。
    """
    np.random.seed(seed)

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    n_val = int(n_samples * val_fraction)
    n_test = int(n_samples * test_fraction)
    n_train = n_samples - n_val - n_test

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return train_idx, val_idx, test_idx


def count_word_frequencies(tokenized_docs: list) -> dict:
    """统计每个 token 在整个语料中出现的总次数。"""
    counts = {}

    for document in tokenized_docs:
        for token in document:
            counts[token] = counts.get(token, 0) + 1

    return counts


def build_vocabulary(word_counts: dict, max_size: int) -> dict:
    """
    选出词频最高的 max_size 个词。
    词频相同时，按字母顺序排列。
    """
    max_size = max(0, int(max_size))

    ranked_words = sorted(
        word_counts.items(),
        key=lambda item: (-item[1], item[0])
    )

    return {
        word: index
        for index, (word, _) in enumerate(ranked_words[:max_size])
    }


def tokens_to_bow(tokens: list, vocab: dict) -> np.ndarray:
    """将一篇文档转成词袋计数向量，未知词忽略。"""
    vector = np.zeros(len(vocab), dtype=float)

    for token in tokens:
        if token in vocab:
            vector[vocab[token]] += 1.0

    return vector


def corpus_to_bow_matrix(tokenized_docs: list, vocab: dict) -> np.ndarray:
    """将多个文档转成形状为 (文档数, 词表大小) 的词袋矩阵。"""
    matrix = np.zeros(
        (len(tokenized_docs), len(vocab)),
        dtype=float
    )

    for row, tokens in enumerate(tokenized_docs):
        matrix[row] = tokens_to_bow(tokens, vocab)

    return matrix


# ============================================================
# 9-12. 文档频率与 TF-IDF
# ============================================================

def compute_document_frequencies(bow_matrix: np.ndarray) -> np.ndarray:
    """
    计算每个词出现于多少篇文档。
    每篇文档中，一个词出现一次或多次，都只计为出现过。
    """
    bow_matrix = np.asarray(bow_matrix)
    return np.sum(bow_matrix > 0, axis=0).astype(int)


def compute_idf(df: np.ndarray, n_docs: int) -> np.ndarray:
    """计算平滑 IDF，使用自然对数。"""
    df = np.asarray(df, dtype=float)
    return np.log((n_docs + 1) / (df + 1)) + 1.0


def transform_tfidf(
    bow_matrix: np.ndarray,
    idf: np.ndarray
) -> np.ndarray:
    """
    每列的词袋计数乘对应 IDF。
    支持输入一个向量 (V,) 或矩阵 (N, V)。
    """
    bow_matrix = np.asarray(bow_matrix, dtype=float)
    idf = np.asarray(idf, dtype=float)
    return bow_matrix * idf


def fit_tfidf(bow_train: np.ndarray) -> np.ndarray:
    """只根据训练集词袋矩阵拟合 IDF。"""
    bow_train = np.asarray(bow_train, dtype=float)

    df = compute_document_frequencies(bow_train)
    n_docs = bow_train.shape[0]

    return compute_idf(df, n_docs)


# ============================================================
# 13-16. 逻辑回归核心：Sigmoid、概率、损失与梯度
# ============================================================

def sigmoid(z: np.ndarray) -> np.ndarray:
    """数值稳定的 sigmoid，返回与输入形状相同的数组。"""
    z = np.asarray(z, dtype=float)
    flat = z.reshape(-1)
    result = np.empty_like(flat)

    # 正数和负数分开计算，避免 exp 溢出
    positive = flat >= 0
    negative = ~positive

    result[positive] = 1.0 / (1.0 + np.exp(-flat[positive]))

    exp_z = np.exp(flat[negative])
    result[negative] = exp_z / (1.0 + exp_z)

    return result.reshape(z.shape)


def logistic_predict_proba(
    X: np.ndarray,
    w: np.ndarray,
    b: float
) -> np.ndarray:
    """计算每一行属于类别 1 的概率。"""
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)

    return sigmoid(X @ w + b)


def binary_cross_entropy(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    w: np.ndarray,
    l2_lambda: float
) -> float:
    """
    平均二元交叉熵 + L2 权重惩罚。
    bias 不在这里传入，所以不会被正则化。
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    w = np.asarray(w, dtype=float)

    # 避免 log(0)
    y_prob = np.clip(y_prob, 1e-15, 1.0 - 1e-15)

    if y_true.size == 0:
        mean_bce = 0.0
    else:
        mean_bce = -np.mean(
            y_true * np.log(y_prob)
            + (1.0 - y_true) * np.log(1.0 - y_prob)
        )

    l2_penalty = l2_lambda * np.sum(w ** 2) / 2.0

    return float(mean_bce + l2_penalty)


def logistic_gradients(
    X: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    w: np.ndarray,
    l2_lambda: float
) -> tuple:
    """计算平均 BCE + L2 对权重和 bias 的梯度。"""
    X = np.asarray(X, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    w = np.asarray(w, dtype=float)

    n_samples = X.shape[0]

    if n_samples == 0:
        return l2_lambda * w, 0.0

    residual = y_prob - y_true

    # dL/dw：数据损失梯度 + L2 梯度
    dw = (X.T @ residual) / n_samples + l2_lambda * w

    # bias 不加 L2 惩罚
    db = float(np.mean(residual))

    return dw, db


# ============================================================
# 17-20. 初始化、梯度下降训练与阈值预测
# ============================================================

def initialize_logistic_params(n_features: int) -> tuple:
    """权重和 bias 都从 0 开始。"""
    w = np.zeros(n_features, dtype=float)
    b = 0.0

    return w, b


def gradient_descent_step(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    b: float,
    lr: float,
    l2_lambda: float
) -> tuple:
    """
    做一次全批量梯度下降。
    返回的 loss 是更新参数之前计算的损失。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)

    probabilities = logistic_predict_proba(X, w, b)
    loss = binary_cross_entropy(y, probabilities, w, l2_lambda)

    dw, db = logistic_gradients(
        X, y, probabilities, w, l2_lambda
    )

    w_new = w - lr * dw
    b_new = float(b - lr * db)

    return w_new, b_new, loss


def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    lr: float,
    l2_lambda: float,
    n_epochs: int
) -> tuple:
    """从零开始训练逻辑回归，并记录每轮更新前的 loss。"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    w, b = initialize_logistic_params(X.shape[1])
    losses = []

    for _ in range(n_epochs):
        w, b, loss = gradient_descent_step(
            X, y, w, b, lr, l2_lambda
        )
        losses.append(float(loss))

    return w, float(b), losses


def predict_labels(
    proba: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """概率大于等于阈值时预测为 1，否则预测为 0。"""
    proba = np.asarray(proba, dtype=float)
    return (proba >= threshold).astype(int)


# ============================================================
# 21-24. 混淆矩阵计数、分类指标与阈值调优
# ============================================================

def confusion_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> tuple:
    """返回 (TP, FP, TN, FN)，均为 Python int。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    return tp, fp, tn, fn


def metrics_from_counts(
    tp: int,
    fp: int,
    tn: int,
    fn: int
) -> dict:
    """根据混淆矩阵计数计算 precision、recall、F1 和 accuracy。"""
    precision_denominator = tp + fp
    recall_denominator = tp + fn
    f1_denominator = 2 * tp + fp + fn
    total = tp + fp + tn + fn

    precision = (
        tp / precision_denominator
        if precision_denominator > 0 else 0.0
    )
    recall = (
        tp / recall_denominator
        if recall_denominator > 0 else 0.0
    )
    f1 = (
        2 * tp / f1_denominator
        if f1_denominator > 0 else 0.0
    )
    accuracy = (
        (tp + tn) / total
        if total > 0 else 0.0
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


def tune_decision_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    thresholds: np.ndarray = None
) -> tuple:
    """
    在候选阈值中选 F1 最高的那个。
    F1 并列时保留 thresholds 中最先出现的阈值。
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)

    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)
    else:
        thresholds = np.asarray(thresholds, dtype=float)

    if len(thresholds) == 0:
        return 0.0, 0.0

    best_threshold = float(thresholds[0])
    best_f1 = -1.0

    for threshold in thresholds:
        y_pred = predict_labels(proba, threshold)
        tp, fp, tn, fn = confusion_counts(y_true, y_pred)
        f1 = metrics_from_counts(tp, fp, tn, fn)["f1"]

        # 使用严格大于，保证并列时留下先遇到的阈值
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return float(best_threshold), float(best_f1)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """返回混淆矩阵计数和分类指标的汇总字典。"""
    tp, fp, tn, fn = confusion_counts(y_true, y_pred)
    metrics = metrics_from_counts(tp, fp, tn, fn)

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "accuracy": metrics["accuracy"],
    }


# ============================================================
# 25-27. 新文本向量化、单条预测与错误分析
# ============================================================

def vectorize_texts(
    texts: list,
    vocab: dict,
    idf: np.ndarray
) -> np.ndarray:
    """把原始文本转换成训练时同样格式的 TF-IDF 矩阵。"""
    tokenized_docs = tokenize_corpus(texts)
    bow_matrix = corpus_to_bow_matrix(tokenized_docs, vocab)

    return transform_tfidf(bow_matrix, idf)


def predict_text(
    text: str,
    vocab: dict,
    idf: np.ndarray,
    w: np.ndarray,
    b: float,
    threshold: float = 0.5
) -> int:
    """使用训练好的参数预测一条原始文本。"""
    X = vectorize_texts([text], vocab, idf)
    proba = logistic_predict_proba(X, w, b)
    prediction = predict_labels(proba, threshold)

    return int(prediction[0])


def collect_prediction_errors(
    texts: list,
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """收集假阳性和假阴性的原始文本，保持输入顺序。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    false_positives = []
    false_negatives = []

    for text, true_label, predicted_label in zip(texts, y_true, y_pred):
        if true_label == 0 and predicted_label == 1:
            false_positives.append(text)
        elif true_label == 1 and predicted_label == 0:
            false_negatives.append(text)

    return {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }

# Step 19 - train_logistic_regression
import numpy as np


# ============================================================
# 1-3. 文本清洗与分词
# ============================================================

def clean_text(text: str) -> str:
    """
    转成小写，并把每个非字母字符替换为空格。
    注意：保留连续空格以及开头、结尾产生的空格。
    """
    text = text.lower()
    return "".join(ch if ch.isalpha() else " " for ch in text)


def tokenize(text: str) -> list:
    """按任意空白字符切分，并自动丢弃空 token。"""
    return text.split()


def tokenize_corpus(texts: list) -> list:
    """清洗并分词整个文本列表。"""
    return [tokenize(clean_text(text)) for text in texts]


# ============================================================
# 4-8. 数据划分、词频、词表与 Bag-of-Words
# ============================================================

def split_train_val_test_indices(
    n_samples: int,
    val_fraction: float,
    test_fraction: float,
    seed: int = 0
) -> tuple:
    """
    生成互不重叠的 train/validation/test 索引。
    按 train、validation、test 的顺序切分打乱后的索引。
    """
    np.random.seed(seed)

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    n_val = int(n_samples * val_fraction)
    n_test = int(n_samples * test_fraction)
    n_train = n_samples - n_val - n_test

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return train_idx, val_idx, test_idx


def count_word_frequencies(tokenized_docs: list) -> dict:
    """统计每个 token 在整个语料中出现的总次数。"""
    counts = {}

    for document in tokenized_docs:
        for token in document:
            counts[token] = counts.get(token, 0) + 1

    return counts


def build_vocabulary(word_counts: dict, max_size: int) -> dict:
    """
    选出词频最高的 max_size 个词。
    词频相同时，按字母顺序排列。
    """
    max_size = max(0, int(max_size))

    ranked_words = sorted(
        word_counts.items(),
        key=lambda item: (-item[1], item[0])
    )

    return {
        word: index
        for index, (word, _) in enumerate(ranked_words[:max_size])
    }


def tokens_to_bow(tokens: list, vocab: dict) -> np.ndarray:
    """将一篇文档转成词袋计数向量，未知词忽略。"""
    vector = np.zeros(len(vocab), dtype=float)

    for token in tokens:
        if token in vocab:
            vector[vocab[token]] += 1.0

    return vector


def corpus_to_bow_matrix(tokenized_docs: list, vocab: dict) -> np.ndarray:
    """将多个文档转成形状为 (文档数, 词表大小) 的词袋矩阵。"""
    matrix = np.zeros(
        (len(tokenized_docs), len(vocab)),
        dtype=float
    )

    for row, tokens in enumerate(tokenized_docs):
        matrix[row] = tokens_to_bow(tokens, vocab)

    return matrix


# ============================================================
# 9-12. 文档频率与 TF-IDF
# ============================================================

def compute_document_frequencies(bow_matrix: np.ndarray) -> np.ndarray:
    """
    计算每个词出现于多少篇文档。
    每篇文档中，一个词出现一次或多次，都只计为出现过。
    """
    bow_matrix = np.asarray(bow_matrix)
    return np.sum(bow_matrix > 0, axis=0).astype(int)


def compute_idf(df: np.ndarray, n_docs: int) -> np.ndarray:
    """计算平滑 IDF，使用自然对数。"""
    df = np.asarray(df, dtype=float)
    return np.log((n_docs + 1) / (df + 1)) + 1.0


def transform_tfidf(
    bow_matrix: np.ndarray,
    idf: np.ndarray
) -> np.ndarray:
    """
    每列的词袋计数乘对应 IDF。
    支持输入一个向量 (V,) 或矩阵 (N, V)。
    """
    bow_matrix = np.asarray(bow_matrix, dtype=float)
    idf = np.asarray(idf, dtype=float)
    return bow_matrix * idf


def fit_tfidf(bow_train: np.ndarray) -> np.ndarray:
    """只根据训练集词袋矩阵拟合 IDF。"""
    bow_train = np.asarray(bow_train, dtype=float)

    df = compute_document_frequencies(bow_train)
    n_docs = bow_train.shape[0]

    return compute_idf(df, n_docs)


# ============================================================
# 13-16. 逻辑回归核心：Sigmoid、概率、损失与梯度
# ============================================================

def sigmoid(z: np.ndarray) -> np.ndarray:
    """数值稳定的 sigmoid，返回与输入形状相同的数组。"""
    z = np.asarray(z, dtype=float)
    flat = z.reshape(-1)
    result = np.empty_like(flat)

    # 正数和负数分开计算，避免 exp 溢出
    positive = flat >= 0
    negative = ~positive

    result[positive] = 1.0 / (1.0 + np.exp(-flat[positive]))

    exp_z = np.exp(flat[negative])
    result[negative] = exp_z / (1.0 + exp_z)

    return result.reshape(z.shape)


def logistic_predict_proba(
    X: np.ndarray,
    w: np.ndarray,
    b: float
) -> np.ndarray:
    """计算每一行属于类别 1 的概率。"""
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)

    return sigmoid(X @ w + b)


def binary_cross_entropy(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    w: np.ndarray,
    l2_lambda: float
) -> float:
    """
    平均二元交叉熵 + L2 权重惩罚。
    bias 不在这里传入，所以不会被正则化。
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    w = np.asarray(w, dtype=float)

    # 避免 log(0)
    y_prob = np.clip(y_prob, 1e-15, 1.0 - 1e-15)

    if y_true.size == 0:
        mean_bce = 0.0
    else:
        mean_bce = -np.mean(
            y_true * np.log(y_prob)
            + (1.0 - y_true) * np.log(1.0 - y_prob)
        )

    l2_penalty = l2_lambda * np.sum(w ** 2) / 2.0

    return float(mean_bce + l2_penalty)


def logistic_gradients(
    X: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    w: np.ndarray,
    l2_lambda: float
) -> tuple:
    """计算平均 BCE + L2 对权重和 bias 的梯度。"""
    X = np.asarray(X, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    w = np.asarray(w, dtype=float)

    n_samples = X.shape[0]

    if n_samples == 0:
        return l2_lambda * w, 0.0

    residual = y_prob - y_true

    # dL/dw：数据损失梯度 + L2 梯度
    dw = (X.T @ residual) / n_samples + l2_lambda * w

    # bias 不加 L2 惩罚
    db = float(np.mean(residual))

    return dw, db


# ============================================================
# 17-20. 初始化、梯度下降训练与阈值预测
# ============================================================

def initialize_logistic_params(n_features: int) -> tuple:
    """权重和 bias 都从 0 开始。"""
    w = np.zeros(n_features, dtype=float)
    b = 0.0

    return w, b


def gradient_descent_step(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    b: float,
    lr: float,
    l2_lambda: float
) -> tuple:
    """
    做一次全批量梯度下降。
    返回的 loss 是更新参数之前计算的损失。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)

    probabilities = logistic_predict_proba(X, w, b)
    loss = binary_cross_entropy(y, probabilities, w, l2_lambda)

    dw, db = logistic_gradients(
        X, y, probabilities, w, l2_lambda
    )

    w_new = w - lr * dw
    b_new = float(b - lr * db)

    return w_new, b_new, loss


def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    lr: float,
    l2_lambda: float,
    n_epochs: int
) -> tuple:
    """从零开始训练逻辑回归，并记录每轮更新前的 loss。"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    w, b = initialize_logistic_params(X.shape[1])
    losses = []

    for _ in range(n_epochs):
        w, b, loss = gradient_descent_step(
            X, y, w, b, lr, l2_lambda
        )
        losses.append(float(loss))

    return w, float(b), losses


def predict_labels(
    proba: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """概率大于等于阈值时预测为 1，否则预测为 0。"""
    proba = np.asarray(proba, dtype=float)
    return (proba >= threshold).astype(int)


# ============================================================
# 21-24. 混淆矩阵计数、分类指标与阈值调优
# ============================================================

def confusion_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> tuple:
    """返回 (TP, FP, TN, FN)，均为 Python int。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    return tp, fp, tn, fn


def metrics_from_counts(
    tp: int,
    fp: int,
    tn: int,
    fn: int
) -> dict:
    """根据混淆矩阵计数计算 precision、recall、F1 和 accuracy。"""
    precision_denominator = tp + fp
    recall_denominator = tp + fn
    f1_denominator = 2 * tp + fp + fn
    total = tp + fp + tn + fn

    precision = (
        tp / precision_denominator
        if precision_denominator > 0 else 0.0
    )
    recall = (
        tp / recall_denominator
        if recall_denominator > 0 else 0.0
    )
    f1 = (
        2 * tp / f1_denominator
        if f1_denominator > 0 else 0.0
    )
    accuracy = (
        (tp + tn) / total
        if total > 0 else 0.0
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


def tune_decision_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    thresholds: np.ndarray = None
) -> tuple:
    """
    在候选阈值中选 F1 最高的那个。
    F1 并列时保留 thresholds 中最先出现的阈值。
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)

    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)
    else:
        thresholds = np.asarray(thresholds, dtype=float)

    if len(thresholds) == 0:
        return 0.0, 0.0

    best_threshold = float(thresholds[0])
    best_f1 = -1.0

    for threshold in thresholds:
        y_pred = predict_labels(proba, threshold)
        tp, fp, tn, fn = confusion_counts(y_true, y_pred)
        f1 = metrics_from_counts(tp, fp, tn, fn)["f1"]

        # 使用严格大于，保证并列时留下先遇到的阈值
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return float(best_threshold), float(best_f1)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """返回混淆矩阵计数和分类指标的汇总字典。"""
    tp, fp, tn, fn = confusion_counts(y_true, y_pred)
    metrics = metrics_from_counts(tp, fp, tn, fn)

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "accuracy": metrics["accuracy"],
    }


# ============================================================
# 25-27. 新文本向量化、单条预测与错误分析
# ============================================================

def vectorize_texts(
    texts: list,
    vocab: dict,
    idf: np.ndarray
) -> np.ndarray:
    """把原始文本转换成训练时同样格式的 TF-IDF 矩阵。"""
    tokenized_docs = tokenize_corpus(texts)
    bow_matrix = corpus_to_bow_matrix(tokenized_docs, vocab)

    return transform_tfidf(bow_matrix, idf)


def predict_text(
    text: str,
    vocab: dict,
    idf: np.ndarray,
    w: np.ndarray,
    b: float,
    threshold: float = 0.5
) -> int:
    """使用训练好的参数预测一条原始文本。"""
    X = vectorize_texts([text], vocab, idf)
    proba = logistic_predict_proba(X, w, b)
    prediction = predict_labels(proba, threshold)

    return int(prediction[0])


def collect_prediction_errors(
    texts: list,
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """收集假阳性和假阴性的原始文本，保持输入顺序。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    false_positives = []
    false_negatives = []

    for text, true_label, predicted_label in zip(texts, y_true, y_pred):
        if true_label == 0 and predicted_label == 1:
            false_positives.append(text)
        elif true_label == 1 and predicted_label == 0:
            false_negatives.append(text)

    return {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }

# Step 20 - predict_labels
import numpy as np


# ============================================================
# 1-3. 文本清洗与分词
# ============================================================

def clean_text(text: str) -> str:
    """
    转成小写，并把每个非字母字符替换为空格。
    注意：保留连续空格以及开头、结尾产生的空格。
    """
    text = text.lower()
    return "".join(ch if ch.isalpha() else " " for ch in text)


def tokenize(text: str) -> list:
    """按任意空白字符切分，并自动丢弃空 token。"""
    return text.split()


def tokenize_corpus(texts: list) -> list:
    """清洗并分词整个文本列表。"""
    return [tokenize(clean_text(text)) for text in texts]


# ============================================================
# 4-8. 数据划分、词频、词表与 Bag-of-Words
# ============================================================

def split_train_val_test_indices(
    n_samples: int,
    val_fraction: float,
    test_fraction: float,
    seed: int = 0
) -> tuple:
    """
    生成互不重叠的 train/validation/test 索引。
    按 train、validation、test 的顺序切分打乱后的索引。
    """
    np.random.seed(seed)

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    n_val = int(n_samples * val_fraction)
    n_test = int(n_samples * test_fraction)
    n_train = n_samples - n_val - n_test

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return train_idx, val_idx, test_idx


def count_word_frequencies(tokenized_docs: list) -> dict:
    """统计每个 token 在整个语料中出现的总次数。"""
    counts = {}

    for document in tokenized_docs:
        for token in document:
            counts[token] = counts.get(token, 0) + 1

    return counts


def build_vocabulary(word_counts: dict, max_size: int) -> dict:
    """
    选出词频最高的 max_size 个词。
    词频相同时，按字母顺序排列。
    """
    max_size = max(0, int(max_size))

    ranked_words = sorted(
        word_counts.items(),
        key=lambda item: (-item[1], item[0])
    )

    return {
        word: index
        for index, (word, _) in enumerate(ranked_words[:max_size])
    }


def tokens_to_bow(tokens: list, vocab: dict) -> np.ndarray:
    """将一篇文档转成词袋计数向量，未知词忽略。"""
    vector = np.zeros(len(vocab), dtype=float)

    for token in tokens:
        if token in vocab:
            vector[vocab[token]] += 1.0

    return vector


def corpus_to_bow_matrix(tokenized_docs: list, vocab: dict) -> np.ndarray:
    """将多个文档转成形状为 (文档数, 词表大小) 的词袋矩阵。"""
    matrix = np.zeros(
        (len(tokenized_docs), len(vocab)),
        dtype=float
    )

    for row, tokens in enumerate(tokenized_docs):
        matrix[row] = tokens_to_bow(tokens, vocab)

    return matrix


# ============================================================
# 9-12. 文档频率与 TF-IDF
# ============================================================

def compute_document_frequencies(bow_matrix: np.ndarray) -> np.ndarray:
    """
    计算每个词出现于多少篇文档。
    每篇文档中，一个词出现一次或多次，都只计为出现过。
    """
    bow_matrix = np.asarray(bow_matrix)
    return np.sum(bow_matrix > 0, axis=0).astype(int)


def compute_idf(df: np.ndarray, n_docs: int) -> np.ndarray:
    """计算平滑 IDF，使用自然对数。"""
    df = np.asarray(df, dtype=float)
    return np.log((n_docs + 1) / (df + 1)) + 1.0


def transform_tfidf(
    bow_matrix: np.ndarray,
    idf: np.ndarray
) -> np.ndarray:
    """
    每列的词袋计数乘对应 IDF。
    支持输入一个向量 (V,) 或矩阵 (N, V)。
    """
    bow_matrix = np.asarray(bow_matrix, dtype=float)
    idf = np.asarray(idf, dtype=float)
    return bow_matrix * idf


def fit_tfidf(bow_train: np.ndarray) -> np.ndarray:
    """只根据训练集词袋矩阵拟合 IDF。"""
    bow_train = np.asarray(bow_train, dtype=float)

    df = compute_document_frequencies(bow_train)
    n_docs = bow_train.shape[0]

    return compute_idf(df, n_docs)


# ============================================================
# 13-16. 逻辑回归核心：Sigmoid、概率、损失与梯度
# ============================================================

def sigmoid(z: np.ndarray) -> np.ndarray:
    """数值稳定的 sigmoid，返回与输入形状相同的数组。"""
    z = np.asarray(z, dtype=float)
    flat = z.reshape(-1)
    result = np.empty_like(flat)

    # 正数和负数分开计算，避免 exp 溢出
    positive = flat >= 0
    negative = ~positive

    result[positive] = 1.0 / (1.0 + np.exp(-flat[positive]))

    exp_z = np.exp(flat[negative])
    result[negative] = exp_z / (1.0 + exp_z)

    return result.reshape(z.shape)


def logistic_predict_proba(
    X: np.ndarray,
    w: np.ndarray,
    b: float
) -> np.ndarray:
    """计算每一行属于类别 1 的概率。"""
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)

    return sigmoid(X @ w + b)


def binary_cross_entropy(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    w: np.ndarray,
    l2_lambda: float
) -> float:
    """
    平均二元交叉熵 + L2 权重惩罚。
    bias 不在这里传入，所以不会被正则化。
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    w = np.asarray(w, dtype=float)

    # 避免 log(0)
    y_prob = np.clip(y_prob, 1e-15, 1.0 - 1e-15)

    if y_true.size == 0:
        mean_bce = 0.0
    else:
        mean_bce = -np.mean(
            y_true * np.log(y_prob)
            + (1.0 - y_true) * np.log(1.0 - y_prob)
        )

    l2_penalty = l2_lambda * np.sum(w ** 2) / 2.0

    return float(mean_bce + l2_penalty)


def logistic_gradients(
    X: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    w: np.ndarray,
    l2_lambda: float
) -> tuple:
    """计算平均 BCE + L2 对权重和 bias 的梯度。"""
    X = np.asarray(X, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    w = np.asarray(w, dtype=float)

    n_samples = X.shape[0]

    if n_samples == 0:
        return l2_lambda * w, 0.0

    residual = y_prob - y_true

    # dL/dw：数据损失梯度 + L2 梯度
    dw = (X.T @ residual) / n_samples + l2_lambda * w

    # bias 不加 L2 惩罚
    db = float(np.mean(residual))

    return dw, db


# ============================================================
# 17-20. 初始化、梯度下降训练与阈值预测
# ============================================================

def initialize_logistic_params(n_features: int) -> tuple:
    """权重和 bias 都从 0 开始。"""
    w = np.zeros(n_features, dtype=float)
    b = 0.0

    return w, b


def gradient_descent_step(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    b: float,
    lr: float,
    l2_lambda: float
) -> tuple:
    """
    做一次全批量梯度下降。
    返回的 loss 是更新参数之前计算的损失。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)

    probabilities = logistic_predict_proba(X, w, b)
    loss = binary_cross_entropy(y, probabilities, w, l2_lambda)

    dw, db = logistic_gradients(
        X, y, probabilities, w, l2_lambda
    )

    w_new = w - lr * dw
    b_new = float(b - lr * db)

    return w_new, b_new, loss


def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    lr: float,
    l2_lambda: float,
    n_epochs: int
) -> tuple:
    """从零开始训练逻辑回归，并记录每轮更新前的 loss。"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    w, b = initialize_logistic_params(X.shape[1])
    losses = []

    for _ in range(n_epochs):
        w, b, loss = gradient_descent_step(
            X, y, w, b, lr, l2_lambda
        )
        losses.append(float(loss))

    return w, float(b), losses


def predict_labels(
    proba: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """概率大于等于阈值时预测为 1，否则预测为 0。"""
    proba = np.asarray(proba, dtype=float)
    return (proba >= threshold).astype(int)


# ============================================================
# 21-24. 混淆矩阵计数、分类指标与阈值调优
# ============================================================

def confusion_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> tuple:
    """返回 (TP, FP, TN, FN)，均为 Python int。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    return tp, fp, tn, fn


def metrics_from_counts(
    tp: int,
    fp: int,
    tn: int,
    fn: int
) -> dict:
    """根据混淆矩阵计数计算 precision、recall、F1 和 accuracy。"""
    precision_denominator = tp + fp
    recall_denominator = tp + fn
    f1_denominator = 2 * tp + fp + fn
    total = tp + fp + tn + fn

    precision = (
        tp / precision_denominator
        if precision_denominator > 0 else 0.0
    )
    recall = (
        tp / recall_denominator
        if recall_denominator > 0 else 0.0
    )
    f1 = (
        2 * tp / f1_denominator
        if f1_denominator > 0 else 0.0
    )
    accuracy = (
        (tp + tn) / total
        if total > 0 else 0.0
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


def tune_decision_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    thresholds: np.ndarray = None
) -> tuple:
    """
    在候选阈值中选 F1 最高的那个。
    F1 并列时保留 thresholds 中最先出现的阈值。
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)

    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)
    else:
        thresholds = np.asarray(thresholds, dtype=float)

    if len(thresholds) == 0:
        return 0.0, 0.0

    best_threshold = float(thresholds[0])
    best_f1 = -1.0

    for threshold in thresholds:
        y_pred = predict_labels(proba, threshold)
        tp, fp, tn, fn = confusion_counts(y_true, y_pred)
        f1 = metrics_from_counts(tp, fp, tn, fn)["f1"]

        # 使用严格大于，保证并列时留下先遇到的阈值
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return float(best_threshold), float(best_f1)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """返回混淆矩阵计数和分类指标的汇总字典。"""
    tp, fp, tn, fn = confusion_counts(y_true, y_pred)
    metrics = metrics_from_counts(tp, fp, tn, fn)

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "accuracy": metrics["accuracy"],
    }


# ============================================================
# 25-27. 新文本向量化、单条预测与错误分析
# ============================================================

def vectorize_texts(
    texts: list,
    vocab: dict,
    idf: np.ndarray
) -> np.ndarray:
    """把原始文本转换成训练时同样格式的 TF-IDF 矩阵。"""
    tokenized_docs = tokenize_corpus(texts)
    bow_matrix = corpus_to_bow_matrix(tokenized_docs, vocab)

    return transform_tfidf(bow_matrix, idf)


def predict_text(
    text: str,
    vocab: dict,
    idf: np.ndarray,
    w: np.ndarray,
    b: float,
    threshold: float = 0.5
) -> int:
    """使用训练好的参数预测一条原始文本。"""
    X = vectorize_texts([text], vocab, idf)
    proba = logistic_predict_proba(X, w, b)
    prediction = predict_labels(proba, threshold)

    return int(prediction[0])


def collect_prediction_errors(
    texts: list,
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """收集假阳性和假阴性的原始文本，保持输入顺序。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    false_positives = []
    false_negatives = []

    for text, true_label, predicted_label in zip(texts, y_true, y_pred):
        if true_label == 0 and predicted_label == 1:
            false_positives.append(text)
        elif true_label == 1 and predicted_label == 0:
            false_negatives.append(text)

    return {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }

# Step 21 - confusion_counts (not yet solved)
# TODO: implement

# Step 22 - metrics_from_counts (not yet solved)
# TODO: implement

# Step 23 - tune_decision_threshold (not yet solved)
# TODO: implement

# Step 24 - evaluate_predictions (not yet solved)
# TODO: implement

# Step 25 - vectorize_texts (not yet solved)
# TODO: implement

# Step 26 - predict_text (not yet solved)
# TODO: implement

# Step 27 - collect_prediction_errors (not yet solved)
# TODO: implement

