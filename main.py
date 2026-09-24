"""实验一：基于 TF-IDF 的新闻文本分类。"""

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


SEED = 42
VAL_SIZE = 0.2
DATA_DIR = Path(__file__).resolve().parent

# 模型比较阶段统一使用 5000 维 unigram，保证比较公平。
BASE_TFIDF = {"max_features": 5000, "ngram_range": (1, 1)}

# 根据验证集实验选出的最终配置。
BEST_TFIDF = {"max_features": 10000, "ngram_range": (1, 1)}
BEST_MODEL_C = 10.0


def load_data():
    """读取训练集和无标签测试集，并完成基本数据检查。"""
    train_df = pd.read_csv(DATA_DIR / "train_data.csv")
    test_df = pd.read_csv(DATA_DIR / "test_data_unlabeled.csv")

    required_train_columns = {"text", "target"}
    if not required_train_columns.issubset(train_df.columns):
        raise ValueError("训练数据必须包含 text 和 target 两列")
    if "text" not in test_df.columns:
        raise ValueError("测试数据必须包含 text 列")
    if train_df["target"].isna().any():
        raise ValueError("训练数据存在缺失标签")

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["text"] = train_df["text"].fillna("").astype(str)
    test_df["text"] = test_df["text"].fillna("").astype(str)

    print("训练数据形状：", train_df.shape)
    print("测试数据形状：", test_df.shape)
    print("各类别样本数量：")
    print(train_df["target"].value_counts().sort_index())
    return train_df, test_df


def split_data(train_df):
    """按照类别比例，将有标签数据划分为训练集和验证集。"""
    return train_test_split(
        train_df["text"],
        train_df["target"],
        test_size=VAL_SIZE,
        random_state=SEED,
        stratify=train_df["target"],
    )


def build_tfidf(X_train, X_val, config):
    """仅在训练集上拟合 TF-IDF，再转换验证集。"""
    vectorizer = TfidfVectorizer(**config)
    start = perf_counter()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    elapsed = perf_counter() - start
    return vectorizer, X_train_vec, X_val_vec, elapsed


def evaluate_model(model, X_train_vec, y_train, X_val_vec, y_val):
    """训练单个模型，返回模型、预测和主要评价指标。"""
    start = perf_counter()
    model.fit(X_train_vec, y_train)
    train_seconds = perf_counter() - start

    train_pred = model.predict(X_train_vec)
    val_pred = model.predict(X_val_vec)
    metrics = {
        "train_accuracy": accuracy_score(y_train, train_pred),
        "val_accuracy": accuracy_score(y_val, val_pred),
        "val_macro_f1": f1_score(y_val, val_pred, average="macro"),
        "train_seconds": train_seconds,
    }
    return model, val_pred, metrics


def run_model_tuning(X_train, X_val, y_train, y_val):
    """在相同 TF-IDF 特征下比较逻辑回归和线性 SVM。"""
    _, X_train_vec, X_val_vec, _ = build_tfidf(
        X_train, X_val, BASE_TFIDF
    )
    results = []

    for model_name in ["LogisticRegression", "Linear SVM"]:
        for c_value in [0.1, 1.0, 10.0]:
            if model_name == "LogisticRegression":
                model = LogisticRegression(
                    C=c_value, max_iter=1000, random_state=SEED
                )
            else:
                model = SVC(
                    kernel="linear", C=c_value, random_state=SEED
                )

            _, _, metrics = evaluate_model(
                model, X_train_vec, y_train, X_val_vec, y_val
            )
            results.append({
                "model": model_name,
                "C": c_value,
                **metrics,
            })

    results_df = pd.DataFrame(results).sort_values(
        ["val_accuracy", "val_macro_f1"], ascending=False
    )
    results_df.to_csv(
        DATA_DIR / "all_tuning_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print("\n模型与参数对比：")
    print(results_df.to_string(index=False, float_format="%.4f"))
    return results_df


def run_feature_experiments(X_train, X_val, y_train, y_val):
    """比较特征数量以及是否加入 bigram。"""
    configs = [
        ("unigram_5000", (1, 1), 5000),
        ("unigram_bigram_5000", (1, 2), 5000),
        ("unigram_10000", (1, 1), 10000),
        ("unigram_bigram_10000", (1, 2), 10000),
    ]
    results = []

    for name, ngram_range, max_features in configs:
        config = {
            "ngram_range": ngram_range,
            "max_features": max_features,
        }
        _, X_train_vec, X_val_vec, vectorize_seconds = build_tfidf(
            X_train, X_val, config
        )
        model = LogisticRegression(
            C=BEST_MODEL_C, max_iter=1000, random_state=SEED
        )
        _, _, metrics = evaluate_model(
            model, X_train_vec, y_train, X_val_vec, y_val
        )
        results.append({
            "feature_name": name,
            "ngram_range": str(ngram_range),
            "max_features": max_features,
            "actual_features": X_train_vec.shape[1],
            "vectorize_seconds": vectorize_seconds,
            **metrics,
        })

    results_df = pd.DataFrame(results).sort_values(
        ["val_accuracy", "val_macro_f1"], ascending=False
    )
    results_df.to_csv(
        DATA_DIR / "feature_tuning_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print("\nTF-IDF 特征配置对比：")
    print(results_df.to_string(index=False, float_format="%.4f"))
    return results_df


def save_result_charts(model_results, feature_results):
    """保存模型参数对比图和特征配置对比图。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name, group in model_results.groupby("model"):
        group = group.sort_values("C")
        ax.plot(
            group["C"],
            group["val_accuracy"],
            marker="o",
            linewidth=2,
            label=model_name,
        )
    ax.set_xscale("log")
    ax.set_xticks([0.1, 1.0, 10.0])
    ax.set_xticklabels(["0.1", "1", "10"])
    ax.set_xlabel("C")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Model and Parameter Comparison")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(DATA_DIR / "model_comparison.png", dpi=200)
    plt.close(fig)

    chart_data = feature_results.sort_values("val_accuracy")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(chart_data["feature_name"], chart_data["val_accuracy"])
    ax.set_xlabel("Validation accuracy")
    ax.set_title("TF-IDF Feature Comparison")
    ax.set_xlim(0.88, 0.94)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(DATA_DIR / "feature_comparison.png", dpi=200)
    plt.close(fig)


def analyze_best_model(X_train, X_val, y_train, y_val):
    """评估最佳配置，并保存混淆矩阵与错误样本。"""
    _, X_train_vec, X_val_vec, _ = build_tfidf(
        X_train, X_val, BEST_TFIDF
    )
    model = LogisticRegression(
        C=BEST_MODEL_C, max_iter=1000, random_state=SEED
    )
    _, val_pred, metrics = evaluate_model(
        model, X_train_vec, y_train, X_val_vec, y_val
    )

    print("\n最终最佳配置的验证结果：")
    print(f"验证集准确率：{metrics['val_accuracy']:.4f}")
    print(f"验证集 Macro-F1：{metrics['val_macro_f1']:.4f}")
    print(classification_report(y_val, val_pred, digits=4, zero_division=0))

    labels = sorted(y_val.unique())
    report_df = pd.DataFrame(
        classification_report(
            y_val,
            val_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()
    report_df.to_csv(
        DATA_DIR / "classification_report_best.csv",
        encoding="utf-8-sig",
    )

    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_val,
        val_pred,
        labels=labels,
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title(
        "Best Model Confusion Matrix\n"
        "Logistic Regression, C=10, TF-IDF unigram 10000"
    )
    fig.tight_layout()
    fig.savefig(DATA_DIR / "confusion_matrix_best.png", dpi=200)
    plt.close(fig)

    cm = confusion_matrix(y_val, val_pred, labels=labels)
    np.fill_diagonal(cm, 0)
    confusion_rows = []
    for true_index, true_label in enumerate(labels):
        for pred_index, pred_label in enumerate(labels):
            count = int(cm[true_index, pred_index])
            if count:
                confusion_rows.append({
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "error_count": count,
                })

    confusion_df = pd.DataFrame(confusion_rows).sort_values(
        "error_count", ascending=False
    )
    confusion_df.to_csv(
        DATA_DIR / "confusion_pairs_best.csv",
        index=False,
        encoding="utf-8-sig",
    )

    errors_df = pd.DataFrame({
        "text": X_val.to_numpy(),
        "true_label": y_val.to_numpy(),
        "predicted_label": val_pred,
    })
    errors_df = errors_df[
        errors_df["true_label"] != errors_df["predicted_label"]
    ]
    errors_df.to_csv(
        DATA_DIR / "validation_errors_best.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print("最佳模型错误样本数：", len(errors_df))
    print("主要混淆类别：")
    print(confusion_df.head(10).to_string(index=False))


def train_final_model(train_df, test_df):
    """在全部有标签数据上训练最佳配置，并生成提交文件。"""
    vectorizer = TfidfVectorizer(**BEST_TFIDF)
    X_all_vec = vectorizer.fit_transform(train_df["text"])
    X_test_vec = vectorizer.transform(test_df["text"])

    model = LogisticRegression(
        C=BEST_MODEL_C, max_iter=1000, random_state=SEED
    )
    model.fit(X_all_vec, train_df["target"])
    predictions = model.predict(X_test_vec)

    if len(predictions) != len(test_df):
        raise ValueError("预测数量与测试集数量不一致")
    if not set(predictions).issubset(set(train_df["target"].unique())):
        raise ValueError("预测结果中出现未知标签")

    output_path = DATA_DIR / "predictions_best.csv"
    pd.DataFrame(predictions).to_csv(
        output_path, index=False, header=False
    )
    saved = pd.read_csv(output_path, header=None)
    if saved.shape != (len(test_df), 1):
        raise ValueError(f"预测文件格式错误：{saved.shape}")

    print("\n最终预测文件：", output_path)
    print("预测文件形状：", saved.shape)
    print("预测类别数量：")
    print(pd.Series(predictions).value_counts().sort_index())


def main():
    train_df, test_df = load_data()
    X_train, X_val, y_train, y_val = split_data(train_df)
    print(f"训练集数量：{len(X_train)}，验证集数量：{len(X_val)}")

    model_results = run_model_tuning(
        X_train, X_val, y_train, y_val
    )
    feature_results = run_feature_experiments(
        X_train, X_val, y_train, y_val
    )
    save_result_charts(model_results, feature_results)
    analyze_best_model(X_train, X_val, y_train, y_val)
    train_final_model(train_df, test_df)


if __name__ == "__main__":
    main()
