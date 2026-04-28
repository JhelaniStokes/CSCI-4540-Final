import os
import warnings
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve,
    f1_score, recall_score, precision_score, ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
os.makedirs("figures", exist_ok=True)

SEED = 42
np.random.seed(SEED)


# ── 1. Load ───────────────────────────────────────────────────────────────────
def load_data(path="network_data.csv"):
    df = pd.read_csv(path)
    return df


# ── 2. EDA ────────────────────────────────────────────────────────────────────
def eda(df):
    print("=== Dataset Shape ===")
    print(df.shape)
    print("\n=== Head ===")
    print(df.head(3).to_string())
    print("\n=== Describe ===")
    print(df.describe().to_string())
    print("\n=== Missing Values ===")
    print(df.isnull().sum())
    print("\n=== Class Distribution ===")
    counts = df["is_malicious"].value_counts()
    print(counts)

    # Correlation heatmap
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(corr.columns, fontsize=7)
    plt.colorbar(im, ax=ax)
    ax.set_title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig("figures/correlation_heatmap.png", dpi=150)
    plt.close()

    return counts


# ── 3. Feature Engineering ────────────────────────────────────────────────────
def engineer_features(df):
    df = df.copy()
    df["byte_ratio"]       = df["src_bytes"] / (df["dst_bytes"] + 1)
    df["bytes_per_packet"] = (df["src_bytes"] + df["dst_bytes"]) / (df["packet_count"] + 1)
    df["failure_rate"]     = df["num_failed_connections"] / (df["packet_count"] + 1)
    return df


# ── 4. Preprocessing ──────────────────────────────────────────────────────────
def preprocess(df):
    feature_cols = [c for c in df.columns if c != "is_malicious"]
    X = df[feature_cols].values
    y = df["is_malicious"].values
    return X, y, feature_cols


# ── 5. Models + Grid Search ───────────────────────────────────────────────────
def build_models(X_train, y_train):
    cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

    # Logistic Regression — tune regularization strength
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")),
    ])
    lr = GridSearchCV(lr_pipe, {"clf__C": [0.01, 0.1, 1, 10, 100]},
                      cv=cv_inner, scoring="f1", n_jobs=-1)
    lr.fit(X_train, y_train)
    print(f"LR best params: {lr.best_params_}  (CV F1={lr.best_score_:.4f})")

    # Random Forest — tune depth and tree count
    rf_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(random_state=SEED, class_weight="balanced")),
    ])
    rf = GridSearchCV(rf_pipe,
                      {"clf__n_estimators": [100, 200],
                       "clf__max_depth": [None, 10, 20],
                       "clf__min_samples_split": [2, 5]},
                      cv=cv_inner, scoring="f1", n_jobs=-1)
    rf.fit(X_train, y_train)
    print(f"RF best params: {rf.best_params_}  (CV F1={rf.best_score_:.4f})")

    return {"Logistic Regression": lr, "Random Forest": rf}


# ── 6. Evaluate ───────────────────────────────────────────────────────────────
def evaluate(models, X_train, X_test, y_train, y_test, feature_cols):
    results = {}
    fig_roc, ax_roc = plt.subplots(figsize=(6, 5))

    colors = ["steelblue", "tomato"]

    for (name, model), color in zip(models.items(), colors):
        y_proba    = model.predict_proba(X_test)[:, 1]
        thresholds = np.linspace(0.01, 0.99, 200)

        # F1-optimal threshold
        f1_scores   = [f1_score(y_test, (y_proba >= t).astype(int), zero_division=0) for t in thresholds]
        thresh_f1   = thresholds[np.argmax(f1_scores)]
        y_pred_f1   = (y_proba >= thresh_f1).astype(int)

        # Min-FN threshold: max recall, then max precision among ties
        recalls     = [recall_score(y_test, (y_proba >= t).astype(int), zero_division=0) for t in thresholds]
        max_recall  = max(recalls)
        candidates  = [t for t, r in zip(thresholds, recalls) if r >= max_recall]
        thresh_minfn = max(candidates, key=lambda t: precision_score(y_test, (y_proba >= t).astype(int), zero_division=0))
        y_pred_minfn = (y_proba >= thresh_minfn).astype(int)

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ax_roc.plot(fpr, tpr, color=color, lw=2, label=name)

        report_f1    = classification_report(y_test, y_pred_f1,    output_dict=True)
        report_minfn = classification_report(y_test, y_pred_minfn, output_dict=True)

        results[name] = {
            "f1_thresh": {
                "threshold": thresh_f1,
                "report":    report_f1,
                "cm":        confusion_matrix(y_test, y_pred_f1).tolist(),
            },
            "minfn_thresh": {
                "threshold": thresh_minfn,
                "report":    report_minfn,
                "cm":        confusion_matrix(y_test, y_pred_minfn).tolist(),
            },
        }

        print(f"\n=== {name} — F1-optimal (t={thresh_f1:.3f}) ===")
        print(classification_report(y_test, y_pred_f1, target_names=["Benign", "Malicious"]))
        print(f"=== {name} — Min-FN (t={thresh_minfn:.3f}) ===")
        print(classification_report(y_test, y_pred_minfn, target_names=["Benign", "Malicious"]))

        # Confusion matrices for both strategies
        for label, y_pred, thresh in [("f1", y_pred_f1, thresh_f1), ("minfn", y_pred_minfn, thresh_minfn)]:
            fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
            disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred),
                                          display_labels=["Benign", "Malicious"])
            disp.plot(ax=ax_cm, colorbar=False, cmap="Blues")
            safe_name = name.lower().replace(" ", "_")
            ax_cm.set_title(f"{name} ({label}, t={thresh:.3f})")
            plt.tight_layout()
            plt.savefig(f"figures/cm_{safe_name}_{label}.png", dpi=150)
            plt.close()

    # Finalize ROC plot
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curves")
    ax_roc.legend(loc="lower right")
    plt.tight_layout()
    fig_roc.savefig("figures/roc_curves.png", dpi=150)
    plt.close(fig_roc)

    # Feature importance (Random Forest)
    rf_model = models["Random Forest"].best_estimator_.named_steps["clf"]
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(indices)), importances[indices], color="steelblue")
    ax.set_xticks(range(len(indices)))
    ax.set_xticklabels([feature_cols[i] for i in indices], rotation=45, ha="right", fontsize=8)
    ax.set_title("Importances (Random Forest)")
    ax.set_ylabel("Importance")
    plt.tight_layout()
    plt.savefig("figures/feature_importance.png", dpi=150)
    plt.close()

    return results


# ── 7. Cross-Validation ───────────────────────────────────────────────────────
def cross_validate_models(models, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_results = {}
    scoring = ["accuracy", "precision", "recall", "f1"]

    print("\n=== 5-Fold Cross-Validation ===")
    for name, model in models.items():
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        cv_results[name] = {s: (scores[f"test_{s}"].mean(), scores[f"test_{s}"].std()) for s in scoring}
        print(f"\n{name}:")
        for s in scoring:
            m, sd = cv_results[name][s]
            print(f"  {s:12s}: {m:.4f} ± {sd:.4f}")

    # CV comparison bar chart
    metrics_to_plot = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (name, scores) in enumerate(cv_results.items()):
        means = [scores[m][0] for m in metrics_to_plot]
        errs  = [scores[m][1] for m in metrics_to_plot]
        ax.bar(x + i * width, means, width, yerr=errs, label=name, capsize=4)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1"])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("5-Fold Cross-Validation Comparison")
    ax.legend()
    plt.tight_layout()
    plt.savefig("figures/cv_comparison.png", dpi=150)
    plt.close()

    return cv_results


# ── 8. Save metrics for LaTeX ─────────────────────────────────────────────────
def save_metrics(results, cv_results, counts):
    metrics = {
        "test_results": {
            name: {
                strat: {
                    "threshold":  r[strat]["threshold"],
                    "precision":  r[strat]["report"]["1"]["precision"],
                    "recall":     r[strat]["report"]["1"]["recall"],
                    "f1":         r[strat]["report"]["1"]["f1-score"],
                    "accuracy":   r[strat]["report"]["accuracy"],
                    "cm":         r[strat]["cm"],
                }
                for strat in ("f1_thresh", "minfn_thresh")
            }
            for name, r in results.items()
        },
        "cv_results": {
            name: {s: list(v) for s, v in cv.items() if s != "roc_auc"}
            for name, cv in cv_results.items()
        },
        "class_counts": {int(k): int(v) for k, v in counts.items()},
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nMetrics saved to metrics.json")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    df = load_data()
    counts = eda(df)

    df = engineer_features(df)
    X, y, feature_cols = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    models = build_models(X_train, y_train)

    results = evaluate(models, X_train, X_test, y_train, y_test, feature_cols)

    cv_results = cross_validate_models(models, X, y)

    save_metrics(results, cv_results, counts)
    print("\nAll figures saved to figures/")


if __name__ == "__main__":
    main()
