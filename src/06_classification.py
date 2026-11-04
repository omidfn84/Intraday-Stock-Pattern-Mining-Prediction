"""Step 6: predict tomorrow's volatility tier (Low / Medium / High).

Two classifiers from the course (Module 3) on Classification_Data:
  - Decision tree: settings chosen on the validation days (pre-pruning)
  - Naive Bayes (Gaussian): no settings to tune
Both are compared with two baselines:
  - Majority: always predict the most common tier in the training days
  - Persistence: predict that tomorrow's tier equals today's
Final models are refit on train + validation and scored ONCE on the test days,
which come after every training day (never shuffled).

Outputs:
  data/processed/classification_results.csv   test predictions + probabilities
  data/processed/classification_metrics.json  every score in the report
  data/processed/tree_tuning.csv              validation score for each setting tried
  figures/fig3_decision_tree.png, fig4_confusion_matrices.png, fig5_roc_curves.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, auc, classification_report, confusion_matrix,
                             f1_score, roc_curve)
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree

from common import FIGURES, PROCESSED, load_config, to_tier

CLASSES = ["Low", "Medium", "High"]
FEATURES = ["realized_volatility", "range_pct", "return_pct", "log_volume",
            "rel_volume", "liquidity_rank"]
TARGET = "next_volatility_tier"
SEED = 42


def scores(y_true, y_pred):
    """Accuracy plus macro-averaged and per-class precision / recall / F1."""
    report = classification_report(y_true, y_pred, labels=CLASSES, output_dict=True,
                                   zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred),
            "macro_precision": report["macro avg"]["precision"],
            "macro_recall": report["macro avg"]["recall"],
            "macro_f1": report["macro avg"]["f1-score"],
            "per_class": {c: {k: report[c][k] for k in ["precision", "recall", "f1-score", "support"]}
                          for c in CLASSES}}


def tune_tree(train, val):
    """Try each setting on the training days, score it on the validation days."""
    rows = []
    for criterion in ["gini", "entropy"]:
        for depth in range(2, 9):
            for leaf in [5, 10, 20, 30]:
                tree = DecisionTreeClassifier(criterion=criterion, max_depth=depth,
                                              min_samples_leaf=leaf, random_state=SEED)
                tree.fit(train[FEATURES], train[TARGET])
                pred = tree.predict(val[FEATURES])
                rows.append({"criterion": criterion, "max_depth": depth, "min_samples_leaf": leaf,
                             "val_macro_f1": f1_score(val[TARGET], pred, average="macro"),
                             "val_accuracy": accuracy_score(val[TARGET], pred),
                             "leaves": tree.get_n_leaves()})
    grid = pd.DataFrame(rows)
    # Best macro F1; ties go to the simpler tree (fewer leaves), as Occam's razor suggests.
    best = grid.sort_values(["val_macro_f1", "leaves"], ascending=[False, True]).iloc[0]
    grid.to_csv(PROCESSED / "tree_tuning.csv", index=False)
    return best


def main():
    config = load_config()
    data = pd.read_csv(PROCESSED / "classification_data.csv")
    data["today_tier"] = to_tier(data["realized_volatility"], config["volatility_cutoffs"])
    train, val, test = (data[data["split"] == s] for s in ["train", "validation", "test"])
    fit_data = pd.concat([train, val])                    # final models learn from both
    metrics = {"rows": {"train": len(train), "validation": len(val), "test": len(test)},
               "test_class_counts": test[TARGET].value_counts().reindex(CLASSES).to_dict()}

    # ---- baselines
    majority = train[TARGET].mode()[0]
    baselines = {"majority": pd.Series(majority, index=test.index),
                 "persistence": test["today_tier"]}
    metrics["baselines"] = {name: scores(test[TARGET], pred) for name, pred in baselines.items()}
    metrics["baselines"]["majority"]["predicts"] = majority

    # ---- decision tree: tune on validation, refit on train + validation
    best = tune_tree(train, val)
    tree = DecisionTreeClassifier(criterion=best["criterion"], max_depth=int(best["max_depth"]),
                                  min_samples_leaf=int(best["min_samples_leaf"]), random_state=SEED)
    tree.fit(fit_data[FEATURES], fit_data[TARGET])
    metrics["tree_settings"] = {"criterion": best["criterion"], "max_depth": int(best["max_depth"]),
                                "min_samples_leaf": int(best["min_samples_leaf"]),
                                "validation_macro_f1": float(best["val_macro_f1"]),
                                "leaves_after_refit": int(tree.get_n_leaves())}
    metrics["tree_feature_importance"] = dict(sorted(
        zip(FEATURES, (round(float(v), 4) for v in tree.feature_importances_)), key=lambda kv: -kv[1]))

    # ---- naive Bayes
    nb = GaussianNB().fit(fit_data[FEATURES], fit_data[TARGET])
    nb_val = GaussianNB().fit(train[FEATURES], train[TARGET]).predict(val[FEATURES])
    metrics["nb_validation_macro_f1"] = f1_score(val[TARGET], nb_val, average="macro")

    # ---- test scores
    models = {"decision_tree": tree, "naive_bayes": nb}
    results = test[["ticker", "feature_date", "label_date", "liquidity_rank", TARGET]].copy()
    for name, model in models.items():
        results[f"pred_{name}"] = model.predict(test[FEATURES])
        proba = pd.DataFrame(model.predict_proba(test[FEATURES]), columns=model.classes_,
                             index=test.index)[CLASSES]
        for c in CLASSES:
            results[f"prob_{name}_{c}"] = proba[c]
        metrics[name] = scores(test[TARGET], results[f"pred_{name}"])
        metrics[name]["accuracy_by_liquidity_tier"] = {
            tier: accuracy_score(g[TARGET], g[f"pred_{name}"])
            for tier, g in results.groupby(results["liquidity_rank"].map({1: "Low", 2: "Medium", 3: "High"}))}
        metrics[name]["confusion_matrix"] = confusion_matrix(
            test[TARGET], results[f"pred_{name}"], labels=CLASSES).tolist()

    # ---- ROC: one class against the rest, both models on the same axes
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, c in zip(axes, CLASSES):
        y = (test[TARGET] == c).astype(int)
        for name, style in [("decision_tree", "-"), ("naive_bayes", "--")]:
            fpr, tpr, _ = roc_curve(y, results[f"prob_{name}_{c}"])
            area = auc(fpr, tpr)
            metrics[name].setdefault("auc", {})[c] = area
            ax.plot(fpr, tpr, style, label=f"{name.replace('_', ' ').title()} (AUC = {area:.2f})")
        ax.plot([0, 1], [0, 1], ":", color="gray", label="Random guessing")
        ax.set_title(f"{c} vs. rest")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate (recall)")
        ax.legend(loc="lower right", fontsize=9)
    fig.suptitle("ROC curves on the test days (Oct 9 - 23), next-day volatility tier")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig5_roc_curves.png", dpi=200, bbox_inches="tight")
    for name in models:
        metrics[name]["macro_auc"] = float(np.mean(list(metrics[name]["auc"].values())))

    # ---- confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, name in zip(axes, models):
        cm = np.array(metrics[name]["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks(range(3), CLASSES)
        ax.set_yticks(range(3), CLASSES)
        ax.set_xlabel("Predicted tier")
        ax.set_ylabel("Actual tier")
        ax.set_title(f"{name.replace('_', ' ').title()} "
                     f"(accuracy {metrics[name]['accuracy']:.1%})")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig4_confusion_matrices.png", dpi=200, bbox_inches="tight")

    # ---- the whole tree (it is only 4 levels deep, so every leaf fits)
    fig, ax = plt.subplots(figsize=(30, 11))
    plot_tree(tree, feature_names=FEATURES, class_names=list(tree.classes_), filled=True,
              fontsize=8, impurity=True, proportion=False, rounded=True, ax=ax)
    ax.set_title(f"Decision tree ({metrics['tree_settings']['criterion']}, max depth "
                 f"{metrics['tree_settings']['max_depth']}, {tree.get_n_leaves()} leaves). "
                 f"value = number of training days per tier, in the order {list(tree.classes_)}; "
                 f"class = the tier the node predicts", fontsize=14)
    fig.savefig(FIGURES / "fig3_decision_tree.png", dpi=200, bbox_inches="tight")

    results.to_csv(PROCESSED / "classification_results.csv", index=False)
    (PROCESSED / "classification_metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    # ---- summary
    print(f"Rows: {metrics['rows']}; test tiers {metrics['test_class_counts']}")
    print(f"Tree settings (chosen on validation): {metrics['tree_settings']}")
    print(f"Tree feature importance: {metrics['tree_feature_importance']}")
    table = pd.DataFrame({
        name: {"accuracy": m["accuracy"], "macro_F1": m["macro_f1"], "macro_AUC": m.get("macro_auc")}
        for name, m in [("majority", metrics["baselines"]["majority"]),
                        ("persistence", metrics["baselines"]["persistence"]),
                        ("decision_tree", metrics["decision_tree"]),
                        ("naive_bayes", metrics["naive_bayes"])]}).T
    print(table.round(3).to_string())
    for name in models:
        print(f"{name}: AUC {dict((k, round(v, 3)) for k, v in metrics[name]['auc'].items())}; "
              f"recall {dict((c, round(metrics[name]['per_class'][c]['recall'], 3)) for c in CLASSES)}; "
              f"accuracy by liquidity {dict((k, round(v, 3)) for k, v in metrics[name]['accuracy_by_liquidity_tier'].items())}")


if __name__ == "__main__":
    main()
