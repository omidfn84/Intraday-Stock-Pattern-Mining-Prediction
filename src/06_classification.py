"""Step 6: predict tomorrow's volatility tier (Low / Medium / High).

Two classifiers from the course (Module 3) on Classification_Data:
  - Decision tree: settings chosen on the validation days (pre-pruning)
  - Naive Bayes (Gaussian): no settings to tune

Two feature versions, run side by side:
  - v1 "today only": the 6 features describing today
  - v2 "today + history": v1 plus the 5-day average volatility, today/yesterday change,
    and market-wide volatility (today and 5-day). Added because v1 only tied the
    persistence baseline; v2 was checked on the validation days before the test days.

Three baselines (no learning):
  - Majority:       always predict the most common tier in the training days
  - Persistence:    tomorrow's tier = today's tier
  - 5-day average:  tomorrow's tier = the tier of the stock's 5-day average volatility

Final models are refit on train + validation and scored on the test days, which come
after every training day (never shuffled).

Outputs:
  data/processed/classification_results.csv   test predictions + probabilities
  data/processed/classification_metrics.json  every score in the report
  data/processed/tree_tuning.csv              validation score for each setting tried
  figures/fig3_decision_tree.png (v2, final), fig3_decision_tree_v1.png,
  figures/fig4_confusion_matrices.png, fig5_roc_curves.png
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
TODAY = ["realized_volatility", "range_pct", "return_pct", "log_volume",
         "rel_volume", "liquidity_rank"]
HISTORY = ["vol_5d", "vol_change", "market_vol", "market_vol_5d"]
VERSIONS = {"v1": TODAY, "v2": TODAY + HISTORY}          # feature list for each version
LABELS = {"v1": "today only", "v2": "today + history"}
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


def tune_tree(train, val, features, version):
    """Try each setting on the training days, score it on the validation days."""
    rows = []
    for criterion in ["gini", "entropy"]:
        for depth in range(2, 9):
            for leaf in [5, 10, 20, 30]:
                tree = DecisionTreeClassifier(criterion=criterion, max_depth=depth,
                                              min_samples_leaf=leaf, random_state=SEED)
                tree.fit(train[features], train[TARGET])
                pred = tree.predict(val[features])
                rows.append({"version": version, "criterion": criterion, "max_depth": depth,
                             "min_samples_leaf": leaf,
                             "val_macro_f1": f1_score(val[TARGET], pred, average="macro"),
                             "val_accuracy": accuracy_score(val[TARGET], pred),
                             "leaves": tree.get_n_leaves()})
    grid = pd.DataFrame(rows)
    # Best macro F1; ties go to the simpler tree (fewer leaves), as Occam's razor suggests.
    best = grid.sort_values(["val_macro_f1", "leaves"], ascending=[False, True]).iloc[0]
    return best, grid


def draw_tree(tree, features, version, path):
    """The whole tree (shallow enough that every leaf fits)."""
    fig, ax = plt.subplots(figsize=(30, 11))
    plot_tree(tree, feature_names=features, class_names=list(tree.classes_), filled=True,
              fontsize=8, impurity=True, proportion=False, rounded=True, ax=ax)
    ax.set_title(f"Decision tree {version} ({LABELS[version]}; {tree.criterion}, max depth "
                 f"{tree.max_depth}, {tree.get_n_leaves()} leaves). value = number of training "
                 f"days per tier, in the order {list(tree.classes_)}; class = the tier the node "
                 f"predicts.\nAt every split, the LEFT branch means the condition is true "
                 f"(value <= threshold) and the RIGHT branch means it is false.", fontsize=14)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    config = load_config()
    cut = config["volatility_cutoffs"]
    data = pd.read_csv(PROCESSED / "classification_data.csv")
    data["today_tier"] = to_tier(data["realized_volatility"], cut)    # for persistence
    data["tier_5d"] = to_tier(data["vol_5d"], cut)                     # for the 5-day rule
    train, val, test = (data[data["split"] == s] for s in ["train", "validation", "test"])
    fit_data = pd.concat([train, val])                    # final models learn from both
    metrics = {"rows": {"train": len(train), "validation": len(val), "test": len(test)},
               "test_class_counts": test[TARGET].value_counts().reindex(CLASSES).to_dict(),
               "validation": {}, "baselines": {}}

    # ---- baselines, scored on validation (to justify v2) and on test
    majority = train[TARGET].mode()[0]
    for part, rows in [("validation", val), ("test", test)]:
        rules = {"majority": pd.Series(majority, index=rows.index),
                 "persistence": rows["today_tier"], "five_day_average": rows["tier_5d"]}
        for name, pred in rules.items():
            s = scores(rows[TARGET], pred)
            if part == "validation":
                metrics["validation"][name] = {"accuracy": s["accuracy"], "macro_f1": s["macro_f1"]}
            else:
                metrics["baselines"][name] = s
    metrics["baselines"]["majority"]["predicts"] = majority

    results = test[["ticker", "feature_date", "label_date", "liquidity_rank", TARGET]].copy()
    models, grids = {}, []
    for version, features in VERSIONS.items():
        # ---- decision tree: tune on validation, refit on train + validation
        best, grid = tune_tree(train, val, features, version)
        grids.append(grid)
        tree = DecisionTreeClassifier(criterion=best["criterion"], max_depth=int(best["max_depth"]),
                                      min_samples_leaf=int(best["min_samples_leaf"]),
                                      random_state=SEED).fit(fit_data[features], fit_data[TARGET])
        # ---- naive Bayes (validation score uses a model trained on train only)
        nb_val = GaussianNB().fit(train[features], train[TARGET]).predict(val[features])
        nb = GaussianNB().fit(fit_data[features], fit_data[TARGET])
        metrics["validation"][f"decision_tree_{version}"] = {
            "accuracy": float(best["val_accuracy"]), "macro_f1": float(best["val_macro_f1"])}
        metrics["validation"][f"naive_bayes_{version}"] = {
            "accuracy": accuracy_score(val[TARGET], nb_val),
            "macro_f1": f1_score(val[TARGET], nb_val, average="macro")}
        metrics[f"tree_settings_{version}"] = {
            "features": features, "criterion": best["criterion"],
            "max_depth": int(best["max_depth"]), "min_samples_leaf": int(best["min_samples_leaf"]),
            "leaves_after_refit": int(tree.get_n_leaves())}
        metrics[f"tree_feature_importance_{version}"] = dict(sorted(
            zip(features, (round(float(v), 4) for v in tree.feature_importances_)),
            key=lambda kv: -kv[1]))
        models[f"decision_tree_{version}"] = (tree, features)
        models[f"naive_bayes_{version}"] = (nb, features)
        draw_tree(tree, features, version, FIGURES / ("fig3_decision_tree.png" if version == "v2"
                                                      else "fig3_decision_tree_v1.png"))
    pd.concat(grids).to_csv(PROCESSED / "tree_tuning.csv", index=False)

    # ---- test scores for the four models
    for name, (model, features) in models.items():
        results[f"pred_{name}"] = model.predict(test[features])
        proba = pd.DataFrame(model.predict_proba(test[features]), columns=model.classes_,
                             index=test.index)[CLASSES]
        for c in CLASSES:
            results[f"prob_{name}_{c}"] = proba[c]
        metrics[name] = scores(test[TARGET], results[f"pred_{name}"])
        metrics[name]["accuracy_by_liquidity_tier"] = {
            tier: accuracy_score(g[TARGET], g[f"pred_{name}"])
            for tier, g in results.groupby(results["liquidity_rank"].map({1: "Low", 2: "Medium", 3: "High"}))}
        metrics[name]["confusion_matrix"] = confusion_matrix(
            test[TARGET], results[f"pred_{name}"], labels=CLASSES).tolist()

    # ---- ROC: one class against the rest; solid = v2, dashed = v1
    styles = {"decision_tree_v2": ("tab:blue", "-"), "naive_bayes_v2": ("tab:orange", "-"),
              "decision_tree_v1": ("tab:blue", "--"), "naive_bayes_v1": ("tab:orange", "--")}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, c in zip(axes, CLASSES):
        y = (test[TARGET] == c).astype(int)
        for name, (color, style) in styles.items():
            fpr, tpr, _ = roc_curve(y, results[f"prob_{name}_{c}"])
            area = auc(fpr, tpr)
            metrics[name].setdefault("auc", {})[c] = area
            model, version = name.rsplit("_", 1)
            ax.plot(fpr, tpr, style, color=color,
                    label=f"{model.replace('_', ' ').title()} {version} (AUC = {area:.2f})")
        ax.plot([0, 1], [0, 1], ":", color="gray", label="Random guessing")
        ax.set_title(f"{c} vs. rest")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate (recall)")
        ax.legend(loc="lower right", fontsize=8.5)
    fig.suptitle("ROC curves on the test days, next-day volatility tier "
                 "(solid = v2 today + history, dashed = v1 today only)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig5_roc_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    for name in models:
        metrics[name]["macro_auc"] = float(np.mean(list(metrics[name]["auc"].values())))

    # ---- confusion matrices: rows = version, columns = model
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5))
    for ax, name in zip(axes.flat, ["decision_tree_v1", "naive_bayes_v1",
                                    "decision_tree_v2", "naive_bayes_v2"]):
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
        model, version = name.rsplit("_", 1)
        ax.set_title(f"{model.replace('_', ' ').title()} {version} ({LABELS[version]}), "
                     f"accuracy {metrics[name]['accuracy']:.1%}")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig4_confusion_matrices.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    results.to_csv(PROCESSED / "classification_results.csv", index=False)
    (PROCESSED / "classification_metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    # ---- summary
    print(f"Rows: {metrics['rows']}; test tiers {metrics['test_class_counts']}")
    for v in VERSIONS:
        print(f"Tree {v} settings: { {k: x for k, x in metrics[f'tree_settings_{v}'].items() if k != 'features'} }")
        print(f"Tree {v} importance: {metrics[f'tree_feature_importance_{v}']}")
    print("\nVALIDATION (used to choose settings and to justify v2):")
    print(pd.DataFrame(metrics["validation"]).T.round(3).to_string())
    print("\nTEST:")
    rows = {**{k: metrics["baselines"][k] for k in ["majority", "persistence", "five_day_average"]},
            **{k: metrics[k] for k in models}}
    print(pd.DataFrame({k: {"accuracy": m["accuracy"], "macro_F1": m["macro_f1"],
                            "macro_AUC": m.get("macro_auc")} for k, m in rows.items()}).T.round(3).to_string())
    for name in models:
        print(f"{name}: AUC {dict((k, round(v, 3)) for k, v in metrics[name]['auc'].items())}; "
              f"recall {dict((c, round(metrics[name]['per_class'][c]['recall'], 3)) for c in CLASSES)}; "
              f"accuracy by liquidity {dict((k, round(v, 3)) for k, v in metrics[name]['accuracy_by_liquidity_tier'].items())}")


if __name__ == "__main__":
    main()
