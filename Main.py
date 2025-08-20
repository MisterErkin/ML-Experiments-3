#!/usr/bin/env python3
import os
import sys
import json
import time
import platform
from datetime import datetime
from typing import List, Optional, Tuple, Dict

import pandas as pd
import numpy as np
import sklearn  # for version logging

# GUI
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Optional resampling
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    IMBLEARN_AVAILABLE = True
except Exception:
    IMBLEARN_AVAILABLE = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


APP_TITLE = "ML-Experiments-3 — Fraud Detection"
ANALYSIS_SAMPLE_CAP = 100_000
LARGE_ROWS_THRESHOLD = 1_000_000
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42

# where to write results (next to Main.py)
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))

# --- simple console logging ---
LOG_TO_TERMINAL = True  # set False to silence prints


def log(msg: str):
    if LOG_TO_TERMINAL:
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {msg}", flush=True)


def log_step(i: int, n: int, msg: str):
    if not LOG_TO_TERMINAL:
        return
    pct = int((i / max(1, n)) * 100)
    sys.stdout.write(f"\r[{datetime.now().strftime('%H:%M:%S')}] {msg} ... {pct:3d}%")
    sys.stdout.flush()
    if i >= n:
        sys.stdout.write("\n")


# ---------------- utils ----------------
def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def open_folder(path: str):
    try:
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            os.system(f'open "{path}"')
        else:
            os.system(f'xdg-open "{path}"')
    except Exception:
        pass


def estimate_memory_mb(df: pd.DataFrame) -> float:
    try:
        return round(df.memory_usage(deep=True).sum() / (1024 ** 2), 2)
    except Exception:
        return float("nan")


def quick_read_csv_head(path: str, nrows: int = 10_000) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows)


def read_full_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def infer_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if c not in num_cols]
    return num_cols, cat_cols


def brief_summary(df_sample: pd.DataFrame) -> Dict:
    n_rows, n_cols = df_sample.shape
    num_cols, cat_cols = infer_types(df_sample)
    missing_top = df_sample.isna().mean().sort_values(ascending=False).head(10).to_dict()
    mem_mb = estimate_memory_mb(df_sample)
    return {
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "memory_mb": mem_mb,
        "n_numeric": len(num_cols),
        "n_categorical": len(cat_cols),
        "missing_top10": missing_top
    }


def detect_target_candidates(df: pd.DataFrame) -> List[str]:
    lower_cols = {c.lower(): c for c in df.columns}
    candidates_lower = [k for k in lower_cols.keys() if k in ("is_fraud", "fraud", "label", "target", "class")]
    return [lower_cols[k] for k in candidates_lower]


def compute_recommendations(df_sample: pd.DataFrame, y_col: str, pos_label) -> List[str]:
    recs = []
    y = df_sample[y_col]
    vc = y.value_counts(dropna=True)
    if pos_label not in vc.index and len(vc) >= 1:
        pos_label = vc.index[0]
    pos_count = int(vc.get(pos_label, 0))
    neg_count = int(vc.sum() - pos_count)
    if pos_count == 0:
        recs.append("No positive samples detected in sample. Ensure target & positive label are correct.")
        return recs
    imb = neg_count / max(pos_count, 1)
    if imb >= 5:
        recs.append("Strong imbalance: prefer PR-AUC, tune for recall; start with class_weight or SMOTE.")
    else:
        recs.append("Moderate imbalance: class_weight likely sufficient; prefer PR-AUC over Accuracy.")
    num_cols, cat_cols = infer_types(df_sample.drop(columns=[y_col]))
    if cat_cols:
        recs.append("Categoricals detected: One-hot will be applied. Trees (RF/GB) are robust; LR needs scaling.")
    if df_sample.shape[0] > 300_000:
        recs.append("Large dataset: avoid SVC/KNN or use subset; prefer LR/RF/GB for speed.")
    recs.append("Suggested models: LogisticRegression, RandomForest, GradientBoosting.")
    return recs


def save_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_text(path: str, text: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# ---------------- preprocessing ----------------
def build_preprocessor(
    X: pd.DataFrame,
    numeric_strategy: str = "median",
    categorical_strategy: str = "most_frequent",
    scale_numeric: bool = True,
) -> ColumnTransformer:
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    transformers = []

    # Numeric: impute (+indicator) -> optional scale
    if num_cols:
        num_steps = [("imputer", SimpleImputer(strategy=numeric_strategy, add_indicator=True))]
        if scale_numeric:
            num_steps.append(("scaler", StandardScaler()))
        transformers.append(("num", Pipeline(steps=num_steps), num_cols))

    # Categorical: impute -> one-hot
    if cat_cols:
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)  # older sklearn
        transformers.append(("cat", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy=categorical_strategy, fill_value="Unknown")),
            ("onehot", ohe),
        ]), cat_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


# ---------------- models / eval ----------------
def get_models(selected: List[str], class_weight_mode: bool) -> Dict[str, object]:
    models = {}
    cw = "balanced" if class_weight_mode else None
    for key in selected:
        k = key.lower()
        if k == "logisticregression":
            models["logreg"] = LogisticRegression(max_iter=300, class_weight=cw, n_jobs=None)
        elif k == "randomforest":
            models["rf"] = RandomForestClassifier(n_estimators=250, class_weight=cw, n_jobs=-1, random_state=42)
        elif k == "gradientboosting":
            models["gb"] = GradientBoostingClassifier(random_state=42)
        elif k in ("kneighbors", "knn"):
            models["knn"] = KNeighborsClassifier(n_neighbors=11)
        elif k == "decisiontree":
            models["dt"] = DecisionTreeClassifier(class_weight=cw, random_state=42)
        elif k == "svc":
            from sklearn.svm import SVC
            models["svc"] = SVC(probability=True, class_weight=cw, random_state=42)
    return models


def resample_if_needed(X, y, sampler_choice: str):
    sampler_choice = (sampler_choice or "class_weight").lower()
    if sampler_choice in ["none", "class_weight"]:
        return X, y, sampler_choice
    if not IMBLEARN_AVAILABLE:
        return X, y, "class_weight"
    if sampler_choice == "smote":
        X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)
        return X_res, y_res, "smote"
    if sampler_choice in ["under", "randomunder", "random_under"]:
        X_res, y_res = RandomUnderSampler(random_state=42).fit_resample(X, y)
        return X_res, y_res, "under"
    return X, y, "class_weight"


def evaluate_all(y_true, preds: Dict[str, np.ndarray], probas: Dict[str, Optional[np.ndarray]]) -> Dict[str, Dict]:
    out = {}
    for name in preds:
        y_pred = preds[name]
        y_proba = probas.get(name)
        met = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            spec = tn / (tn + fp + 1e-9)
        else:
            spec = float("nan")
        met["specificity"] = float(spec)
        if y_proba is not None:
            try:
                met["roc_auc"] = float(roc_auc_score(y_true, y_proba))
            except Exception:
                met["roc_auc"] = None
            try:
                met["pr_auc"] = float(average_precision_score(y_true, y_proba))
            except Exception:
                met["pr_auc"] = None
        else:
            met["roc_auc"] = None
            met["pr_auc"] = None
        out[name] = met
    return out


# -------- plotting --------
def plot_confusion(
    y_true, y_pred, out_path: str, model_name: str,
    normalize: bool = True, labels=("0","1")
):
    cm_norm = confusion_matrix(y_true, y_pred, normalize="true" if normalize else None)
    cm_raw  = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    im = ax.imshow(cm_norm, interpolation="nearest")

    title = f"{model_name}\nConfusion Matrix" + (" (normalized)" if normalize else "")
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1]); ax.set_xticklabels(list(labels))
    ax.set_yticks([0, 1]); ax.set_yticklabels(list(labels))

    for (i, j), v in np.ndenumerate(cm_norm):
        ax.text(j, i, f"{v:.2f}\n({cm_raw[i, j]})", ha="center", va="center")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pr_single(y_true, y_proba, out_path: str, model_name: str, ap: Optional[float] = None):
    if y_proba is None:
        return
    if ap is None:
        try:
            ap = average_precision_score(y_true, y_proba)
        except Exception:
            ap = None
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(rec, prec, label=model_name)
    ttl = f"{model_name} — Precision-Recall"
    if ap is not None:
        ttl += f" (AP={ap:.3f})"
    ax.set_title(ttl)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc_single(y_true, y_proba, out_path: str, model_name: str, auc_roc: Optional[float] = None):
    if y_proba is None:
        return
    if auc_roc is None:
        try:
            auc_roc = roc_auc_score(y_true, y_proba)
        except Exception:
            auc_roc = None
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=model_name)
    ttl = f"{model_name} — ROC Curve"
    if auc_roc is not None:
        ttl += f" (AUC={auc_roc:.3f})"
    ax.set_title(ttl)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_curves_combined(y_true, probas: Dict[str, Optional[np.ndarray]], out_dir: str):
    fig1, ax1 = plt.subplots(figsize=(5, 4))
    any_plot = False
    for name, p in probas.items():
        if p is None:
            continue
        prec, rec, _ = precision_recall_curve(y_true, p)
        ax1.plot(rec, prec, label=name)
        any_plot = True
    if any_plot:
        ax1.set_title("Precision–Recall (all models)")
        ax1.set_xlabel("Recall")
        ax1.set_ylabel("Precision")
        ax1.legend()
        fig1.tight_layout()
        fig1.savefig(os.path.join(out_dir, "pr_curve.png"), dpi=150)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(5, 4))
    any_plot = False
    for name, p in probas.items():
        if p is None:
            continue
        fpr, tpr, _ = roc_curve(y_true, p)
        ax2.plot(fpr, tpr, label=name)
        any_plot = True
    if any_plot:
        ax2.set_title("ROC (all models)")
        ax2.set_xlabel("FPR")
        ax2.set_ylabel("TPR")
        ax2.legend()
        fig2.tight_layout()
        fig2.savefig(os.path.join(out_dir, "roc_curve.png"), dpi=150)
    plt.close(fig2)


# ---------------- GUI helpers ----------------
def ask_dataset_path(root: tk.Tk) -> Optional[str]:
    while True:
        path = filedialog.askopenfilename(
            parent=root, title="Select a dataset (.csv)",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if path:
            return path
        res = messagebox.askyesno(APP_TITLE, "No file selected.\n\nDo you want to re-select?\n(Yes = Re-select, No = Exit)")
        if not res:
            return None


def popup_summary(root: tk.Tk, summary: Dict, recommendations: List[str]):
    txt = []
    txt.append(f"Rows (sample): {summary['n_rows']}")
    txt.append(f"Columns: {summary['n_cols']} | Memory (MB, sample): {summary['memory_mb']}")
    txt.append(f"Numeric: {summary['n_numeric']} | Categorical: {summary['n_categorical']}")
    if summary['missing_top10']:
        top = ", ".join([f"{k}={v:.1%}" for k, v in summary['missing_top10'].items()])
        txt.append(f"Top missingness: {top}")
    if recommendations:
        txt.append("\nRecommendations:")
        for r in recommendations:
            txt.append(f"- {r}")
    messagebox.showinfo(APP_TITLE, "\n".join(txt), parent=root)


def ask_target_and_pos(root: tk.Tk, df: pd.DataFrame, candidates: List[str]) -> Tuple[Optional[str], Optional[object]]:
    win = tk.Toplevel(root)
    win.title("Select target column")
    win.resizable(False, False)

    tk.Label(win, text="Target column:").grid(row=0, column=0, padx=8, pady=6, sticky="w")
    target_var = tk.StringVar(value=candidates[0] if candidates else (df.columns[0] if len(df.columns) else ""))
    target_cb = ttk.Combobox(win, values=list(df.columns), textvariable=target_var, state="readonly", width=40)
    target_cb.grid(row=0, column=1, padx=8, pady=6)

    pos_label_var = tk.StringVar(value="")
    val_cb = ttk.Combobox(win, values=[], textvariable=pos_label_var, state="readonly", width=40)
    tk.Label(win, text="Positive label:").grid(row=1, column=0, padx=8, pady=6, sticky="w")
    val_cb.grid(row=1, column=1, padx=8, pady=6)

    info = tk.Label(win, text="", foreground="gray")
    info.grid(row=2, columnspan=2, padx=8, pady=6, sticky="w")

    done = {"ok": False}

    def update_values(*_):
        tcol = target_var.get()
        if tcol in df.columns:
            uniq = pd.Series(df[tcol].dropna().unique()).tolist()
            uniq = [x.item() if hasattr(x, "item") else x for x in uniq]
            val_cb["values"] = uniq
            if 1 in uniq:
                pos_label_var.set("1")
            elif "1" in uniq:
                pos_label_var.set("1")
            elif len(uniq) > 0:
                pos_label_var.set(str(uniq[0]))
            info.config(text=f"Unique values: {len(uniq)} (needs binary)")
        else:
            val_cb["values"] = []
            pos_label_var.set("")
            info.config(text="")

    def on_ok():
        tcol = target_var.get()
        if tcol not in df.columns:
            messagebox.showerror(APP_TITLE, "Please select a valid target column.", parent=win)
            return
        uniq = pd.Series(df[tcol].dropna().unique()).tolist()
        if len(uniq) != 2:
            messagebox.showerror(APP_TITLE, f"Target must be binary. Selected has {len(uniq)} unique values.", parent=win)
            return
        pos_val = pos_label_var.get()
        if pos_val == "":
            messagebox.showerror(APP_TITLE, "Please select a positive label.", parent=win)
            return
        try:
            s = df[tcol].dropna()
            example = s.iloc[0]
            if isinstance(example, (int, np.integer)):
                pos = int(pos_val)
            elif isinstance(example, (float, np.floating)):
                pos = float(pos_val)
            else:
                pos = pos_val
        except Exception:
            pos = pos_val
        done["ok"] = True
        done["target"] = tcol
        done["pos"] = pos
        win.destroy()

    def on_cancel():
        win.destroy()

    target_cb.bind("<<ComboboxSelected>>", update_values)
    update_values()

    btn = tk.Frame(win)
    btn.grid(row=3, columnspan=2, pady=8)
    tk.Button(btn, text="OK", width=12, command=on_ok).pack(side="left", padx=6)
    tk.Button(btn, text="Cancel", width=12, command=on_cancel).pack(side="left", padx=6)

    win.grab_set()
    win.wait_window()
    if done["ok"]:
        return done["target"], done["pos"]
    return None, None


def ask_training_options(root: tk.Tk, suggest_large_sample: Optional[int], imblearn_available: bool) -> Optional[Dict]:
    win = tk.Toplevel(root)
    win.title("Training options")
    win.resizable(False, False)

    tk.Label(win, text="Models:").grid(row=0, column=0, sticky="nw", padx=8, pady=6)
    models_frame = tk.Frame(win); models_frame.grid(row=0, column=1, sticky="w", padx=8, pady=6)

    var_lr = tk.BooleanVar(value=True)
    var_rf = tk.BooleanVar(value=True)
    var_gb = tk.BooleanVar(value=True)
    var_knn = tk.BooleanVar(value=False)
    var_dt = tk.BooleanVar(value=False)
    var_svc = tk.BooleanVar(value=False)

    tk.Checkbutton(models_frame, text="LogisticRegression", variable=var_lr).grid(row=0, column=0, sticky="w")
    tk.Checkbutton(models_frame, text="RandomForest", variable=var_rf).grid(row=1, column=0, sticky="w")
    tk.Checkbutton(models_frame, text="GradientBoosting", variable=var_gb).grid(row=2, column=0, sticky="w")
    tk.Checkbutton(models_frame, text="KNN", variable=var_knn).grid(row=0, column=1, sticky="w")
    tk.Checkbutton(models_frame, text="DecisionTree", variable=var_dt).grid(row=1, column=1, sticky="w")
    tk.Checkbutton(models_frame, text="SVC (slow)", variable=var_svc).grid(row=2, column=1, sticky="w")

    def select_all():
        var_lr.set(True); var_rf.set(True); var_gb.set(True)
        var_knn.set(True); var_dt.set(True); var_svc.set(False)
    tk.Button(models_frame, text="Select All (no SVC)", command=select_all).grid(row=3, column=0, pady=4, sticky="w")

    tk.Label(win, text="Sampling:").grid(row=1, column=0, sticky="w", padx=8, pady=6)
    samp_var = tk.StringVar(value="class_weight")
    for i, (txt, val) in enumerate([("class_weight (default)", "class_weight"),
                                    ("SMOTE", "smote"),
                                    ("RandomUnder", "under"),
                                    ("None", "none")]):
        tk.Radiobutton(win, text=txt, variable=samp_var, value=val).grid(row=1, column=1+i, sticky="w", padx=4, pady=6)

    scale_var = tk.BooleanVar(value=True)
    tk.Checkbutton(win, text="Use StandardScaler (numeric)", variable=scale_var).grid(row=2, column=1, sticky="w", padx=8, pady=6)

    tk.Label(win, text="Metric focus:").grid(row=2, column=2, sticky="e")
    metric_var = tk.StringVar(value="pr_auc")
    metric_cb = ttk.Combobox(win, values=["pr_auc", "roc_auc", "f1"], textvariable=metric_var, state="readonly", width=12)
    metric_cb.grid(row=2, column=3, sticky="w", padx=8)

    tk.Label(win, text="Test size:").grid(row=3, column=0, sticky="e", padx=8)
    test_size_var = tk.DoubleVar(value=DEFAULT_TEST_SIZE)
    tk.Entry(win, textvariable=test_size_var, width=8).grid(row=3, column=1, sticky="w")

    tk.Label(win, text="Random state:").grid(row=3, column=2, sticky="e")
    rs_var = tk.IntVar(value=DEFAULT_RANDOM_STATE)
    tk.Entry(win, textvariable=rs_var, width=8).grid(row=3, column=3, sticky="w")

    tk.Label(win, text="Training data size:").grid(row=4, column=0, sticky="e", padx=8)
    size_mode = tk.StringVar(value="full")
    rb_full = tk.Radiobutton(win, text="Use full dataset", variable=size_mode, value="full")
    rb_sample = tk.Radiobutton(win, text="Use sampled subset (rows):", variable=size_mode, value="sample")
    rb_full.grid(row=4, column=1, sticky="w")
    rb_sample.grid(row=4, column=2, sticky="w")
    sample_rows_var = tk.IntVar(value=suggest_large_sample or 200_000)
    tk.Entry(win, textvariable=sample_rows_var, width=12).grid(row=4, column=3, sticky="w")

    warn = tk.Label(win, text="", fg="gray")
    warn.grid(row=5, column=0, columnspan=4, sticky="w", padx=8)
    if not IMBLEARN_AVAILABLE:
        warn.config(text="Note: imbalanced-learn not installed. SMOTE/Under won't work.")

    done = {"ok": False, "config": None}

    def on_ok():
        chosen_models = []
        if var_lr.get(): chosen_models.append("LogisticRegression")
        if var_rf.get(): chosen_models.append("RandomForest")
        if var_gb.get(): chosen_models.append("GradientBoosting")
        if var_knn.get(): chosen_models.append("KNN")
        if var_dt.get(): chosen_models.append("DecisionTree")
        if var_svc.get(): chosen_models.append("SVC")

        if not chosen_models:
            messagebox.showerror(APP_TITLE, "Please select at least one model.", parent=win); return

        if samp_var.get() == "smote" and not IMBLEARN_AVAILABLE:
            res = messagebox.askyesno(
                APP_TITLE,
                "SMOTE requires 'imbalanced-learn'.\n\nSwitch to class_weight now?\n(Yes = Switch, No = Back to options)",
                parent=win
            )
            if res:
                samp_val = "class_weight"
            else:
                return
        else:
            samp_val = samp_var.get()

        cfg = {
            "models": chosen_models,
            "sampler": samp_val,
            "scale": bool(scale_var.get()),
            "metric": metric_var.get(),
            "test_size": float(test_size_var.get()),
            "random_state": int(rs_var.get()),
            "size_mode": size_mode.get(),
            "sample_rows": int(sample_rows_var.get())
        }
        done["ok"] = True
        done["config"] = cfg
        win.destroy()

    def on_cancel():
        win.destroy()

    btn = tk.Frame(win); btn.grid(row=6, column=0, columnspan=4, pady=10)
    tk.Button(btn, text="OK", width=12, command=on_ok).pack(side="left", padx=6)
    tk.Button(btn, text="Cancel", width=12, command=on_cancel).pack(side="left", padx=6)

    win.grab_set()
    win.wait_window()
    return done["config"] if done["ok"] else None


def popup_results(root: tk.Tk, metrics: Dict[str, Dict], sort_key: str, out_dir: str):
    items = list(metrics.items())
    items.sort(key=lambda kv: (kv[1].get(sort_key) is None, -(kv[1].get(sort_key) or 0)))

    lines = [f"Results sorted by {sort_key} (desc):"]
    for name, m in items:
        lines.append(f"- {name}: "
                     f"Acc={m.get('accuracy'):.4f}, P={m.get('precision'):.4f}, R={m.get('recall'):.4f}, "
                     f"F1={m.get('f1'):.4f}, PR-AUC={(m.get('pr_auc') or 0):.4f}, ROC-AUC={(m.get('roc_auc') or 0):.4f}")
    lines.append("\nOpen results folder?")
    res = messagebox.askyesno(APP_TITLE, "\n".join(lines), parent=root)
    if res:
        open_folder(out_dir)


# ---------------- main ----------------
def main():
    root = tk.Tk()
    root.withdraw()
    root.update()

    # 1) dataset
    csv_path = ask_dataset_path(root)
    if not csv_path:
        return

    # 2) quick sample scan
    try:
        log("Reading CSV sample…")
        sample_df = quick_read_csv_head(csv_path, nrows=ANALYSIS_SAMPLE_CAP)
    except Exception as e:
        messagebox.showerror(APP_TITLE, f"Failed to read CSV sample:\n{e}", parent=root)
        return

    sum_info = brief_summary(sample_df)
    candidates = detect_target_candidates(sample_df)

    # 3) target
    target, pos_label = ask_target_and_pos(root, sample_df, candidates)
    if target is None:
        return

    # 4) summary & recs
    try:
        recommendations = compute_recommendations(sample_df, target, pos_label)
    except Exception:
        recommendations = []
    popup_summary(root, sum_info, recommendations)

    # 5) training options
    large_suggest = None
    try:
        full_head = quick_read_csv_head(csv_path, nrows=1_000_001)
        if len(full_head) >= 1_000_000:
            large_suggest = 200_000
    except Exception:
        pass

    opts = ask_training_options(root, large_suggest, IMBLEARN_AVAILABLE)
    if not opts:
        return

    # 6) new session folder
    sess = f"session_{timestamp()}"
    out_dir = os.path.join(BASE_DIR, "results", sess)
    ensure_dir(out_dir)
    brief_dir = os.path.join(out_dir, "brief"); ensure_dir(brief_dir)
    plots_dir = os.path.join(out_dir, "plots"); ensure_dir(plots_dir)

    try:
        log_lines = [
            f"Dataset path: {csv_path}",
            f"Target: {target} | Positive label: {pos_label}",
            f"Python: {platform.python_version()}",
            f"Platform: {platform.platform()}",
            f"Pandas: {pd.__version__} | Numpy: {np.__version__} | Scikit-learn: {sklearn.__version__}"
        ]
        if IMBLEARN_AVAILABLE:
            import imblearn
            log_lines.append(f"Imbalanced-learn: {imblearn.__version__}")
        save_text(os.path.join(out_dir, "session_log.txt"), "\n".join(log_lines))
    except Exception:
        pass

    # 7) save brief (sample)
    try:
        y = sample_df[target]
        vc = y.value_counts(dropna=False).to_dict()
        sum_info2 = dict(sum_info); sum_info2["target_counts_sample"] = vc
        save_json(os.path.join(brief_dir, "brief.json"), {"summary": sum_info2, "recommendations": recommendations})
        lines = [
            f"Summary (sample): rows={sum_info2['n_rows']}, cols={sum_info2['n_cols']}",
            f"Target counts (sample): {vc}",
            "Recommendations:",
            *[f"- {r}" for r in recommendations]
        ]
        save_text(os.path.join(brief_dir, "summary.txt"), "\n".join(lines))
    except Exception:
        pass

    # 8) heavy work with console logs
    metrics, probas, preds, timings = {}, {}, {}, {}
    used_sampler = opts["sampler"]

    log("Reading full CSV…")
    try:
        full_df = read_full_csv(csv_path)
    except Exception as e:
        messagebox.showerror(APP_TITLE, f"Failed to read full CSV:\n{e}", parent=root)
        return

    log("Dropping rows with missing target…")
    full_df = full_df[~full_df[target].isna()].copy()

    # Large CSV guard
    nrows_full = len(full_df)
    if opts["size_mode"] == "full" and nrows_full >= LARGE_ROWS_THRESHOLD:
        cont = messagebox.askyesno(
            APP_TITLE,
            f"Dataset seems large (rows≈{nrows_full}). Proceed with full data?\n\n"
            f"Yes = Proceed (may be slow)\nNo = Switch to sampled subset",
            parent=root
        )
        if not cont:
            opts["size_mode"] = "sample"

    if opts["size_mode"] == "sample":
        log(f"Taking stratified sample to ~{opts['sample_rows']} rows…")
        sample_rows = min(int(opts["sample_rows"]), nrows_full)
        full_df = full_df.groupby(full_df[target]).apply(lambda g: g.sample(
            n=max(1, int(len(g) * (sample_rows / nrows_full))), random_state=opts["random_state"]
        )).reset_index(drop=True)

    log("Splitting train/test…")
    X = full_df.drop(columns=[target])
    y = (full_df[target] == pos_label).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=opts["test_size"], random_state=opts["random_state"], stratify=y
    )

    log("Fitting preprocessor on train…")
    pre = build_preprocessor(
        X_train,
        numeric_strategy="median",
        categorical_strategy="most_frequent",
        scale_numeric=opts["scale"],
    )
    pre.fit(X_train, y_train)

    log("Transforming train/test…")
    X_train_t = pre.transform(X_train)
    X_test_t = pre.transform(X_test)

    log(f"Resampling on train (mode={opts['sampler']})…")
    try:
        X_train_res, y_train_res, used_sampler = resample_if_needed(X_train_t, y_train.values, opts["sampler"])
        if opts["sampler"] == "smote" and not IMBLEARN_AVAILABLE:
            save_text(os.path.join(out_dir, "warning.txt"),
                      "SMOTE selected but 'imbalanced-learn' missing. Switching to class_weight.")
            used_sampler = "class_weight"
    except Exception as e:
        messagebox.showwarning(APP_TITLE, f"Resampling failed, continuing without it.\n\n{e}", parent=root)
        X_train_res, y_train_res, used_sampler = X_train_t, y_train.values, "class_weight"

    model_names_selected = opts["models"]
    models = get_models(model_names_selected, class_weight_mode=(used_sampler == "class_weight"))

    model_items = list(models.items())
    total = len(model_items)
    for i, (mname, model) in enumerate(model_items, start=1):
        friendly = {
            "logreg": "LogisticRegression",
            "rf": "RandomForest",
            "gb": "GradientBoosting",
            "knn": "KNN",
            "dt": "DecisionTree",
            "svc": "SVC",
        }.get(mname, mname)

        log_step(i - 1, total, f"Training models — fitting {friendly}")
        t0 = time.time()
        try:
            model.fit(X_train_res, y_train_res)
        except Exception as e:
            metrics[mname] = {"error": str(e)}
            continue
        fit_sec = time.time() - t0

        log(f"Predicting with {friendly}…")
        t1 = time.time()
        try:
            y_pred = model.predict(X_test_t)
        except Exception as e:
            metrics[mname] = {"error": f"predict failed: {e}"}
            continue
        pred_sec = time.time() - t1

        try:
            y_proba = model.predict_proba(X_test_t)[:, 1]
        except Exception:
            y_proba = None

        preds[mname] = y_pred
        probas[mname] = y_proba
        timings[mname] = {"fit_seconds": round(fit_sec, 3), "predict_seconds": round(pred_sec, 3)}
        log(f"{friendly} done. fit={fit_sec:.2f}s, predict={pred_sec:.2f}s")

        # per-model folder
        model_dir = os.path.join(plots_dir, mname)
        ensure_dir(model_dir)

        # confusion
        try:
            plot_confusion(
                y_test.values, y_pred,
                os.path.join(model_dir, "cm_norm.png"),
                model_name=friendly, normalize=True
            )
        except Exception:
            pass

        # per-model PR/ROC
        try:
            plot_pr_single(y_test.values, y_proba, os.path.join(model_dir, "pr_curve.png"), friendly)
            plot_roc_single(y_test.values, y_proba, os.path.join(model_dir, "roc_curve.png"), friendly)
        except Exception:
            pass

    log_step(total, total, "Training models")

    # combined curves
    log("Generating combined PR/ROC plots…")
    try:
        plot_curves_combined(y_test.values, probas, plots_dir)
    except Exception:
        pass

    # score + save
    log("Scoring and saving artifacts…")
    metrics_out = evaluate_all(y_test.values, preds, probas)
    try:
        save_json(os.path.join(out_dir, "metrics.json"), metrics_out)
        save_json(os.path.join(out_dir, "timings.json"), timings)
        cfg_dump = {
            "dataset_path": csv_path,
            "target": target,
            "positive_label": pos_label,
            "options": opts,
            "sampler_used": used_sampler
        }
        save_json(os.path.join(out_dir, "config.json"), cfg_dump)

        lines = ["model\taccuracy\tprecision\trecall\tf1\tpr_auc\troc_auc"]
        for name, m in metrics_out.items():
            lines.append(
                f"{name}\t{m.get('accuracy'):.4f}\t{m.get('precision'):.4f}\t{m.get('recall'):.4f}\t"
                f"{m.get('f1'):.4f}\t{(m.get('pr_auc') or 0):.4f}\t{(m.get('roc_auc') or 0):.4f}"
            )
        save_text(os.path.join(out_dir, "summary.tsv"), "\n".join(lines))
    except Exception:
        pass

    log(f"All done. Results folder: {out_dir}")

    # results popup
    sort_key = opts["metric"]
    popup_results(root, metrics_out, sort_key, out_dir)

    # rerun?
    res = messagebox.askyesno(APP_TITLE, "Run another configuration?\n\nYes = New session (same app)\nNo = Exit", parent=root)
    if res:
        main()
    else:
        return


if __name__ == "__main__":
    main()
