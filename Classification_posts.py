import os
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight

import xgboost as xgb
from sentence_transformers import SentenceTransformer


# ===============================
# CONFIG
# ===============================
CSV_PATH = "/content/drive/MyDrive/Validation_set_10k_final1.csv"
TEXT_COL = "text"

LAYER1_COL = "lens1"   # organic / inorganic
LAYER2_COL = "lens"    # individual / brand / restaurant / influencer

ST_MODEL = "all-mpnet-base-v2"
BATCH_SIZE = 64
NORMALIZE = True

SEED = 42
TEST_SIZE = 0.2

ARTIFACT_PATH = "two_layer_model3.joblib"


# ===============================
# LAYER 1 CONFIG (organic / inorganic)
# ===============================
#Do not touch anymore
LAYER1_CONFIG = {
    "n_estimators": 1400,
    "max_depth": 5,             # ↓ very important
    "learning_rate": 0.03,
    "subsample": 0.9,
    "colsample_bytree": 0.9,

    "min_child_weight": 6,      # ↑ smooth splits
    "gamma": 0.3,               # ↓ allow splits
    "reg_lambda": 4.0,
    "reg_alpha": 0.3,

    "class_weights": {
        "organic": 1.0,
        "inorganic": 0.55,      # 🔑 bias toward inorganic recall
    },

    # 🔑 MOST IMPORTANT CHANGE
    "thresholds": {
        "organic": 0.85,        # require high confidence to discard Layer-2
    },
}


# ===============================
# LAYER 2 CONFIG (brand / restaurant / influencer)
# ===============================
LAYER2_CONFIG = {
    "n_estimators": 1000,
    "max_depth": 5,
    "learning_rate": 0.04,
    "subsample": 0.85,
    "colsample_bytree": 0.85,

    "min_child_weight": 10,
    "gamma": 1.5,
    "reg_lambda": 5.0,
    "reg_alpha": 2.0,

    "class_weights": {
        "restaurant": 1.0,
        "influencer": 1.25,
        "brand": 1.4,
    },

    "thresholds": None,
}

# ===============================
# INFERENCE WRAPPER (saved with artifact)
# ===============================
class TwoLayerClassifier:
    """
    Self-contained inference wrapper.
    Loads SentenceTransformer, normalizes embeddings, runs Layer1 → Layer2.
    """

    def __init__(self, artifact_path: str):
        art = joblib.load(artifact_path)
        self.clf_l1 = art["layer1_model"]
        self.clf_l2 = art["layer2_model"]
        self.le_l1 = art["le_layer1"]
        self.le_l2 = art["le_layer2"]
        self.st_model_name = art["sentence_model"]
        self.normalize = art["normalize"]
        self.batch_size = art["batch_size"]
        self.layer1_thresholds = art.get("layer1_config", {}).get("thresholds")
        self.layer2_thresholds = art.get("layer2_config", {}).get("thresholds")
        self._st = None  # lazy-loaded

    def _get_st(self):
        if self._st is None:
            device = "cpu"
            try:
                import torch
                if torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = "mps"
            except Exception:
                pass
            self._st = SentenceTransformer(self.st_model_name, device=device)
        return self._st

    def _embed(self, texts):
        st = self._get_st()
        X = st.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        if self.normalize:
            X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        return X

    def _predict_with_thresholds(self, clf, X, thresholds, le):
        proba = clf.predict_proba(X)
        preds = np.argmax(proba, axis=1)
        if thresholds is not None:
            for cls_name, thr in thresholds.items():
                cls_idx = np.where(le.classes_ == cls_name)[0]
                if len(cls_idx) == 0:
                    continue
                cls_idx = cls_idx[0]
                for i in range(len(preds)):
                    if preds[i] == cls_idx and proba[i, cls_idx] < thr:
                        remaining = proba[i].copy()
                        remaining[cls_idx] = -1.0
                        preds[i] = np.argmax(remaining)
        return preds, proba

    def predict(self, texts):
        """
        Predict final labels for a list of strings.
        Returns: list of labels (individual / brand / restaurant / influencer)
        """
        if isinstance(texts, str):
            texts = [texts]

        X = self._embed(texts)

        # Layer 1: organic vs inorganic
        l1_preds, _ = self._predict_with_thresholds(
            self.clf_l1, X, self.layer1_thresholds, self.le_l1
        )
        l1_labels = self.le_l1.inverse_transform(l1_preds)

        # Default: everything predicted organic → "individual"
        final = np.array(["individual"] * len(texts), dtype=object)

        # Layer 2: only for predicted inorganic
        inorganic_mask = l1_labels == "inorganic"
        if inorganic_mask.any():
            X_inorg = X[inorganic_mask]
            l2_preds, _ = self._predict_with_thresholds(
                self.clf_l2, X_inorg, self.layer2_thresholds, self.le_l2
            )
            final[inorganic_mask] = self.le_l2.inverse_transform(l2_preds)

        return final.tolist()

    def predict_detail(self, texts):
        """
        Returns dict with layer1 label, final label, and probabilities.
        """
        if isinstance(texts, str):
            texts = [texts]

        X = self._embed(texts)

        l1_preds, l1_proba = self._predict_with_thresholds(
            self.clf_l1, X, self.layer1_thresholds, self.le_l1
        )
        l1_labels = self.le_l1.inverse_transform(l1_preds)

        results = []
        inorganic_mask = l1_labels == "inorganic"
        l2_labels = np.array(["N/A"] * len(texts), dtype=object)
        l2_proba_all = np.zeros((len(texts), len(self.le_l2.classes_)))

        if inorganic_mask.any():
            X_inorg = X[inorganic_mask]
            l2_preds, l2_proba = self._predict_with_thresholds(
                self.clf_l2, X_inorg, self.layer2_thresholds, self.le_l2
            )
            l2_labels[inorganic_mask] = self.le_l2.inverse_transform(l2_preds)
            l2_proba_all[inorganic_mask] = l2_proba

        for i in range(len(texts)):
            final_label = "individual" if l1_labels[i] == "organic" else l2_labels[i]
            results.append({
                "text": texts[i][:80] + "..." if len(texts[i]) > 80 else texts[i],
                "layer1": l1_labels[i],
                "layer1_proba": dict(zip(self.le_l1.classes_, l1_proba[i].round(4))),
                "layer2": l2_labels[i] if l1_labels[i] == "inorganic" else "N/A",
                "layer2_proba": dict(zip(self.le_l2.classes_, l2_proba_all[i].round(4))) if l1_labels[i] == "inorganic" else {},
                "final_label": final_label,
            })

        return results


# ===============================
# HELPERS
# ===============================
def build_xgb_from_config(num_classes, seed, config):
    return xgb.XGBClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        learning_rate=config["learning_rate"],
        subsample=config["subsample"],
        colsample_bytree=config["colsample_bytree"],
        reg_lambda=config["reg_lambda"],
        reg_alpha=config["reg_alpha"],
        min_child_weight=config["min_child_weight"],
        gamma=config["gamma"],
        tree_method="hist",
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        random_state=seed,
    )


def make_sample_weights(y_encoded, config_weights, label_encoder):
    if config_weights == "balanced":
        return compute_sample_weight("balanced", y_encoded)
    if isinstance(config_weights, dict):
        weight_map = np.ones(len(label_encoder.classes_))
        for cls_name, w in config_weights.items():
            cls_idx = np.where(label_encoder.classes_ == cls_name)[0]
            if len(cls_idx) == 0:
                raise ValueError(
                    f"Class '{cls_name}' not in encoder. "
                    f"Available: {list(label_encoder.classes_)}"
                )
            weight_map[cls_idx[0]] = w
        return np.array([weight_map[label] for label in y_encoded])
    raise ValueError(f"Unsupported class_weights: {config_weights}")


def predict_with_thresholds(clf, X, thresholds, label_encoder):
    proba = clf.predict_proba(X)
    preds = np.argmax(proba, axis=1)
    if thresholds is not None:
        for cls_name, thr in thresholds.items():
            cls_idx = np.where(label_encoder.classes_ == cls_name)[0]
            if len(cls_idx) == 0:
                continue
            cls_idx = cls_idx[0]
            for i in range(len(preds)):
                if preds[i] == cls_idx and proba[i, cls_idx] < thr:
                    remaining = proba[i].copy()
                    remaining[cls_idx] = -1.0
                    preds[i] = np.argmax(remaining)
    return preds, proba


# ===============================
# DATA LOADING
# ===============================
def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df = df[[TEXT_COL, LAYER1_COL, LAYER2_COL]].dropna()
    df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
    df[LAYER1_COL] = df[LAYER1_COL].str.lower()
    df[LAYER2_COL] = df[LAYER2_COL].str.lower()
    return df


df = load_data(CSV_PATH)
print("Rows:", len(df))
print("Layer1 distribution:\n", df[LAYER1_COL].value_counts())
print("Layer2 distribution:\n", df[LAYER2_COL].value_counts())


def pick_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


device = pick_device()
print("Device:", device)


# ===============================
# EMBEDDINGS (no KPCA)
# ===============================
st = SentenceTransformer(ST_MODEL, device=device)

X = st.encode(
    df[TEXT_COL].tolist(),
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
)

if NORMALIZE:
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

print("Embeddings shape:", X.shape)


# ===============================
# TRAIN / TEST SPLIT
# ===============================
idx = np.arange(len(df))

X_train, X_test, idx_train, idx_test = train_test_split(
    X,
    idx,
    test_size=TEST_SIZE,
    stratify=df[LAYER1_COL] + "::" + df[LAYER2_COL],
    random_state=SEED,
)


# =====================================================
# ================= LAYER 1 ============================
# =====================================================
print("\n================ LAYER 1 =================")
print(f"Config: {LAYER1_CONFIG}\n")

le_l1 = LabelEncoder()
y1 = le_l1.fit_transform(df[LAYER1_COL])

y1_train = y1[idx_train]
y1_test = y1[idx_test]

w1 = make_sample_weights(y1_train, LAYER1_CONFIG["class_weights"], le_l1)

clf_l1 = build_xgb_from_config(len(le_l1.classes_), SEED, LAYER1_CONFIG)
clf_l1.fit(X_train, y1_train, sample_weight=w1)

y1_pred_train, _ = predict_with_thresholds(
    clf_l1, X_train, LAYER1_CONFIG["thresholds"], le_l1
)
y1_pred_test, _ = predict_with_thresholds(
    clf_l1, X_test, LAYER1_CONFIG["thresholds"], le_l1
)

print("\nLAYER 1 – TRAIN METRICS")
print(classification_report(
    y1_train, y1_pred_train, target_names=le_l1.classes_, digits=4
))

print("\nLAYER 1 – TEST METRICS")
print(classification_report(
    y1_test, y1_pred_test, target_names=le_l1.classes_, digits=4
))

print("\n================ LAYER 2 =================")
print(f"Config: {LAYER2_CONFIG}\n")

# ---- TRAIN: ground-truth inorganic & not individual
train_mask_gt = (
    (df.iloc[idx_train][LAYER1_COL] == "inorganic") &
    (df.iloc[idx_train][LAYER2_COL] != "individual")
)

X2_train_raw = X_train[train_mask_gt.values]
y2_train_raw = df.iloc[idx_train][train_mask_gt][LAYER2_COL]

# ---- TEST: predicted inorganic & not individual (this stays prediction-based — matches inference)
test_mask = (
    (le_l1.inverse_transform(y1_pred_test) == "inorganic") &
    (df.iloc[idx_test][LAYER2_COL] != "individual")
)

X2_test_raw = X_test[test_mask.values]
y2_test_raw = df.iloc[idx_test][test_mask][LAYER2_COL]

le_l2 = LabelEncoder()
y2_train = le_l2.fit_transform(y2_train_raw)
y2_test = le_l2.transform(y2_test_raw)

print(f"L2 train samples: {len(y2_train)} (ground-truth inorganic)")
print(f"L2 test samples:  {len(y2_test)} (L1-predicted inorganic)")

w2 = make_sample_weights(y2_train, LAYER2_CONFIG["class_weights"], le_l2)

clf_l2 = build_xgb_from_config(len(le_l2.classes_), SEED, LAYER2_CONFIG)
clf_l2.fit(X2_train_raw, y2_train, sample_weight=w2)

y2_pred_train, _ = predict_with_thresholds(
    clf_l2, X2_train_raw, LAYER2_CONFIG["thresholds"], le_l2
)
y2_pred_test, _ = predict_with_thresholds(
    clf_l2, X2_test_raw, LAYER2_CONFIG["thresholds"], le_l2
)

print("\nLAYER 2 – TRAIN METRICS")
print(classification_report(
    y2_train, y2_pred_train, target_names=le_l2.classes_, digits=4
))

print("\nLAYER 2 – TEST METRICS")
print(classification_report(
    y2_test, y2_pred_test, target_names=le_l2.classes_, digits=4
))



# =====================================================
# ================= FINAL METRICS ======================
# =====================================================
print("\n================ FINAL METRICS =================")

final_preds = np.array(["individual"] * len(idx_test), dtype=object)
final_preds[test_mask.values] = le_l2.inverse_transform(y2_pred_test)

final_true = df.iloc[idx_test][LAYER2_COL].values

print(classification_report(
    final_true,
    final_preds,
    labels=["individual", "brand", "restaurant", "influencer"],
    digits=4,
))


# ===============================
# SAVE (fully reproducible artifact)
# ===============================
artifact = {
    "layer1_model": clf_l1,
    "layer2_model": clf_l2,
    "le_layer1": le_l1,
    "le_layer2": le_l2,
    "sentence_model": ST_MODEL,
    "normalize": NORMALIZE,
    "batch_size": BATCH_SIZE,
    "layer1_config": LAYER1_CONFIG,
    "layer2_config": LAYER2_CONFIG,
}

joblib.dump(artifact, ARTIFACT_PATH)
print(f"\nModel saved to: {ARTIFACT_PATH}")
print(f"File size: {os.path.getsize(ARTIFACT_PATH) / 1024 / 1024:.2f} MB")

print("\n================ SMOKE TEST =================")
classifier = TwoLayerClassifier(ARTIFACT_PATH)

sample_texts = [
    "Just had the best pizza at this new place downtown! 🍕",
    "Use code SAVE20 for 20% off all menu items this weekend only!",
    "Morning coffee and journaling before work ☕",
]

results = classifier.predict_detail(sample_texts)
for r in results:
    print(f"\n  Text:   {r['text']}")
    print(f"  L1:     {r['layer1']} {r['layer1_proba']}")
    print(f"  L2:     {r['layer2']} {r['layer2_proba']}")
    print(f"  Final:  {r['final_label']}")