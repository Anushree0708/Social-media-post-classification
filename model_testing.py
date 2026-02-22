import os
import numpy as np
import pandas as pd
import joblib

from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report


# ===============================
# CONFIG
# ===============================
MODEL_PATH = "two_layer_model_mpnet.joblib"
UNSEEN_DATA_PATH = "Social_data_testing_mpnet_other.csv"
TEXT_COL = "text"

# OPTIONAL: set to None if no ground-truth column exists
GT_COL = None

OUTPUT_PATH = "Organic_Inorganic_predictions_mpnet_individual_fill.csv"


# ===============================
# LOAD MODEL ARTIFACT
# ===============================
artifact = joblib.load(MODEL_PATH)

clf_l1 = artifact["layer1_model"]
clf_l2 = artifact["layer2_model"]
le_l1 = artifact["le_layer1"]
le_l2 = artifact["le_layer2"]
ST_MODEL = artifact["sentence_model"]
NORMALIZE = artifact["normalize"]
BATCH_SIZE = artifact["batch_size"]
LAYER1_THRESHOLDS = artifact.get("layer1_config", {}).get("thresholds")
LAYER2_THRESHOLDS = artifact.get("layer2_config", {}).get("thresholds")

print(f"Loaded artifact: {MODEL_PATH}")
print(f"  Sentence model:    {ST_MODEL}")
print(f"  Normalize:         {NORMALIZE}")
print(f"  Layer1 classes:    {list(le_l1.classes_)}")
print(f"  Layer2 classes:    {list(le_l2.classes_)}")
print(f"  Layer1 thresholds: {LAYER1_THRESHOLDS}")
print(f"  Layer2 thresholds: {LAYER2_THRESHOLDS}")


# ===============================
# DEVICE
# ===============================
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
print(f"Using device: {device}")


# ===============================
# THRESHOLD-AWARE PREDICTION
# ===============================
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
# LOAD UNSEEN DATA
# ===============================
df = pd.read_csv(UNSEEN_DATA_PATH)

if TEXT_COL not in df.columns:
    raise ValueError(f"'{TEXT_COL}' column not found in unseen data")

df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
print(f"\nUnseen rows: {len(df)}")


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

print(f"Embeddings shape: {X.shape}")


# ===============================
# LAYER 1 PREDICTION
# ===============================
y1_pred, y1_proba = predict_with_thresholds(clf_l1, X, LAYER1_THRESHOLDS, le_l1)
layer1_preds = le_l1.inverse_transform(y1_pred)

print(f"\nLayer1 prediction distribution:")
unique, counts = np.unique(layer1_preds, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  {u}: {c} ({c / len(layer1_preds) * 100:.1f}%)")


# ===============================
# LAYER 2 PREDICTION (ONLY INORGANIC)
# ===============================
final_preds = np.array(["individual"] * len(df), dtype=object)

inorg_mask = layer1_preds == "inorganic"

if inorg_mask.any():
    X_inorg = X[inorg_mask]
    y2_pred, y2_proba = predict_with_thresholds(clf_l2, X_inorg, LAYER2_THRESHOLDS, le_l2)
    final_preds[inorg_mask] = le_l2.inverse_transform(y2_pred)

print(f"\nFinal prediction distribution:")
unique, counts = np.unique(final_preds, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  {u}: {c} ({c / len(final_preds) * 100:.1f}%)")


# ===============================
# OUTPUT RESULTS
# ===============================
df_out = df.copy()
df_out["layer1_pred"] = layer1_preds
df_out["final_pred"] = final_preds

# Add per-class probabilities for debugging
for i, cls in enumerate(le_l1.classes_):
    df_out[f"l1_prob_{cls}"] = y1_proba[:, i].round(4)

print("\nSample predictions:")
print(df_out[[TEXT_COL, "layer1_pred", "final_pred"]].head(10).to_string(index=False))


# ===============================
# OPTIONAL: METRICS (IF GT EXISTS)
# ===============================
if GT_COL and GT_COL in df.columns:
    gt = df[GT_COL].astype(str).str.lower().str.strip()

    print("\n================ FINAL METRICS ON UNSEEN DATA =================")
    print(classification_report(
        gt,
        final_preds,
        labels=["individual", "brand", "restaurant", "influencer"],
        digits=4,
    ))
else:
    if GT_COL:
        print(f"\nWarning: GT column '{GT_COL}' not found — skipping metrics.")


# ===============================
# SAVE OUTPUT
# ===============================
df_out.to_csv(OUTPUT_PATH, index=False)
print(f"\nPredictions saved to: {OUTPUT_PATH}")
print(f"Columns: {list(df_out.columns)}")




# import os
# import numpy as np
# import pandas as pd
# import joblib

# from sentence_transformers import SentenceTransformer
# from sklearn.metrics import classification_report


# # ===============================
# # CONFIG
# # ===============================
# MODEL_PATH = "two_layer_model_mpnet.joblib"
# UNSEEN_DATA_PATH = "Validation_set_10k_update_4.csv"  
# TEXT_COL = "text"

# # OPTIONAL (only if GT labels exist)
# GT_COL = "lens_pf"  # set to None if not available

# BATCH_SIZE = 64
# NORMALIZE = True


# # ===============================
# # LOAD MODEL ARTIFACT
# # ===============================
# artifact = joblib.load(MODEL_PATH)

# clf_l1 = artifact["layer1_model"]
# clf_l2 = artifact["layer2_model"]
# kpca1 = artifact["kpca1"]
# kpca2 = artifact["kpca2"]
# le_l1 = artifact["le_layer1"]
# le_l2 = artifact["le_layer2"]
# ST_MODEL = artifact["sentence_model"]


# # ===============================
# # DEVICE
# # ===============================
# def pick_device():
#     try:
#         import torch
#         if torch.cuda.is_available():
#             return "cuda"
#         if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
#             return "mps"
#     except Exception:
#         pass
#     return "cpu"


# device = pick_device()
# print("Using device:", device)


# # ===============================
# # LOAD UNSEEN DATA
# # ===============================
# df = pd.read_csv(UNSEEN_DATA_PATH)

# if TEXT_COL not in df.columns:
#     raise ValueError(f"'{TEXT_COL}' column not found in unseen data")

# df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()

# print("Unseen rows:", len(df))


# # ===============================
# # EMBEDDINGS
# # ===============================
# st = SentenceTransformer(ST_MODEL, device=device)

# X = st.encode(
#     df[TEXT_COL].tolist(),
#     batch_size=BATCH_SIZE,
#     show_progress_bar=True,
#     convert_to_numpy=True
# )

# if NORMALIZE:
#     X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


# # ===============================
# # LAYER 1 PREDICTION
# # ===============================
# X1 = kpca1.transform(X)
# y1_prob = clf_l1.predict_proba(X1)
# y1_pred = np.argmax(y1_prob, axis=1)

# layer1_preds = le_l1.inverse_transform(y1_pred)


# # ===============================
# # LAYER 2 PREDICTION (ONLY INORGANIC)
# # ===============================
# final_preds = np.array(["individual"] * len(df), dtype=object)

# inorg_mask = layer1_preds == "inorganic"

# if inorg_mask.any():
#     X2 = kpca2.transform(X[inorg_mask])
#     y2_prob = clf_l2.predict_proba(X2)
#     y2_pred = np.argmax(y2_prob, axis=1)

#     final_preds[inorg_mask] = le_l2.inverse_transform(y2_pred)


# # ===============================
# # OUTPUT RESULTS
# # ===============================
# df_out = df.copy()
# df_out["layer1_pred"] = layer1_preds
# df_out["final_pred"] = final_preds

# print("\nSample predictions:")
# print(df_out[[TEXT_COL, "layer1_pred", "final_pred"]].head())


# # ===============================
# # OPTIONAL: METRICS (IF GT EXISTS)
# # ===============================
# if GT_COL and GT_COL in df.columns:
#     print("\nFINAL METRICS ON UNSEEN DATA")
#     print(classification_report(
#         df[GT_COL].str.lower(),
#         final_preds,
#         labels=["individual", "brand", "restaurant", "influencer"],
#         digits=4
#     ))


# # ===============================
# # SAVE OUTPUT
# # ===============================
# OUTPUT_PATH = "Organic_Inorganic_mpnet.csv"
# df_out.to_csv(OUTPUT_PATH, index=False)

# print(f"\nPredictions saved to: {OUTPUT_PATH}")