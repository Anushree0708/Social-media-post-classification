TWO-LAYER HIERARCHICAL TEXT CLASSIFIER
======================================

Overview
========
This project implements a production-ready, two-layer hierarchical text classification system.

It classifies text into:

- individual
- brand
- restaurant
- influencer

The system first determines whether content is organic or inorganic (commercial).
If inorganic, it then classifies into brand / restaurant / influencer.


Architecture
============
The model uses a hybrid architecture:

1) Sentence Embeddings
   - Model: all-mpnet-base-v2
   - Library: SentenceTransformers
   - Outputs normalized dense vector embeddings

2) Layer 1 Classifier (XGBoost)
   - Task: organic vs inorganic
   - Uses class weighting
   - Uses probability thresholding to control routing

3) Layer 2 Classifier (XGBoost)
   - Task: brand / restaurant / influencer
   - Runs only on inorganic predictions
   - Strong regularization + class weights


Pipeline Flow
=============

TEXT
  ↓
SentenceTransformer Embedding
  ↓
Layer 1: organic vs inorganic
  ↓
If organic → final = "individual"
If inorganic → Layer 2
  ↓
brand / restaurant / influencer


Why Hierarchical?
=================
Instead of directly training a 4-class classifier, this design:

- Improves precision
- Improves recall for commercial content
- Allows threshold-based business control
- Mirrors real-world routing logic
- Makes tuning easier per layer


Key Features
============

✔ Transformer + Gradient Boosting hybrid
✔ Hierarchical decision cascade
✔ Confidence threshold gating
✔ Class-weighted training
✔ Stratified train/test split
✔ Fully reproducible artifact (.joblib)
✔ Lazy-loaded inference wrapper
✔ CPU / CUDA / MPS device detection


Configuration
=============

Layer 1 Config
- n_estimators: 1400
- max_depth: 5
- learning_rate: 0.03
- threshold on "organic": 0.85

This ensures high confidence is required before discarding Layer 2.

Layer 2 Config
- n_estimators: 1000
- max_depth: 5
- learning_rate: 0.04
- Strong regularization
- Class weights for imbalance


Dataset Requirements
====================

CSV file must contain:

- text column
- lens1 column (organic / inorganic)
- lens column (individual / brand / restaurant / influencer)

Example:

text,lens1,lens
"Morning coffee ☕",organic,individual
"Use code SAVE20",inorganic,brand


Training
========

1) Update CSV_PATH in config
2) Run the script
3) The script will:

   - Load data
   - Generate embeddings
   - Train Layer 1
   - Train Layer 2
   - Print evaluation metrics
   - Save model artifact

Saved artifact:

two_layer_model3.joblib


Inference Usage
===============

After training, load and use:

    classifier = TwoLayerClassifier("two_layer_model3.joblib")
    predictions = classifier.predict(["Your text here"])

For detailed output:

    classifier.predict_detail(["Your text here"])

Returns:

- Layer 1 prediction + probabilities
- Layer 2 prediction + probabilities
- Final label


Use Cases
=========

- Social media content classification
- Sponsored content detection
- Influencer marketing analysis
- Brand monitoring
- Restaurant promotion detection
- Creator ecosystem segmentation
- Ad detection pipelines
- Content moderation routing


Why XGBoost Instead of Fine-Tuning?
===================================

This design:

- Works well on ~10k samples
- Requires no GPU for training
- Is fast to retrain
- Is stable on small/medium datasets
- Keeps inference efficient


Dependencies
============

Python 3.8+

Required packages:

- numpy
- pandas
- scikit-learn
- xgboost
- sentence-transformers
- joblib
- torch

Install:

    pip install numpy pandas scikit-learn xgboost sentence-transformers joblib torch


Production Considerations
=========================

- Thresholds allow business control over recall vs precision
- Layer 2 only runs when necessary (efficient routing)
- Fully serializable and portable
- Device auto-detection built in
- Embedding normalization improves tree performance


Future Improvements
===================

- Probability calibration
- Confidence-based abstention
- Larger transformer model
- End-to-end fine-tuning
- REST API wrapper
- Batch inference pipeline


License
=======

Add your license here (MIT recommended).


Author
======

Your Name
