# Plan: CNN-MCL → BI-LSTM → FWA (BI-ATT) → Random Forest on NSL-KDD

Reproduce the hybrid model from Hashmi, Barukab & Hamza Osman (PLOS ONE 19(5), 2024, e0302294), *"A hybrid feature weighted attention based deep learning approach for an intrusion detection system using the random forest algorithm"*. Paper text in `paper_extracted.txt`; equations and shapes are cited by line number below.

## Corrected architecture (per paper Algorithm 3, line 482)

```
raw NSL-KDD row  (41 features + label)
   │
   ▼ preprocessing   (label-encode categoricals, drop 2 constant cols, min-max scale, one-hot → 120-d)
   │
Input  N × 120                                                       (line 394)
   │
   ▼ CNN-MCL block   (Algorithm 1 + Eqs 1–4, lines 263–394)
   │    F(L) = Mean( W(L)(Cx,Cy) )         Eq 1 — per-filter mean
   │    sigmoid on the mean                Eq 3
   │    Adam updates filter weights        Eq 4
   ▼
MCL out  N × 10 × 120                     → Conv1 → Pool → Conv2 → Flatten → v  (N × 56)
   │                                                       weighted feature vector Wc
   ▼
BI-LSTM   (§I, Eq 5, lines 412–421)       non-linear mapping UcFt
   │                                       forward + backward hidden states
   ▼
FWA / BI-ATT   (Algorithm 2, lines 395–405)   — feature-weighted attention
   │    FB  = tanh(Wc·Ft + Uc·Ft-1 + bc)     forward
   │    FBL = tanh(Wc·Ft + Uc·Ft-1 + bc)     backward
   │    FA, FAT from Eqs 1–2 of the attention section
   │    Fe  = concat(FBL, FAT)              mapped features
   ▼
Random Forest classifier   (§K, line 484; Algorithm 3 step 4)
   │    C <- RandomForest(Fe, X)
   ▼
prediction: 2-class (binary) or 5-class (Normal/DoS/Probe/R2L/U2R)
```

Paper's exact pseudocode, Algorithm 3 (line 482):
```
Wc <- CNN-MCL(F)
Fe <- BI-ATT(Wc, F)
C  <- RandomForest(Fe, X)
```

So the DL stack (CNN-MCL + BI-LSTM + BI-ATT) is a **feature extractor**; the classifier is a scikit-learn RandomForest fit on the extracted features. Training is therefore **two-phase**, not end-to-end.

## Two-phase training

1. **Phase 1 — train the DL feature extractor.** The paper doesn't specify a loss for this phase explicitly, but the standard approach (and the only one that makes the MCL weights and BI-LSTM/BI-ATT parameters learn anything meaningful) is to attach a temporary softmax head during DL training with cross-entropy on the target labels, using Adam (paper §F says MCL weights are updated by Adam). Once trained, the softmax head is discarded.
2. **Phase 2 — extract `Fe` for every row, fit RandomForest.** Features are frozen; RF is trained with sklearn defaults first, then tuned.

I'll flag this clearly in code comments — it's the only pragmatic reading of Algorithm 3, but it is an interpretation. If the user later finds the authors' reference code doing something different (e.g., contrastive loss, autoencoder pretraining), swapping in that objective only touches `train.py`.

## What exists today

- `NSL-KDD/KDDTrain+.txt`, `KDDTest+.txt`, 20% / Test-21 variants, `.arff` copies.
- `kdd-classification.ipynb` — classical-ML EDA + binary SVM, useful for column list and label mapping.
- `preprocessing.ipynb` — empty.
- `requirements.txt` — numpy/pandas/sklearn/plotting only. No PyTorch installed.
- `paper_extracted.txt` — source of truth for equations.

## Deliverables

1. **`src/nids_dl/`** — Python package:
   - `data.py` — NSL-KDD loader + preprocessor → tensors of shape `(N, 120)`.
   - `layers/mcl.py` — `CNNMCL` (Eqs 1–4 + the two conv layers + pool + flatten, producing the 56-d weighted vector `Wc`).
   - `layers/bilstm.py` — thin wrapper around `nn.LSTM(bidirectional=True)` producing `UcFt`.
   - `layers/fwa.py` — `FWA` / BI-ATT layer implementing Algorithm 2 (lines 395–405): `FB`, `FBL`, `FA`, `FAT`, `Fe = concat(FBL, FAT)`.
   - `extractor.py` — `FeatureExtractor` composing MCL + BI-LSTM + FWA + a temporary softmax head for phase-1 training.
   - `train_extractor.py` — phase-1 training loop (Adam, CE loss, early stopping).
   - `rf_head.py` — phase-2: pass dataset through trained extractor with softmax head detached, fit `sklearn.ensemble.RandomForestClassifier`, evaluate.
   - `eval.py` — precision, recall, F1, FPR, detection rate, confusion matrix (paper's Table 5 metrics, line 743).
2. **`preprocessing.ipynb`** — fills the empty notebook: EDA + preprocessing, saves `data/processed/{train,test}.pt`.
3. **`notebooks/train_mcl_fwa_rf.ipynb`** — thin driver: phase-1 train, phase-2 RF fit, report metrics for binary + 5-class.
4. **`tests/`** — pytest:
   - `test_mcl.py` — shape flow `N×120 → N×10×120 → ... → N×56`; Eq 1 numeric check on a tiny input.
   - `test_fwa.py` — attention weights sum to 1; Algorithm 2 concat produces expected shape.
   - `test_extractor.py` — forward + backward pass on a 32-row batch.
   - `test_rf_head.py` — pipeline end-to-end on 1000 rows, asserts > random accuracy.
5. **`requirements-dl.txt`** — `torch`, `torchmetrics`, `pytest`. Leaves existing `requirements.txt` untouched.

## Key implementation decisions

- **120-feature input.** 41 raw cols − 3 categorical + one-hot(`protocol_type`=3, `service`≈70, `flag`=11) = 122, minus the two near-constant columns `num_outbound_cmds` and `is_host_login` = 120. Documented in `data.py`.
- **Label sets.** Binary: `normal` vs `attack`. 5-class: Normal / DoS / Probe / R2L / U2R via standard NSL-KDD family map. Same extractor, two separate RF heads.
- **Train/test split.** Train on `KDDTrain+.txt`, test on `KDDTest+.txt`, 10% of train held out for phase-1 validation.
- **MCL layer.** `nn.Conv1d(in_channels=1, out_channels=10, kernel_size=1)` → sigmoid → per-filter mean along Cx/Cy to realise Eq 1/3. Adam on its parameters as paper §F specifies. Followed by Conv2d stack producing the 56-d vector.
- **BI-LSTM.** `nn.LSTM(input_size=120, hidden_size=H, bidirectional=True)` — `H` small (e.g. 64) since input is tabular. Output shape `(N, 2H)` fed to FWA.
- **FWA/BI-ATT.** Implements Algorithm 2 literally: two `tanh(Wc·Ft + Uc·Ft-1 + bc)` terms (forward and backward) plus the attention weights `FA`/`FAT` from the block's Eqs 1–2 (softmax over `u^T · v`). Final output `Fe = concat(FBL, FAT)`.
- **RF head.** `RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=0)` as starting point; can grid-search if metrics miss.
- **Class imbalance.** Use `class_weight='balanced'` in RF and weighted CE in phase-1. Paper doesn't use SMOTE.
- **Reproducibility.** Seed numpy + torch + sklearn; cache processed tensors.
- **No GPU assumed.** Code guards on `torch.cuda.is_available()`; CPU is fine for 125k rows.

## Execution order

1. `requirements-dl.txt`; `pip install -r requirements-dl.txt`.
2. `data.py` + fill `preprocessing.ipynb`; verify `(N, 120)` tensors saved.
3. `layers/mcl.py` + test.
4. `layers/bilstm.py` + `layers/fwa.py` + tests.
5. `extractor.py` + `train_extractor.py`; 1-epoch smoke run to confirm loss drops.
6. Full phase-1 train on `KDDTrain+`; save weights.
7. `rf_head.py` — extract `Fe`, fit RF, evaluate on `KDDTest+` for binary + 5-class.
8. Driver notebook reproduces paper's Table 5 metrics (binary target ≈99%).

## Open items flagged to the user

- **Phase-1 loss.** Paper is silent on the objective used to train the DL extractor. I'm assuming a temporary softmax + cross-entropy head because §F explicitly says Adam updates the MCL weights, which requires a differentiable loss on labels. If you want a different phase-1 objective (autoencoder reconstruction, contrastive), say so before I code `extractor.py`.
- **Algorithm 2 Eqs 1–2 ambiguity.** Lines 436–446 of the paper describe the attention weight computation in abstract terms (`exp(u^T · ...) / sum`). I'll implement the standard Bahdanau-style attention that matches the description and note the exact formula in a docstring.

## Out of scope for v1

- UNSW-NB15 dataset.
- Tuning RF hyperparameters beyond defaults + `class_weight='balanced'`.
- Containerisation / CI.
