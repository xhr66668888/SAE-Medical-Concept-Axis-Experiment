"""
Model Validation: 5-Way Comparison
====================================
Methods:
  B1  : xcorr features (6-dim) + Logistic Regression   [signal baseline]
  B2  : xcorr features (6-dim) + RBF-SVM               [signal baseline]
  B3  : Single shared Encoder   + shallow MLP probe          [no dual-pool]
  B4  : Dual-Pool, no CrossPoolAttention + shallow MLP probe [no cross-pool]
  Ours: Full Dual-Pool          + shallow MLP probe          [proposed]

Experiment 1 — Trial-level (stratified 80/20 split)
  Encoder-decoder trained ONLY on training trials → zero test leakage.
  Classifier trained ONLY on frozen training embeddings.

Experiment 2 — Subject-level LOOCV (18 folds: 4 patient + 14 healthy)
  All 5 methods retrained from scratch in every fold.
  Normalisation fitted on training subjects only.
  Subject-level AND trial-level metrics reported per method.

Imbalance (4 patients vs 14 healthy) handled everywhere:
  BCEWithLogitsLoss(pos_weight = n_H / n_P)
  WeightedRandomSampler for balanced mini-batches
  Metrics: AUROC, balanced accuracy, weighted F1, sensitivity, specificity
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, roc_curve
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from dual_pool_model import (
    DualPoolModel, PoolEncoder, PoolDecoder, ProjectionHead,
    PoolDataset, fit_channelwise_normalizer, apply_channelwise_normalizer,
    train_dual_pool,
)
import random

JAW_CH = slice(0, 9)
TT_CH  = slice(9, 18)

METHODS = [
    "B1_xcorr_LR",
    "B2_xcorr_SVM",
    "B3_single_enc",
    "B4_dual_no_xattn",
    "Ours_full_dual",
]
METHOD_LABELS = {
    "B1_xcorr_LR":      "B1  xcorr features + LR",
    "B2_xcorr_SVM":     "B2  xcorr features + SVM",
    "B3_single_enc":    "B3  Single Encoder",
    "B4_dual_no_xattn": "B4  Dual-Pool (no cross-attn)",
    "Ours_full_dual":   "Ours  Full Dual-Pool",
}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — xcorr feature extraction (signal-level baselines)
# ══════════════════════════════════════════════════════════════════════════════

def extract_xcorr_features(X_norm: np.ndarray, fs: int = 250) -> np.ndarray:
    """
    6 cross-correlation features per trial  → (N, 6)

    Feature layout:
      [0] global peak lag (ms)          — overall JAW–TT timing
      [1] global peak xcorr value       — overall coupling strength
      [2] early-segment peak lag (ms)   — 0   … T/3
      [3] mid-segment   peak lag (ms)   — T/3 … 2T/3
      [4] late-segment  peak lag (ms)   — 2T/3 … T
      [5] xcorr value at lag = 0        — instantaneous synchrony

    Sign convention: negative lag → JAW leads TT.
    """
    jaw = X_norm[:, 1,  :]   # JAW-y  (ch 1)
    tt  = X_norm[:, 10, :]   # TT-y   (ch 10)
    N, T = jaw.shape
    max_lag = int(0.4 * fs)   # ±400 ms
    lags_ms = np.arange(-max_lag, max_lag + 1) / fs * 1000.0
    feats = np.zeros((N, 6), dtype=np.float32)

    def _xcorr(a, b, ml):
        """Normalised xcorr between z-scored a and b, symmetric lag ±ml samples."""
        a = (a - a.mean()) / (a.std() + 1e-8)
        b = (b - b.mean()) / (b.std() + 1e-8)
        full   = np.correlate(a, b, mode="full")
        c      = len(b) - 1
        return full[c - ml : c + ml + 1] / len(b)

    seg = T // 3
    seg_slices = [slice(0, seg), slice(seg, 2 * seg), slice(2 * seg, T)]

    for n in range(N):
        xc = _xcorr(tt[n], jaw[n], max_lag)
        feats[n, 0] = lags_ms[np.argmax(xc)]
        feats[n, 1] = xc.max()
        feats[n, 5] = xc[max_lag]          # value at lag=0 (centre)

        for k, sl in enumerate(seg_slices):
            j_s = jaw[n, sl]; t_s = tt[n, sl]
            ml_s = min(max_lag, len(j_s) // 3)
            lags_s = np.arange(-ml_s, ml_s + 1) / fs * 1000.0
            xc_s = _xcorr(t_s, j_s, ml_s)
            feats[n, 2 + k] = lags_s[np.argmax(xc_s)]

    return feats


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Architecture ablation model classes
# ══════════════════════════════════════════════════════════════════════════════

class SingleEncoderModel(nn.Module):
    """
    Ablation B3: one shared encoder–decoder for ALL trials (no dual-pool).
    Trained jointly on patient + healthy; no group-specific representations.
    embed_dim-dimensional embeddings (half of dual-pool's 2×embed_dim).
    """
    def __init__(self, in_channels: int = 9, d_model: int = 128, nhead: int = 4,
                 num_enc_layers: int = 4, num_dec_layers: int = 4,
                 dim_ff: int = 256, dropout: float = 0.1,
                 seq_len: int = 400, embed_dim: int = 64):
        super().__init__()
        enc_kw = dict(in_channels=in_channels, d_model=d_model, nhead=nhead,
                      num_layers=num_enc_layers, dim_ff=dim_ff,
                      dropout=dropout, seq_len=seq_len)
        dec_kw = dict(out_channels=in_channels, d_model=d_model, nhead=nhead,
                      num_layers=num_dec_layers, dim_ff=dim_ff,
                      dropout=dropout, seq_len=seq_len)
        self.encoder = PoolEncoder(**enc_kw)
        self.decoder = PoolDecoder(**dec_kw)
        self.proj    = ProjectionHead(d_model, embed_dim)

    def forward(self, tt, jaw):
        enc, _ = self.encoder(tt, jaw)
        recon  = self.decoder(tt, enc)
        return recon, enc

    @torch.no_grad()
    def extract_embeddings(self, X_norm: np.ndarray,
                            device: str = "cpu", batch_size: int = 32) -> np.ndarray:
        self.eval()
        tt  = torch.tensor(X_norm[:, TT_CH,  :], dtype=torch.float32)
        jaw = torch.tensor(X_norm[:, JAW_CH, :], dtype=torch.float32)
        embs = []
        for i in range(0, len(tt), batch_size):
            enc, _ = self.encoder(tt[i:i+batch_size].to(device),
                                   jaw[i:i+batch_size].to(device))
            embs.append(self.proj(enc).cpu().numpy())
        return np.concatenate(embs, axis=0)   # (N, embed_dim)


class DualPoolNoXAttn(DualPoolModel):
    """
    Ablation B4: dual-pool encoders WITHOUT CrossPoolAttention.
    Both encoders are trained independently with no cross-pool information
    exchange. Isolates the contribution of cross-pool attention.
    """
    def forward(self, tt_p, jaw_p, tt_h, jaw_h):
        enc_p, _ = self.patient_encoder(tt_p, jaw_p)
        enc_h, _ = self.healthy_encoder(tt_h, jaw_h)
        # CrossPoolAttention intentionally skipped
        recon_p = self.patient_decoder(tt_p, enc_p)
        recon_h = self.healthy_decoder(tt_h, enc_h)
        return recon_p, recon_h, enc_p, enc_h


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Classification head + dataset
# ══════════════════════════════════════════════════════════════════════════════

class ClassificationHead(nn.Module):
    """
    2-layer MLP probe on top of frozen encoder embeddings.
    input_dim = embed_dim   for SingleEncoderModel  (e.g. 64)
              = 2*embed_dim for DualPool models      (e.g. 128)
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ClfDataset(Dataset):
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        self.emb    = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels,     dtype=torch.float32)

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx): return self.emb[idx], self.labels[idx]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Imbalance utilities + weighted metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_pos_weight(labels: np.ndarray) -> torch.Tensor:
    n_pos = labels.sum() + 1e-8
    n_neg = len(labels) - labels.sum()
    return torch.tensor([n_neg / n_pos], dtype=torch.float32)


def make_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    n_pos = labels.sum() + 1e-8
    n_neg = len(labels) - labels.sum() + 1e-8
    w = np.where(labels == 1, len(labels) / (2 * n_pos),
                              len(labels) / (2 * n_neg))
    return WeightedRandomSampler(torch.tensor(w, dtype=torch.float64),
                                  len(labels), replacement=True)


def compute_weighted_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                              threshold: float = 0.5) -> dict:
    """Comprehensive class-imbalance-aware metrics. Patient=1, Healthy=0."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    n_pos = int(y_true.sum())
    n_neg = int((1 - y_true).sum())

    if n_pos == 0 or n_neg == 0:          # degenerate fold
        return dict(auroc=float("nan"), balanced_acc=float("nan"),
                    weighted_f1=float("nan"), sensitivity=float("nan"),
                    specificity=float("nan"),
                    confusion=np.zeros((2, 2), int),
                    n_patient=n_pos, n_healthy=n_neg)

    cm       = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens     = tp / (tp + fn + 1e-8)
    spec     = tn / (tn + fp + 1e-8)
    bal_acc  = (sens + spec) / 2.0

    try:
        auroc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auroc = float("nan")

    w   = np.where(y_true == 1, len(y_true) / (2 * n_pos),
                                len(y_true) / (2 * n_neg))
    wf1 = f1_score(y_true, y_pred, average="weighted",
                   sample_weight=w, zero_division=0)

    return dict(auroc=auroc, balanced_acc=bal_acc, weighted_f1=wf1,
                sensitivity=float(sens), specificity=float(spec),
                confusion=cm, n_patient=n_pos, n_healthy=n_neg)


def print_metrics(m: dict, title: str = ""):
    sep = "=" * 54
    print(f"\n{sep}")
    if title: print(f"  {title}")
    print(sep)
    cm = m["confusion"]
    print(f"  AUROC            : {m['auroc']:.4f}")
    print(f"  Balanced Accuracy: {m['balanced_acc']:.4f}")
    print(f"  Weighted F1      : {m['weighted_f1']:.4f}")
    print(f"  Sensitivity (TPR): {m['sensitivity']:.4f}  ← patient recall")
    print(f"  Specificity (TNR): {m['specificity']:.4f}  ← healthy recall")
    print(f"  Confusion Matrix:")
    print(f"              Pred H  Pred P")
    print(f"    True H     {cm[0,0]:>5}   {cm[0,1]:>5}")
    print(f"    True P     {cm[1,0]:>5}   {cm[1,1]:>5}")
    print(f"  (N patient={m['n_patient']}, N healthy={m['n_healthy']})")


def train_classifier(head: ClassificationHead,
                     embeddings: np.ndarray, labels: np.ndarray,
                     pos_weight: torch.Tensor,
                     epochs: int = 30, lr: float = 1e-3,
                     batch_size: int = 32, device: str = "cpu") -> ClassificationHead:
    sampler = make_weighted_sampler(labels)
    loader  = DataLoader(ClfDataset(embeddings, labels),
                         batch_size=batch_size, sampler=sampler)
    head    = head.to(device)
    optim   = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    crit    = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    head.train()
    for epoch in range(epochs):
        total, n = 0.0, 0
        for emb_b, lbl_b in loader:
            emb_b = emb_b.to(device); lbl_b = lbl_b.to(device)
            optim.zero_grad()
            loss = crit(head(emb_b), lbl_b)
            loss.backward(); optim.step()
            total += loss.item() * len(lbl_b); n += len(lbl_b)
        if (epoch + 1) % 10 == 0:
            print(f"      clf epoch {epoch+1:>3}/{epochs}  loss={total/n:.4f}")
    return head


@torch.no_grad()
def predict_proba(head: ClassificationHead, embeddings: np.ndarray,
                  device: str = "cpu", batch_size: int = 256) -> np.ndarray:
    head.eval()
    probs = []
    for i in range(0, len(embeddings), batch_size):
        emb_b = torch.tensor(embeddings[i:i+batch_size], dtype=torch.float32, device=device)
        probs.append(torch.sigmoid(head(emb_b)).cpu().numpy())
    return np.concatenate(probs)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Per-method runners (shared by Exp 1 and Exp 2)
# ══════════════════════════════════════════════════════════════════════════════

def _run_xcorr_baseline(X_tr: np.ndarray, y_tr: np.ndarray,
                         X_te: np.ndarray, y_te: np.ndarray,
                         clf_type: str = "LR") -> np.ndarray:
    """
    Extract 6-dim xcorr features, fit sklearn classifier on training data,
    return predicted P(patient) for test trials.
    """
    feat_tr = extract_xcorr_features(X_tr)
    feat_te = extract_xcorr_features(X_te)

    scaler   = StandardScaler()
    feat_tr  = scaler.fit_transform(feat_tr)
    feat_te  = scaler.transform(feat_te)

    if clf_type == "LR":
        clf = LogisticRegression(class_weight="balanced", max_iter=500,
                                  solver="lbfgs", C=1.0)
    else:   # SVM
        clf = SVC(kernel="rbf", class_weight="balanced",
                  probability=True, C=1.0, gamma="scale")

    clf.fit(feat_tr, y_tr.astype(int))
    return clf.predict_proba(feat_te)[:, 1].astype(np.float32)


def _train_single_encoder(model: SingleEncoderModel, X_all: np.ndarray,
                            device: str, lr: float, epochs: int,
                            batch_size: int = 16) -> SingleEncoderModel:
    """Train SingleEncoderModel (TT reconstruction, all trials jointly)."""
    loader = DataLoader(PoolDataset(X_all), batch_size=batch_size,
                        shuffle=True, drop_last=False)
    model  = model.to(device)
    optim  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    crit   = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        total, n = 0.0, 0
        for tt_b, jaw_b in loader:
            tt_b = tt_b.to(device); jaw_b = jaw_b.to(device)
            optim.zero_grad()
            recon, _ = model(tt_b, jaw_b)
            loss = crit(recon, tt_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total += loss.item() * len(tt_b); n += len(tt_b)
        if (epoch + 1) % 10 == 0:
            print(f"      enc epoch {epoch+1:>3}/{epochs}  loss={total/n:.6f}")
    return model


@torch.no_grad()
def _extract_dual_embeddings(model: DualPoolModel, X_norm: np.ndarray,
                               device: str, batch_size: int = 32) -> np.ndarray:
    """Concat [patient_proj ∥ healthy_proj] → (N, 2*embed_dim)."""
    ep = model.extract_embeddings(X_norm, "patient", device, batch_size)
    eh = model.extract_embeddings(X_norm, "healthy", device, batch_size)
    return np.concatenate([ep, eh], axis=1)


def run_all_methods(
    X_tr_p:     np.ndarray,   # patient training trials  (N_p, 18, T)
    X_tr_h:     np.ndarray,   # healthy training trials  (N_h, 18, T)
    X_te:       np.ndarray,   # test trials              (N_te, 18, T)
    y_te:       np.ndarray,   # test labels
    model_kwargs: dict,
    device:       str   = "cpu",
    seed:         int   = 42,
    epochs_enc:   int   = 60,
    epochs_clf:   int   = 30,
    methods:      list  = None,   # None → run all 5
) -> dict:
    """
    Run all requested methods on the given train/test split.

    Returns
    -------
    preds : {method_name: np.ndarray of shape (N_te,)}
        Raw P(patient) predictions for each test trial.
        Call compute_weighted_metrics(y_te, preds[m]) to get metrics.
    """
    if methods is None:
        methods = METHODS

    X_tr_all = np.concatenate([X_tr_p, X_tr_h], axis=0)
    y_tr_all = np.concatenate([np.ones(len(X_tr_p)),
                                np.zeros(len(X_tr_h))]).astype(np.float32)
    pos_w    = compute_pos_weight(y_tr_all)
    embed_dim = model_kwargs.get("embed_dim", 64)

    preds = {}

    # ── B1: xcorr + LR ───────────────────────────────────────────────────
    if "B1_xcorr_LR" in methods:
        print("    [B1] xcorr + Logistic Regression")
        preds["B1_xcorr_LR"] = _run_xcorr_baseline(
            X_tr_all, y_tr_all, X_te, y_te, "LR")

    # ── B2: xcorr + SVM ──────────────────────────────────────────────────
    if "B2_xcorr_SVM" in methods:
        print("    [B2] xcorr + SVM (RBF)")
        preds["B2_xcorr_SVM"] = _run_xcorr_baseline(
            X_tr_all, y_tr_all, X_te, y_te, "SVM")

    # ── B3: Single Encoder ────────────────────────────────────────────────
    if "B3_single_enc" in methods:
        print("    [B3] Single Encoder")
        m3 = SingleEncoderModel(**model_kwargs)
        m3 = _train_single_encoder(m3, X_tr_all, device,
                                    lr=1e-4, epochs=epochs_enc)
        emb3_tr = m3.extract_embeddings(X_tr_all, device)
        emb3_te = m3.extract_embeddings(X_te,     device)
        head3   = ClassificationHead(input_dim=embed_dim,
                                      hidden_dim=64, dropout=0.2)
        head3   = train_classifier(head3, emb3_tr, y_tr_all, pos_w,
                                    epochs=epochs_clf, device=device)
        preds["B3_single_enc"] = predict_proba(head3, emb3_te, device)

    # ── B4: Dual-Pool, no cross-attn ─────────────────────────────────────
    if "B4_dual_no_xattn" in methods:
        print("    [B4] Dual-Pool (no cross-attn)")
        m4 = DualPoolNoXAttn(**model_kwargs)
        p_loader = DataLoader(PoolDataset(X_tr_p), batch_size=16,
                              shuffle=True, drop_last=False)
        h_loader = DataLoader(PoolDataset(X_tr_h), batch_size=16,
                              shuffle=True, drop_last=False)
        m4, _ = train_dual_pool(m4, p_loader, h_loader,
                                 device=device, lr=1e-4, epochs=epochs_enc)
        emb4_tr = _extract_dual_embeddings(m4, X_tr_all, device)
        emb4_te = _extract_dual_embeddings(m4, X_te,     device)
        head4   = ClassificationHead(input_dim=2 * embed_dim,
                                      hidden_dim=64, dropout=0.2)
        head4   = train_classifier(head4, emb4_tr, y_tr_all, pos_w,
                                    epochs=epochs_clf, device=device)
        preds["B4_dual_no_xattn"] = predict_proba(head4, emb4_te, device)

    # ── Ours: Full Dual-Pool ──────────────────────────────────────────────
    if "Ours_full_dual" in methods:
        print("    [Ours] Full Dual-Pool")
        m5 = DualPoolModel(**model_kwargs)
        p_loader = DataLoader(PoolDataset(X_tr_p), batch_size=16,
                              shuffle=True, drop_last=False)
        h_loader = DataLoader(PoolDataset(X_tr_h), batch_size=16,
                              shuffle=True, drop_last=False)
        m5, _ = train_dual_pool(m5, p_loader, h_loader,
                                 device=device, lr=1e-4, epochs=epochs_enc)
        emb5_tr = _extract_dual_embeddings(m5, X_tr_all, device)
        emb5_te = _extract_dual_embeddings(m5, X_te,     device)
        head5   = ClassificationHead(input_dim=2 * embed_dim,
                                      hidden_dim=64, dropout=0.2)
        head5   = train_classifier(head5, emb5_tr, y_tr_all, pos_w,
                                    epochs=epochs_clf, device=device)
        preds["Ours_full_dual"] = predict_proba(head5, emb5_te, device)

    return preds


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Experiment 1: Trial-level classification
# ══════════════════════════════════════════════════════════════════════════════

def experiment1_trial_level(
    patient_X_raw: np.ndarray,
    healthy_X_raw: np.ndarray,
    model_kwargs:   dict,
    test_size:      float = 0.20,
    epochs_enc:     int   = 60,
    epochs_clf:     int   = 30,
    device:         str   = "cpu",
    seed:           int   = 42,
    save_dir:       str   = ".",
    methods:        list  = None,
) -> dict:
    """
    Trial-level 5-way comparison.

    Critically: the test split is NEVER seen during encoder-decoder training
    or classifier training → zero leakage. The remaining subject-overlap
    (same person's other trials may appear in train) is acceptable here;
    subject-level generalisation is assessed by Experiment 2.

    Returns
    -------
    results : {method_name: metrics_dict}
    """
    if methods is None:
        methods = METHODS

    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Trial-Level 5-Way Comparison")
    print("=" * 60)

    n_p   = len(patient_X_raw)
    n_h   = len(healthy_X_raw)
    X_all = np.concatenate([patient_X_raw, healthy_X_raw], axis=0)
    y_all = np.concatenate([np.ones(n_p), np.zeros(n_h)]).astype(np.float32)
    print(f"  Total: {len(y_all)} trials  "
          f"(P={n_p}, H={n_h}, imbalance ratio {n_h/n_p:.1f}:1)")

    # Stratified split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(sss.split(X_all, y_all))

    X_tr_raw = X_all[train_idx];  y_tr = y_all[train_idx]
    X_te_raw = X_all[test_idx];   y_te = y_all[test_idx]

    # Fit normalization on TRAIN only
    mu, sigma = fit_channelwise_normalizer(X_tr_raw)
    X_tr = apply_channelwise_normalizer(X_tr_raw, mu, sigma)
    X_te = apply_channelwise_normalizer(X_te_raw, mu, sigma)

    # Patient / healthy subsets of training data (for dual-pool encoders)
    X_tr_p = X_tr[y_tr == 1]
    X_tr_h = X_tr[y_tr == 0]

    for tag, y in [("Train", y_tr), ("Test", y_te)]:
        print(f"  {tag}: {len(y):>5} trials  "
              f"(P={int(y.sum()):>4}, H={int((1-y).sum()):>4})")

    preds   = run_all_methods(X_tr_p, X_tr_h, X_te, y_te,
                               model_kwargs, device, seed,
                               epochs_enc, epochs_clf, methods)
    results = {}
    for method, prob in preds.items():
        results[method] = compute_weighted_metrics(y_te, prob)
        print_metrics(results[method], f"Exp1 | {METHOD_LABELS[method]}")

    plot_comparison_table(results, title="Experiment 1: Trial-Level Results",
                          save_path=f"{save_dir}/exp1_comparison_table.png")
    _plot_roc_comparison(preds, y_te,
                          title="Experiment 1: ROC Curves (All Methods)",
                          save_path=f"{save_dir}/exp1_roc_comparison.png")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Experiment 2: Subject-level LOOCV
# ══════════════════════════════════════════════════════════════════════════════

def experiment2_subject_loocv(
    patient_subjects: dict,
    healthy_subjects: dict,
    model_kwargs:     dict,
    epochs_enc:       int  = 60,
    epochs_clf:       int  = 30,
    device:           str  = "cpu",
    seed:             int  = 42,
    save_dir:         str  = ".",
    methods:          list = None,
) -> dict:
    """
    Subject-level LOOCV, all 5 methods compared.

    Per fold: all methods share the same normalisation and train/test split,
    ensuring a fair comparison.  pos_weight is recomputed each fold.

    Returns
    -------
    {
      'trial'  : {method: metrics_dict},   # aggregate over all test trials
      'subject': {method: metrics_dict},   # subject-level (mean prob per subject)
    }
    """
    if methods is None:
        methods = METHODS

    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Subject-Level LOOCV  (18 folds)")
    print("=" * 60)

    all_subs = []
    for name, payload in patient_subjects.items():
        all_subs.append((name, payload[0], 1))
    for name, payload in healthy_subjects.items():
        all_subs.append((name, payload[0], 0))

    n_sub = len(all_subs)
    n_pat = sum(l == 1 for _, _, l in all_subs)
    n_hlt = sum(l == 0 for _, _, l in all_subs)
    print(f"  Subjects: {n_sub}  (P={n_pat}, H={n_hlt})")
    print(f"  Methods per fold: {[METHOD_LABELS[m] for m in methods]}\n")

    # Accumulators: trial-level and subject-level per method
    fold_trial   = {m: {"probs": [], "labels": []} for m in methods}
    fold_subject = {m: {"probs": [], "labels": []} for m in methods}

    for fold, (left_name, left_X_raw, left_label) in enumerate(all_subs):
        group = "patient" if left_label == 1 else "healthy"
        print(f"\n  Fold {fold+1:>2}/{n_sub} — [{left_name}] ({group})")

        # Build train arrays
        tr_p_list, tr_h_list = [], []
        for name, X_raw, label in all_subs:
            if name == left_name: continue
            (tr_p_list if label == 1 else tr_h_list).append(X_raw)

        tr_p_raw = np.concatenate(tr_p_list, axis=0)
        tr_h_raw = np.concatenate(tr_h_list, axis=0)

        # Normalise using training subjects only
        mu, sigma  = fit_channelwise_normalizer(
            np.concatenate([tr_p_raw, tr_h_raw], axis=0))
        tr_p_norm  = apply_channelwise_normalizer(tr_p_raw,  mu, sigma)
        tr_h_norm  = apply_channelwise_normalizer(tr_h_raw,  mu, sigma)
        left_norm  = apply_channelwise_normalizer(left_X_raw, mu, sigma)

        y_te_fold = np.array([left_label] * len(left_norm), dtype=np.float32)

        # Dummy y_te for run_all_methods (metrics computed later via accumulation)
        preds_fold = run_all_methods(
            tr_p_norm, tr_h_norm, left_norm, y_te_fold,
            model_kwargs, device, seed, epochs_enc, epochs_clf, methods,
        )

        for m, prob in preds_fold.items():
            fold_trial[m]["probs"].extend(prob.tolist())
            fold_trial[m]["labels"].extend([left_label] * len(prob))
            fold_subject[m]["probs"].append(float(prob.mean()))
            fold_subject[m]["labels"].append(left_label)
            pred = "patient" if prob.mean() >= 0.5 else "healthy"
            ok   = "✓" if (prob.mean() >= 0.5) == bool(left_label) else "✗"
            print(f"    {METHOD_LABELS[m]:<34} "
                  f"mean_prob={prob.mean():.3f}  pred={pred}  {ok}")

    # Compute aggregate metrics
    trial_results   = {}
    subject_results = {}
    for m in methods:
        trial_results[m]   = compute_weighted_metrics(
            np.array(fold_trial[m]["labels"]),
            np.array(fold_trial[m]["probs"]),
        )
        subject_results[m] = compute_weighted_metrics(
            np.array(fold_subject[m]["labels"]),
            np.array(fold_subject[m]["probs"]),
        )
        print_metrics(trial_results[m],
                      f"LOOCV Trial-Level   | {METHOD_LABELS[m]}")
        print_metrics(subject_results[m],
                      f"LOOCV Subject-Level | {METHOD_LABELS[m]}")

    # Plots
    plot_comparison_table(
        trial_results,
        title="Experiment 2 LOOCV: Trial-Level Results",
        save_path=f"{save_dir}/exp2_loocv_trial_table.png",
    )
    plot_comparison_table(
        subject_results,
        title="Experiment 2 LOOCV: Subject-Level Results (N=18)",
        save_path=f"{save_dir}/exp2_loocv_subject_table.png",
    )
    _plot_roc_comparison(
        {m: np.array(fold_trial[m]["probs"]) for m in methods},
        np.array(fold_trial[methods[0]]["labels"]),
        title="Experiment 2 LOOCV: Trial-Level ROC (All Methods)",
        save_path=f"{save_dir}/exp2_loocv_roc_comparison.png",
    )
    _plot_loocv_subject_bars(
        all_subs,
        {m: fold_subject[m]["probs"] for m in methods},
        subject_results,
        save_path=f"{save_dir}/exp2_loocv_subject_bars.png",
    )

    return {"trial": trial_results, "subject": subject_results}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Plotting
# ══════════════════════════════════════════════════════════════════════════════

def plot_comparison_table(results: dict, title: str = "",
                           save_path: str = "comparison_table.png"):
    """
    Render a clean matplotlib table with all methods × metrics.
    Rows: methods (best value per column bolded).
    Columns: AUROC, Balanced Acc, Weighted F1, Sensitivity, Specificity.
    """
    metric_keys  = ["auroc", "balanced_acc", "weighted_f1",
                    "sensitivity", "specificity"]
    metric_labels = ["AUROC", "Bal. Acc", "W-F1",
                     "Sensitivity\n(TPR)", "Specificity\n(TNR)"]

    row_labels = [METHOD_LABELS[m] for m in results]
    data = []
    for m in results:
        row = [results[m].get(k, float("nan")) for k in metric_keys]
        data.append(row)
    data = np.array(data, dtype=float)

    fig, ax = plt.subplots(figsize=(12, 0.55 * len(results) + 1.8))
    ax.axis("off")

    tbl = ax.table(
        cellText  = [[f"{v:.4f}" if not np.isnan(v) else "—"
                      for v in row] for row in data],
        rowLabels = row_labels,
        colLabels = metric_labels,
        cellLoc   = "center",
        loc       = "center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.6)

    # Highlight best (highest) per column
    for col_i, col_key in enumerate(metric_keys):
        best_row = int(np.nanargmax(data[:, col_i]))
        tbl[best_row + 1, col_i].set_facecolor("#d4edda")    # light green
        tbl[best_row + 1, col_i].set_text_props(fontweight="bold")

    # Header style
    for col_i in range(len(metric_keys)):
        tbl[0, col_i].set_facecolor("#2c3e50")
        tbl[0, col_i].set_text_props(color="white", fontweight="bold")

    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def _plot_roc_comparison(preds: dict, y_true: np.ndarray,
                          title: str = "ROC Curves",
                          save_path: str = "roc_comparison.png"):
    """Overlay ROC curves for all methods on one axes."""
    colors = ["#e74c3c", "#e67e22", "#3498db", "#9b59b6", "#27ae60"]
    fig, ax = plt.subplots(figsize=(6, 6))

    for (method, prob), color in zip(preds.items(), colors):
        try:
            fpr, tpr, _ = roc_curve(y_true, prob, pos_label=1)
            auroc = roc_auc_score(y_true, prob)
            ax.plot(fpr, tpr, lw=2, color=color,
                    label=f"{METHOD_LABELS[method]}  (AUC={auroc:.3f})")
        except ValueError:
            pass

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Chance")
    ax.set_xlabel("False Positive Rate  (1 − Specificity)", fontsize=10)
    ax.set_ylabel("True Positive Rate  (Sensitivity)", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Saved: {save_path}")


def _plot_loocv_subject_bars(all_subs, subj_probs_per_method,
                               subject_results, save_path):
    """
    One row of bar charts per method showing per-subject P(patient).
    Subjects sorted: patients first, then healthy.
    """
    n_methods = len(subj_probs_per_method)
    order     = sorted(range(len(all_subs)),
                        key=lambda i: (all_subs[i][2], all_subs[i][0]))
    names_s   = [all_subs[i][0] for i in order]
    labels_s  = [all_subs[i][2] for i in order]
    n         = len(names_s)

    fig, axes = plt.subplots(n_methods, 1,
                              figsize=(14, 2.8 * n_methods),
                              sharex=True)
    if n_methods == 1: axes = [axes]

    for ax, (method, probs_raw) in zip(axes, subj_probs_per_method.items()):
        probs_s = [probs_raw[i] for i in order]
        colors  = ["C0" if l == 1 else "C1" for l in labels_s]
        ax.bar(range(n), probs_s, color=colors, edgecolor="k", linewidth=0.5)
        ax.axhline(0.5, color="red", lw=1.2, ls="--")

        for i, (p, l) in enumerate(zip(probs_s, labels_s)):
            ok = "★" if (p >= 0.5) == bool(l) else "✗"
            ax.text(i, p + 0.03, ok, ha="center", fontsize=8,
                    color="k" if (p >= 0.5) == bool(l) else "red")

        sm = subject_results[method]
        ax.set_ylim([0, 1.15])
        ax.set_ylabel("P(patient)", fontsize=8)
        ax.set_title(
            f"{METHOD_LABELS[method]}   "
            f"AUROC={sm['auroc']:.3f}  "
            f"BalAcc={sm['balanced_acc']:.3f}  "
            f"Sens={sm['sensitivity']:.3f}  "
            f"Spec={sm['specificity']:.3f}",
            fontsize=8, loc="left",
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xticks(range(n))
    axes[-1].set_xticklabels(
        [f"{nm}\n({'P' if lb==1 else 'H'})"
         for nm, lb in zip(names_s, labels_s)],
        rotation=45, ha="right", fontsize=7,
    )
    fig.suptitle("Experiment 2 LOOCV: Subject-Level P(patient) — All Methods\n"
                 "Blue=patient  Orange=healthy  ★=correct  ✗=error",
                 fontsize=10, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — Summary print
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(exp1_results: dict, exp2_results: dict):
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    header = f"{'Method':<38} {'AUROC':>7} {'BalAcc':>8} {'Sens':>7} {'Spec':>7}"
    sep    = "-" * 70

    if exp1_results:
        print("\nExperiment 1 — Trial-Level Test Set")
        print(header); print(sep)
        for m in METHODS:
            if m not in exp1_results: continue
            r = exp1_results[m]
            print(f"  {METHOD_LABELS[m]:<36} "
                  f"{r['auroc']:>7.4f} {r['balanced_acc']:>8.4f} "
                  f"{r['sensitivity']:>7.4f} {r['specificity']:>7.4f}")

    if exp2_results:
        for level in ["trial", "subject"]:
            res = exp2_results.get(level, {})
            if not res: continue
            label = "Trial-Level" if level == "trial" else "Subject-Level (N=18)"
            print(f"\nExperiment 2 — LOOCV {label}")
            print(header); print(sep)
            for m in METHODS:
                if m not in res: continue
                r = res[m]
                print(f"  {METHOD_LABELS[m]:<36} "
                      f"{r['auroc']:>7.4f} {r['balanced_acc']:>8.4f} "
                      f"{r['sensitivity']:>7.4f} {r['specificity']:>7.4f}")


def set_all_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    SAVE_DIR = "/home/liu0193/Data/Ziming/EMA/new_multi_processed"
    SEQ_LEN  = 400
    DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE}\n")

    # ── Load raw per-subject data ──────────────────────────────────────────
    from read_mat import load_group_data
    patient_subjects = load_group_data(SAVE_DIR, "patient")
    healthy_subjects = load_group_data(SAVE_DIR, "healthy")
    print(f"Patient subjects : {len(patient_subjects)}")
    print(f"Healthy subjects : {len(healthy_subjects)}")

    # ── Raw arrays for Experiment 1 (normalization will be fit on TRAIN only) ──
    set_all_seeds(42)
    p_raw = np.concatenate([v[0] for v in patient_subjects.values()], axis=0)
    h_raw = np.concatenate([v[0] for v in healthy_subjects.values()], axis=0)

    model_kwargs = dict(
        in_channels=9, d_model=128, nhead=4,
        num_enc_layers=4, num_dec_layers=4,
        dim_ff=256, dropout=0.1,
        seq_len=SEQ_LEN, embed_dim=64,
    )

    # ── Experiment 1 ──────────────────────────────────────────────────────
    exp1_results = experiment1_trial_level(
        p_raw, h_raw,
        model_kwargs = model_kwargs,
        test_size    = 0.20,
        epochs_enc   = 60,
        epochs_clf   = 30,
        device       = DEVICE,
        seed         = 42,
        save_dir     = SAVE_DIR,
        methods      = METHODS,
    )

    # ── Experiment 2 ──────────────────────────────────────────────────────
    # 18 folds × 5 methods → 3 deep-learning training runs per fold.
    # Budget: ~1.5–3 h on a single GPU.
    exp2_results = experiment2_subject_loocv(
        patient_subjects = patient_subjects,
        healthy_subjects = healthy_subjects,
        model_kwargs     = model_kwargs,
        epochs_enc       = 60,
        epochs_clf       = 30,
        device           = DEVICE,
        seed             = 42,
        save_dir         = SAVE_DIR,
        methods          = METHODS,
    )

    print_summary(exp1_results, exp2_results)
