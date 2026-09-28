"""
Dual-Pool Transformer for EMA Coordination Analysis
=====================================================
Architecture:
  - Two separate encoders (patient pool / healthy pool)
  - TT (pos+vel+acc, 9ch) as main sequence input
  - JAW (pos+vel+acc, 9ch) as condition via causal cross-attention
  - Learnable pool token per encoder → cross-pool attention between summaries
  - Symmetric decoders reconstruct TT (training signal)
  - After training: encoder → projection head → PCA

Channel layout in (N, 18, T) tensors:
  0-8  : JAW position(3) + velocity(3) + acceleration(3)
  9-17 : TT  position(3) + velocity(3) + acceleration(3)
"""

import math
import re
from itertools import cycle

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import linregress

import torch
import os

print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.cuda.device_count():", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. This job is not running on a GPU-enabled environment.")

DEVICE = "cuda"
print("Using GPU:", torch.cuda.get_device_name(0))


# ── Channel slices ─────────────────────────────────────────────────────────────
JAW_CH = slice(0, 9)
TT_CH  = slice(9, 18)


# ── Normalisation (reused from pilot) ──────────────────────────────────────────
def fit_channelwise_normalizer(X_all):
    """X_all: (N, 18, T)  →  mean/std: (1, 18, 1)"""
    mean = X_all.mean(axis=(0, 2), keepdims=True)
    std  = X_all.std(axis=(0, 2),  keepdims=True)
    std[std < 1e-8] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def apply_channelwise_normalizer(X, mean, std):
    return ((X - mean) / std).astype(np.float32)


# ── Dataset ────────────────────────────────────────────────────────────────────
class PoolDataset(Dataset):
    """Each sample: (TT tensor 9×T, JAW tensor 9×T) for one trial."""

    def __init__(self, X):
        # X: (N, 18, T)
        self.tt  = torch.tensor(X[:, TT_CH,  :], dtype=torch.float32)
        self.jaw = torch.tensor(X[:, JAW_CH, :], dtype=torch.float32)

    def __len__(self):
        return len(self.tt)

    def __getitem__(self, idx):
        return self.tt[idx], self.jaw[idx]


# ── Model components ───────────────────────────────────────────────────────────

class InputProjection(nn.Module):
    """(B, C, T) → (B, T, d_model) via pointwise 1-D conv."""

    def __init__(self, in_channels: int, d_model: int):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, d_model, kernel_size=1)

    def forward(self, x):               # x: (B, C, T)
        return self.proj(x).transpose(1, 2)   # (B, T, d_model)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):               # x: (B, T, d_model)
        return self.drop(x + self.pe[:, : x.size(1)])


class EncoderLayer(nn.Module):
    """
    One transformer encoder layer:
      1. Self-attention on [pool_token | TT] sequence
      2. Causal cross-attention: [pool_token | TT] attends to JAW
         - pool token can attend to ALL jaw positions (global summary)
         - TT at step t only attends to jaw steps 0..t  (causal)
      3. Feed-forward network
    """

    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float = 0.1):
        super().__init__()
        kw = dict(dropout=dropout, batch_first=True)
        self.self_attn  = nn.MultiheadAttention(d_model, nhead, **kw)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, **kw)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, jaw_enc, causal_mask, return_attn: bool = False):
        """
        x          : (B, T+1, d)   — [pool_token | TT tokens]
        jaw_enc    : (B, T,   d)   — JAW encoding  (no pool token)
        causal_mask: (T+1, T)      — -inf where attention is blocked
        return_attn: bool          — whether to return cross-attention weights

        returns (x, cross_w)
          cross_w : (B, T+1, T) averaged over heads, or None
        """
        x2, _ = self.self_attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.drop(x2))

        x2, cross_w = self.cross_attn(
            x, jaw_enc, jaw_enc,
            attn_mask=causal_mask,
            need_weights=return_attn,
            average_attn_weights=True,
        )
        x = self.norm2(x + self.drop(x2))
        x = self.norm3(x + self.drop(self.ffn(x)))
        return x, cross_w   # cross_w is None when return_attn=False


class PoolEncoder(nn.Module):
    """
    Transformer encoder for one pool (patient OR healthy).
    Prepends a learnable pool token to every trial's TT sequence.
    """

    def __init__(
        self,
        in_channels: int = 9,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_ff: int = 256,
        dropout: float = 0.1,
        seq_len: int = 400,
    ):
        super().__init__()
        self.seq_len  = seq_len
        self.tt_proj  = InputProjection(in_channels, d_model)
        self.jaw_proj = InputProjection(in_channels, d_model)
        self.pos_enc  = PositionalEncoding(d_model, max_len=seq_len + 2, dropout=dropout)

        # Learnable pool token — represents this pool's overall pattern
        self.pool_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, nhead, dim_ff, dropout) for _ in range(num_layers)]
        )

        # Build causal cross-attention mask once and register as buffer
        #   Shape (T+1, T):
        #     row 0   (pool token): attend to ALL jaw positions  → all 0
        #     row t+1 (TT step t): attend to jaw steps 0..t     → -inf for j > t
        T = seq_len
        rows = torch.arange(T + 1).unsqueeze(1)   # (T+1, 1)
        cols = torch.arange(T).unsqueeze(0)        # (1, T)
        allowed = (rows == 0) | (cols < rows)      # pool token OR causal
        mask = torch.where(
            allowed,
            torch.zeros(T + 1, T),
            torch.full((T + 1, T), float("-inf")),
        )
        self.register_buffer("causal_mask", mask)  # moved to device automatically

    def forward(self, tt, jaw, return_attn: bool = False):
        """
        tt, jaw     : (B, 9, T)
        return_attn : bool — collect cross-attention weights from every layer

        returns (x, attn_stack)
          x          : (B, T+1, d_model)
          attn_stack : (B, num_layers, T+1, T)  or None
        """
        B = tt.size(0)

        tt_enc  = self.pos_enc(self.tt_proj(tt))    # (B, T, d)
        jaw_enc = self.pos_enc(self.jaw_proj(jaw))  # (B, T, d)

        pool = self.pool_token.expand(B, -1, -1)    # (B, 1, d)
        x    = torch.cat([pool, tt_enc], dim=1)     # (B, T+1, d)

        layer_attns = []
        for layer in self.layers:
            x, attn_w = layer(x, jaw_enc, self.causal_mask, return_attn=return_attn)
            if return_attn:
                layer_attns.append(attn_w)          # (B, T+1, T)

        if return_attn:
            return x, torch.stack(layer_attns, dim=1)   # (B, num_layers, T+1, T)
        return x, None


class CrossPoolAttention(nn.Module):
    """
    Each trial's encoding attends to the *other* pool's summary.
    Pool summary = mean of pool-token outputs across the current batch.

    This fuses the two pools' overall patterns while preserving
    individual trial structure.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, enc, other_pool_summary):
        """
        enc               : (B, T+1, d)
        other_pool_summary: (1, 1,   d)  — mean pool token from the other pool
        """
        kv  = other_pool_summary.expand(enc.size(0), -1, -1)  # (B, 1, d)
        x2, _ = self.attn(enc, kv, kv)
        return self.norm(enc + self.drop(x2))


class DecoderLayer(nn.Module):
    """Standard transformer decoder layer (causal self-attn + cross-attn + FFN)."""

    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float = 0.1):
        super().__init__()
        kw = dict(dropout=dropout, batch_first=True)
        self.self_attn  = nn.MultiheadAttention(d_model, nhead, **kw)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, **kw)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, tgt, memory):
        """
        tgt   : (B, T,   d)
        memory: (B, T+1, d)  — encoder output (includes pool token)
        """
        T      = tgt.size(1)
        causal = torch.triu(
            torch.full((T, T), float("-inf"), device=tgt.device), diagonal=1
        )

        x2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=causal)
        tgt = self.norm1(tgt + self.drop(x2))

        x2, _ = self.cross_attn(tgt, memory, memory)
        tgt = self.norm2(tgt + self.drop(x2))

        tgt = self.norm3(tgt + self.drop(self.ffn(tgt)))
        return tgt


class PoolDecoder(nn.Module):
    """Reconstructs TT signal from encoder memory."""

    def __init__(
        self,
        out_channels: int = 9,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_ff: int = 256,
        dropout: float = 0.1,
        seq_len: int = 400,
    ):
        super().__init__()
        self.tgt_proj = InputProjection(out_channels, d_model)
        self.pos_enc  = PositionalEncoding(d_model, max_len=seq_len + 2, dropout=dropout)
        self.layers   = nn.ModuleList(
            [DecoderLayer(d_model, nhead, dim_ff, dropout) for _ in range(num_layers)]
        )
        self.out_proj = nn.Linear(d_model, out_channels)

    def forward(self, tt, memory):
        """
        tt    : (B, 9, T)    — original TT (teacher forcing)
        memory: (B, T+1, d)  — encoder output
        returns (B, 9, T)    — reconstructed TT
        """
        tgt = self.pos_enc(self.tgt_proj(tt))   # (B, T, d)
        for layer in self.layers:
            tgt = layer(tgt, memory)
        out = self.out_proj(tgt)                 # (B, T, 9)
        return out.transpose(1, 2)               # (B, 9, T)


class ProjectionHead(nn.Module):
    """
    Encoder output → fixed-size per-trial embedding for PCA.
    Skips the pool token (pos 0) and global-average-pools the TT positions.
    """

    def __init__(self, d_model: int, embed_dim: int = 64):
        super().__init__()
        self.conv = nn.Conv1d(d_model, embed_dim, kernel_size=1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, enc):
        # enc: (B, T+1, d)  →  (B, embed_dim)
        x = enc[:, 1:, :].transpose(1, 2)   # skip pool token → (B, d, T)
        x = self.conv(x).mean(dim=-1)        # (B, embed_dim)
        return self.norm(x)


# ── Full model ─────────────────────────────────────────────────────────────────

class DualPoolModel(nn.Module):
    """
    Two symmetric encoder–decoder pairs (patient / healthy) connected
    by a cross-pool attention module between their pool-token summaries.
    """

    def __init__(
        self,
        in_channels: int = 9,
        d_model: int = 128,
        nhead: int = 4,
        num_enc_layers: int = 4,
        num_dec_layers: int = 4,
        dim_ff: int = 256,
        dropout: float = 0.1,
        seq_len: int = 400,
        embed_dim: int = 64,
    ):
        super().__init__()

        enc_kw = dict(
            in_channels=in_channels, d_model=d_model, nhead=nhead,
            num_layers=num_enc_layers, dim_ff=dim_ff, dropout=dropout, seq_len=seq_len,
        )
        dec_kw = dict(
            out_channels=in_channels, d_model=d_model, nhead=nhead,
            num_layers=num_dec_layers, dim_ff=dim_ff, dropout=dropout, seq_len=seq_len,
        )

        self.patient_encoder = PoolEncoder(**enc_kw)
        self.healthy_encoder = PoolEncoder(**enc_kw)

        self.patient_cross   = CrossPoolAttention(d_model, nhead, dropout)
        self.healthy_cross   = CrossPoolAttention(d_model, nhead, dropout)

        self.patient_decoder = PoolDecoder(**dec_kw)
        self.healthy_decoder = PoolDecoder(**dec_kw)

        self.patient_proj    = ProjectionHead(d_model, embed_dim)
        self.healthy_proj    = ProjectionHead(d_model, embed_dim)

    def forward(self, tt_p, jaw_p, tt_h, jaw_h):
        """
        tt_p, jaw_p : (B_p, 9, T) — patient batch
        tt_h, jaw_h : (B_h, 9, T) — healthy batch
        returns: recon_p, recon_h, enc_p, enc_h
        """
        # Encode both pools
        enc_p, _ = self.patient_encoder(tt_p, jaw_p)    # (B_p, T+1, d)
        enc_h, _ = self.healthy_encoder(tt_h, jaw_h)    # (B_h, T+1, d)

        # Pool summaries: mean of pool-token outputs across batch
        pool_p = enc_p[:, :1, :].mean(dim=0, keepdim=True)   # (1, 1, d)
        pool_h = enc_h[:, :1, :].mean(dim=0, keepdim=True)   # (1, 1, d)

        # Cross-pool attention (each pool attends to the other's summary)
        enc_p = self.patient_cross(enc_p, pool_h)
        enc_h = self.healthy_cross(enc_h, pool_p)

        # Decode → reconstruct TT
        recon_p = self.patient_decoder(tt_p, enc_p)   # (B_p, 9, T)
        recon_h = self.healthy_decoder(tt_h, enc_h)   # (B_h, 9, T)

        return recon_p, recon_h, enc_p, enc_h

    @torch.no_grad()
    def extract_embeddings(self, X_norm, pool: str = "patient",
                           device: str = "cpu", batch_size: int = 32):
        """
        X_norm : (N, 18, T) normalised data
        pool   : "patient" or "healthy"
        returns: (N, embed_dim) embeddings
        """
        self.eval()
        tt  = torch.tensor(X_norm[:, TT_CH,  :], dtype=torch.float32)
        jaw = torch.tensor(X_norm[:, JAW_CH, :], dtype=torch.float32)

        encoder = self.patient_encoder if pool == "patient" else self.healthy_encoder
        proj    = self.patient_proj    if pool == "patient" else self.healthy_proj

        embs = []
        for i in range(0, len(tt), batch_size):
            tt_b  = tt[i : i + batch_size].to(device)
            jaw_b = jaw[i : i + batch_size].to(device)
            enc, _ = encoder(tt_b, jaw_b)
            embs.append(proj(enc).cpu().numpy())

        return np.concatenate(embs, axis=0)

    @torch.no_grad()
    def extract_encoder_sequence(self, X_norm, pool: str = "patient",
                                  device: str = "cpu", batch_size: int = 32):
        """
        Return the full encoder output sequence for each trial.

        Returns
        -------
        seqs : np.ndarray  (N, T+1, d_model)
          pos 0   = pool token output
          pos 1..T = TT timestep outputs (after cross-attention with JAW)
        """
        self.eval()
        tt  = torch.tensor(X_norm[:, TT_CH,  :], dtype=torch.float32)
        jaw = torch.tensor(X_norm[:, JAW_CH, :], dtype=torch.float32)
        encoder = self.patient_encoder if pool == "patient" else self.healthy_encoder

        seqs = []
        for i in range(0, len(tt), batch_size):
            tt_b  = tt[i : i + batch_size].to(device)
            jaw_b = jaw[i : i + batch_size].to(device)
            enc, _ = encoder(tt_b, jaw_b)
            seqs.append(enc.cpu().numpy())
        return np.concatenate(seqs, axis=0)  # (N, T+1, d_model)

    @torch.no_grad()
    def extract_attention_maps(self, X_norm, pool: str = "patient",
                               device: str = "cpu", batch_size: int = 16):
        """
        Extract per-trial cross-attention weights (TT attending to JAW).

        Returns
        -------
        attn : np.ndarray  (N, num_layers, T+1, T_jaw)
          dim-2 row 0      = pool token → JAW attention
          dim-2 rows 1..T  = TT[t]     → JAW attention  (causal)
        """
        self.eval()
        tt  = torch.tensor(X_norm[:, TT_CH,  :], dtype=torch.float32)
        jaw = torch.tensor(X_norm[:, JAW_CH, :], dtype=torch.float32)

        encoder = self.patient_encoder if pool == "patient" else self.healthy_encoder

        all_attns = []
        for i in range(0, len(tt), batch_size):
            tt_b  = tt[i : i + batch_size].to(device)
            jaw_b = jaw[i : i + batch_size].to(device)
            _, attn = encoder(tt_b, jaw_b, return_attn=True)
            all_attns.append(attn.cpu().numpy())

        return np.concatenate(all_attns, axis=0)   # (N, num_layers, T+1, T)


# ── Training ───────────────────────────────────────────────────────────────────

def train_dual_pool(
    model,
    patient_loader,
    healthy_loader,
    device: str = "cpu",
    lr: float = 1e-4,
    epochs: int = 60,
):
    model     = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    losses    = []

    # Run for max(len(p_loader), len(h_loader)) steps per epoch,
    # cycling the shorter loader so both pools are seen equally.
    n_steps = max(len(patient_loader), len(healthy_loader))

    for epoch in range(epochs):
        model.train()
        total_loss, n = 0.0, 0

        p_iter = cycle(patient_loader)
        h_iter = cycle(healthy_loader)

        for _ in range(n_steps):
            tt_p, jaw_p = next(p_iter)
            tt_h, jaw_h = next(h_iter)

            tt_p  = tt_p.to(device);  jaw_p = jaw_p.to(device)
            tt_h  = tt_h.to(device);  jaw_h = jaw_h.to(device)

            optimizer.zero_grad()
            recon_p, recon_h, _, _ = model(tt_p, jaw_p, tt_h, jaw_h)

            loss = criterion(recon_p, tt_p) + criterion(recon_h, tt_h)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * (len(tt_p) + len(tt_h))
            n          += len(tt_p) + len(tt_h)

        avg = total_loss / n
        losses.append(avg)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:>3}/{epochs}  Loss: {avg:.6f}")

    return model, losses


# ── Data loading (multi-subject) ──────────────────────────────────────────────

def load_and_concat_group(save_dir, group_name):
    """
    Load all per-subject npz files for a group and concatenate in subject order,
    preserving within-subject trial ordering (required for time-series analysis).

    Returns
    -------
    X           : (N_total, 18, T)
    subject_ids : list[str]   — subject name for every trial
    boundaries  : list[int]   — trial index where each new subject starts
    """
    from read_mat import load_group_data
    subjects = load_group_data(save_dir, group_name)

    X_list, subject_ids, boundaries = [], [], []
    cursor = 0
    for subject_name, (X, *_) in subjects.items():
        boundaries.append(cursor)
        X_list.append(X)
        subject_ids.extend([subject_name] * len(X))
        cursor += len(X)

    return np.concatenate(X_list, axis=0), subject_ids, boundaries


def parse_within_subject_trial_id(trial_name):
    """
    Parse the within-subject trial number from both naming conventions:
      - T0003          -> 3
      - T7_T3_SUed_   -> 3
      - T2_T100_SUet_ -> 100
    """
    matches = re.findall(r"T0*(\d+)", str(trial_name))
    if not matches:
        raise ValueError(f"Cannot parse trial id from trial name: {trial_name}")
    return int(matches[-1])


def load_trial_aligned_groups(save_dir, group_names=("patient", "healthy")):
    """
    Load groups and keep only trial IDs present in every subject.

    Every subject is ordered by the same numeric within-subject trial IDs before
    concatenation, so downstream within-subject curves compare the same trials
    instead of interpolating over uneven or lexicographically sorted sequences.
    """
    from read_mat import load_group_data

    group_subjects = {
        group_name: load_group_data(save_dir, group_name)
        for group_name in group_names
    }

    trial_sets = []
    subject_trial_maps = {}
    for group_name, subjects in group_subjects.items():
        subject_trial_maps[group_name] = {}
        for subject_name, (X, trial_names, *_) in subjects.items():
            trial_ids = [parse_within_subject_trial_id(name) for name in trial_names]
            if len(set(trial_ids)) != len(trial_ids):
                raise ValueError(
                    f"Duplicate parsed trial IDs in {group_name}/{subject_name}"
                )

            id_to_index = {trial_id: idx for idx, trial_id in enumerate(trial_ids)}
            subject_trial_maps[group_name][subject_name] = id_to_index
            trial_sets.append(set(id_to_index))

    common_trial_ids = sorted(set.intersection(*trial_sets))
    if not common_trial_ids:
        raise ValueError("No common trial IDs found across all subjects.")

    print(
        f"\nTrial alignment: keeping {len(common_trial_ids)} trials present in "
        f"every subject ({common_trial_ids[0]}-{common_trial_ids[-1]} by ID)."
    )

    aligned = {}
    for group_name, subjects in group_subjects.items():
        X_list, subject_ids, boundaries = [], [], []
        cursor = 0
        removed_total = 0

        for subject_name, (X, *_rest) in subjects.items():
            id_to_index = subject_trial_maps[group_name][subject_name]
            keep_idx = [id_to_index[trial_id] for trial_id in common_trial_ids]

            boundaries.append(cursor)
            X_list.append(X[keep_idx])
            subject_ids.extend([subject_name] * len(keep_idx))
            cursor += len(keep_idx)
            removed_total += len(X) - len(keep_idx)

        X_aligned = np.concatenate(X_list, axis=0)
        aligned[group_name] = (X_aligned, subject_ids, boundaries)
        print(
            f"{group_name}: {X_aligned.shape} after alignment "
            f"({removed_total} trials removed)."
        )

    return aligned, common_trial_ids


# ── Analysis utilities ─────────────────────────────────────────────────────────

def compute_joint_pca(Z_p, Z_h, n_components: int = 2):
    Z_all  = np.vstack([Z_p, Z_h])
    scaler = StandardScaler()
    Z_sc   = scaler.fit_transform(Z_all)
    pca    = PCA(n_components=n_components)
    Z_pca  = pca.fit_transform(Z_sc)
    return Z_pca[: len(Z_p)], Z_pca[len(Z_p) :], pca, scaler


def _group_avg_pca(Z_pca, boundaries, n_common: int = 250):
    """
    Average PCA coordinates across subjects, aligned by within-subject trial position.

    Parameters
    ----------
    Z_pca      : (N_total, n_components)
    boundaries : list[int]  — trial index where each subject starts
    n_common   : number of trial positions to interpolate to

    Returns
    -------
    mean_curve : (n_common, n_components)
    std_curve  : (n_common, n_components)
    """
    if boundaries is None:
        boundaries = [0]
    ends   = list(boundaries[1:]) + [len(Z_pca)]
    curves = [Z_pca[s:e] for s, e in zip(boundaries, ends) if e > s]
    n_common = min(n_common, min(len(c) for c in curves))

    interpolated = []
    for c in curves:
        x_old = np.linspace(0, 1, len(c))
        x_new = np.linspace(0, 1, n_common)
        interpolated.append(
            np.stack([np.interp(x_new, x_old, c[:, i])
                      for i in range(c.shape[1])], axis=1)
        )
    stacked = np.stack(interpolated)         # (n_subjects, n_common, n_comp)
    return stacked.mean(axis=0), stacked.std(axis=0)


def _shifted_trial_axis(trial_ids, n_points):
    """Plot trial IDs shifted to start at zero, e.g. trial 5 -> 0."""
    if trial_ids is None:
        return np.arange(1, n_points + 1)

    shifted = np.asarray(trial_ids, dtype=float)
    shifted = shifted - shifted[0]
    if len(shifted) == n_points:
        return shifted

    return np.linspace(shifted[0], shifted[-1], n_points)


def plot_trajectory(Z_p, Z_h,
                    p_boundaries=None, h_boundaries=None,
                    n_common: int = 250,
                    title="Mean Trial Trajectory in PCA Space",
                    save_path="pca_trajectory.png"):
    """
    Plot group-average PC1 vs PC2 trajectory across within-subject trial position.
    Each subject's curve is interpolated to `n_common` points then averaged.
    Shaded band = ±1 SD across subjects.
    """
    mean_p, std_p = _group_avg_pca(Z_p, p_boundaries, n_common)
    mean_h, std_h = _group_avg_pca(Z_h, h_boundaries, n_common)

    fig, ax = plt.subplots(figsize=(8, 6))

    for mean, std, color, label in [
            (mean_p, std_p, "C0", "Patient"),
            (mean_h, std_h, "C1", "Healthy")]:
        ax.plot(mean[:, 0], mean[:, 1], "-o", ms=3, lw=1.5,
                color=color, label=f"{label} (mean)")
        ax.scatter(mean[0, 0],  mean[0, 1],  marker="s", s=80,
                   color=color, zorder=5, label=f"{label} start")
        ax.scatter(mean[-1, 0], mean[-1, 1], marker="*", s=120,
                   color=color, zorder=5, label=f"{label} end")

    ax.set(xlabel="PC1", ylabel="PC2", title=title)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def plot_pc1_over_trials(Z_p, Z_h,
                          p_boundaries=None, h_boundaries=None,
                          n_common: int = 250,
                          trial_ids=None,
                          save_path="pc1_trend.png"):
    mean_p, std_p = _group_avg_pca(Z_p, p_boundaries, n_common)
    mean_h, std_h = _group_avg_pca(Z_h, h_boundaries, n_common)

    x_p = _shifted_trial_axis(trial_ids, len(mean_p))
    x_h = _shifted_trial_axis(trial_ids, len(mean_h))

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(x_p, mean_p[:, 0], lw=2, color="C0", label="Patient (mean)")
    ax.fill_between(x_p, mean_p[:, 0] - std_p[:, 0], mean_p[:, 0] + std_p[:, 0],
                    alpha=0.2, color="C0", label="Patient ±1 SD")

    ax.plot(x_h, mean_h[:, 0], lw=2, color="C1", label="Healthy (mean)")
    ax.fill_between(x_h, mean_h[:, 0] - std_h[:, 0], mean_h[:, 0] + std_h[:, 0],
                    alpha=0.2, color="C1", label="Healthy ±1 SD")

    ax.set_xlabel("Within-subject trial index (first retained trial = 0)")
    ax.set_ylabel("PC1")
    ax.set_title("PC1 Trend Across Trials\n(group mean ± 1 SD across subjects)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def summarize_trajectory(Z, name="Subject"):
    x  = np.arange(len(Z))
    res = linregress(x, Z[:, 0])
    n  = len(Z)
    k  = max(1, int(n * 0.2))
    early_late = float(np.linalg.norm(Z[-k:].mean(0) - Z[:k].mean(0)))
    return {
        "name":               name,
        "trajectory_length":  float(np.linalg.norm(np.diff(Z, axis=0), axis=1).sum()),
        "dispersion":         float(np.linalg.norm(Z - Z.mean(0), axis=1).mean()),
        "early_late_dist":    early_late,
        "pc1_slope":          float(res.slope),
        "pc1_r":              float(res.rvalue),
        "pc1_p":              float(res.pvalue),
    }


# ── XAI: Cross-correlation based coordination analysis ────────────────────────
#
# Why cross-correlation instead of learned attention weights?
# The transformer's cross-attention (TT→JAW) collapsed to attention-sink
# behaviour (all weight on t=0) because MSE reconstruction of TT does not
# force the model to actually USE JAW. The PCA embeddings are still valid,
# but attention maps are uninformative.
#
# Cross-correlation is computed directly from the normalised input signals and
# answers the same four scientific questions reliably without depending on
# whether the model learned a meaningful attention distribution.
#
# Channel indices in X_norm (N, 18, T):
#   JAW y-position  → ch 1   (vertical jaw opening, speech-relevant)
#   TT  y-position  → ch 10  (tongue-tip height, speech-relevant)
# ─────────────────────────────────────────────────────────────────────────────

JAW_Y_CH = 1    # JAW vertical position
TT_Y_CH  = 10   # TT  vertical position


def _compute_xcorr(X_norm, fs: int = 250, max_lag_ms: float = 400.0):
    """
    Compute normalised cross-correlation between JAW-y and TT-y for every trial.

    Parameters
    ----------
    X_norm     : (N, 18, T)
    fs         : sampling rate (Hz)
    max_lag_ms : symmetric lag window in milliseconds

    Returns
    -------
    xcorr   : (N, 2*max_lag+1)  normalised cross-correlation curves
    lags_ms : (2*max_lag+1,)    lag axis in ms
              negative → JAW leads TT (JAW moves before TT)
              positive → TT leads JAW
    """
    jaw = X_norm[:, JAW_Y_CH, :]   # (N, T)
    tt  = X_norm[:, TT_Y_CH,  :]   # (N, T)
    T        = jaw.shape[1]
    max_lag  = int(max_lag_ms / 1000.0 * fs)
    lags_ms  = np.arange(-max_lag, max_lag + 1) / fs * 1000.0

    xcorr = np.zeros((len(X_norm), len(lags_ms)), dtype=np.float32)
    for n in range(len(X_norm)):
        j = jaw[n]; j = (j - j.mean()) / (j.std() + 1e-8)
        t = tt[n];  t = (t - t.mean()) / (t.std() + 1e-8)
        # np.correlate(t, j): positive lag → t leads j (TT leads JAW)
        full = np.correlate(t, j, mode="full")   # len = 2T-1
        center = T - 1
        xcorr[n] = full[center - max_lag : center + max_lag + 1] / T

    return xcorr, lags_ms


def plot_xcorr_profile(X_p, X_h, fs: int = 250, max_lag_ms: float = 400.0,
                       save_path: str = "xai_q1_xcorr_profile.png"):
    """
    Q1 — Within a general trial, at what lag do JAW and TT most strongly covary?
         And where does the patient profile deviate from healthy?

    Left  : Mean cross-correlation ± 95 % CI  (patient vs healthy)
    Right : Difference curve (patient − healthy) with ±2 SD band

    X-axis : Lag (ms)  negative = JAW leads TT
    Y-axis : Normalised cross-correlation coefficient
    """
    xcorr_p, lags_ms = _compute_xcorr(X_p, fs, max_lag_ms)
    xcorr_h, _       = _compute_xcorr(X_h, fs, max_lag_ms)

    mean_p = xcorr_p.mean(axis=0);  sem_p = xcorr_p.std(axis=0) / np.sqrt(len(xcorr_p))
    mean_h = xcorr_h.mean(axis=0);  sem_h = xcorr_h.std(axis=0) / np.sqrt(len(xcorr_h))
    diff   = mean_p - mean_h
    diff_sd = np.sqrt(xcorr_p.std(axis=0)**2 / len(xcorr_p) +
                      xcorr_h.std(axis=0)**2 / len(xcorr_h))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Left: mean profiles
    for ax_l, (mean, sem, label, color) in zip(
            [axes[0], axes[0]],
            [(mean_p, sem_p, "Patient", "C0"), (mean_h, sem_h, "Healthy", "C1")]):
        ax_l.plot(lags_ms, mean, lw=1.5, label=label, color=color)
        ax_l.fill_between(lags_ms, mean - 1.96*sem, mean + 1.96*sem,
                          alpha=0.2, color=color)

    # Mark peak lag for each group
    peak_p = lags_ms[np.argmax(mean_p)]
    peak_h = lags_ms[np.argmax(mean_h)]
    axes[0].axvline(peak_p, color="C0", lw=1, ls="--",
                    label=f"Patient peak = {peak_p:.0f} ms")
    axes[0].axvline(peak_h, color="C1", lw=1, ls="--",
                    label=f"Healthy peak = {peak_h:.0f} ms")
    axes[0].axvline(0, color="k", lw=0.6, ls=":", alpha=0.5)
    axes[0].set_xlabel("Lag (ms)\n[negative = JAW leads TT]")
    axes[0].set_ylabel("Normalised cross-correlation")
    axes[0].set_title("Q1: Mean JAW–TT Cross-Correlation Profile\n"
                      "(shaded = 95 % CI across trials)")
    axes[0].legend(fontsize=8)

    # Right: difference
    axes[1].plot(lags_ms, diff, lw=1.5, color="purple", label="Patient − Healthy")
    axes[1].fill_between(lags_ms, diff - 2*diff_sd, diff + 2*diff_sd,
                         alpha=0.2, color="purple")
    axes[1].axhline(0, color="k", lw=0.6, ls=":")
    axes[1].axvline(0, color="k", lw=0.6, ls=":", alpha=0.5)
    axes[1].set_xlabel("Lag (ms)\n[negative = JAW leads TT]")
    axes[1].set_ylabel("Correlation difference")
    axes[1].set_title("Q1: Patient − Healthy Deviation\n"
                      "(positive = patient more correlated at this lag)")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def plot_peak_lag_across_trials(X_p, X_h,
                                 p_boundaries=None, h_boundaries=None,
                                 n_common: int = 250,
                                 trial_ids=None,
                                 fs: int = 250, max_lag_ms: float = 400.0,
                                 save_path: str = "xai_q2_peak_lag_trials.png"):
    """
    Q2 — Does the JAW–TT coordination lag change across trials?

    Peak lag per trial is extracted, then averaged across subjects
    (each subject's curve interpolated to `n_common` points).
    Shaded band = ±1 SD.  Linear trend fitted to each group mean.

    X-axis : Within-subject trial position (1 … n_common)
    Y-axis : Peak lag (ms)  [negative = JAW leads TT]
    """
    xcorr_p, lags_ms = _compute_xcorr(X_p, fs, max_lag_ms)
    xcorr_h, _       = _compute_xcorr(X_h, fs, max_lag_ms)

    peak_lag_p = lags_ms[np.argmax(xcorr_p, axis=1)]   # (N_p,)
    peak_lag_h = lags_ms[np.argmax(xcorr_h, axis=1)]   # (N_h,)

    def _subject_avg_lag(peak_lag, boundaries, n_common):
        if boundaries is None:
            boundaries = [0]
        ends   = list(boundaries[1:]) + [len(peak_lag)]
        curves = [peak_lag[s:e] for s, e in zip(boundaries, ends) if e > s]
        n_common = min(n_common, min(len(c) for c in curves))
        interp = []
        for c in curves:
            x_old = np.linspace(0, 1, len(c))
            x_new = np.linspace(0, 1, n_common)
            interp.append(np.interp(x_new, x_old, c))
        stacked = np.stack(interp)   # (n_subjects, n_common)
        return stacked.mean(0), stacked.std(0), n_common

    mean_p, std_p, nc_p = _subject_avg_lag(peak_lag_p, p_boundaries, n_common)
    mean_h, std_h, nc_h = _subject_avg_lag(peak_lag_h, h_boundaries, n_common)

    def _trend(x, y):
        res = linregress(x, y)
        return res.slope * x + res.intercept, res

    x_p = _shifted_trial_axis(trial_ids, nc_p)
    x_h = _shifted_trial_axis(trial_ids, nc_h)
    trend_p, res_p = _trend(x_p, mean_p)
    trend_h, res_h = _trend(x_h, mean_h)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x_p, mean_p, lw=1.5, color="C0", label="Patient (mean)")
    ax.fill_between(x_p, mean_p - std_p, mean_p + std_p, alpha=0.2, color="C0",
                    label="Patient ±1 SD")
    ax.plot(x_h, mean_h, lw=1.5, color="C1", label="Healthy (mean)")
    ax.fill_between(x_h, mean_h - std_h, mean_h + std_h, alpha=0.2, color="C1",
                    label="Healthy ±1 SD")
    ax.plot(x_p, trend_p, lw=2, ls="--", color="C0",
            label=f"Patient trend  slope={res_p.slope:.3f} ms/trial  "
                  f"r={res_p.rvalue:.2f}  p={res_p.pvalue:.3f}")
    ax.plot(x_h, trend_h, lw=2, ls="--", color="C1",
            label=f"Healthy trend  slope={res_h.slope:.3f} ms/trial  "
                  f"r={res_h.rvalue:.2f}  p={res_h.pvalue:.3f}")
    ax.axhline(0, color="k", lw=0.6, ls=":", alpha=0.6,
               label="Lag = 0 ms (synchronous)")
    ax.set_xlabel("Within-subject trial index (first retained trial = 0)")
    ax.set_ylabel("Peak Cross-Correlation Lag (ms)\n[negative = JAW leads TT]")
    ax.set_title("Q2: JAW–TT Coordination Lag Across Trials\n"
                 "(group mean ± 1 SD across subjects)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def plot_pool_token_xcorr(model, X_p, X_h, fs: int = 250, max_lag_ms: float = 400.0,
                           save_path: str = "xai_q3_pool_token.png"):
    """
    Q3 — Group-level coordination: pool token vectors + group xcorr distribution.

    Left  : Distribution of peak lag per group (violin + individual points)
            → which group coordinates earlier/later?
    Right : Learned pool-token cosine similarity (group prototype divergence)

    X-axis left  : Group
    Y-axis left  : Peak lag (ms)
    X-axis right : Embedding dimension
    Y-axis right : Parameter value
    """
    xcorr_p, lags_ms = _compute_xcorr(X_p, fs, max_lag_ms)
    xcorr_h, _       = _compute_xcorr(X_h, fs, max_lag_ms)

    peak_p = lags_ms[np.argmax(xcorr_p, axis=1)]
    peak_h = lags_ms[np.argmax(xcorr_h, axis=1)]

    pt_p = model.patient_encoder.pool_token.detach().cpu().numpy().squeeze()
    pt_h = model.healthy_encoder.pool_token.detach().cpu().numpy().squeeze()
    cos_sim = float((pt_p @ pt_h) /
                    (np.linalg.norm(pt_p) * np.linalg.norm(pt_h) + 1e-8))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: violin plot of peak lag distributions
    parts = axes[0].violinplot([peak_p, peak_h], positions=[0, 1],
                                showmedians=True, showextrema=True)
    parts["bodies"][0].set_facecolor("C0"); parts["bodies"][0].set_alpha(0.5)
    parts["bodies"][1].set_facecolor("C1"); parts["bodies"][1].set_alpha(0.5)
    # Overlay jittered scatter
    rng = np.random.default_rng(0)
    axes[0].scatter(rng.normal(0, 0.05, len(peak_p)), peak_p,
                    s=4, alpha=0.3, color="C0")
    axes[0].scatter(rng.normal(1, 0.05, len(peak_h)), peak_h,
                    s=4, alpha=0.3, color="C1")
    axes[0].axhline(0, color="k", lw=0.8, ls=":", alpha=0.6,
                    label="Lag = 0 ms")
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["Patient", "Healthy"])
    axes[0].set_ylabel("Peak Cross-Correlation Lag (ms)\n[negative = JAW leads TT]")
    axes[0].set_title("Q3a: Group-Level Coordination Lag Distribution\n"
                      f"Patient median={np.median(peak_p):.1f} ms   "
                      f"Healthy median={np.median(peak_h):.1f} ms")
    axes[0].legend(fontsize=8)

    # Right: learned pool token vectors
    d = len(pt_p)
    axes[1].plot(np.arange(d), pt_p, lw=1, alpha=0.8, color="C0",
                 label="Patient pool token")
    axes[1].plot(np.arange(d), pt_h, lw=1, alpha=0.8, color="C1",
                 label="Healthy pool token")
    axes[1].set_xlabel("Embedding dimension index")
    axes[1].set_ylabel("Parameter value")
    axes[1].set_title(f"Q3b: Learned Pool Token Vectors\n"
                      f"Cosine similarity = {cos_sim:.4f}  "
                      f"({'similar' if cos_sim > 0.8 else 'distinct'} group prototypes)")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def plot_xcorr_within_trial_segments(X_p, X_h, fs: int = 250,
                                      max_lag_ms: float = 300.0,
                                      save_path: str = "xai_q4_xcorr_segments.png"):
    """
    Q4 — How does JAW–TT coupling change within a general trial?

    The trial is split into three equal segments (early / mid / late).
    Cross-correlation is computed within each segment separately,
    showing whether the coupling strength and preferred lag change
    across the trial phases.

    X-axis : Lag (ms)  [negative = JAW leads TT]
    Y-axis : Normalised cross-correlation coefficient
    """
    jaw_all_p = X_p[:, JAW_Y_CH, :]   # (N_p, T)
    tt_all_p  = X_p[:, TT_Y_CH,  :]
    jaw_all_h = X_h[:, JAW_Y_CH, :]
    tt_all_h  = X_h[:, TT_Y_CH,  :]

    T        = jaw_all_p.shape[1]
    seg_size = T // 3
    segments = {
        f"Early  (0 – {seg_size/fs:.2f} s)":             slice(0,           seg_size),
        f"Mid  ({seg_size/fs:.2f} – {2*seg_size/fs:.2f} s)": slice(seg_size,   2 * seg_size),
        f"Late  ({2*seg_size/fs:.2f} – {T/fs:.2f} s)":   slice(2 * seg_size, T),
    }

    max_lag = int(max_lag_ms / 1000.0 * fs)
    lags_ms = np.arange(-max_lag, max_lag + 1) / fs * 1000.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (label, seg) in zip(axes, segments.items()):
        seg_len = seg.stop - seg.start

        def _seg_xcorr(jaw_all, tt_all):
            out = np.zeros((len(jaw_all), len(lags_ms)), dtype=np.float32)
            for n in range(len(jaw_all)):
                j = jaw_all[n, seg]; j = (j - j.mean()) / (j.std() + 1e-8)
                t = tt_all[n, seg];  t = (t - t.mean()) / (t.std() + 1e-8)
                full   = np.correlate(t, j, mode="full")
                center = seg_len - 1
                # Clamp if segment is shorter than max_lag
                lo = max(0, center - max_lag)
                hi = min(len(full), center + max_lag + 1)
                needed_lo = max_lag - (center - lo)
                needed_hi = needed_lo + (hi - lo)
                out[n, needed_lo:needed_hi] = full[lo:hi] / seg_len
            return out

        xc_p = _seg_xcorr(jaw_all_p, tt_all_p)
        xc_h = _seg_xcorr(jaw_all_h, tt_all_h)

        mean_p = xc_p.mean(axis=0);  sem_p = xc_p.std(axis=0) / np.sqrt(len(xc_p))
        mean_h = xc_h.mean(axis=0);  sem_h = xc_h.std(axis=0) / np.sqrt(len(xc_h))

        peak_p_lag = lags_ms[np.argmax(mean_p)]
        peak_h_lag = lags_ms[np.argmax(mean_h)]

        ax.plot(lags_ms, mean_p, lw=1.5, color="C0", label="Patient")
        ax.plot(lags_ms, mean_h, lw=1.5, color="C1", label="Healthy")
        ax.fill_between(lags_ms, mean_p - 1.96*sem_p, mean_p + 1.96*sem_p,
                        alpha=0.15, color="C0")
        ax.fill_between(lags_ms, mean_h - 1.96*sem_h, mean_h + 1.96*sem_h,
                        alpha=0.15, color="C1")
        ax.axvline(peak_p_lag, color="C0", lw=1, ls="--",
                   label=f"Patient peak {peak_p_lag:.0f} ms")
        ax.axvline(peak_h_lag, color="C1", lw=1, ls="--",
                   label=f"Healthy peak {peak_h_lag:.0f} ms")
        ax.axvline(0, color="k", lw=0.6, ls=":", alpha=0.5)
        ax.set_xlabel("Lag (ms)\n[negative = JAW leads TT]")
        ax.set_ylabel("Normalised cross-correlation")
        ax.set_title(f"Q4: {label}")
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


# ── XAI: Embedding-based analysis ────────────────────────────────────────────
#
# These four functions use the TRAINED MODEL directly:
#   - encoder sequence outputs (N, T+1, d_model)  via extract_encoder_sequence()
#   - projected embeddings     (N, embed_dim)      via extract_embeddings()
#   - raw cross-attention maps (N, L, T+1, T)      via extract_attention_maps()
#
# Q1: Which time interval has the largest patient vs healthy difference?
# Q2: Do encoder representations change systematically across trials?
# Q3: Pool token group-level embedding analysis (PCA + discriminative dims)
# Q4: Mean JAW→TT cross-attention heatmap + gradient attribution on pool token
# ─────────────────────────────────────────────────────────────────────────────


def plot_emb_q1_time_interval_diff(model, patient_X_norm, healthy_X_norm,
                                    device: str = "cpu", batch_size: int = 32,
                                    fs: int = 250,
                                    save_path: str = "xai_emb_q1_time_diff.png"):
    """
    Q1 (Embedding) — Which time interval shows the largest patient vs healthy difference?

    For each encoder output position t ∈ {pool_token, TT_0 … TT_{T-1}}:
      mean_p[t] = mean over all patient trials of encoder output at position t
      mean_h[t] = mean over all healthy trials
      diff[t]   = L2( mean_p[t] − mean_h[t] )

    Left  : diff vs time (ms) for TT positions (pool token marked separately)
    Right : bar comparing pool-token separation vs mean TT separation
    """
    seq_p = model.extract_encoder_sequence(patient_X_norm, "patient", device, batch_size)
    seq_h = model.extract_encoder_sequence(healthy_X_norm, "healthy", device, batch_size)
    # shape: (N, T+1, d_model)

    mean_p = seq_p.mean(axis=0)   # (T+1, d)
    mean_h = seq_h.mean(axis=0)   # (T+1, d)

    diff_l2  = np.linalg.norm(mean_p - mean_h, axis=1)   # (T+1,)
    T = seq_p.shape[1] - 1    # number of TT time steps

    # Time axis for TT positions (position 0 = pool token, skip it)
    time_ms = np.arange(T) / fs * 1000.0   # 0 .. (T-1)/fs * 1000 ms

    pool_diff    = diff_l2[0]
    tt_diff      = diff_l2[1:]               # (T,)
    peak_t       = int(np.argmax(tt_diff))
    peak_ms      = time_ms[peak_t]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Left: L2 diff across time
    axes[0].plot(time_ms, tt_diff, lw=1.2, color="purple")
    axes[0].axvline(peak_ms, color="red", lw=1.2, ls="--",
                    label=f"Peak at {peak_ms:.0f} ms")
    axes[0].axhline(tt_diff.mean(), color="k", lw=0.8, ls=":",
                    alpha=0.6, label="Mean diff")
    axes[0].set_xlabel("Time within trial (ms)")
    axes[0].set_ylabel("L2 distance between group mean embeddings")
    axes[0].set_title("Q1 (Embedding): Patient vs Healthy Difference\nat Each Time Step")
    axes[0].legend(fontsize=8)

    # Right: pool token vs TT comparison
    axes[1].bar(["Pool Token\n(global summary)", f"TT mean\n(all {T} steps)"],
                [pool_diff, tt_diff.mean()],
                color=["steelblue", "coral"], edgecolor="k", linewidth=0.8)
    axes[1].set_ylabel("L2 distance (group mean difference)")
    axes[1].set_title("Pool Token vs TT:\nGroup Separation Strength")
    for i, v in enumerate([pool_diff, tt_diff.mean()]):
        axes[1].text(i, v + 0.005, f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def plot_emb_q2_trial_dynamics(model, patient_X_norm, healthy_X_norm,
                                p_boundaries=None, h_boundaries=None,
                                n_common: int = 250,
                                trial_ids=None,
                                device: str = "cpu", batch_size: int = 32,
                                save_path: str = "xai_emb_q2_trial_dynamics.png"):
    """
    Q2 (Embedding) — Do encoder representations change across trials?

    Left  : Embedding drift — L2 distance of each trial's embedding from group
            mean, smoothed and averaged across subjects.
            A rising curve → representations diverge from the prototype over time.

    Right : Trial-to-trial stability — cosine similarity between consecutive
            trials' embeddings within each subject, shown as violin.
            Low similarity → large inter-trial variability.
    """
    emb_p = model.extract_embeddings(patient_X_norm, "patient", device, batch_size)
    emb_h = model.extract_embeddings(healthy_X_norm, "healthy", device, batch_size)

    def _drift(emb):
        return np.linalg.norm(emb - emb.mean(axis=0), axis=1)   # (N,)

    def _consec_cosine(emb, boundaries):
        if boundaries is None:
            boundaries = [0]
        ends = list(boundaries[1:]) + [len(emb)]
        result = []
        for s, e in zip(boundaries, ends):
            sub = emb[s:e]
            if len(sub) < 2:
                continue
            nrm = sub / (np.linalg.norm(sub, axis=1, keepdims=True) + 1e-8)
            result.append((nrm[:-1] * nrm[1:]).sum(axis=1))   # (n-1,)
        return result

    def _subject_avg_smooth(values, boundaries, n_common, window=15):
        if boundaries is None:
            boundaries = [0]
        ends   = list(boundaries[1:]) + [len(values)]
        curves = [values[s:e] for s, e in zip(boundaries, ends) if e > s]
        n_common = min(n_common, min(len(c) for c in curves))
        interp = []
        for c in curves:
            x_old = np.linspace(0, 1, len(c))
            x_new = np.linspace(0, 1, n_common)
            interp.append(np.interp(x_new, x_old, c))
        stacked = np.stack(interp)   # (n_subjects, n_common)
        # smooth each subject's curve
        def _smooth(x, w):
            return np.convolve(x, np.ones(w) / w, mode="same")
        smoothed = np.stack([_smooth(c, window) for c in stacked])
        return smoothed.mean(0), smoothed.std(0), n_common

    drift_p = _drift(emb_p)
    drift_h = _drift(emb_h)
    mean_drift_p, std_drift_p, nc_p = _subject_avg_smooth(drift_p, p_boundaries, n_common)
    mean_drift_h, std_drift_h, nc_h = _subject_avg_smooth(drift_h, h_boundaries, n_common)
    x_p = _shifted_trial_axis(trial_ids, nc_p)
    x_h = _shifted_trial_axis(trial_ids, nc_h)

    cos_p_list = _consec_cosine(emb_p, p_boundaries)
    cos_h_list = _consec_cosine(emb_h, h_boundaries)
    cos_p_flat = np.concatenate(cos_p_list) if cos_p_list else np.array([0.0])
    cos_h_flat = np.concatenate(cos_h_list) if cos_h_list else np.array([0.0])

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Left: embedding drift
    axes[0].plot(x_p, mean_drift_p, lw=2, color="C0", label="Patient (mean)")
    axes[0].fill_between(x_p, mean_drift_p - std_drift_p, mean_drift_p + std_drift_p,
                         alpha=0.2, color="C0", label="Patient ±1 SD")
    axes[0].plot(x_h, mean_drift_h, lw=2, color="C1", label="Healthy (mean)")
    axes[0].fill_between(x_h, mean_drift_h - std_drift_h, mean_drift_h + std_drift_h,
                         alpha=0.2, color="C1", label="Healthy ±1 SD")
    axes[0].set_xlabel("Within-subject trial index (first retained trial = 0)")
    axes[0].set_ylabel("L2 distance from group mean embedding")
    axes[0].set_title("Q2 (Embedding): Embedding Drift Across Trials\n"
                      "(group mean ± 1 SD, smoothed)")
    axes[0].legend(fontsize=8)

    # Right: consecutive cosine similarity violin
    parts = axes[1].violinplot([cos_p_flat, cos_h_flat], positions=[0, 1],
                               showmedians=True, showextrema=True)
    parts["bodies"][0].set_facecolor("C0"); parts["bodies"][0].set_alpha(0.5)
    parts["bodies"][1].set_facecolor("C1"); parts["bodies"][1].set_alpha(0.5)
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["Patient", "Healthy"])
    axes[1].set_ylabel("Cosine similarity (consecutive trials within subject)")
    axes[1].set_title("Q2: Trial-to-Trial Embedding Stability\n"
                      f"Patient median={np.median(cos_p_flat):.3f}   "
                      f"Healthy median={np.median(cos_h_flat):.3f}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def plot_emb_q3_pool_token_analysis(model, patient_X_norm, healthy_X_norm,
                                     device: str = "cpu", batch_size: int = 32,
                                     save_path: str = "xai_emb_q3_pool_token.png"):
    """
    Q3 (Embedding) — Pool token group-level analysis.

    The pool token (encoder output position 0) is the per-trial group summary —
    it attends to ALL JAW positions and all TT positions via self-attention.

    Left  : PCA of pool token vectors for all trials, coloured by group.
            Tight, well-separated clusters → clear group prototype.
    Right : Top-20 embedding dimensions that most discriminate patient vs healthy
            (ranked by |mean_patient − mean_healthy|), shown as bar ± SD.
    """
    seq_p = model.extract_encoder_sequence(patient_X_norm, "patient", device, batch_size)
    seq_h = model.extract_encoder_sequence(healthy_X_norm, "healthy", device, batch_size)

    pt_p = seq_p[:, 0, :]   # (N_p, d_model) — pool token outputs
    pt_h = seq_h[:, 0, :]   # (N_h, d_model)

    # PCA of all pool tokens
    all_pt = np.vstack([pt_p, pt_h])
    scaler = StandardScaler()
    all_pt_sc = scaler.fit_transform(all_pt)
    pca_pt = PCA(n_components=2)
    coords = pca_pt.fit_transform(all_pt_sc)
    coords_p = coords[: len(pt_p)]
    coords_h = coords[len(pt_p) :]

    # Discriminative dimensions
    diff_abs  = np.abs(pt_p.mean(0) - pt_h.mean(0))
    top_dims  = np.argsort(diff_abs)[-20:][::-1]
    mean_p_d  = pt_p.mean(0)[top_dims];  std_p_d = pt_p.std(0)[top_dims]
    mean_h_d  = pt_h.mean(0)[top_dims];  std_h_d = pt_h.std(0)[top_dims]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: PCA scatter
    axes[0].scatter(coords_p[:, 0], coords_p[:, 1], s=10, alpha=0.4,
                    color="C0", label=f"Patient (N={len(pt_p)})")
    axes[0].scatter(coords_h[:, 0], coords_h[:, 1], s=10, alpha=0.4,
                    color="C1", label=f"Healthy (N={len(pt_h)})")
    # Group centroids
    for coords_g, color in [(coords_p, "C0"), (coords_h, "C1")]:
        axes[0].scatter(*coords_g.mean(0), marker="*", s=250,
                        color=color, edgecolors="k", linewidths=0.8, zorder=6)
    axes[0].set_xlabel(f"PC1 ({pca_pt.explained_variance_ratio_[0]*100:.1f}%)")
    axes[0].set_ylabel(f"PC2 ({pca_pt.explained_variance_ratio_[1]*100:.1f}%)")
    axes[0].set_title("Q3 (Embedding): Pool Token PCA\n"
                      "(each point = one trial's group-summary vector;\n"
                      "★ = group centroid)")
    axes[0].legend(fontsize=8)

    # Right: top discriminative dims
    x = np.arange(len(top_dims))
    axes[1].bar(x - 0.2, mean_p_d, 0.38, yerr=std_p_d, color="C0",
                alpha=0.75, label="Patient", capsize=2, error_kw={"lw": 0.8})
    axes[1].bar(x + 0.2, mean_h_d, 0.38, yerr=std_h_d, color="C1",
                alpha=0.75, label="Healthy", capsize=2, error_kw={"lw": 0.8})
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(d) for d in top_dims], rotation=45, fontsize=7)
    axes[1].set_xlabel("Embedding dimension index\n(top-20 by |mean patient − healthy|)")
    axes[1].set_ylabel("Pool token value (mean ± 1 SD)")
    axes[1].set_title("Q3: Most Discriminative Dimensions\nin Pool Token Embeddings")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def plot_emb_q4_jaw_attention(model, patient_X_norm, healthy_X_norm,
                               device: str = "cpu", batch_size: int = 16,
                               fs: int = 250,
                               save_path: str = "xai_emb_q4_jaw_attention.png"):
    """
    Q4 (Embedding) — JAW→TT attention distribution in a general trial.

    Two complementary views:

    Left  : Mean cross-attention heatmap (last encoder layer, averaged over all
            trials per group).  Rows = TT positions + pool token (query),
            Cols = JAW positions (key/value).  Causal mask → upper triangle = 0.
            Bright columns → JAW time steps that TT attends to most.

    Right : Gradient attribution of the pool token embedding w.r.t. JAW input.
            |d‖pool_token‖/d JAW_channel[t]| averaged over trials, then summed
            over JAW channels → shows which JAW time steps have the strongest
            causal influence on the group summary regardless of attention sink.
            (Bypass attention-sink problem via gradients.)
    """
    # ── Left: mean cross-attention heatmap (last layer) ─────────────────────
    attn_p = model.extract_attention_maps(patient_X_norm, "patient", device, batch_size)
    attn_h = model.extract_attention_maps(healthy_X_norm, "healthy", device, batch_size)
    # (N, num_layers, T+1, T)

    mean_attn_p = attn_p[:, -1, :, :].mean(axis=0)    # last layer (T+1, T)
    mean_attn_h = attn_h[:, -1, :, :].mean(axis=0)

    T = mean_attn_p.shape[1]

    # Subsample for display (400 steps → 80 steps)
    stride = 5
    sub_p = mean_attn_p[::stride, ::stride]
    sub_h = mean_attn_h[::stride, ::stride]
    jaw_ticks_ms = np.arange(sub_p.shape[1]) * stride / fs * 1000.0
    tt_ticks_ms  = np.arange(sub_p.shape[0]) * stride / fs * 1000.0

    # ── Right: gradient attribution on JAW → pool token ─────────────────────
    def _gradient_attribution(X_norm, pool):
        """
        Returns mean |gradient| of pool-token L2 norm w.r.t. JAW input.
        Shape: (T,) — summed over 9 JAW channels, averaged over trials.
        """
        model.eval()
        tt_t  = torch.tensor(X_norm[:, TT_CH,  :], dtype=torch.float32, device=device)
        jaw_t = torch.tensor(X_norm[:, JAW_CH, :], dtype=torch.float32, device=device)
        jaw_t.requires_grad_(True)

        encoder = model.patient_encoder if pool == "patient" else model.healthy_encoder
        enc, _  = encoder(tt_t, jaw_t)         # (N, T+1, d)
        pool_tok = enc[:, 0, :]                 # (N, d)
        loss = pool_tok.norm(dim=1).sum()
        loss.backward()

        grad = jaw_t.grad.detach().abs().cpu().numpy()   # (N, 9, T)
        return grad.mean(axis=0).sum(axis=0)              # (T,) mean over trials, sum over channels

    grad_p = _gradient_attribution(patient_X_norm, "patient")   # (T,)
    grad_h = _gradient_attribution(healthy_X_norm, "healthy")   # (T,)
    time_ms = np.arange(T) / fs * 1000.0

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    vmax = max(sub_p.max(), sub_h.max())
    for ax, data, label in [(axes[0, 0], sub_p, "Patient"),
                             (axes[0, 1], sub_h, "Healthy")]:
        im = ax.imshow(data, aspect="auto", origin="upper",
                       cmap="hot", vmin=0, vmax=vmax)
        plt.colorbar(im, ax=ax, label="Mean attention weight (last layer)")
        # x-axis: JAW time
        n_xticks = 6
        xi = np.linspace(0, sub_p.shape[1] - 1, n_xticks, dtype=int)
        ax.set_xticks(xi)
        ax.set_xticklabels([f"{jaw_ticks_ms[i]:.0f}" for i in xi])
        # y-axis: TT + pool token
        n_yticks = 6
        yi = np.linspace(0, sub_p.shape[0] - 1, n_yticks, dtype=int)
        ax.set_yticks(yi)
        ax.set_yticklabels(["Pool" if i == 0 else f"{tt_ticks_ms[i]:.0f}" for i in yi])
        ax.set_xlabel("JAW time (ms)")
        ax.set_ylabel("TT / pool token position (ms)")
        ax.set_title(f"Q4a: Mean JAW→TT Cross-Attention\n{label} (last encoder layer)")

    # Bottom: gradient attribution
    for ax, grad, color, label in [(axes[1, 0], grad_p, "C0", "Patient"),
                                    (axes[1, 1], grad_h, "C1", "Healthy")]:
        ax.plot(time_ms, grad, lw=1.2, color=color)
        ax.fill_between(time_ms, 0, grad, alpha=0.3, color=color)
        peak_t = int(np.argmax(grad))
        ax.axvline(time_ms[peak_t], color="red", lw=1.2, ls="--",
                   label=f"Peak at {time_ms[peak_t]:.0f} ms")
        ax.set_xlabel("JAW time (ms)")
        ax.set_ylabel("|Gradient| of pool token\n(summed over JAW channels)")
        ax.set_title(f"Q4b: JAW Gradient Attribution → Pool Token\n{label}")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


# ── Signal-level PCA dimensionality analysis ───────────────────────────────────

def plot_signal_pca_dimensionality(
    patient_X_norm,
    healthy_X_norm,
    patient_boundaries,
    healthy_boundaries,
    variance_thresholds=(0.80, 0.90, 0.95),
    channels=None,
    save_path: str = "signal_pca_dimensionality.png",
):
    """
    Intrinsic dimensionality analysis (collaborator suggestion).

    For each subject, reshape their trials into [N_trials*T, n_channels],
    run PCA, and record how many PCs are needed to exceed each variance
    threshold (e.g. 80 / 90 / 95 %).

    Hypothesis: If OC patients show *structured reorganisation* rather than
    simple added noise, their movement space may occupy fewer effective
    dimensions (lower intrinsic dimensionality) compared to healthy controls.
    This would NOT be captured by STI-style variability measures.

    Parameters
    ----------
    patient_X_norm  : (N_p, 18, T) float32 — all patient trials, normalised
    healthy_X_norm  : (N_h, 18, T) float32 — all healthy trials, normalised
    patient_boundaries  : list[int] — trial index where each patient starts
    healthy_boundaries  : list[int] — trial index where each healthy starts
    variance_thresholds : iterable of floats in (0, 1)
    channels : slice or array-like or None
        Channel indices to include. None = all 18 channels.
        Use slice(9, 18) for TT-only, slice(0, 9) for JAW-only.
    save_path : output figure path
    """
    from scipy.stats import mannwhitneyu

    # Apply channel selection
    ch = channels if channels is not None else slice(None)
    patient_X = patient_X_norm[:, ch, :]
    healthy_X = healthy_X_norm[:, ch, :]
    n_ch = patient_X.shape[1]

    def _subject_n_pcs(X_all, boundaries, threshold):
        """
        Returns list of n_PCs (one per subject) needed to exceed `threshold`
        of cumulative explained variance.
        """
        ends = list(boundaries[1:]) + [X_all.shape[0]]
        n_pcs_list = []
        for s, e in zip(boundaries, ends):
            if e - s < 2:          # skip subjects with < 2 trials
                continue
            X_subj = X_all[s:e]    # (n_trials, n_ch, T)
            n_trials, n_ch_s, T = X_subj.shape
            # reshape to [n_trials*T, n_ch] — each time point is one observation
            mat = X_subj.transpose(0, 2, 1).reshape(-1, n_ch_s)
            pca = PCA(n_components=n_ch_s)
            pca.fit(mat)
            cumvar = np.cumsum(pca.explained_variance_ratio_)
            n_pc = int(np.searchsorted(cumvar, threshold) + 1)  # 1-indexed
            n_pcs_list.append(n_pc)
        return n_pcs_list

    n_thresh = len(variance_thresholds)
    fig, axes = plt.subplots(1, n_thresh + 1, figsize=(5 * (n_thresh + 1), 5))

    all_p_results, all_h_results = [], []

    for ax, thr in zip(axes[:n_thresh], variance_thresholds):
        n_pcs_p = _subject_n_pcs(patient_X, patient_boundaries, thr)
        n_pcs_h = _subject_n_pcs(healthy_X, healthy_boundaries, thr)
        all_p_results.append(n_pcs_p)
        all_h_results.append(n_pcs_h)

        stat, pval = mannwhitneyu(n_pcs_p, n_pcs_h, alternative="two-sided")

        bp = ax.boxplot(
            [n_pcs_p, n_pcs_h],
            labels=["Patient", "Healthy"],
            patch_artist=True,
            widths=0.5,
            medianprops={"color": "k", "lw": 2},
        )
        for patch, color in zip(bp["boxes"], ["C0", "C1"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        # overlay individual subject dots with jitter
        rng = np.random.default_rng(42)
        for xi, vals in enumerate([n_pcs_p, n_pcs_h], start=1):
            jitter = rng.uniform(-0.12, 0.12, len(vals))
            ax.scatter(xi + jitter, vals, s=40, zorder=5,
                       color=["C0", "C1"][xi - 1], edgecolors="k", linewidths=0.5)

        sig_label = (
            "***" if pval < 0.001 else
            "**"  if pval < 0.01  else
            "*"   if pval < 0.05  else
            "n.s."
        )
        y_top = ax.get_ylim()[1]
        ax.annotate(
            f"U={stat:.0f}\np={pval:.3f} {sig_label}",
            xy=(1.5, y_top * 0.97),
            ha="center", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.8),
        )
        ax.set_ylabel("# PCs needed")
        ax.set_title(f"PCs for {int(thr*100)}% variance\n"
                     f"(mean: P={np.mean(n_pcs_p):.1f}, H={np.mean(n_pcs_h):.1f})")
        ax.set_ylim(bottom=0)

    # ── Right panel: cumulative explained variance curves per group ───────────
    ax_cv = axes[-1]
    def _mean_cumvar_curve(X_all, boundaries):
        ends = list(boundaries[1:]) + [X_all.shape[0]]
        curves = []
        for s, e in zip(boundaries, ends):
            if e - s < 2:
                continue
            X_subj = X_all[s:e]
            n_trials, n_ch_s, T = X_subj.shape
            mat = X_subj.transpose(0, 2, 1).reshape(-1, n_ch_s)
            pca = PCA(n_components=n_ch_s)
            pca.fit(mat)
            curves.append(np.cumsum(pca.explained_variance_ratio_))
        return np.array(curves)   # (n_subjects, n_ch)

    curves_p = _mean_cumvar_curve(patient_X, patient_boundaries)
    curves_h = _mean_cumvar_curve(healthy_X, healthy_boundaries)

    x_axis = np.arange(1, n_ch + 1)
    for label, curves, color in [("Patient", curves_p, "C0"),
                                  ("Healthy", curves_h, "C1")]:
        mean_cv = curves.mean(0)
        std_cv  = curves.std(0)
        ax_cv.plot(x_axis, mean_cv, color=color, lw=2, label=f"{label} (n={len(curves)})")
        ax_cv.fill_between(x_axis, mean_cv - std_cv, mean_cv + std_cv,
                           color=color, alpha=0.15)

    for thr in variance_thresholds:
        ax_cv.axhline(thr, color="gray", lw=0.8, ls="--", alpha=0.6)
        ax_cv.text(n_ch + 0.1, thr, f"{int(thr*100)}%", va="center", fontsize=7, color="gray")

    ax_cv.set_xlabel("Number of principal components")
    ax_cv.set_ylabel("Cumulative explained variance")
    ax_cv.set_title("Mean cumulative variance\n(shaded = ±1 SD across subjects)")
    ax_cv.set_xlim(1, n_ch)
    ax_cv.set_ylim(0, 1.05)
    ax_cv.legend(fontsize=9)

    plt.suptitle("Signal-Space Intrinsic Dimensionality\n"
                 "(lower # PCs = more constrained / reorganised movement)",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n── PCA Dimensionality Summary ──────────────────────────────────────")
    for thr, n_pcs_p, n_pcs_h in zip(variance_thresholds, all_p_results, all_h_results):
        stat, pval = mannwhitneyu(n_pcs_p, n_pcs_h, alternative="two-sided")
        print(f"  {int(thr*100)}% var  →  "
              f"Patient: {np.mean(n_pcs_p):.1f}±{np.std(n_pcs_p):.1f}  "
              f"Healthy: {np.mean(n_pcs_h):.1f}±{np.std(n_pcs_h):.1f}  "
              f"U={stat:.0f} p={pval:.3f}")
    print("────────────────────────────────────────────────────────────────────\n")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SAVE_DIR  = "/home/liu0193/Data/Ziming/EMA/new_multi_processed"
    SEQ_LEN   = 400
    if not torch.cuda.is_available():
        raise RuntimeError("GPU requested but CUDA is not available.")
    DEVICE = "cuda"
    print("Device:", DEVICE)

    # ── Load multi-subject data aligned by true within-subject trial ID ────────
    # Keep only trial IDs present in every subject across both pools. Trial names
    # like T0003 and T7_T3_SUed_ are both parsed as within-subject trial 3.
    aligned_groups, common_trial_ids = load_trial_aligned_groups(
        SAVE_DIR,
        group_names=("patient", "healthy"),
    )
    patient_X, patient_subject_ids, patient_boundaries = aligned_groups["patient"]
    healthy_X, healthy_subject_ids, healthy_boundaries = aligned_groups["healthy"]

    print(f"\nAligned trial IDs: {len(common_trial_ids)} common trials")
    print(f"Aligned Patient: {patient_X.shape}  ({len(set(patient_subject_ids))} subjects)")
    print(f"Aligned Healthy: {healthy_X.shape}  ({len(set(healthy_subject_ids))} subjects)")

    # ── Normalise (joint, per-channel) ─────────────────────────────────────────
    X_all  = np.concatenate([patient_X, healthy_X], axis=0)
    mean, std = fit_channelwise_normalizer(X_all)
    patient_X_norm = apply_channelwise_normalizer(patient_X, mean, std)
    healthy_X_norm = apply_channelwise_normalizer(healthy_X, mean, std)

    # ── DataLoaders (shuffle=True for training only) ───────────────────────────
    patient_loader = DataLoader(PoolDataset(patient_X_norm), batch_size=16,
                                shuffle=True, drop_last=False)
    healthy_loader = DataLoader(PoolDataset(healthy_X_norm), batch_size=16,
                                shuffle=True, drop_last=False)

    # ── Model ──────────────────────────────────────────────────────────────────
    model = DualPoolModel(
        in_channels=9,
        d_model=128,
        nhead=4,
        num_enc_layers=4,
        num_dec_layers=4,
        dim_ff=256,
        dropout=0.1,
        seq_len=SEQ_LEN,
        embed_dim=64,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # ── Train ──────────────────────────────────────────────────────────────────
    model, losses = train_dual_pool(
        model, patient_loader, healthy_loader,
        device=DEVICE, lr=1e-4, epochs=60,
    )

    # Loss curve
    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Loss")
    plt.title("Training Loss")
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/loss_curve.png", dpi=300)
    plt.close()

    # ── Extract embeddings ─────────────────────────────────────────────────────
    patient_Z = model.extract_embeddings(patient_X_norm, pool="patient", device=DEVICE)
    healthy_Z = model.extract_embeddings(healthy_X_norm, pool="healthy", device=DEVICE)
    print("Patient embeddings:", patient_Z.shape)
    print("Healthy embeddings:", healthy_Z.shape)

    # ── PCA + Plots ────────────────────────────────────────────────────────────
    patient_Z_pca, healthy_Z_pca, pca, _ = compute_joint_pca(patient_Z, healthy_Z)
    print("PCA explained variance:", pca.explained_variance_ratio_)

    plot_trajectory(patient_Z_pca, healthy_Z_pca,
                    p_boundaries=patient_boundaries,
                    h_boundaries=healthy_boundaries,
                    n_common=len(common_trial_ids),
                    save_path=f"{SAVE_DIR}/pca_trajectory.png")
    plot_pc1_over_trials(patient_Z_pca, healthy_Z_pca,
                         p_boundaries=patient_boundaries,
                         h_boundaries=healthy_boundaries,
                         n_common=len(common_trial_ids),
                         trial_ids=common_trial_ids,
                         save_path=f"{SAVE_DIR}/pc1_trend.png")

    print(summarize_trajectory(patient_Z_pca, name="Patient"))
    print(summarize_trajectory(healthy_Z_pca, name="Healthy"))

    # ── XAI: Cross-correlation analysis (directly on normalised signals) ─────────
    # NOTE: The four functions below take X_norm arrays (N, 18, T), NOT attention
    # maps. Cross-correlation is computed from JAW-y (ch 1) and TT-y (ch 10).

    # Q1 — Mean JAW–TT xcorr profile + patient vs healthy deviation
    print("\nComputing XAI Q1: cross-correlation profile...")
    plot_xcorr_profile(
        patient_X_norm, healthy_X_norm, fs=250, max_lag_ms=400.0,
        save_path=f"{SAVE_DIR}/xai_q1_xcorr_profile.png",
    )

    # Q2 — Peak lag across trials (temporal trend, group averages)
    print("Computing XAI Q2: peak lag across trials...")
    plot_peak_lag_across_trials(
        patient_X_norm, healthy_X_norm,
        p_boundaries=patient_boundaries,
        h_boundaries=healthy_boundaries,
        n_common=len(common_trial_ids),
        trial_ids=common_trial_ids,
        fs=250, max_lag_ms=400.0,
        save_path=f"{SAVE_DIR}/xai_q2_peak_lag_trials.png",
    )

    # Q3 — Group-level lag distribution + pool token vector comparison
    print("Computing XAI Q3: group-level pool token analysis...")
    plot_pool_token_xcorr(
        model, patient_X_norm, healthy_X_norm, fs=250, max_lag_ms=400.0,
        save_path=f"{SAVE_DIR}/xai_q3_pool_token.png",
    )

    # Q4 — Within-trial xcorr by segment (early / mid / late)
    print("Computing XAI Q4: within-trial segment coupling...")
    plot_xcorr_within_trial_segments(
        patient_X_norm, healthy_X_norm, fs=250, max_lag_ms=300.0,
        save_path=f"{SAVE_DIR}/xai_q4_xcorr_segments.png",
    )

    # ── XAI: Embedding-based analysis (model-derived) ─────────────────────────
    # These use encoder outputs / gradients directly from the trained model.

    # Emb-Q1 — Which time interval has the largest patient vs healthy difference?
    print("\nComputing Emb-XAI Q1: time-interval embedding difference...")
    plot_emb_q1_time_interval_diff(
        model, patient_X_norm, healthy_X_norm,
        device=DEVICE, fs=250,
        save_path=f"{SAVE_DIR}/xai_emb_q1_time_diff.png",
    )

    # Emb-Q2 — Do encoder representations change systematically across trials?
    print("Computing Emb-XAI Q2: trial dynamics in embedding space...")
    plot_emb_q2_trial_dynamics(
        model, patient_X_norm, healthy_X_norm,
        p_boundaries=patient_boundaries,
        h_boundaries=healthy_boundaries,
        n_common=len(common_trial_ids),
        trial_ids=common_trial_ids,
        device=DEVICE,
        save_path=f"{SAVE_DIR}/xai_emb_q2_trial_dynamics.png",
    )

    # Emb-Q3 — Pool token group-level embedding analysis
    print("Computing Emb-XAI Q3: pool token embedding analysis...")
    plot_emb_q3_pool_token_analysis(
        model, patient_X_norm, healthy_X_norm,
        device=DEVICE,
        save_path=f"{SAVE_DIR}/xai_emb_q3_pool_token.png",
    )

    # Emb-Q4 — JAW→TT cross-attention heatmap + gradient attribution
    print("Computing Emb-XAI Q4: JAW attention distribution...")
    plot_emb_q4_jaw_attention(
        model, patient_X_norm, healthy_X_norm,
        device=DEVICE, fs=250,
        save_path=f"{SAVE_DIR}/xai_emb_q4_jaw_attention.png",
    )

    # ── Signal-level PCA dimensionality (collaborator suggestion) ─────────────
    print("\nComputing signal-space intrinsic dimensionality (all 18 channels)...")
    plot_signal_pca_dimensionality(
        patient_X_norm, healthy_X_norm,
        patient_boundaries, healthy_boundaries,
        variance_thresholds=(0.80, 0.90, 0.95),
        channels=None,
        save_path=f"{SAVE_DIR}/signal_pca_dimensionality_all.png",
    )

    print("\nComputing signal-space intrinsic dimensionality (TT channels only)...")
    plot_signal_pca_dimensionality(
        patient_X_norm, healthy_X_norm,
        patient_boundaries, healthy_boundaries,
        variance_thresholds=(0.80, 0.90, 0.95),
        channels=slice(9, 18),   # TT: pos+vel+acc (9 channels)
        save_path=f"{SAVE_DIR}/signal_pca_dimensionality_tt_only.png",
    )
