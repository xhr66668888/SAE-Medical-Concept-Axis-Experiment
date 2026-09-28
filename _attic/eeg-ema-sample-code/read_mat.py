import scipy.io
import numpy as np
import os
import glob
from scipy.signal import resample


# ── Low-level signal utilities ─────────────────────────────────────────────────

def extract_signal_matrix(x):
    x = np.asarray(x)
    x = np.squeeze(x)
    if x.ndim == 1:
        x = x[:, None]
    elif x.ndim != 2:
        raise ValueError(f"Signal must be 1D or 2D, got shape {x.shape}")
    return x.astype(np.float32)


def compute_velocity_matrix(x, fs=250):
    x = extract_signal_matrix(x)
    v = np.gradient(x, 1.0 / fs, axis=0)
    return v.astype(np.float32)


def compute_acceleration_matrix(x, fs=250):
    x = extract_signal_matrix(x)
    v   = np.gradient(x,   1.0 / fs, axis=0)
    acc = np.gradient(v,   1.0 / fs, axis=0)
    return acc.astype(np.float32)


# ── Audio epoch detection ──────────────────────────────────────────────────────

def detect_speech_epoch(audio, audio_fs, ema_fs=250,
                         energy_thresh=0.02, min_dur_s=0.05, pad_s=0.02):
    """
    Detect word onset/offset from audio amplitude envelope using energy thresholding.

    Parameters
    ----------
    audio        : (T_audio, 1) or (T_audio,) array
    audio_fs     : audio sample rate (Hz)
    ema_fs       : EMA sample rate (Hz), used for index conversion
    energy_thresh: fraction of peak RMS to use as threshold (0–1)
    min_dur_s    : minimum valid epoch duration in seconds
    pad_s        : padding added to each side of the detected epoch (seconds)

    Returns
    -------
    onset_ema  : int  — onset sample index in EMA time
    offset_ema : int  — offset sample index in EMA time
    duration_s : float — epoch duration in seconds (after padding)
    """
    audio = np.squeeze(np.asarray(audio, dtype=np.float64))

    peak = np.abs(audio).max()
    if peak < 1e-8:
        raise ValueError("Audio signal is near-silent; cannot detect epoch")
    audio_norm = audio / peak

    # Short-window RMS envelope (20 ms window)
    win = max(1, int(0.02 * audio_fs))
    energy = np.convolve(audio_norm ** 2, np.ones(win) / win, mode="same")
    energy = np.sqrt(np.maximum(energy, 0.0))

    above = energy > energy_thresh
    idx   = np.where(above)[0]
    if len(idx) == 0:
        raise ValueError(
            f"No speech detected above threshold {energy_thresh}. "
            f"Peak energy: {energy.max():.4f}"
        )

    pad_samples   = int(pad_s * audio_fs)
    onset_audio   = max(0,               idx[0]  - pad_samples)
    offset_audio  = min(len(audio) - 1,  idx[-1] + pad_samples)

    if (offset_audio - onset_audio) < int(min_dur_s * audio_fs):
        raise ValueError(
            f"Detected epoch too short: "
            f"{(offset_audio - onset_audio) / audio_fs:.3f}s < {min_dur_s}s"
        )

    # Convert to EMA sample indices
    onset_ema  = int(onset_audio  / audio_fs * ema_fs)
    offset_ema = int(offset_audio / audio_fs * ema_fs)
    duration_s = (offset_audio - onset_audio) / audio_fs

    return onset_ema, offset_ema, duration_s


# ── Trial-level loading ────────────────────────────────────────────────────────

def build_trial_dict_from_mat(mat_path, trial_key, fs=250,
                               use_audio_epoch=False, energy_thresh=0.02):
    """
    Load one trial from a .mat file.

    If use_audio_epoch=True, the EMA signals are sliced to the acoustically
    detected word onset/offset before any further processing.

    Returns a dict with keys:
        JAW, JAW_v, JAW_acc, TT, TT_v, TT_acc  — (T_epoch, 3) each
        duration_s  — epoch duration in seconds
        onset_ema   — onset sample in EMA time (0 when use_audio_epoch=False)
        offset_ema  — offset sample in EMA time
    """
    mat_data   = scipy.io.loadmat(mat_path)
    trial_data = mat_data[trial_key][0]

    tt_full  = extract_signal_matrix(trial_data[1]["SIGNAL"][:, :3])
    jaw_full = extract_signal_matrix(trial_data[2]["SIGNAL"][:, :3])

    if use_audio_epoch:
        audio_entry = trial_data[0]
        audio_fs    = int(audio_entry[1][0, 0])
        audio       = extract_signal_matrix(audio_entry[2])
        onset_ema, offset_ema, duration_s = detect_speech_epoch(
            audio, audio_fs, ema_fs=fs, energy_thresh=energy_thresh
        )
        # Clamp to EMA signal bounds
        onset_ema  = max(0,              onset_ema)
        offset_ema = min(len(tt_full),   offset_ema)
        if onset_ema >= offset_ema:
            raise ValueError(
                f"Audio epoch [{onset_ema}, {offset_ema}] is invalid after clamping "
                f"(EMA length={len(tt_full)})"
            )
    else:
        onset_ema  = 0
        offset_ema = len(tt_full)
        duration_s = len(tt_full) / fs

    tt  = tt_full [onset_ema:offset_ema]
    jaw = jaw_full[onset_ema:offset_ema]

    return {
        "JAW":       jaw,
        "JAW_v":     compute_velocity_matrix(jaw,  fs=fs),
        "JAW_acc":   compute_acceleration_matrix(jaw, fs=fs),
        "TT":        tt,
        "TT_v":      compute_velocity_matrix(tt,   fs=fs),
        "TT_acc":    compute_acceleration_matrix(tt,  fs=fs),
        "duration_s": float(duration_s),
        "onset_ema":  int(onset_ema),
        "offset_ema": int(offset_ema),
    }


def resample_trial_matrix(X, target_len=400):
    """X: (T, F)  →  (target_len, F)"""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D matrix (T, F), got {X.shape}")
    if X.shape[0] == target_len:
        return X
    return resample(X, target_len, axis=0).astype(np.float32)


def trial_dict_to_matrix(trial, target_len=400):
    """
    Concatenates JAW + TT (pos, vel, acc) into a single matrix.

    Channel layout in the returned (18, target_len) array:
      0-2  : JAW position  (x, y, z)
      3-5  : JAW velocity
      6-8  : JAW acceleration
      9-11 : TT  position  (x, y, z)
      12-14: TT  velocity
      15-17: TT  acceleration
    """
    X = np.concatenate([
        trial["JAW"],
        trial["JAW_v"],
        trial["JAW_acc"],
        trial["TT"],
        trial["TT_v"],
        trial["TT_acc"],
    ], axis=1)                                     # (T_epoch, 18)
    X = resample_trial_matrix(X, target_len)       # (target_len, 18)
    return X.T.astype(np.float32)                  # (18, target_len)


# ── Subject-level loading ──────────────────────────────────────────────────────

def load_subject_tensor(subject_folder, fs=250, target_len=400,
                         use_audio_epoch=False, energy_thresh=0.02):
    """
    Load all trials from one subject folder (*.mat files, sorted by filename).

    Returns
    -------
    X            : (N_trials, 18, target_len)  — trials in file-sort order
    trial_names  : list[str]                   — one name per trial
    durations_s  : list[float]                 — epoch duration in seconds per trial
    """
    trial_files = sorted(glob.glob(os.path.join(subject_folder, "*.mat")))
    if not trial_files:
        raise ValueError(f"No .mat files found in {subject_folder}")

    X_list, trial_names, durations_s = [], [], []
    for mat_path in trial_files:
        trial_key = os.path.splitext(os.path.basename(mat_path))[0]
        try:
            trial = build_trial_dict_from_mat(
                mat_path, trial_key, fs=fs,
                use_audio_epoch=use_audio_epoch,
                energy_thresh=energy_thresh,
            )
            X_list.append(trial_dict_to_matrix(trial, target_len))
            trial_names.append(trial_key)
            durations_s.append(trial["duration_s"])
        except Exception as e:
            print(f"  Skipping {trial_key}: {e}")

    if not X_list:
        raise ValueError(f"No valid trials loaded from {subject_folder}")

    return (
        np.stack(X_list, axis=0),   # (N, 18, T)
        trial_names,                 # list[str]
        durations_s,                 # list[float]
    )


# ── Group-level loading (multiple subjects) ────────────────────────────────────

def load_group_subjects(group_folder, fs=250, target_len=400,
                         use_audio_epoch=False, energy_thresh=0.02):
    """
    Discover all subject sub-folders inside group_folder and load each one.

    Folder structure expected:
        group_folder/
            subject-A-MAT/
                trial_001.mat
                trial_002.mat
                ...
            subject-B-MAT/
                ...

    Returns
    -------
    subjects : dict  { subject_name (str) : (X, trial_names, durations_s) }
               X is (N_trials, 18, target_len), trials in temporal order.
    """
    subject_dirs = sorted([
        d for d in glob.glob(os.path.join(group_folder, "*"))
        if os.path.isdir(d)
    ])
    if not subject_dirs:
        raise ValueError(f"No sub-folders found in {group_folder}")

    subjects = {}
    for subject_dir in subject_dirs:
        subject_name = os.path.basename(subject_dir)
        print(f"  Loading subject: {subject_name} ...", end=" ")
        try:
            X, trial_names, durations_s = load_subject_tensor(
                subject_dir, fs=fs, target_len=target_len,
                use_audio_epoch=use_audio_epoch, energy_thresh=energy_thresh,
            )
            subjects[subject_name] = (X, trial_names, durations_s)
            print(f"{X.shape[0]} trials loaded.")
        except Exception as e:
            print(f"FAILED: {e}")

    return subjects


# ── Save / load helpers ────────────────────────────────────────────────────────

def save_subject_data(save_path, X, trial_names, durations_s=None):
    """Save one subject's tensor to a compressed .npz file."""
    kwargs = dict(
        X=X,
        trial_names=np.array(trial_names, dtype=object),
    )
    if durations_s is not None:
        kwargs["durations_s"] = np.array(durations_s, dtype=np.float32)
    np.savez_compressed(save_path, **kwargs)
    print(f"  Saved: {save_path}  {X.shape}")


def load_saved_subject_data(save_path):
    """
    Load one subject's tensor from a .npz file.

    Returns
    -------
    X           : (N, 18, T)
    trial_names : list[str]
    durations_s : list[float] or None (if file was saved without durations)
    """
    data = np.load(save_path, allow_pickle=True)
    durations_s = (
        data["durations_s"].tolist()
        if "durations_s" in data
        else None
    )
    return data["X"], data["trial_names"].tolist(), durations_s


def save_group_data(save_dir, group_name, subjects):
    """
    Save every subject in `subjects` dict to:
        {save_dir}/{group_name}/{subject_name}.npz

    Parameters
    ----------
    save_dir   : root directory for processed data
    group_name : "patient" or "healthy"
    subjects   : dict returned by load_group_subjects()
    """
    out_dir = os.path.join(save_dir, group_name)
    os.makedirs(out_dir, exist_ok=True)
    for subject_name, payload in subjects.items():
        X, trial_names = payload[0], payload[1]
        durations_s    = payload[2] if len(payload) > 2 else None
        save_path      = os.path.join(out_dir, f"{subject_name}.npz")
        save_subject_data(save_path, X, trial_names, durations_s)


def load_group_data(save_dir, group_name):
    """
    Reload all saved subjects for a group.

    Returns
    -------
    subjects : dict  { subject_name : (X, trial_names, durations_s) }
    """
    in_dir    = os.path.join(save_dir, group_name)
    npz_files = sorted(glob.glob(os.path.join(in_dir, "*.npz")))
    if not npz_files:
        raise ValueError(f"No .npz files found in {in_dir}")

    subjects = {}
    for npz_path in npz_files:
        subject_name           = os.path.splitext(os.path.basename(npz_path))[0]
        X, trial_names, durs   = load_saved_subject_data(npz_path)
        subjects[subject_name] = (X, trial_names, durs)
        print(f"  Loaded: {subject_name}  {X.shape}")
    return subjects


# ── Phase 3: group duration comparison ────────────────────────────────────────

def compare_group_durations(patient_subjects, healthy_subjects, plot=True):
    """
    Collect per-trial word durations from both groups and compare.

    Parameters
    ----------
    patient_subjects : dict from load_group_subjects() for cancer patients
    healthy_subjects : dict from load_group_subjects() for healthy controls

    Returns
    -------
    p_durs : np.ndarray  — all patient trial durations (seconds)
    h_durs : np.ndarray  — all healthy trial durations (seconds)
    stats  : dict        — mean, std, median for each group; Mann-Whitney U p-value
    """
    from scipy.stats import mannwhitneyu

    def collect(subjects):
        all_durs = []
        for _, payload in subjects.items():
            durs = payload[2] if len(payload) > 2 else None
            if durs:
                all_durs.extend(durs)
        return np.array(all_durs, dtype=np.float32)

    p_durs = collect(patient_subjects)
    h_durs = collect(healthy_subjects)

    stat, pval = mannwhitneyu(p_durs, h_durs, alternative="two-sided")
    stats = {
        "patient_mean":   float(p_durs.mean()),
        "patient_std":    float(p_durs.std()),
        "patient_median": float(np.median(p_durs)),
        "healthy_mean":   float(h_durs.mean()),
        "healthy_std":    float(h_durs.std()),
        "healthy_median": float(np.median(h_durs)),
        "mannwhitney_U":  float(stat),
        "p_value":        float(pval),
    }

    print("\n=== Word Duration Comparison (Phase 3) ===")
    print(f"  Cancer patients : {stats['patient_mean']:.3f} ± {stats['patient_std']:.3f} s "
          f"(median {stats['patient_median']:.3f} s,  n={len(p_durs)} trials)")
    print(f"  Healthy controls: {stats['healthy_mean']:.3f} ± {stats['healthy_std']:.3f} s "
          f"(median {stats['healthy_median']:.3f} s,  n={len(h_durs)} trials)")
    print(f"  Mann-Whitney U={stat:.1f}, p={pval:.4f}")

    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.boxplot(
            [p_durs, h_durs],
            labels=["Cancer\nPatients", "Healthy\nControls"],
            patch_artist=True,
            boxprops=dict(facecolor="#f4a582"),
            medianprops=dict(color="black", linewidth=2),
        )
        ax.set_ylabel("Word duration (s)")
        ax.set_title(
            f"Acoustically-defined epoch duration\n"
            f"Mann-Whitney p = {pval:.4f}"
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.savefig("duration_comparison.png", dpi=150)
        plt.show()
        print("  Saved: duration_comparison.png")

    return p_durs, h_durs, stats


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    fs            = 250
    target_len    = 400
    use_audio     = True      # set False to use fixed 1.6 s window as before
    energy_thresh = 0.02      # tune if needed (fraction of peak RMS)

    patient_group_folder = "Ziming/EMA/EMA-cancer-project-files-Ziming/cancer-patient-MAT-files"
    healthy_group_folder = "Ziming/EMA/EMA-cancer-project-files-Ziming/healthy-control-MAT-files"
    save_dir             = "Ziming/EMA/new_multi_processed"

    # ── Cancer patients ───────────────────────────────────────────────────────
    print("=== Loading cancer patients ===")
    patient_subjects = load_group_subjects(
        patient_group_folder, fs=fs, target_len=target_len,
        use_audio_epoch=use_audio, energy_thresh=energy_thresh,
    )
    save_group_data(save_dir, "patient", patient_subjects)

    print(f"\nPatient subjects loaded: {len(patient_subjects)}")
    for name, (X, _, durs) in patient_subjects.items():
        mean_dur = np.mean(durs) if durs else float("nan")
        print(f"  {name}: {X.shape[0]} trials,  mean duration {mean_dur:.3f} s")

    # ── Healthy controls ──────────────────────────────────────────────────────
    print("\n=== Loading healthy controls ===")
    healthy_subjects = load_group_subjects(
        healthy_group_folder, fs=fs, target_len=target_len,
        use_audio_epoch=use_audio, energy_thresh=energy_thresh,
    )
    save_group_data(save_dir, "healthy", healthy_subjects)

    print(f"\nHealthy subjects loaded: {len(healthy_subjects)}")
    for name, (X, _, durs) in healthy_subjects.items():
        mean_dur = np.mean(durs) if durs else float("nan")
        print(f"  {name}: {X.shape[0]} trials,  mean duration {mean_dur:.3f} s")

    # ── Phase 3: duration comparison ──────────────────────────────────────────
    if use_audio:
        compare_group_durations(patient_subjects, healthy_subjects, plot=True)
