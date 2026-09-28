import scipy.io
import numpy as np
import os
import glob
from scipy.signal import resample
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import linregress

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def fit_channelwise_normalizer(X_all):
    """
    X_all: (N, C, T)
    return:
        mean: (1, C, 1)
        std:  (1, C, 1)
    """
    mean = X_all.mean(axis=(0, 2), keepdims=True)
    std = X_all.std(axis=(0, 2), keepdims=True)
    std[std < 1e-8] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def apply_channelwise_normalizer(X, mean, std):
    return ((X - mean) / std).astype(np.float32)

class EMADataset(Dataset):
    def __init__(self, X):
        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        return x, x   # autoencoder: target = input
    
class ConvAutoencoder(nn.Module):
    def __init__(self, in_channels=18, seq_len=400, latent_dim=32):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, stride=2, padding=2),   # -> (32, 200)
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),            # -> (64, 100)
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),           # -> (128, 50)
            nn.ReLU(),
        )

        self.enc_out_len = seq_len
        for _ in range(3):
            self.enc_out_len = int(np.ceil(self.enc_out_len / 2))  # 400 -> 200 -> 100 -> 50

        self.flatten_dim = 128 * self.enc_out_len

        self.fc_enc = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.flatten_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),  # 50 -> 100
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),   # 100 -> 200
            nn.ReLU(),
            nn.ConvTranspose1d(32, in_channels, kernel_size=4, stride=2, padding=1),  # 200 -> 400
        )

        self.seq_len = seq_len
        self.latent_dim = latent_dim

    def encode(self, x):
        h = self.encoder(x)
        h = h.reshape(h.size(0), -1)
        z = self.fc_enc(h)
        return z

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.reshape(h.size(0), 128, self.enc_out_len)
        x_hat = self.decoder(h)

        if x_hat.shape[-1] > self.seq_len:
            x_hat = x_hat[:, :, :self.seq_len]
        elif x_hat.shape[-1] < self.seq_len:
            x_hat = nn.functional.interpolate(
                x_hat, size=self.seq_len, mode="linear", align_corners=False
            )
        return x_hat

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

def train_autoencoder(model, dataloader, device="cpu", lr=1e-3, epochs=60):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for x, target in dataloader:
            x = x.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            x_hat, z = model(x)
            loss = criterion(x_hat, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(dataloader.dataset)
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

    return model, losses

def extract_embeddings(model, X, device="cpu", batch_size=64):
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32)
    embeddings = []

    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size].to(device)
            z = model.encode(batch)
            embeddings.append(z.cpu().numpy())

    return np.concatenate(embeddings, axis=0)

def compute_joint_pca(Z1, Z2, n_components=2):
    Z_all = np.vstack([Z1, Z2])

    scaler = StandardScaler()
    Z_all_scaled = scaler.fit_transform(Z_all)

    pca = PCA(n_components=n_components)
    Z_all_pca = pca.fit_transform(Z_all_scaled)

    Z1_pca = Z_all_pca[:len(Z1)]
    Z2_pca = Z_all_pca[len(Z1):]

    return Z1_pca, Z2_pca, pca, scaler

def plot_trajectory(Z1, Z2, label1="Patient", label2="Healthy", title="Trial Trajectory in PCA Space"):
    plt.figure(figsize=(8, 6))

    plt.plot(Z1[:, 0], Z1[:, 1], '-o', markersize=3, linewidth=1, label=label1)
    plt.plot(Z2[:, 0], Z2[:, 1], '-o', markersize=3, linewidth=1, label=label2)

    plt.scatter(Z1[0, 0], Z1[0, 1], marker='s', s=80, label=f'{label1} Start')
    plt.scatter(Z1[-1, 0], Z1[-1, 1], marker='*', s=120, label=f'{label1} End')

    plt.scatter(Z2[0, 0], Z2[0, 1], marker='s', s=80, label=f'{label2} Start')
    plt.scatter(Z2[-1, 0], Z2[-1, 1], marker='*', s=120, label=f'{label2} End')

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Ziming/EMA/result/PCA_space.png", dpi=300)
    plt.close()

def plot_pc1_over_trials(Z1, Z2, label1="Patient", label2="Healthy"):
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(len(Z1)), Z1[:, 0], label=f"{label1} PC1")
    plt.plot(np.arange(len(Z2)), Z2[:, 0], label=f"{label2} PC1")

    plt.xlabel("Trial Index")
    plt.ylabel("PC1")
    plt.title("PC1 Trend Across Trials")
    plt.legend()
    plt.savefig("Ziming/EMA/result/PC1_Trend.png", dpi=300)
    plt.close()

def trajectory_length(Z):
    diffs = np.diff(Z, axis=0)
    step_lengths = np.linalg.norm(diffs, axis=1)
    return float(np.sum(step_lengths))


def trajectory_dispersion(Z):
    center = np.mean(Z, axis=0)
    dists = np.linalg.norm(Z - center, axis=1)
    return float(np.mean(dists))


def early_late_distance(Z, fraction=0.2):
    n = len(Z)
    k = max(1, int(n * fraction))
    early_center = np.mean(Z[:k], axis=0)
    late_center = np.mean(Z[-k:], axis=0)
    return float(np.linalg.norm(late_center - early_center))


def pc1_trend(Z):
    x = np.arange(len(Z))
    y = Z[:, 0]
    res = linregress(x, y)
    return {
        "slope": float(res.slope),
        "r": float(res.rvalue),
        "p": float(res.pvalue)
    }


def summarize_trajectory(Z, name="Subject"):
    trend = pc1_trend(Z)
    summary = {
        "name": name,
        "trajectory_length": trajectory_length(Z),
        "dispersion": trajectory_dispersion(Z),
        "early_late_distance": early_late_distance(Z),
        "pc1_slope": trend["slope"],
        "pc1_r": trend["r"],
        "pc1_p": trend["p"]
    }
    return summary


patient_data = np.load("Ziming/EMA/processed/patient_data.npz", allow_pickle=True)
patient_X = patient_data["X"]
patient_trial_names = patient_data["trial_names"].tolist()

healthy_data = np.load("Ziming/EMA/processed/healthy_data.npz", allow_pickle=True)
healthy_X = healthy_data["X"]
healthy_trial_names = healthy_data["trial_names"].tolist()


X_all = np.concatenate([patient_X, healthy_X], axis=0)   # (491, 18, 400)
mean, std = fit_channelwise_normalizer(X_all)

patient_X_norm = apply_channelwise_normalizer(patient_X, mean, std)
healthy_X_norm = apply_channelwise_normalizer(healthy_X, mean, std)

print("Normalized patient shape:", patient_X_norm.shape)
print("Normalized healthy shape:", healthy_X_norm.shape)

X_train = np.concatenate([patient_X_norm, healthy_X_norm], axis=0)
print("Training tensor shape:", X_train.shape)

dataset = EMADataset(X_train)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model = ConvAutoencoder(in_channels=18, seq_len=400, latent_dim=32)
model, losses = train_autoencoder(model, dataloader, device=device, lr=1e-3, epochs=60)

patient_Z = extract_embeddings(model, patient_X_norm, device=device)
healthy_Z = extract_embeddings(model, healthy_X_norm, device=device)

print("Patient embedding shape:", patient_Z.shape)
print("Healthy embedding shape:", healthy_Z.shape)

patient_Z_pca, healthy_Z_pca, pca, pca_scaler = compute_joint_pca(patient_Z, healthy_Z)

print("PCA explained variance ratio:", pca.explained_variance_ratio_)

plot_trajectory(patient_Z_pca, healthy_Z_pca, label1="Patient", label2="Healthy")

plot_pc1_over_trials(patient_Z_pca, healthy_Z_pca)

patient_summary = summarize_trajectory(patient_Z_pca, name="Patient")
healthy_summary = summarize_trajectory(healthy_Z_pca, name="Healthy")

print(patient_summary)
print(healthy_summary)


'''
plt.figure(figsize=(6, 4))
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Reconstruction Loss")
plt.title("Autoencoder Training Loss")
plt.tight_layout()

plt.savefig("Ziming/EMA/processed/ae_loss.png", dpi=300)
plt.close()
'''
