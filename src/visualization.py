# src/dataset.py

import numpy as np
import torch
import torch.utils.data as data
from sklearn.preprocessing import StandardScaler

K_COULOMB = 8.9875517923e9

class ThreeChargeDataset(data.Dataset):
    def __init__(self, num_samples, transform=None):
        self.X, self.y = self.generate_dataset(num_samples)
        self.scaler = transform or StandardScaler()
        self.X = self.scaler.fit_transform(self.X)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

    def generate_dataset(self, num_samples):
        positions = np.random.uniform(-10, 10, (num_samples, 3, 3))
        charges = np.random.uniform(-5, 5, (num_samples, 3))

        q1, q2, q_target = charges[:, 0:1], charges[:, 1:2], charges[:, 2:3]
        r1, r2, r_target = positions[:, 0], positions[:, 1], positions[:, 2]

        r_t1, r_t2 = r_target - r1, r_target - r2
        dist_t1_sq = np.sum(r_t1**2, axis=1, keepdims=True)
        dist_t2_sq = np.sum(r_t2**2, axis=1, keepdims=True)
        dist_t1_sq[dist_t1_sq < 1e-6] = 1e-6
        dist_t2_sq[dist_t2_sq < 1e-6] = 1e-6

        E1 = (K_COULOMB * q1 / dist_t1_sq) * (r_t1 / np.sqrt(dist_t1_sq))
        E2 = (K_COULOMB * q2 / dist_t2_sq) * (r_t2 / np.sqrt(dist_t2_sq))

        E_total = E1 + E2
        F_total = E_total * q_target

        X = np.hstack((r1, r2, r_target, E_total, F_total))
        y = np.hstack((q1, q2))
        return X, y


# Ensure this file only defines the dataset; no other imports from `src` should appear here to avoid circular imports.