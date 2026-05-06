# src/dataset.py
import numpy as np
import torch
import torch.utils.data as data
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple

K_COULOMB = 8.9875517923e9  # SI units (N·m²/C²)

def _to_float_or_none(x):
    """
    Convert YAML-loaded values (which may be strings) to float or None.
    Accepts scientific notation strings like '1e3', '1.0e-4', and 'null'/'None'.
    """
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        xs = x.strip().lower()
        if xs in {"", "none", "null"}:
            return None
        return float(xs)
    return float(x)


class ThreeChargeDataset(data.Dataset):
    """
    Two source charges (q1, q2) and one target point/charge (q_target).
    X (15): r1(3), r2(3), r_target(3), E_total(3), F_total(3)
    y (2):  q1, q2
    Filters: keep only samples with |E_total| >= e_min and |F_total| >= f_min (if set).
    Adaptive oversampling ensures requested num_samples after filtering.
    """

    def __init__(
        self,
        num_samples: int,
        transform: Optional[StandardScaler] = None,
        position_range: float = 10.0,
        charge_range: float = 5.0,
        e_min: Optional[float] = None,
        f_min: Optional[float] = None,
        oversample: int = 2,
        max_oversample: int = 64,
        seed: Optional[int] = None,
        min_distance: float = 1e-3,   # distance floor to avoid extreme fields (meters)
    ) -> None:

        # Robust to YAML strings
        e_min = _to_float_or_none(e_min)
        f_min = _to_float_or_none(f_min)

        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

        X, y, stats = self.generate_dataset(
            num_samples=num_samples,
            rng=rng,
            position_range=position_range,
            charge_range=charge_range,
            e_min=e_min,
            f_min=f_min,
            oversample=oversample,
            max_oversample=max_oversample,
            min_distance=min_distance,
        )

        self.scaler = transform or StandardScaler()
        self.X = self.scaler.fit_transform(X)
        self.y = y

        self.accept_rate = stats["accept_rate"]
        self.generated = stats["generated"]
        self.kept = stats["kept"]
        self.e_min = e_min
        self.f_min = f_min

        # Percentile report (pre-filter distribution snapshot)
        e_p = stats["E_percentiles"]
        f_p = stats["F_percentiles"]
        print(
            "[ThreeChargeDataset] |E_total| percentiles (V/m): "
            f"p01={e_p[0]:.3e}, p10={e_p[1]:.3e}, p50={e_p[2]:.3e}, p90={e_p[3]:.3e}, p99={e_p[4]:.3e}"
        )
        print(
            "[ThreeChargeDataset] |F_total| percentiles (N):   "
            f"p01={f_p[0]:.3e}, p10={f_p[1]:.3e}, p50={f_p[2]:.3e}, p90={f_p[3]:.3e}, p99={f_p[4]:.3e}"
        )
        print(
            f"[ThreeChargeDataset] kept {self.kept}/{self.generated} "
            f"({self.accept_rate*100:.1f}%) after thresholds (e_min={self.e_min}, f_min={self.f_min})."
        )

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )

    @staticmethod
    def feature_names():
        return [
            "r1_x", "r1_y", "r1_z",
            "r2_x", "r2_y", "r2_z",
            "r_target_x", "r_target_y", "r_target_z",
            "E_total_x", "E_total_y", "E_total_z",
            "F_total_x", "F_total_y", "F_total_z",
        ]

    @staticmethod
    def _sample_positions_and_charges(N: int, rng, position_range: float, charge_range: float):
        positions = rng.uniform(-position_range, position_range, size=(N, 3, 3))
        charges   = rng.uniform(-charge_range, charge_range,  size=(N, 3))
        r1, r2, r_target = positions[:, 0], positions[:, 1], positions[:, 2]
        q1 = charges[:, 0:1]
        q2 = charges[:, 1:2]
        q_target = charges[:, 2:3]
        return r1, r2, r_target, q1, q2, q_target

    @staticmethod
    def _compute_fields_and_forces(r1, r2, r_target, q1, q2, q_target, min_distance: float):
        # Vectors from charges to target
        r_t1 = r_target - r1
        r_t2 = r_target - r2

        # Distances squared with floor to avoid division by ~0
        dist_t1_sq = np.sum(r_t1**2, axis=1, keepdims=True)
        dist_t2_sq = np.sum(r_t2**2, axis=1, keepdims=True)
        floor = max(min_distance**2, 1e-12)
        dist_t1_sq = np.maximum(dist_t1_sq, floor)
        dist_t2_sq = np.maximum(dist_t2_sq, floor)

        # Coulomb fields: E = k q r_vec / r^3
        inv_r1 = 1.0 / np.sqrt(dist_t1_sq)
        inv_r2 = 1.0 / np.sqrt(dist_t2_sq)
        E1 = (K_COULOMB * q1 * inv_r1**3) * r_t1
        E2 = (K_COULOMB * q2 * inv_r2**3) * r_t2
        E_total = E1 + E2

        # Force on target charge
        F_total = E_total * q_target
        return E_total, F_total

    def generate_dataset(
        self,
        num_samples: int,
        rng,
        position_range: float,
        charge_range: float,
        e_min: Optional[float],
        f_min: Optional[float],
        oversample: int,
        max_oversample: int,
        min_distance: float,
    ):
        current_oversample = max(1, int(oversample))
        X_list, y_list = [], []
        generated_total = 0

        # First, one quick draw to estimate scale (for logging only)
        N_probe = max(4096, num_samples // 10)
        r1p, r2p, r_tp, q1p, q2p, q_tp = self._sample_positions_and_charges(N_probe, rng, position_range, charge_range)
        E_probe, F_probe = self._compute_fields_and_forces(r1p, r2p, r_tp, q1p, q2p, q_tp, min_distance)
        E_mag_p = np.linalg.norm(E_probe, axis=1)
        F_mag_p = np.linalg.norm(F_probe, axis=1)
        E_percentiles = np.percentile(E_mag_p, [1, 10, 50, 90, 99]).tolist()
        F_percentiles = np.percentile(F_mag_p, [1, 10, 50, 90, 99]).tolist()

        while True:
            N = num_samples * current_oversample
            r1, r2, r_target, q1, q2, q_target = self._sample_positions_and_charges(
                N, rng, position_range, charge_range
            )
            E_total, F_total = self._compute_fields_and_forces(r1, r2, r_target, q1, q2, q_target, min_distance)

            # Magnitudes for thresholding
            E_mag = np.linalg.norm(E_total, axis=1, keepdims=True)
            F_mag = np.linalg.norm(F_total, axis=1, keepdims=True)

            mask = np.ones((N, 1), dtype=bool)
            if e_min is not None:
                mask &= (E_mag >= e_min)
            if f_min is not None:
                mask &= (F_mag >= f_min)
            mask = mask.ravel()

            X = np.hstack((r1, r2, r_target, E_total, F_total))
            y = np.hstack((q1, q2))

            X_list.append(X[mask])
            y_list.append(y[mask])

            generated_total += N
            X_all = np.vstack(X_list) if X_list else np.empty((0, 15))
            y_all = np.vstack(y_list) if y_list else np.empty((0, 2))

            if len(X_all) >= num_samples:
                X_out = X_all[:num_samples]
                y_out = y_all[:num_samples]
                kept = len(X_out)
                accept_rate = kept / generated_total
                stats = {
                    "accept_rate": accept_rate,
                    "generated": generated_total,
                    "kept": kept,
                    "E_percentiles": E_percentiles,
                    "F_percentiles": F_percentiles,
                }
                return X_out, y_out, stats

            current_oversample *= 2
            if current_oversample > max_oversample:
                kept = len(X_all)
                accept_rate = kept / generated_total if generated_total > 0 else 0.0
                stats = {
                    "accept_rate": accept_rate,
                    "generated": generated_total,
                    "kept": kept,
                    "E_percentiles": E_percentiles,
                    "F_percentiles": F_percentiles,
                }
                print(
                    "[ThreeChargeDataset][WARN] returning fewer samples than requested: "
                    f"requested={num_samples}, kept={kept}. Consider relaxing thresholds or increasing ranges."
                )
                return X_all, y_all, stats
