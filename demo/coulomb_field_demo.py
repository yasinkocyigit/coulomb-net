"""Small educational Coulomb-field demo.

This standalone example is intentionally separate from the private research
implementation. It visualizes the superposition of two point-charge fields.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

K_COULOMB = 8.9875517923e9


def electric_field(charge: float, source: tuple[float, float], points: np.ndarray):
    displacement = points - np.asarray(source)
    distance = np.linalg.norm(displacement, axis=-1, keepdims=True)
    distance = np.maximum(distance, 1e-6)
    return K_COULOMB * charge * displacement / distance**3


def main() -> None:
    axis = np.linspace(-2.5, 2.5, 80)
    x_grid, y_grid = np.meshgrid(axis, axis)
    points = np.stack((x_grid, y_grid), axis=-1)

    field = (
        electric_field(1.0e-9, (-0.8, 0.0), points)
        + electric_field(-1.0e-9, (0.8, 0.0), points)
    )
    magnitude = np.linalg.norm(field, axis=-1)

    figure, axes = plt.subplots(figsize=(8, 6))
    axes.streamplot(x_grid, y_grid, field[..., 0], field[..., 1], color=magnitude, cmap="viridis")
    axes.scatter([-0.8, 0.8], [0.0, 0.0], c=["tab:red", "tab:blue"], s=90, edgecolors="black")
    axes.set(title="Electric-field superposition demo", xlabel="x (m)", ylabel="y (m)")
    axes.set_aspect("equal")
    figure.tight_layout()
    output_path = Path(__file__).with_name("coulomb_field_demo.png")
    figure.savefig(output_path, dpi=160)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
