# Synthetic-data figures (SIEVE paper)

Reproduces the synthetic-data experiment figures in the SIEVE paper: the 2D Figure 1 panels,
the 3D appendix loss grid, and the 4D appendix loss panels. Each notebook is self-contained.
Open it and run all cells to write the figures to disk.

## Environment

Tested with Python 3.9.12 and `numpy` 1.26.4, `torch` 1.12.1, `matplotlib` 3.9.2.

## Notebooks

| Notebook | Run All produces |
|---|---|
| `synthetic_2d_data.ipynb` | `loss_dynamics.pdf`, `decision_boundary.pdf` |
| `synthetic_3d_loss_grid.ipynb` | `img_results/loss_p{0.9,0.8,0.7,0.5}_sigma{0.0,0.5,1.0,2.0}.pdf` (16) and `img_results/delta_conf_sigma{0.0,0.5,1.0,2.0}.pdf` (4) |
| `synthetic_4d_loss.ipynb` | `img_results-2-x4/loss_p23_0.8_s23_0.0_p4_0.8_s4_{0.0,0.5,1.0,2.0}.pdf` (4) |

The 2D `decision_boundary.pdf` is written un-annotated; the rotation arrow and the
"Epoch 1 / Epoch 30" labels in the published figure are added by hand.
