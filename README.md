# SIEVE

Supplementary material for the paper **"Robust Spuriousness-Aware Group Discovery for Worst-Group Generalization"**.

SIEVE constructs a group-labeled validation set without any manual group labels, so that existing
methods can perform model selection for strong worst-group accuracy under spurious correlations.
It identifies a small set of near-boundary training examples and splits them into spurious and
non-spurious groups using their early-training loss dynamics; the resulting validation set replaces
group-labeled validation for model selection.

## Contents

| Path | Description |
|---|---|
| [`SIEVE-extended-version.pdf`](SIEVE-extended-version.pdf) | Extended version of the paper, with full dataset details, hyperparameter and hardware configurations, the complete sensitivity grid, runtime measurements, and additional analyses. |
| [`sieve/`](sieve/) | Code for the real-data benchmark experiments on six datasets (Waterbirds, CelebA, MetaShift, Dominoes, MultiNLI, CivilComments). |
| [`synthetic-experiments/`](synthetic-experiments/) | Notebooks reproducing the synthetic-data figures (the 2D main-text panels and the 3D / 4D appendix loss panels). |

Each code folder is self-contained, with its own environment and run instructions in its `README.md`.
