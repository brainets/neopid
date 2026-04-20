# SPID: Spectral Partial Information Decomposition

Spectral Partial Information Decomposition (SPID) combines frequency-domain
mutual information with Partial Information Decomposition (PID) to decompose
oscillatory interactions between brain regions into unique, redundant, and
synergistic components with respect to a cognitive variable. This repository
contains the code and notebooks accompanying the paper:

> *citation / DOI — to be added*

SPID is validated on Kuramoto and Stuart–Landau oscillator networks, extended
to a whole-brain model constrained by macaque anatomical connectivity (Markov
et al., 2014), and applied to local field potentials (LFPs) recorded from the
macaque fronto-parietal network during a working memory task.

---

## Repository layout

```
spid/
├── notebooks/          # Figure notebooks (Figure1–Figure5_6) + helpers
│   ├── Figure1.ipynb
│   ├── Figure2.ipynb
│   ├── Figure3.ipynb
│   ├── Figure4.ipynb
│   ├── Figure5_6.ipynb
│   ├── plot.py         # Shared plotting utilities
│   ├── session.py      # GrayLab session loader
│   └── run.sh          # Convenience wrapper
├── src/                # Shared Python modules
│   ├── models.py       # Kuramoto and Stuart–Landau simulators
│   ├── models_setup.py # Parameter setup and initialisation
│   └── utils.py        # RNG and general utilities
├── interareal/         # Macaque structural connectivity (Markov et al., 2014)
├── data/               # Additional connectivity data
├── figures/            # Output directory — PDFs written here by notebooks
├── paper/              # Manuscript PDF
└── README.md
```

---

## Installation

A Python 3.10 environment with the following packages is required:

```bash
pip install numpy scipy matplotlib xarray tqdm joblib scikit-learn \
            jax jaxlib mne frites hoi jupyterlab
```

---

## Reproducing the figures

Each `FigureXX.ipynb` notebook is a self-contained, deterministic pipeline.
Run all cells top-to-bottom; outputs are written to `figures/`.

| Notebook | Output | What it shows |
|---|---|---|
| `Figure1.ipynb` | `figures/Figure1.pdf` | Two-node Kuramoto model: coherence and MI as a function of coupling strength |
| `Figure2.ipynb` | `figures/Figure2.pdf` | Phase–amplitude encoding in a two-node Stuart–Landau model |
| `Figure3.ipynb` | `figures/Figure3.pdf` | PID decomposition (redundancy, synergy, unique) in a two-node oscillator |
| `Figure4.ipynb` | `figures/Figure4.pdf` | Whole-brain Stuart–Landau model: pairwise and higher-order (2- and 3-plet) PID |
| `Figure5_6.ipynb` | `figures/Figure5.pdf`, `figures/Figure6.pdf` | LFP recordings: pairwise and higher-order PID in macaque fronto-parietal cortex |

> **Note for Figure5_6.ipynb:** this notebook requires access to the GrayLab LFP
> dataset, which is not distributed with this repository. Set the path in cell 11
> before running.

---

## Source modules (`src/`)

| Module | Purpose |
|---|---|
| `models.py` | Euler–Maruyama simulation of Kuramoto and Stuart–Landau oscillator networks |
| `models_setup.py` | Parameter initialisation, delay matrices, and coupling profiles |
| `utils.py` | JAX-compatible random number generation utilities |

---

## Data

Structural connectivity (FLN matrix, area hierarchy) is derived from Markov et al.
(2014) and stored in `interareal/` and `data/`.
