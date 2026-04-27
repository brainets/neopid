# NeOPID: Neural Oscillatory Partial Information Decomposition

Neural Oscillatory Partial Information Decomposition (NeOPID) can be used to break down information about about cognitive variables in neural oscillations into power/phase contributions and to quantify redundant and synergistic information in brain relations, from pairwise to higher-order interactions.NeOPID
This repository contains the code and notebooks accompanying the paper:

> *citation / DOI — to be added*

We validated the approach on Kuramoto and Stuart–Landau oscillator networks, including a whole-brain model constrained by macaque anatomical connectivity. NeOPID accurately recovers ground-truth encoding schemes and reveals that phase relations and amplitude co-modulations act as complementary coding channels with both redundant and synergistic components. NeOPID further extends this decomposition to higher-order functional interactions, beyond pairwise relations, enabling the characterization of how cognitive information is collectively distributed across multiple oscillatory edges via redundant and synergistic encoding. To illustrate biological applicability, we applied NeOPID to local field potentials (LFPs) recorded from the macaque fronto-parietal network during a working memory task. In this dataset, NeOPID identified beta-band amplitude co-modulations as the primary carrier of stimulus information, and revealed that higher-order phase interactions exhibit both redundant and synergistic structure during the memory delay.

---

## Repository layout

```
neopid/
├── notebooks/          # Figure notebooks (Figure1–Figure6_7) + helpers
│   ├── Figure1.ipynb
│   ├── Figure2.ipynb
│   ├── Figure3.ipynb
│   ├── Figure4_5.ipynb
│   ├── Figure6_7.ipynb
│   ├── plot.py         # Shared plotting utilities
│   ├── session.py      # GrayLab session loader
│   └── run.sh          # Convenience wrapper
├── src/                # Shared Python modules
│   ├── models.py       # Kuramoto and Stuart–Landau simulators
│   ├── models_setup.py # Parameter setup and initialisation
│   ├── utils.py        # RNG and general utilities
│   └── flatmap/        # Macaque cortical flatmap plotting utilities
├── interareal/         # Macaque structural connectivity (Markov et al., 2014)
├── data/               # Additional connectivity data
├── figures/            # Output directory — PDFs written here by notebooks
├── paper/              # Manuscript PDF
├── LICENSE             # BSD 3-Clause
└── README.md
```

---

## Installation

A Python 3.10+ environment with the following packages is required:

```bash
pip install numpy scipy matplotlib xarray tqdm joblib scikit-learn \
            jax jaxlib mne frites hoi jupyterlab
```

---

## Reproducing the figures

Each notebook is a self-contained, deterministic pipeline.
Run all cells top-to-bottom; outputs are written to `figures/`.

| Notebook | Output | What it shows |
|---|---|---|
| `Figure1.ipynb` | `figures/Figure1.pdf` | Two-node Kuramoto model: coherence and MI as a function of coupling strength |
| `Figure2.ipynb` | `figures/Figure2.pdf` | Phase–amplitude encoding in a two-node Stuart–Landau model |
| `Figure3.ipynb` | `figures/Figure3.pdf` | PID decomposition (redundancy, synergy, unique) in a two-node oscillator |
| `Figure4_5.ipynb` | `figures/Figure4.pdf`, `figures/Figure5.pdf` | Whole-brain Stuart–Landau model: pairwise and higher-order (2- and 3-plet) PID |
| `Figure6_7.ipynb` | `figures/Figure6.pdf`, `figures/Figure7.pdf` | LFP recordings: pairwise and higher-order PID in macaque fronto-parietal cortex |

> **Note for Figure6_7.ipynb:** this notebook requires access to the GrayLab LFP
> dataset, which is not distributed with this repository. Set the data path
> before running.

---

## Source modules (`src/`)

| Module | Purpose |
|---|---|
| `models.py` | Euler–Maruyama simulation of Kuramoto and Stuart–Landau oscillator networks |
| `models_setup.py` | Parameter initialisation, delay matrices, and coupling profiles |
| `utils.py` | JAX-compatible random number generation utilities |
| `flatmap/` | Macaque cortical flatmap visualisation (area coordinates and outline plotting) |

---

## Data

Structural connectivity (FLN matrix, area hierarchy) is derived from Markov et al.
(2014) and stored in `interareal/` and `data/`.

---

## License

BSD 3-Clause. See [LICENSE](LICENSE) for details.
