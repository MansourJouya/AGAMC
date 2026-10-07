# AGAMC — Adaptive Graph Augmentation for Multi-View Clustering

Official implementation of the paper
**"Adaptive Graph Augmentation for Multi-View Clustering: A Contrastive Learning Approach with View-Specific Augmentations"**
by Mansour Jouya and Alireza Abdollahpouri (Department of Computer Engineering, University of Kurdistan).

![Python](https://img.shields.io/badge/python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-ee4c2c)
![PyG](https://img.shields.io/badge/PyG-2.8.0-3c2179)
![License](https://img.shields.io/badge/license-MIT-green)

AGAMC clusters nodes of a multi-view graph with a shared GCN encoder trained by a multi-term contrastive loss. Its key idea is that **each view gets augmentations that suit its semantics** (a taxonomy of four view types: *sparse*, *dense*, *perceptual*, *structured*), chosen either by fixed rules or by a small learnable controller.

---

## Table of contents

1. [Highlights](#highlights)
2. [Repository structure](#repository-structure)
3. [Installation](#installation)
4. [Datasets](#datasets)
5. [Quick start](#quick-start)
6. [How configuration works](#how-configuration-works)
7. [Reproducing the paper's tables](#reproducing-the-papers-tables)
8. [Output files](#output-files)
9. [Visualisations](#visualisations)
10. [Hardware and reproducibility notes](#hardware-and-reproducibility-notes)
11. [Known notes and limitations](#known-notes-and-limitations)
12. [Troubleshooting](#troubleshooting)
13. [Citation](#citation)
14. [License](#license)

---

## Highlights

- **View-specific augmentation** driven by a view-type taxonomy (rule-based mode), or **learnable augmentation** via an MLP controller with straight-through estimators.
- **Multi-term contrastive loss**: intra-view (α), cross-view (β) and hybrid (γ) terms.
- **Two fusion modes** for the final embedding: plain average or attention fusion.
- **Optional automatic cluster-number estimation** (silhouette or eigengap).
- **One file, one config dict, no command-line arguments**: everything is controlled from `CONFIG` in `agamc.py`.
- **Six datasets** supported: Cora, CiteSeer, ACM, DBLP, Amazon, YouTube.
- **Built-in experiment suites** that regenerate the paper's analyses (Tables 4–15) and a full visual suite (t-SNE/UMAP, confusion matrix, attention weights, interactive 3D, …).

---

## Repository structure

```
AGAMC/
├── agamc.py                  # Main pipeline: data loading, augmentation, training, evaluation, all studies
├── download_data.py          # One-time downloader/preprocessor for ACM, DBLP, Amazon, YouTube
├── requirements.txt          # Core dependencies
├── requirements-optional.txt # Extras for some plots (umap-learn, plotly)
├── docs/
│   └── CONFIG_GUIDE.md       # Complete reference for every CONFIG option
├── CITATION.cff
├── LICENSE
└── README.md
```

Created at runtime (git-ignored): `data/` (datasets) and `outputs/` (results of each run).

---

## Installation

### Prerequisites

| Requirement | Notes |
|---|---|
| Python | 3.12 (the version used for the paper). Other recent 3.x versions will likely work. |
| PyTorch | 2.5.1. A GPU is optional but strongly recommended. |
| PyTorch Geometric | 2.8.0 |
| Internet access | Needed once, to download datasets. |
| Disk space | A few hundred MB for all datasets (the raw SNAP YouTube file is the largest). |

GPU memory: the paper's experiments ran on a 4 GB laptop GPU (Quadro T2000 Max-Q). If you run out of memory, lower `cl_batch_size` (see the [Config Guide](docs/CONFIG_GUIDE.md)).

### Step 1 — Clone

```bash
git clone https://github.com/<your-username>/AGAMC.git
cd AGAMC
```

### Step 2 — Create a virtual environment (recommended)

```bash
# Linux / macOS
python3.12 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

### Step 3 — Install PyTorch

Pick the command for your CUDA version from <https://pytorch.org/get-started/locally/>. For example (CUDA 12.1):

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

CPU-only:

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

### Step 4 — Install the remaining dependencies

```bash
pip install -r requirements.txt
```

### Step 5 (optional) — Extras for some visualisations

```bash
pip install -r requirements-optional.txt
```

These are only needed for the `tsne_umap` UMAP panel and the `interactive_3d` plots. If a package is missing, the script prints a warning and continues.

### Verify the installation

```bash
python -c "import torch, torch_geometric; print(torch.__version__, torch_geometric.__version__, torch.cuda.is_available())"
```

---

## Datasets

| Dataset | Classes | Views | Source | Preparation |
|---|---:|---:|---|---|
| `cora` | 7 | 4 | PyG `Planetoid` | **Automatic** on first run |
| `citeseer` | 6 | 4 | PyG `Planetoid` | **Automatic** on first run |
| `acm` | 3 | 3 | PyG `HGBDataset("ACM")` | `python download_data.py` |
| `dblp` | 4 | 2 | PyG `DBLP` | `python download_data.py` |
| `amazon` | 10 | 2 | PyG `Amazon("Computers")` | `python download_data.py` |
| `youtube` | 5 | 3 | SNAP `com-youtube` | `python download_data.py` |

**Cora and CiteSeer** need no preparation: they are downloaded into `data/<name>/` the first time you run `agamc.py`. Their four views are the citation graph (sparse), a kNN graph on content features (dense), a PPR-diffusion graph (perceptual) and a structured clone of the citation graph.

**ACM, DBLP, Amazon and YouTube** are prepared once with:

```bash
python download_data.py
```

This builds `data/acm/`, `data/dblp/`, `data/amazon/` and `data/youtube/` in the standardized file format that `agamc.py` expects (`features.npy`, `labels.npy`, `graph_<name>.npz`, `feat_<name>.npy`). If one dataset fails (e.g. a network error), the others are unaffected and you can simply re-run the script.

What each dataset contains:

- **ACM**: paper features; co-author and co-subject meta-path graphs; term kNN view.
- **DBLP**: author features; co-author meta-path graph; per-author term-usage kNN view (two views).
- **Amazon**: Computers co-purchase graph; PCA-256 product features for the kNN view.
- **YouTube**: real SNAP friendship graph and real SNAP ground-truth communities (5 communities, about 5,000 nodes).

> **Important — YouTube features.** No standard public release of per-node thumbnail-CNN or comment-TF-IDF features exists for the SNAP graph. `download_data.py` therefore creates *structural proxies* (a graph-spectral embedding and degree/clustering/triangle descriptors) and labels them as such. To use real content features, overwrite `data/youtube/feat_thumbnail_cnn.npy` and `data/youtube/feat_comment_tfidf.npy` with your own `(N, D)` float32 arrays.

> The YouTube step downloads a large SNAP file and parses it, which can take several minutes.

### Using your own dataset

Add a folder `data/<name>/` with the files below and register it in `DATASET_REGISTRY` in `agamc.py` (copy the `dblp` entry as a template):

| File | Shape / type | Purpose |
|---|---|---|
| `features.npy` | `(N, D)` float32 | Node features fed to the shared GCN encoder |
| `labels.npy` | `(N,)` int | Ground-truth classes (used **only** for reporting) |
| `graph_<name>.npz` | scipy sparse `(N, N)` | A view that is a real relation (co-author, co-purchase, …) |
| `feat_<name>.npy` | `(N, D_v)` float32 | A view built as a kNN graph over another feature space |

Each registry entry lists, in order, the views, their file, build type (`graph_file` or `knn_file`) and the view type tag (`sparse` / `dense` / `perceptual` / `structured`).

---

## Quick start

AGAMC has **no command-line arguments**. You edit the `CONFIG` dictionary near the top of `agamc.py`, then run:

```bash
python agamc.py
```

### Example 1 — Main result on Cora (Table 4 row and Tables 5–7)

In `agamc.py`, set:

```python
CONFIG = {
    "dataset_name": "cora",
    "seeds": [0, 22, 62, 77, 99],
    ...
    "reports": {
        "main_comparison": {"enabled": True},    # Table 4 (ACC)
        "table5_nmi":      {"enabled": True},    # Table 5 (NMI)
        "table6_ari":      {"enabled": True},    # Table 6 (ARI)
        "table7_f1":       {"enabled": True},    # Table 7 (Macro-F1)
        ...
    },
}
```

Then run `python agamc.py`. The script trains one model per seed, prints mean ± std for ACC / NMI / ARI / F1, and prints a consolidated **FINAL REPORT** at the end.

> The file as shipped has only `augmentation_category.sparse` enabled (Table 9), so a bare `python agamc.py` runs that study. Enable the reports you want as shown above, and disable the ones you don't.

### Example 2 — Run on another dataset

```bash
python download_data.py        # once
```

then set `"dataset_name": "acm"` (and the matching seed list, see below) in `CONFIG` and run `python agamc.py`.

### Example 3 — Also save plots

```python
"output_artifacts": {
    "enabled": True,
    "include": {"training_curves": True, "tsne_umap": True, "confusion_matrix": True, ...},
},
```

The full list of switches is in the [Config Guide](docs/CONFIG_GUIDE.md#output_artifacts).

### Seeds used in the experiments

`CONFIG["seeds"]` is the single seed list used by every study. The script file contains the per-dataset lists (commented out) that were used for each dataset:

| Dataset | Seeds |
|---|---|
| Cora | `[0, 22, 62, 77, 99]` |
| ACM | `[22, 110, 119, 127, 148]` |
| CiteSeer | `[5, 6, 19, 28, 29, 37]` |
| DBLP | `[5, 29, 31, 81, 64, 96]` |
| Amazon | `[3, 5, 42, 55, 63]` |
| YouTube | `[8, 13, 48, 89, 97]` |

---

## How configuration works

Everything is controlled by one dictionary, `CONFIG`, in `agamc.py`. There are no command-line flags. The key groups are:

| Group | Keys (examples) | Purpose |
|---|---|---|
| Dataset & run | `dataset_name`, `seeds`, `device`, `data_root`, `output_dir` | Which data, which seeds, where to read/write |
| Augmentation | `augmentation_mode`, `rule_based_view_aug`, `edge_drop_p`, `feature_mask_p` | How views are perturbed |
| Model | `hidden_dim`, `embed_dim`, `encoder_dropout`, `proj_*` | Encoder and projection head |
| Loss | `temperature`, `alpha_intra`, `beta_cross`, `gamma_hybrid` | Multi-term contrastive loss |
| Optimization | `epochs`, `lr`, `cl_batch_size`, `early_stop_patience` | Training schedule |
| Fusion & clusters | `fusion_mode`, `auto_estimate_k` | Final embedding and k-means |
| `output_artifacts` | `enabled`, `include.*` | Which **plots** are saved |
| `reports` | `main_comparison`, `ablation_study`, … | Which **tables / studies** are run |

**Two things to remember:**

1. `output_artifacts` controls *plots*; `reports` controls *which experiments run* (and therefore which paper tables you get).
2. Studies are skipped automatically (with a message) if the dataset lacks the required view type, e.g. asking for `augmentation_category.structured` on ACM.

**The complete, option-by-option reference — every key, its allowed values, default and what it does — is in [`docs/CONFIG_GUIDE.md`](docs/CONFIG_GUIDE.md).** When you start a run, the script also prints the full active configuration so you can confirm what was used.

---

## Reproducing the paper's tables

Set the listed `reports` entries to `True` (everything else `False`), choose the dataset, and run `python agamc.py`.

| Paper table / figure | What it shows | `reports` setting |
|---|---|---|
| Table 4 | Main ACC comparison (AGAMC row) | `main_comparison.enabled` |
| Table 5 / 6 / 7 | NMI / ARI / Macro-F1 | `table5_nmi` / `table6_ari` / `table7_f1` `.enabled` |
| Fig. (4.2.2) | Radar chart ACC/NMI/ARI/F1 | `main_comparison_radar.enabled` |
| Table 8 | Ablation (Full, Uniform, NoAug, RandomAug, NoHybrid, NoCross, IntraOnly) | `ablation_study.enabled` |
| Table 9 | Single-operator study, **sparse** views | `augmentation_category.sparse.enabled` |
| Table 10 | Single-operator study, **dense** views | `augmentation_category.dense.enabled` |
| Table 11 | Single-operator study, **perceptual** views | `augmentation_category.perceptual.enabled` |
| Table 12 | Single-operator study, **structured** views | `augmentation_category.structured.enabled` |
| Table 13 | Rule-based vs. learnable augmentation | `rule_vs_learnable.enabled` |
| Table 14 | Hyperparameter sensitivity sweeps | `hyperparameter_sensitivity.<param>.enabled` |
| Table 15 / Fig. | Scalability (time & GPU memory) | `scalability_analysis.vs_batch_size / vs_num_views / vs_num_nodes` |

The baselines in the paper's comparison tables (K-means, SC, GAE, RMSC, O2MA, GDGCA, CHCC) are **not** implemented in this repository; the script trains and reports AGAMC only, and the AGAMC row is the one populated with real numbers.

**Example: reproduce Table 8 on Cora**

```python
"dataset_name": "cora",
"seeds": [0, 22, 62, 77, 99],
"reports": {
    "ablation_study": {"enabled": True},
    # everything else False
}
```

**Example: reproduce Table 13 (rule-based vs. learnable)**

```python
"reports": {
    "rule_vs_learnable": {"enabled": True, "output_table": True},
}
```

**Example: hyperparameter sensitivity of the temperature**

```python
"hyperparameter_sensitivity": {
    "temperature": {"enabled": True, "values": [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]},
    "output_tables": True,
}
# plus: "output_artifacts": {"enabled": True, "include": {"hyperparameter_sensitivity_curves": True}}
```

Note that studies multiply the number of training runs (variants × seeds). At the start of each run the script prints the total number of runs it will perform and then an overall progress line with elapsed time and ETA after each one. The paper reports that early stopping (patience 8) typically ends training long before the 300-epoch limit.

---

## Output files

Each invocation creates a timestamped folder:

```
outputs/<YYYYMMDD_HHMMSS>_<dataset>_<augmentation_mode>/
└── <dataset>/
    ├── seed_<s>/                  # per-seed artifacts (training log, plots, summary)
    ├── multi_seed_summary.csv/.json
    ├── table4_main_comparison.csv
    ├── table5_*.csv ... table7_*.csv
    ├── table8_ablation.csv/.json  (+ bar/radar charts, per-seed detail)
    ├── category_study/            # Tables 9-12 (+ combined figure)
    ├── table13_rule_vs_learnable.csv (+ charts)
    ├── hyperparameter_sensitivity/ # Table 14 results and curves
    └── scalability_analysis/       # Table 15 results and curves
```

Notes:

- Tables are always **printed** to the console, and a consolidated *FINAL REPORT* is printed at the end. Whether the CSV/plot **files** are written depends on the `output_tables` / `output_table` flags of each study and on `output_artifacts.enabled`. See the Config Guide.
- Per-seed folders hold `training_log.csv`, `probe_log.csv`, `results_summary.json` and any enabled plots.
- At the end, a timing summary lists how long each section took.

---

## Visualisations

When `output_artifacts.enabled = True`, the individual plots are switched on or off under `output_artifacts.include`:

| Switch | Output |
|---|---|
| `training_curves` | Loss / Silhouette curves per seed |
| `tsne_umap` | 2-D t-SNE (+ UMAP if installed), coloured by predicted and true class |
| `per_view_vs_fused` | One t-SNE per view next to the fused embedding |
| `confusion_matrix` | Hungarian-matched confusion matrix |
| `similarity_heatmap` | Pairwise similarity of the fused embedding, sorted by cluster |
| `attention_weights` | Fusion attention weights per view (when `fusion_mode="attention"`) |
| `graph_structure` | NetworkX drawing of views (off by default; slow on large graphs) |
| `interactive_3d` | Interactive 3-D explorer (HTML, via Plotly) |
| `study_bar_charts`, `radar_charts` | Grouped bar / radar charts for Tables 8–13 |
| `silhouette_diagram` | Per-sample silhouette plot |
| `cluster_size_distribution` | Predicted vs. true cluster sizes |
| `view_degree_distribution` | Log-log degree histogram per view |
| `multi_seed_comparison` | Box / bar / radar across all seeds |
| `hyperparameter_sensitivity_curves`, `loss_weight_heatmap` | Table 14 curves and joint heatmap |
| `scalability_curves` | Table 15 time / memory curves |

---

## Hardware and reproducibility notes

- Paper experiments: single NVIDIA Quadro T2000 Max-Q (4 GB), Intel Core i7-10850H, 32 GB RAM, Windows 11, PyTorch 2.5.1, PyG 2.8.0, CUDA 12.1.
- All seeds are set via the `seeds` list; each seed is trained independently and results are reported as mean ± std.
- GPU non-determinism in sparse message passing can cause tiny numerical differences across hardware and driver versions, so your numbers may differ slightly in the last digit.
- Cluster labels are used **only for reporting** (Hungarian-matched ACC, NMI, ARI, F1). Early stopping and model selection use the label-free Silhouette probe.

---

## Known notes and limitations

- Only the AGAMC model is implemented here (no baselines).
- YouTube content features are structural proxies unless you supply real ones (see [Datasets](#datasets)).
- The `edge_drop_rate` and `feature_mask_rate` sweeps (Table 14) write to the global `edge_drop_p` / `feature_mask_p` settings. These are read by the `uniform`, `random` and single-operator paths, whereas `rule_based` mode reads its per-view strengths from `rule_based_view_aug`. To measure sensitivity to these two parameters, run those sweeps with `"augmentation_mode": "uniform"`.
- `rule_vs_learnable.learnable_cost_pct` is an informational annotation only, not measured automatically.
- Scalability runs are short fixed-length measurements (default 10 epochs, early stopping off) and are not meant to report converged clustering quality.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError` in `data/acm/...` etc. | Run `python download_data.py` first. |
| CUDA out of memory | Lower `cl_batch_size` (e.g. 512 or 256), or set `device` to `"cpu"`. |
| A dataset failed during `download_data.py` | Re-run the script; finished datasets are kept. For YouTube, check your connection to `snap.stanford.edu`. |
| "WARNING: augmentation_category.X was requested but dataset has no X view" | Expected: that dataset lacks the view type. Pick another type or dataset. |
| UMAP / Plotly warning | Install `requirements-optional.txt`, or ignore. |
| Non-finite loss warning | The NaN guard skips that step and continues; see `nan_guard_enabled` in the Config Guide. |
| Plots do not appear | Set `output_artifacts.enabled = True` and enable the specific `include` switches. Plots are written to files (headless backend). |

---

## Citation

If you use this code, please cite the paper (see also `CITATION.cff`):

```bibtex
@article{jouya2026agamc,
  title   = {Adaptive Graph Augmentation for Multi-View Clustering: A Contrastive Learning Approach with View-Specific Augmentations},
  author  = {Jouya, Mansour and Abdollahpouri, Alireza},
  year    = {2026},
  note    = {Code: https://github.com/<your-username>/AGAMC}
}
```

> Update the journal/venue, volume and DOI fields once the paper is published.

## License

Released under the [MIT License](LICENSE).

## Contact

Mansour Jouya, Alireza Abdollahpouri — Department of Computer Engineering, University of Kurdistan, Sanandaj, Iran. Please open a GitHub issue for questions or bug reports.
