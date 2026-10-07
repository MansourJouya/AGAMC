# AGAMC Configuration Guide

AGAMC has **no command-line arguments**. Every setting lives in the `CONFIG` dictionary near the top of `agamc.py`. Edit the values there, save, and run:

```bash
python agamc.py
```

At startup the script prints the full active configuration, so you can always confirm what a run used.

**Contents**

- [Mental model](#mental-model)
- [1. Dataset and run](#1-dataset-and-run)
- [2. Multi-view construction](#2-multi-view-construction)
- [3. Augmentation](#3-augmentation)
- [4. Model (encoder and projection head)](#4-model-encoder-and-projection-head)
- [5. Contrastive loss](#5-contrastive-loss)
- [6. Optimization and early stopping](#6-optimization-and-early-stopping)
- [7. Fusion and cluster number](#7-fusion-and-cluster-number)
- [8. `output_artifacts` (plots)](#output_artifacts)
- [9. `reports` (which tables / studies run)](#reports)
- [10. `DATASET_REGISTRY`](#dataset_registry)
- [Ready-to-use recipes](#ready-to-use-recipes)
- [Common mistakes](#common-mistakes)

---

## Mental model

Think of `CONFIG` as three layers:

| Layer | Question it answers | Where |
|---|---|---|
| **Setup** | Which dataset? Which seeds? Which model and training settings? | Top-level keys (`dataset_name`, `seeds`, `hidden_dim`, `lr`, …) |
| **Experiments** | Which paper tables / studies should be run? | `CONFIG["reports"]` |
| **Plots** | Which figures should be saved to disk? | `CONFIG["output_artifacts"]` |

`reports` decides **what is computed**; `output_artifacts` decides **which plots are saved**. A study can be enabled while its plots are off (you still get the printed table), and vice versa.

Dataset-dependent values (number of classes, number of views) are **not** set in `CONFIG`; they come automatically from `DATASET_REGISTRY` once you set `dataset_name`.

---

## 1. Dataset and run

| Key | Allowed values | Default | Description |
|---|---|---|---|
| `dataset_name` | `"cora"`, `"citeseer"`, `"acm"`, `"dblp"`, `"amazon"`, `"youtube"` | `"cora"` | Dataset to run. Sets the number of classes and the views automatically. ACM/DBLP/Amazon/YouTube require `python download_data.py` first. |
| `seeds` | list of ints | `[0, 22, 62, 77, 99]` | **The single seed list** used by the main run and every study. Each seed is a full independent training run; results are reported as mean ± std. |
| `device` | `"cuda"` / `"cpu"` | auto (`cuda` if available) | Compute device. |
| `data_root` | path | `./data` next to the script | Where datasets are stored/read. |
| `output_dir` | path | `./outputs` next to the script | Base folder for results. Each run creates `outputs/<timestamp>_<dataset>_<augmentation_mode>/`. |

Seed lists used per dataset in the experiments (commented in the file):

| Dataset | Seeds |
|---|---|
| cora | `[0, 22, 62, 77, 99]` |
| acm | `[22, 110, 119, 127, 148]` |
| citeseer | `[5, 6, 19, 28, 29, 37]` |
| dblp | `[5, 29, 31, 81, 64, 96]` |
| amazon | `[3, 5, 42, 55, 63]` |
| youtube | `[8, 13, 48, 89, 97]` |

---

## 2. Multi-view construction

| Key | Default | Description |
|---|---:|---|
| `num_views` | auto | **Do not edit.** Overwritten from `DATASET_REGISTRY`. |
| `knn_k` | `20` | Neighbours `k` for the kNN view of **Cora/CiteSeer** (the "dense" view). For ACM/DBLP/Amazon/YouTube, `knn_k` is set per view inside `DATASET_REGISTRY`. |
| `ppr_alpha` | `0.15` | Teleport probability of the Personalized-PageRank diffusion view (Cora/CiteSeer "perceptual" view). |
| `ppr_topk` | `128` | Number of top PPR neighbours kept per node. |

---

## 3. Augmentation

### `augmentation_mode`

| Value | Meaning |
|---|---|
| `"rule_based"` | **AGAMC-Full.** Each view is augmented according to its view type using `rule_based_view_aug` (below). |
| `"learnable"` | An MLP controller predicts augmentation strengths from simple view statistics, trained jointly (straight-through estimators). Used in Table 13. |
| `"uniform"` | Ablation: the same augmentation (`edge_drop_p`, `feature_mask_p`) on every view. |
| `"random"` | Ablation: random strengths in [0.1, 0.4] for edge drop + feature mask on every view. |
| `"none"` | Ablation: no augmentation (direct contrastive learning on original graphs). |

### `rule_based_view_aug` (used by `"rule_based"` mode)

Per-view-type strengths. Tuples are `(branch_a, branch_b)`: the two augmented branches get different strengths.

| View type | Keys | Default |
|---|---|---|
| `sparse` | `edge_drop_p`, `node_drop_p` | `(0.2, 0.4)`, `(0.1, 0.2)` |
| `dense` | `edge_add_ratio`, `feature_mask_p` | `0.05`, `(0.2, 0.4)` |
| `perceptual` | `subgraph_walk_length`, `feature_mask_p` | `50`, `(0.2, 0.4)` |
| `structured` | `feature_mask_p` | `(0.2, 0.3)` |

### Global augmentation strengths

| Key | Default | Used by |
|---|---|---|
| `edge_drop_p` | `(0.20, 0.40)` | `"uniform"` mode, single-operator study (Tables 9–12), `edge_drop_rate` sweep |
| `feature_mask_p` | `(0.20, 0.40)` | `"uniform"` mode, single-operator study, `feature_mask_rate` sweep |
| `node_drop_p` | `(0.10, 0.20)` | Single-operator study |
| `edge_add_ratio` | `0.05` | Single-operator study |
| `feature_noise_std` | `0.05` | Single-operator study (structured views) |

> In `"rule_based"` mode, strengths come from `rule_based_view_aug`, **not** from the global keys above.

### Learnable controller (used by `"learnable"` mode)

| Key | Default | Description |
|---|---:|---|
| `controller_hidden_dim` | `32` | Hidden size of the controller MLP. |
| `controller_stat_dim` | `3` | Number of per-view statistics fed to the controller. |
| `controller_num_params` | `8` | Number of augmentation parameters it predicts. |
| `learnable_param_bounds` | dict | Allowed `(min, max)` range of each predicted parameter (`edge_drop_p_a/b`, `node_drop_p_a/b`, `feature_mask_p_a/b`, `edge_add_ratio`, `feature_noise_std`). |
| `edge_add_pool_ratio` | `0.20` | Size of the candidate pool used for learnable edge addition. |

---

## 4. Model (encoder and projection head)

| Key | Default | Description |
|---|---:|---|
| `hidden_dim` | `256` | Hidden size of the shared GCN encoder. |
| `embed_dim` | `128` | Output (embedding) size. |
| `encoder_dropout` | `0.3` | Dropout between GCN layers. |
| `proj_hidden_dim` | `128` | Hidden size of the projection head. |
| `proj_out_dim` | `64` | Output size of the projection head (space where the contrastive loss is computed). |

The number of GCN layers defaults to 2 and is changed only by the `gcn_layers` sweep (Table 14).

---

## 5. Contrastive loss

Total loss = α·L_intra + β·L_cross + γ·L_hybrid.

| Key | Default | Description |
|---|---:|---|
| `temperature` | `0.4` | Temperature τ of InfoNCE. |
| `alpha_intra` | `1.0` | Weight of the **intra-view** term (same node across two augmentations of a view). |
| `beta_cross` | `1.0` | Weight of the **cross-view** term (same node across views). Set `0` for the NoCross ablation. |
| `gamma_hybrid` | `0.5` | Weight of the **hybrid** term (inter-view separation). Set `0` for the NoHybrid ablation. |
| `normalize_loss_weights` | `True` | Divide by the sum of the active weights (weighted mean rather than raw sum) so ablation variants with fewer terms are trained on the same loss scale. Set `False` to restore the raw weighted sum. |

---

## 6. Optimization and early stopping

| Key | Default | Description |
|---|---:|---|
| `epochs` | `300` | Maximum number of epochs. |
| `lr` | `2e-3` | Learning rate. |
| `lr_schedule_enabled` | `True` | Linear warmup then cosine annealing. `False` = constant LR. |
| `warmup_epochs` | `20` | Warmup length. |
| `min_lr` | `1e-5` | Final LR of the cosine schedule. |
| `weight_decay` | `5e-4` | Weight decay. |
| `cl_batch_size` | `1024` | Number of nodes per contrastive mini-batch. **Lower this if you run out of GPU memory.** Memory rises sharply at large values (see the paper's scalability analysis). |
| `log_every` | `1` | Log training loss every N epochs. |
| `eval_every` | `10` | Run the label-free Silhouette probe (and read-only diagnostics) every N epochs. |
| `grad_clip_norm` | `3.0` | Gradient-norm clipping; `None` disables it. |
| `nan_guard_enabled` | `True` | Skip a step with a non-finite loss instead of crashing. |
| `debug_detect_anomaly` | `False` | Enables PyTorch anomaly detection to locate NaN sources. **Much slower**; only for debugging. |
| `early_stopping_enabled` | `True` | Stop when the Silhouette probe stops improving. Label-free. |
| `early_stop_patience` | `8` | Number of probe evaluations without improvement before stopping. |

---

## 7. Fusion and cluster number

### Fusion of view embeddings

| Key | Allowed values | Default | Description |
|---|---|---|---|
| `fusion_mode` | `"average"`, `"attention"` | `"average"` | How per-view embeddings are combined into the final embedding. `"attention"` learns per-node view weights (shown by the `attention_weights` plot). |
| `fusion_align_loss` | `"infonce"`, `"mse"` | `"infonce"` | Alignment loss for attention fusion. |
| `fusion_entropy_coef` | float | `0.1` | Entropy regularisation on attention weights. |
| `fusion_train_steps` | int | `50` | Optimisation steps for the fusion network. |
| `fusion_lr` | float | `1e-2` | Learning rate of the fusion network. |

The `fusion_*` options other than `fusion_mode` only matter when `fusion_mode="attention"`.

### Number of clusters

| Key | Allowed values | Default | Description |
|---|---|---|---|
| `auto_estimate_k` | `True` / `False` | `False` | `False`: use the dataset's known number of classes (standard setting). `True`: estimate the number of clusters from the embedding. |
| `k_estimation_method` | `"silhouette"`, `"eigengap"` | `"silhouette"` | Estimation method (only if `auto_estimate_k=True`). |
| `k_search_range` | `(min, max)` | `(2, 15)` | Candidate range of cluster counts. |
| `eigengap_knn_k` | int | `10` | kNN size used by the eigengap method. |

### Reporting-only settings

| Key | Default | Description |
|---|---:|---|
| `enable_readonly_diagnostics` | `True` | Compute ACC/NMI/ARI/F1 during training for monitoring. **Never** used for training decisions or model selection. |
| `silhouette_sample_size` | `5000` | Max points used to estimate Silhouette (also subsamples heatmaps / graph drawings on large datasets). `None` = exact full computation. |

---

<a id="output_artifacts"></a>
## 8. `output_artifacts` (plots)

```python
"output_artifacts": {
    "enabled": False,          # MASTER switch
    "include": { ...per-plot switches... },
}
```

- `enabled = False` → **no plots/log files are written**. Final metrics are still computed and printed.
- `enabled = True` → each plot is produced only if its `include` switch is `True`.

| `include` switch | Output | Notes |
|---|---|---|
| `training_curves` | Loss / Silhouette curves per seed | |
| `tsne_umap` | t-SNE (+ UMAP) of fused embedding, by predicted cluster and by true class | UMAP needs `umap-learn` |
| `per_view_vs_fused` | One t-SNE per view next to the fused one | Shows the value of fusion |
| `confusion_matrix` | Hungarian-matched confusion heatmap | |
| `similarity_heatmap` | Pairwise similarity, nodes sorted by cluster | Subsampled for large N |
| `attention_weights` | Fusion attention weights per view | Meaningful for `fusion_mode="attention"` |
| `graph_structure` | NetworkX drawing of views | Default off; slow/cluttered on big graphs; needs `networkx` |
| `interactive_3d` | Interactive 3D explorer (HTML) | Needs `plotly` (+ `umap-learn`) |
| `study_bar_charts` | Grouped bar charts with error bars | Tables 8, 9–12, 13 |
| `radar_charts` | Multi-metric radar charts | Tables 8, 9–12, 13 |
| `silhouette_diagram` | Per-sample silhouette plot | |
| `cluster_size_distribution` | Predicted vs. true cluster sizes | |
| `view_degree_distribution` | Log-log degree histogram per view | |
| `multi_seed_comparison` | Box / bar / radar across all seeds | |
| `hyperparameter_sensitivity_curves` | Table 14 sweep curves (±1 std band) | |
| `loss_weight_heatmap` | Table 14 joint β×γ ACC heatmap | Needs `beta_gamma_joint_heatmap` enabled |
| `scalability_curves` | Table 15 time/memory curves | |

---

<a id="reports"></a>
## 9. `reports` (which tables / studies run)

Each entry enables one study from the paper. Disabled entries cost nothing. The script counts all enabled runs up front and shows overall progress/ETA while running; at the end it prints every enabled table again in one **FINAL REPORT**.

### 9.1 Main results (Tables 4–7, radar)

| Key | Paper item | Notes |
|---|---|---|
| `main_comparison.enabled` | Table 4 (ACC) | Runs the primary multi-seed training |
| `table5_nmi.enabled` | Table 5 (NMI) | Reuses the primary run |
| `table6_ari.enabled` | Table 6 (ARI) | Reuses the primary run |
| `table7_f1.enabled` | Table 7 (Macro-F1) | Reuses the primary run |
| `main_comparison_radar.enabled` | Radar chart (4.2.2) | Reuses the primary run |

The primary run is executed **once** if any of these (or the ablation) is enabled, and shared between them. Only AGAMC is trained; baseline rows are not implemented.

### 9.2 `ablation_study` (Table 8)

`{"enabled": True}` trains seven variants (each × all seeds). The "Full" variant reuses the primary run.

| Variant | What changes |
|---|---|
| AGAMC-Full | Nothing (rule-based, all three loss terms) |
| AGAMC-Uniform | `augmentation_mode="uniform"` |
| AGAMC-NoAug | `augmentation_mode="none"` |
| AGAMC-RandomAug | `augmentation_mode="random"` |
| AGAMC-NoHybrid | `gamma_hybrid=0` |
| AGAMC-NoCross | `beta_cross=0` |
| AGAMC-IntraOnly | `beta_cross=0`, `gamma_hybrid=0` |

### 9.3 `augmentation_category` (Tables 9–12)

Tests **one augmentation operator at a time** on all views of a given type.

| Key | Paper table | Applies to |
|---|---|---|
| `sparse.enabled` | Table 9 | sparse views |
| `dense.enabled` | Table 10 | dense views |
| `perceptual.enabled` | Table 11 | perceptual views |
| `structured.enabled` | Table 12 | structured views |

Other keys:

| Key | Default | Description |
|---|---|---|
| `ops_per_view_type` | dict | The list of operators tested per view type (`edge_drop`, `node_drop`, `edge_add`, `feature_mask`, `feature_noise`, `subgraph_extract`). Edit to test a subset. |
| `subgraph_extract_walk_length` | `50` | Random-walk length for `subgraph_extract`. |
| `output_tables` | `False` | `True` saves the tables (and bar/radar charts if enabled) as files; otherwise they are only printed. |
| `combined_figure` | `False` | One 2×2 figure merging Tables 9–12 (4.3.2). |
| `target_view_types` | auto | **Do not edit.** Filled automatically with the requested types that exist in the chosen dataset. |

If you request a view type the dataset lacks (e.g. `structured` on ACM), it is skipped with a warning.

| Dataset | View types available |
|---|---|
| cora, citeseer | sparse, dense, perceptual, structured |
| acm | sparse, dense, perceptual |
| dblp | sparse, dense |
| amazon | sparse, dense |
| youtube | sparse, perceptual, dense |

### 9.4 `rule_vs_learnable` (Table 13)

| Key | Default | Description |
|---|---|---|
| `enabled` | `False` | Trains rule-based and learnable modes (each × all seeds) and compares them. |
| `output_table` | `False` | Save the table / charts to files. |
| `annotate_cost` | `False` | Add a "+X% training-time cost" annotation to the cost chart (4.5). |
| `learnable_cost_pct` | `30.0` | The percentage shown in that annotation. **Informational only; not measured automatically.** |

### 9.5 `hyperparameter_sensitivity` (Table 14)

Sweeps **one parameter at a time**, holding everything else at the base `CONFIG`, and reports ACC/NMI/ARI/F1 (mean ± std over seeds) at each value.

| Parameter key | Default `values` | Notes |
|---|---|---|
| `edge_drop_rate` | `[0.0 … 0.6]` | Sets both branches of `edge_drop_p`. Takes effect in `"uniform"` mode (see note below). |
| `feature_mask_rate` | `[0.0 … 0.7]` | Sets both branches of `feature_mask_p`. Takes effect in `"uniform"` mode (see note below). |
| `knn_k` | `[3, 5, 8, 10, 15, 20]` | Needs a kNN-built view. |
| `subgraph_walk_length` | `[20 … 200]` | Needs a **perceptual** view. |
| `temperature` | `[0.05 … 1.0]` | |
| `alpha_intra` | `[0.0 … 2.0]` | |
| `beta_cross` | `[0.0 … 2.0]` | |
| `gamma_hybrid` | `[0.0 … 1.5]` | |
| `hidden_dim` | `[32 … 256]` | |
| `gcn_layers` | `[1, 2, 3, 4]` | |

Each parameter is `{"enabled": bool, "values": [...]}`. Edit `values` to change the grid.

Other keys:

| Key | Description |
|---|---|
| `beta_gamma_joint_heatmap.enabled` | Optional 2-D ACC heatmap over β × γ (uses the `values` of `beta_cross` and `gamma_hybrid`). |
| `alpha_beta_joint_heatmap.enabled` | Optional 2-D ACC heatmap over α × β. |
| `seeds` | `None` → use `CONFIG["seeds"]`; or a list to override for this study only. |
| `output_tables` | `True` saves tables as CSV files. |

> **Note on `edge_drop_rate` / `feature_mask_rate`.** These sweeps modify `edge_drop_p` / `feature_mask_p`. Those global keys are read by `"uniform"`, `"random"` and single-operator paths; in `"rule_based"` mode the per-view strengths come from `rule_based_view_aug`. To measure sensitivity to these two parameters, run the sweep with `"augmentation_mode": "uniform"`.

> Number of runs = (number of values) × (number of seeds), per enabled parameter. A joint heatmap multiplies the two value lists. Sweeps that do not apply to the dataset (e.g. `subgraph_walk_length` without a perceptual view) are skipped with a message.

### 9.6 `scalability_analysis` (Table 15)

Short fixed-length runs that measure **wall-clock time per epoch** and **peak GPU memory** (CUDA only). They are not meant to produce converged clustering quality.

| Key | Default | Description |
|---|---|---|
| `vs_batch_size.enabled` / `.values` | `False` / `[256, 512, 1024, 2048]` | Sweep `cl_batch_size`. |
| `vs_num_views.enabled` | `False` | Sweep the number of views K = 1 … (number of real views of the dataset). |
| `vs_num_nodes.enabled` / `.values` | `False` / `[500, 1000, 2000, "full"]` | Subsample the graph to N nodes; `"full"` = whole graph. Reports time/memory only (no quality). |
| `measure_epochs` | `10` | Length of each measurement run (early stopping disabled). |
| `seeds` | `None` | `None` → `CONFIG["seeds"]`. |
| `output_tables` | `False` | Save tables. |
| `combined_figure` | `False` | One 2×2 figure merging time/views/nodes (4.6). |

Memory figures require a CUDA device.

---

<a id="dataset_registry"></a>
## 10. `DATASET_REGISTRY`

`DATASET_REGISTRY` (above `CONFIG` in `agamc.py`) defines, per dataset: number of classes, loader type (`planetoid` or `generic`), number of views, and for each view its **type tag** and **build method**.

```python
"dblp": {
    "num_classes": 4,
    "loader": "generic",
    "num_views": 2,
    "features_file": "features.npy",
    "labels_file": "labels.npy",
    "views": [
        {"type": "sparse", "build": "graph_file", "file": "graph_coauthor.npz"},
        {"type": "dense",  "build": "knn_file",   "file": "feat_term.npy", "knn_k": 10},
    ],
}
```

| Field | Meaning |
|---|---|
| `type` | `sparse` / `dense` / `perceptual` / `structured`: selects which augmentation row applies to the view. |
| `build: "graph_file"` | The view is a real graph loaded from a `.npz` sparse matrix. |
| `build: "knn_file"` | The view is a kNN graph built over a `.npy` feature matrix, with its own `knn_k`. |

You normally never edit this unless you add a custom dataset (see the README).

---

## Ready-to-use recipes

Only the changed keys are shown; leave everything else as is. Set unrelated `reports` entries to `False`.

**Fastest sanity check (quick look, no files):**

```python
"dataset_name": "cora",
"seeds": [0],
"epochs": 50,
"reports": {"main_comparison": {"enabled": True}},
"output_artifacts": {"enabled": False, "include": {...all False...}},
```

**Main results with all four metrics:**

```python
"reports": {
    "main_comparison": {"enabled": True},
    "table5_nmi": {"enabled": True},
    "table6_ari": {"enabled": True},
    "table7_f1": {"enabled": True},
    "main_comparison_radar": {"enabled": True},
}
```

**Full ablation with saved bar/radar charts:**

```python
"reports": {"ablation_study": {"enabled": True}},
"output_artifacts": {"enabled": True, "include": {"study_bar_charts": True, "radar_charts": True}},
```

**Tables 9 and 10 together (Cora):**

```python
"reports": {"augmentation_category": {
    "sparse": {"enabled": True}, "dense": {"enabled": True},
    "perceptual": {"enabled": False}, "structured": {"enabled": False},
    "output_tables": True,
    ...
}}
```

**Rule-based vs. learnable (Table 13):**

```python
"reports": {"rule_vs_learnable": {"enabled": True, "output_table": True}}
```

**Sensitivity of γ with curve:**

```python
"reports": {"hyperparameter_sensitivity": {
    "gamma_hybrid": {"enabled": True, "values": [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]},
    "output_tables": True,
}},
"output_artifacts": {"enabled": True, "include": {"hyperparameter_sensitivity_curves": True}}
```

**Scalability on the GPU:**

```python
"reports": {"scalability_analysis": {
    "vs_batch_size": {"enabled": True},
    "vs_num_views": {"enabled": True},
    "vs_num_nodes": {"enabled": True},
    "output_tables": True, "combined_figure": True,
}},
"output_artifacts": {"enabled": True, "include": {"scalability_curves": True}}
```

**Low-memory GPU (4 GB):**

```python
"cl_batch_size": 512,   # or 256
```

**Attention fusion with attention plot:**

```python
"fusion_mode": "attention",
"output_artifacts": {"enabled": True, "include": {"attention_weights": True}}
```

**Estimate the number of clusters instead of using the known value:**

```python
"auto_estimate_k": True,
"k_estimation_method": "silhouette",   # or "eigengap"
"k_search_range": (2, 15),
```

---

## Common mistakes

| Mistake | What happens | Fix |
|---|---|---|
| Changed `dataset_name` but not `seeds` | Runs with the previous dataset's seeds. Results still valid but won't match the paper's seed set. | Use the per-dataset seed list in section 1. |
| Enabled a study but no files appear | Tables are printed only. | Set that study's `output_tables` / `output_table` to `True`, and `output_artifacts.enabled` for plots. |
| Enabled plots but nothing saved | The master switch is off. | Set `output_artifacts["enabled"] = True`. |
| Asked for a view type the dataset lacks | The study is skipped with a warning. | Choose a type the dataset has (see section 9.3). |
| Edge-drop / feature-mask sweep gives flat results in default mode | Rule-based mode ignores the global keys. | Use `"augmentation_mode": "uniform"` for those sweeps. |
| Edited `num_views` or `target_view_types` | Overwritten automatically. | Edit `DATASET_REGISTRY` / the `enabled` flags instead. |
| Scalability memory shows 0 / N/A | No CUDA device. | Run on a GPU. |
| Very long run | Studies multiply runs (variants × seeds). | Reduce `seeds`, `values` lists, or enable fewer studies. |
