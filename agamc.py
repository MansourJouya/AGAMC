"""
================================================================================
Multi-View Graph Contrastive Clustering (MV-GCC / AGAMC) - MULTI-DATASET VERSION
    WITH PER-TABLE OUTPUT SELECTION (Tables 4, 8, 9-12, 13) + FULL VISUAL SUITE
================================================================================
This is the same 11-stage pipeline as the original Cora-only script, extended
to support 6 datasets, selected with ONE config key:

    CONFIG["dataset_name"] in {"cora", "citeseer", "acm", "dblp", "amazon", "youtube"}

WHAT CHANGED vs the Cora-only version
---------------------------------------------------------------------------
ONLY Stage 1 (data loading / multi-view graph construction) changed in
substance. Stages 2-11 (augmentation, shared encoder, contrastive loss,
fusion, cluster estimation, K-Means++, evaluation, seeds, ablation) are
100% unchanged in their numerical/training behavior - they only consume
    x            : (N, D) node feature tensor
    views         : list of {"type": "sparse"/"dense"/"perceptual"/"structured",
                             "edge_index": LongTensor(2, E)}
    gt_labels     : (N,) ground-truth class ids (read-only, diagnostic only)
    num_classes   : int
regardless of where they came from.

DATASET_REGISTRY (below) is the single source of truth for:
  - how many ground-truth classes a dataset has (replaces the old hardcoded
    NUM_CLUSTERS_KNOWN = 7)
  - how many views it has, and what TYPE each view gets (which controls which
    row of the augmentation table in Stage 2 applies to it - "sparse",
    "dense", "perceptual", or "structured")
  - where each view's graph/features come from

Two loading paths:
  1. "planetoid"  -> Cora, Citeseer. Auto-downloaded via torch_geometric,
     views built exactly like the original script (citation graph, kNN on
     content features, PPR diffusion, structured clone).
  2. "generic"    -> ACM, DBLP, Amazon, YouTube. Loaded from a local folder
     you provide, in a simple standardized file format (see
     "GENERIC DATASET FILE FORMAT" section below). These are NOT
     auto-downloaded because there is no single canonical public file that
     already matches this exact multi-view breakdown for every dataset -
     you assemble the folder yourself from whichever raw source you use
     (suggested starting points are documented next to each entry below).

GENERIC DATASET FILE FORMAT  (folder: <data_root>/<dataset_name>/)
---------------------------------------------------------------------------
  features.npy            : (N, D) float32 array - node features fed to the
                             shared GCN encoder (e.g. BOW / TF-IDF / PCA'd text)
  labels.npy               : (N,) int array - ground-truth class ids
                             (used ONLY for reporting, exactly like
                             gt_labels_readonly elsewhere in this file)
  graph_<name>.npz          : a scipy.sparse adjacency matrix
                             (sp.save_npz(...)) for any view that is a REAL
                             relation (co-author graph, co-purchase graph,
                             friendship graph, paper-subject graph, ...)
  feat_<name>.npy           : (N, D_v) float32 array for any view that is
                             built by taking a k-NN graph over some OTHER
                             feature space (term features, thumbnail CNN
                             embeddings, comment TF-IDF, ...)

Each dataset's entry in DATASET_REGISTRY lists, in order, which of these
files feeds which view and what "type" tag (sparse/dense/perceptual/
structured) that view gets for augmentation purposes.

Suggested raw sources (you still need to preprocess these into the format above):
  - ACM / DBLP  : meta-path adjacency matrices (co-author / subject graphs)
                  as distributed in HAN (https://github.com/Jhy1993/HAN) or
                  HeCo (https://github.com/liun-online/HeCo) style .mat files
                  (e.g. acm.mat / dblp.mat), which bundle several adjacency
                  matrices + a feature matrix + labels per dataset.
  - Amazon      : co-purchase graph from SNAP (https://snap.stanford.edu/data/#amazon)
                  + your own PCA'd text/review features (256-d, as described).
  - YouTube     : friendship graph from SNAP com-youtube
                  (https://snap.stanford.edu/data/com-Youtube.html) - the
                  thumbnail-CNN / comment-TF-IDF features described are not a
                  standard public release as far as I could confirm, so
                  you'll likely need to assemble those yourself (e.g. run a
                  pretrained CNN over thumbnails you collect, and TF-IDF over
                  scraped comments) and save them as feat_*.npy files above.

Everything else (all 11 pipeline stages, CONFIG structure, seeds, ablation
study) is IDENTICAL in spirit to the original Cora-only script - only the
data-loading front end changed to be dataset-driven.

--------------------------------------------------------------------------------
PER-TABLE OUTPUT SELECTION
--------------------------------------------------------------------------------
Everything related to "which tables get produced" lives under ONE nested
dict: CONFIG["reports"]. Each key corresponds to one paper-style table:

    main_comparison                          -> Table 4  - Main ACC comparison table
    ablation_study                           -> Table 8  - Ablation study
    augmentation_category.sparse             -> Table 9  - sparse-view augmentation study
    augmentation_category.dense              -> Table 10 - dense-view augmentation study
    augmentation_category.perceptual         -> Table 11 - perceptual-view augmentation study
    augmentation_category.structured         -> Table 12 - structured-view augmentation study
    rule_vs_learnable                        -> Table 13 - Rule-based vs. Learnable augmentation
    hyperparameter_sensitivity.<param>        -> Table 14 - Hyperparameter sensitivity sweeps
    scalability_analysis.<dimension>          -> Table 15 - Scalability (time/memory) sweeps

There are NO other top-level "ablation_study" / "augmentation_category_study" /
"rule_vs_learnable_study" dicts anywhere in this file any more - CONFIG["reports"]
is the single source of truth, and every function below reads its settings
from there (with sensible .get(..., default) fallbacks).

Any table whose required view-type is not present in the chosen dataset is
skipped automatically with a message (see the sync block right after CONFIG,
which computes reports["augmentation_category"]["target_view_types"]).

At the end of the run (main()), ALL enabled tables are gathered and printed
together one more time in a single consolidated "FINAL REPORT" section, in
addition to being printed/saved as they are computed during the run.

--------------------------------------------------------------------------------
HYPERPARAMETER SENSITIVITY ANALYSIS (Table 14 style, see CONFIG["reports"]["hyperparameter_sensitivity"])
--------------------------------------------------------------------------------
Sweeps one hyperparameter at a time (edge_drop_rate, feature_mask_rate,
knn_k, subgraph_walk_length, temperature, alpha_intra, beta_cross,
gamma_hybrid, hidden_dim, gcn_layers) across its declared "values" list,
holding everything else fixed at the base CONFIG, and reports ACC/NMI/ARI/F1
(mean +/- std over cfg["seeds"]) at every value. Each sub-key has its own
"enabled" flag (default False) and is skipped gracefully with a printed
message when it doesn't structurally apply to the current dataset (e.g.
subgraph_walk_length needs a "perceptual" view; knn_k needs some knn-built
view). Gated plots: output_artifacts.include.hyperparameter_sensitivity_curves
(a curve of each metric vs the swept value, +/-1 std band) and, only when
"beta_gamma_joint_heatmap" is enabled, output_artifacts.include.loss_weight_heatmap
(a 2D ACC heatmap over beta_cross x gamma_hybrid).

--------------------------------------------------------------------------------
SCALABILITY ANALYSIS (Table 15 style, see CONFIG["reports"]["scalability_analysis"])
--------------------------------------------------------------------------------
Measures wall-clock per-epoch time and peak GPU memory (CUDA only) as batch
size, number of views, or number of nodes changes, using a SHORT fixed-length
run (scalability_analysis["measure_epochs"], early stopping disabled) purely
for measurement - these runs are NOT meant to produce converged clustering
quality numbers (vs_num_nodes omits clustering-quality metrics entirely,
since subsampling the graph arbitrarily shifts class balance). Gated plot:
output_artifacts.include.scalability_curves (epoch-time + peak-memory vs the
swept dimension).

--------------------------------------------------------------------------------
VISUALIZATION SUITE (see CONFIG["output_artifacts"])
--------------------------------------------------------------------------------
CONFIG["output_artifacts"]["enabled"] is the master on/off switch. When True,
CONFIG["output_artifacts"]["include"] gives finer-grained control over which
individual visualizations get produced:

    training_curves    -> per-seed loss/silhouette curves (Stage 4-6/10)
    tsne_umap          -> 2D t-SNE (+ UMAP panel if umap-learn is installed)
                          of the final fused embedding, colored by predicted
                          cluster AND by ground-truth class, per seed
    per_view_vs_fused  -> one t-SNE panel per view (pre-fusion) next to the
                          fused-embedding panel - shows Stage 9's value
    confusion_matrix   -> Hungarian-matched predicted-vs-true heatmap
    similarity_heatmap -> pairwise-similarity heatmap of the fused embedding,
                          nodes sorted by predicted cluster (subsampled for
                          large N, reusing the silhouette_sample_size pattern)
    attention_weights  -> Stage 9 fusion attention-weight bar chart
    graph_structure    -> networkx drawing of one or more views (default OFF -
                          slow/cluttered on large datasets; subsampled)
    interactive_3d     -> interactive 3D t-SNE/UMAP HTML via Plotly
    study_bar_charts   -> grouped bar charts (with error bars over seeds) for
                          Table 8 / Table 9-12 / Table 13
    hyperparameter_sensitivity_curves -> Table 14 sweep curves (+/-1 std band)
    loss_weight_heatmap               -> Table 14 optional beta/gamma joint heatmap
    scalability_curves                -> Table 15 time/memory vs swept dimension

OPTIONAL THIRD-PARTY DEPENDENCIES (only needed for the visual extras above -
the core pipeline never needs them, and every call site below degrades
gracefully with a warning if a package is missing):
    pip install umap-learn --break-system-packages   # tsne_umap, interactive_3d
    pip install plotly --break-system-packages       # interactive_3d
    pip install networkx --break-system-packages     # graph_structure
================================================================================
"""

import os
import copy
import time
import math
import random
import csv
import json
from datetime import timedelta

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")  # headless / file-output backend, no display needed
import matplotlib.pyplot as plt

from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected, add_self_loops, degree
from torch_geometric.nn import GCNConv

from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
from sklearn.manifold import TSNE
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score, f1_score
from scipy.optimize import linear_sum_assignment


# ==============================================================================
# RUN-WIDE PROGRESS / TIMING TRACKER
# ==============================================================================
# Tracks (a) how many individual training runs (one per seed of the primary
# run, per ablation variant x seed, per augmentation-category op x seed, per
# rule_vs_learnable mode x seed) will happen in total for this invocation of
# main(), so an overall "X/Y runs done, elapsed/ETA" line can be printed
# after each one finishes; and (b) wall-clock duration of each top-level
# section (main run / ablation study / augmentation-category study /
# rule_vs_learnable study), printed when that section finishes and again in
# one summary table at the very end. Purely informational - never affects
# training or model-selection decisions.
_PROGRESS = {"total_runs": 0, "done_runs": 0, "start_time": None}
_SECTION_TIMES = {}  # {section_name: [start_time, end_time_or_None]}


def _progress_init(total_runs):
    _PROGRESS["total_runs"] = total_runs
    _PROGRESS["done_runs"] = 0
    _PROGRESS["start_time"] = time.time()
    print(f"[Overall Progress] This invocation will run {total_runs} individual "
          f"training run(s) in total (across all enabled seeds/variants/ops/modes).")


def _progress_tick(label):
    if _PROGRESS["start_time"] is None:
        return
    _PROGRESS["done_runs"] += 1
    done, total = _PROGRESS["done_runs"], _PROGRESS["total_runs"]
    elapsed = time.time() - _PROGRESS["start_time"]
    pct = (done / total * 100) if total else 100.0
    eta = (elapsed / done * (total - done)) if (done > 0 and total > done) else 0.0
    print(f"[Overall Progress] {done}/{total} run(s) done ({pct:5.1f}%) - "
          f"just finished: {label} - elapsed={timedelta(seconds=int(elapsed))} - "
          f"est. remaining={timedelta(seconds=int(eta))}")


def _section_start(name):
    _SECTION_TIMES[name] = [time.time(), None]
    print(f"\n[Timing] Section '{name}' started.")


def _section_end(name):
    if name in _SECTION_TIMES:
        _SECTION_TIMES[name][1] = time.time()
        duration = _SECTION_TIMES[name][1] - _SECTION_TIMES[name][0]
        print(f"[Timing] Section '{name}' finished in {timedelta(seconds=int(duration))}.")


def _print_section_timing_summary():
    if not _SECTION_TIMES:
        return
    print("\n" + "=" * 78)
    print(" SECTION TIMING SUMMARY")
    print("=" * 78)
    grand_total = 0.0
    for name, (start, end) in _SECTION_TIMES.items():
        if end is None:
            continue
        duration = end - start
        grand_total += duration
        print(f"  {name:<45s}: {timedelta(seconds=int(duration))}")
    print("-" * 78)
    print(f"  {'TOTAL (sum of sections)':<45s}: {timedelta(seconds=int(grand_total))}")
    print("=" * 78 + "\n")


# ==============================================================================
# DATASET REGISTRY: single source of truth for "which dataset, how many
# classes, how many views, and how is each view built"
# ==============================================================================
DATASET_REGISTRY = {
    "cora": {
        "num_classes": 7,
        "loader": "planetoid",
        "planetoid_name": "Cora",
        "num_views": 4,
        "views": [
            {"type": "sparse",     "build": "planetoid_citation"},
            {"type": "dense",      "build": "planetoid_knn"},
            {"type": "perceptual", "build": "planetoid_ppr"},
            {"type": "structured", "build": "planetoid_structured"},
        ],
    },
    "citeseer": {
        "num_classes": 6,
        "loader": "planetoid",
        "planetoid_name": "CiteSeer",
        "num_views": 4,
        "views": [
            {"type": "sparse",     "build": "planetoid_citation"},
            {"type": "dense",      "build": "planetoid_knn"},
            {"type": "perceptual", "build": "planetoid_ppr"},
            {"type": "structured", "build": "planetoid_structured"},
        ],
    },
    "acm": {
        # Papers from ACM conferences. 3 research areas (DB, DM, Multimedia).
        "num_classes": 3,
        "loader": "generic",
        "num_views": 3,
        "features_file": "features.npy",     # e.g. TF-IDF / BOW paper features
        "labels_file": "labels.npy",
        "views": [
            {"type": "sparse",     "build": "graph_file", "file": "graph_coauthor.npz"},
            {"type": "dense",      "build": "graph_file", "file": "graph_subject.npz"},
            {"type": "perceptual", "build": "knn_file",   "file": "feat_term.npy", "knn_k": 10},
        ],
    },
    "dblp": {
        # Co-authorship network of CS authors. 4 research areas.
        "num_classes": 4,

        "loader": "generic",
        "num_views": 2,
        "features_file": "features.npy",
        "labels_file": "labels.npy",
        "views": [
            {"type": "sparse", "build": "graph_file", "file": "graph_coauthor.npz"},
            {"type": "dense",  "build": "knn_file",   "file": "feat_term.npy", "knn_k": 10},
        ],
    },
    "amazon": {
        # Product co-purchase network. 10 product categories.
        "num_classes": 10,
        "loader": "generic",
        "num_views": 2,
        "features_file": "features.npy",     # e.g. PCA'd text features (256-d)
        "labels_file": "labels.npy",
        "views": [
            {"type": "sparse", "build": "graph_file", "file": "graph_copurchase.npz"},
            {"type": "dense",  "build": "knn_file",   "file": "feat_text_pca256.npy", "knn_k": 15},
        ],
    },
    "youtube": {
        # Subsampled YouTube social network. 5 interest groups.
        "num_classes": 5,
        "loader": "generic",
        "num_views": 3,
        "features_file": "features.npy",
        "labels_file": "labels.npy",
        "views": [
            {"type": "sparse",     "build": "graph_file", "file": "graph_friendship.npz"},
            {"type": "perceptual", "build": "knn_file",   "file": "feat_thumbnail_cnn.npy", "knn_k": 10},
            {"type": "dense",      "build": "knn_file",   "file": "feat_comment_tfidf.npy", "knn_k": 10},
        ],
    },
}


# ==============================================================================
# 0. CONFIGURATION  (all tunable hyper-parameters live here, printed at startup)
#    -> This is the ONLY place you need to edit. There are no command-line
#       arguments in this script by design.
# ==============================================================================

CONFIG = {
    # ---- Dataset selection ---------------------------------------------------
    "dataset_name": "cora",

    # ---- Reproducibility / device -----------------------------------------
    # SINGLE SOURCE OF TRUTH FOR SEEDS: this is the only "seeds" list in the
    # whole script. The primary run, the ablation study (Table 8), the
    # augmentation-category study (Tables 9-12) and the rule-vs-learnable
    # study (Table 13) all read their seeds from here (with .get(..., None) ->
    # base_cfg["seeds"] fallbacks in the relevant functions below).
    
    "seeds": [0, 22, 62, 77, 99],      # cora
    #"seeds": [22,110,119,127,148],     # acm
    #"seeds": [5, 6, 19, 28, 29, 37],   # citeseer
    #"seeds": [5, 29, 31, 81, 64, 96],  # dblp
    #"seeds": [3, 5, 42, 55, 63],       # amazon
    #"seeds": [8, 13, 48, 89, 97],      # youtube

    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # ---- Dataset ------------------------------------------------------------
    "data_root": os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),

    # ---- Multi-view construction (Stage 1) -----------------------------------
    "num_views": 4,                  # overridden per-dataset from DATASET_REGISTRY below
    "knn_k": 20,
    "ppr_alpha": 0.15,
    "ppr_topk": 128,

    # ---- Augmentation (Stage 2 / 3) ------------------------------------------
    "augmentation_mode": "rule_based",   # "rule_based" (fixed table) or "learnable" (MLP controller)

    "edge_drop_p": (0.20, 0.40),
    "node_drop_p": (0.10, 0.20),
    "edge_add_ratio": 0.05,
    "feature_mask_p": (0.20, 0.40),
    "feature_noise_std": 0.05,

    # ---- Adaptive per-view-type augmentation strengths (rule_based / AGAMC-Full ONLY) ----
    # This table is what makes the AGAMC-Full recipe *adaptive*: instead of one
    # blunt augmentation applied identically to every view (the AGAMC-Uniform
    # ablation) or destructive operators, each view type is perturbed on the
    # channel that is REDUNDANT for it while its informative channel is
    # preserved (the adaptive-augmentation principle, cf. GCA):
    #   - sparse structural views: topology carries the signal -> keep edges
    #     almost intact, mask features gently.
    #   - dense similarity views: topology is dense/redundant -> drop edges
    #     harder, mask features lightly.
    #   - perceptual views: moderate perturbation on both channels.
    #   - structured views: no reliable topology to drop -> feature masking only.
    # Every view type uses the SAME two safe, standard operators (edge_drop +
    # feature_mask); only their strengths differ. The earlier recipe injected
    # RANDOM edges into the dense view (random_edge_add -> false semantic links)
    # and ZEROED whole nodes in the sparse view (node_drop_features -> total
    # node erasure); both corrupt the contrastive signal more than they help,
    # which made Full lose to the simpler ablations, so they are intentionally
    # gone from the Full recipe. (random_edge_add / node_drop_features still
    # exist and are still used by the Table 9-12 single-op study and the
    # learnable controller path - only augment_view() stopped calling them.)
    # Each entry is (branch_a_strength, branch_b_strength).
    "rule_based_view_aug": {
        "sparse":     {"edge_drop_p": (0.2, 0.4), "node_drop_p": (0.1, 0.2)},
        "dense":      {"edge_add_ratio": 0.05, "feature_mask_p": (0.2, 0.4)},
        "perceptual": {"subgraph_walk_length": 50, "feature_mask_p": (0.2, 0.4)},
        "structured": {"feature_mask_p": (0.2, 0.3)},
    },


    "controller_hidden_dim": 32,
    "controller_stat_dim": 3,
    "controller_num_params": 8,
    "learnable_param_bounds": {
        "edge_drop_p_a":      (0.05, 0.50),
        "edge_drop_p_b":      (0.05, 0.50),
        "node_drop_p_a":      (0.00, 0.30),
        "node_drop_p_b":      (0.00, 0.30),
        "feature_mask_p_a":   (0.05, 0.50),
        "feature_mask_p_b":   (0.05, 0.50),
        "edge_add_ratio":     (0.00, 0.15),
        "feature_noise_std":  (0.00, 0.15),
    },
    "edge_add_pool_ratio": 0.20,

    # ---- Shared GCN Encoder (Stage 4) ----------------------------------------
    "hidden_dim": 256,
    "embed_dim": 128,
    "encoder_dropout": 0.3,

    # ---- Projection Head (Stage 5) -------------------------------------------
    "proj_hidden_dim": 128,
    "proj_out_dim": 64,

    # ---- Multi-Term Contrastive Loss (Stage 6) --------------------------------
    "temperature": 0.4,
    "alpha_intra": 1.0,
    "beta_cross": 1.0,
    "gamma_hybrid": 0.5,
    # Normalize the multi-term contrastive loss by the sum of its ACTIVE term
    # weights, i.e. use a weighted MEAN of (intra, cross, hybrid) instead of a
    # raw weighted SUM. Without this, AGAMC-Full (3 active terms) carries a
    # systematically larger loss magnitude than the Table-8 ablations that drop
    # a term (NoCross / NoHybrid, 2 terms) or two terms (IntraOnly, 1 term).
    # Under a fixed lr + grad_clip_norm, a larger loss magnitude means a larger
    # gradient norm and therefore HARDER gradient clipping, giving Full a
    # smaller effective step size than the leaner variants - an optimization
    # artifact that penalizes the full model purely for having more terms and
    # makes the Table 8 comparison unfair to it. Normalizing puts every variant
    # on the same loss scale so the ablation measures the terms' VALUE, not a
    # gradient-scale side effect. Set False to recover the old raw-sum behavior.
    "normalize_loss_weights": True,


    # ---- Optimization ----------------------------------------------------------
    "epochs": 300,
    "lr": 2e-3,
    "lr_schedule_enabled": True,
    "min_lr": 1e-5,
    "warmup_epochs": 20,
    "weight_decay": 5e-4,
    "log_every": 1,
    "eval_every": 10,
    "cl_batch_size": 1024,
    # Gradient-norm clipping applied every step (helps most with
    # augmentation_mode="learnable", where the STE/controller path is more
    # prone to occasional gradient spikes than the fixed rule_based/uniform
    # augmentations). Set to None to disable clipping entirely.
    "grad_clip_norm": 3.0,
    # If a step produces a non-finite loss (can still happen very rarely,
    # e.g. an unlucky random augmentation collapsing a whole view), that
    # step is skipped (no optimizer.step()) instead of crashing the whole
    # run - a warning is printed and training continues from the last good
    # weights. This never masks a systematic divergence: if it keeps
    # happening, epochs_without_improvement / early stopping will still
    # kick in based on the label-free Silhouette probe as before.
    "nan_guard_enabled": True,
    # Optional deep-debug switch: wraps training in
    # torch.autograd.set_detect_anomaly(True), which makes PyTorch print the
    # exact forward op responsible the first time a NaN/Inf gradient is
    # produced (very informative, but SIGNIFICANTLY slower - only turn this
    # on temporarily to diagnose a persistent NaN issue, not for normal runs).
    "debug_detect_anomaly": False,

    # ---- Early Stopping (label-free, based on the Stage-10-style Silhouette probe) --
    "early_stopping_enabled": True,
    "early_stop_patience": 8,

    # ---- Fusion Module (Stage 9) ------------------------------------------------
    "fusion_mode": "average",
    "fusion_align_loss": "infonce",
    "fusion_entropy_coef": 0.1,
    "fusion_train_steps": 50,
    "fusion_lr": 1e-2,

    # ---- Cluster Number Estimation (Stage 10) -----------------------------------
    "auto_estimate_k": False,
    "k_estimation_method": "silhouette",
    "k_search_range": (2, 15),
    "eigengap_knn_k": 10,

    # ---- Reporting only (NEVER used for training/model-selection decisions) ------
    "enable_readonly_diagnostics": True,
    "output_dir": os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"),

    # ---- Silhouette scoring: caps the number of points used to ESTIMATE the
    # Silhouette coefficient (also reused below to subsample similarity-heatmap
    # and graph-structure visualizations on large datasets). Set to None to
    # always use the exact full-N computation.
    "silhouette_sample_size": 5000,

    # ---- Output artifacts (plots / CSV logs / JSON summary) -----------------
    # Set enabled=False to skip ALL plotting entirely (plots are the slow part
    # for large N). Final ACC/NMI/ARI/F1/Silhouette numbers are still
    # computed/printed either way - this only controls whether files get
    # written. When enabled, "include" gives finer-grained control over which
    # individual visualizations get produced (see module docstring above).
    "output_artifacts": {
        "enabled": False,
        "include": {
            "training_curves": False,
            "tsne_umap": False,
            "per_view_vs_fused": False,
            "confusion_matrix": False,
            "similarity_heatmap": False,
            "attention_weights": False,
            "graph_structure": False,    # default OFF - slow/cluttered on large datasets
            "interactive_3d": False,      # now a multi-view explorer (dropdown: per view + fused)
            "study_bar_charts": False,    # Table 8 / 9-12 / 13 grouped bar charts
            "radar_charts": False,        # Table 8 / 9-12 / 13 multi-metric radar/spider charts
            "silhouette_diagram": False,  # classic per-sample silhouette plot (Stage 11)
            "cluster_size_distribution": False,   # predicted vs ground-truth cluster sizes
            "view_degree_distribution": False,    # log-log degree histogram per view (Stage 1)
            "multi_seed_comparison": False,       # combined box/bar/radar across ALL seeds
            "hyperparameter_sensitivity_curves": False,  # Table 14 sweep curves
            "loss_weight_heatmap": False,                # Table 14 optional beta/gamma joint heatmap
            "scalability_curves": False,                 # Table 15 time/memory curves
        },
    },

    # =========================================================================
    # ---- SINGLE CONTROL POINT: WHICH PAPER-STYLE TABLES TO PRODUCE ----------
    # =========================================================================
    # Everything related to "which tables get produced" lives under this ONE
    # nested dict. There is no other top-level "ablation_study" /
    # "augmentation_category_study" / "rule_vs_learnable_study" dict anywhere
    # in this file - every function below reads its settings from here.
    #
    #   main_comparison.enabled                  -> Table 4  - main ACC comparison
    #   ablation_study.enabled                    -> Table 8  - component-removal ablation
    #   augmentation_category.sparse.enabled      -> Table 9  - sparse views
    #   augmentation_category.dense.enabled       -> Table 10 - dense views
    #   augmentation_category.perceptual.enabled  -> Table 11 - perceptual views
    #   augmentation_category.structured.enabled  -> Table 12 - structured views
    #   rule_vs_learnable.enabled                 -> Table 13 - rule-based vs learnable
    #   hyperparameter_sensitivity.<param>.enabled -> Table 14 - hyperparameter sweeps
    #   scalability_analysis.<dimension>.enabled   -> Table 15 - scalability sweeps
    #
    # If a table needs a view type that is not present in the currently
    # selected dataset (DATASET_REGISTRY), that table is skipped automatically
    # with a warning message (see the sync block right after CONFIG). At the
    # end of the run, every table enabled here is printed again, together, in
    # one consolidated "FINAL REPORT" section.
    # =========================================================================
    "reports": {
        "main_comparison": {
            "enabled": False,        # Table 4  - main ACC comparison
        },
        "table5_nmi": {
            "enabled": False,        # Table 5  - NMI comparison across datasets
        },
        "table6_ari": {
            "enabled": False,        # Table 6  - ARI comparison across datasets
        },
        "table7_f1": {
            "enabled": False,        # Table 7  - Macro F1 comparison across datasets
        },
        "main_comparison_radar": {
            "enabled": False,        # 4.2.2 radar/spider chart (ACC/NMI/ARI/F1)
        },
        "ablation_study": {
            "enabled": False,    # Table 8
        },
        "augmentation_category": {
            "sparse":     {"enabled": True},   # Table 9 Cora
            "dense":      {"enabled": False},   # Table 10 ACM  
            "perceptual": {"enabled": False},   # Table 11 YouTube
            "structured": {"enabled": False},   # Table 12 Citeseer, Cora
            "ops_per_view_type": {
                "sparse":     ["edge_drop", "node_drop", "edge_add", "feature_mask", "subgraph_extract"],
                "dense":      ["edge_add", "feature_mask", "edge_drop", "node_drop", "subgraph_extract"],
                "perceptual": ["subgraph_extract", "feature_mask", "edge_drop", "edge_add", "node_drop"],
                "structured": ["feature_mask", "feature_noise", "edge_drop", "node_drop", "edge_add"],
            },
            "subgraph_extract_walk_length": 50,
            "output_tables": False,
            "combined_figure": False,   # 4.3.2: single 2x2 figure merging Tables 9-12
            # "target_view_types" is populated automatically in the sync
            # block below - do not edit by hand.
            "target_view_types": [],
        },
        "rule_vs_learnable": {
            "enabled": False,    # Table 13
            "output_table": False,
            "annotate_cost": False,        # 4.5: annotate +X% training-time cost
            "learnable_cost_pct": 30.0,   # informational only - not measured automatically
        },
        "hyperparameter_sensitivity": {
            "edge_drop_rate":       {"enabled": False, "values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]},
            "feature_mask_rate":    {"enabled": False, "values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]},
            "knn_k":                {"enabled": False, "values": [3, 5, 8, 10, 15, 20]},
            "subgraph_walk_length": {"enabled": False, "values": [20, 30, 50, 80, 100, 150, 200]},
            "temperature":          {"enabled": False, "values": [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]},
            "alpha_intra":          {"enabled": False, "values": [0.0, 0.5, 1.0, 1.5, 2.0]},
            "beta_cross":           {"enabled": False, "values": [0.0, 0.5, 1.0, 1.5, 2.0]},
            "gamma_hybrid":         {"enabled": False, "values": [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]},
            "beta_gamma_joint_heatmap": {"enabled": False},   # optional, see plot_loss_weight_joint_heatmap
            "alpha_beta_joint_heatmap": {"enabled": False},   # 4.4.2: alpha_intra x beta_cross ACC heatmap
            "hidden_dim":           {"enabled": False, "values": [32, 64, 128, 192, 256]},
            "gcn_layers":           {"enabled": False, "values": [1, 2, 3, 4]},
            "seeds": None,          # None -> falls back to CONFIG["seeds"]
            "output_tables": False,
        },
        "scalability_analysis": {
            "vs_batch_size": {"enabled": False, "values": [256, 512, 1024, 2048]},
            "vs_num_views":  {"enabled": False},   # sweeps K = 1..len(current dataset's real views)
            "vs_num_nodes":  {"enabled": False, "values": [500, 1000, 2000, "full"]},
            "measure_epochs": 10,     # short run length used ONLY for these measurements
            "seeds": None,
            "output_tables": False,
            "combined_figure": False,  # 4.6: single 2x2 figure merging time/views/nodes
        },
    },
}

# ---- Sync dataset-dependent settings from DATASET_REGISTRY -------------------
if CONFIG["dataset_name"] not in DATASET_REGISTRY:
    raise ValueError(
        f"Unknown dataset_name {CONFIG['dataset_name']!r}. "
        f"Must be one of: {list(DATASET_REGISTRY.keys())}"
    )
_DATASET_SPEC = DATASET_REGISTRY[CONFIG["dataset_name"]]
NUM_CLUSTERS_KNOWN = _DATASET_SPEC["num_classes"]
CONFIG["num_views"] = _DATASET_SPEC["num_views"]

# ---- Sync reports["augmentation_category"] -> target_view_types --------------
# Collects which view types were requested (their own "enabled" flag is
# True) AND are actually present in the chosen dataset. Any requested view
# type absent from DATASET_REGISTRY[dataset_name] is skipped with a warning.
_available_view_types = {v["type"] for v in _DATASET_SPEC["views"]}
_aug_cat_cfg = CONFIG["reports"]["augmentation_category"]
_selected_view_types = []
for _vtype in ("sparse", "dense", "perceptual", "structured"):
    if _aug_cat_cfg.get(_vtype, {}).get("enabled", False):
        if _vtype in _available_view_types:
            _selected_view_types.append(_vtype)
        else:
            print(f"[Reports] WARNING: augmentation_category.{_vtype} was requested but "
                  f"dataset '{CONFIG['dataset_name']}' has no '{_vtype}' view - skipping it.")
_aug_cat_cfg["target_view_types"] = _selected_view_types


# ==============================================================================
# Utility: reproducibility
# ==============================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_config(cfg: dict):
    print("=" * 78)
    print(" MULTI-VIEW GRAPH CONTRASTIVE CLUSTERING - CONFIGURATION")
    print("=" * 78)
    print(f"\n[Dataset - single source of truth: DATASET_REGISTRY[dataset_name]]")
    print(f"  dataset_name        : {cfg['dataset_name']}")
    print(f"  NUM_CLUSTERS_KNOWN  : {NUM_CLUSTERS_KNOWN}")
    print(f"  num_views           : {cfg['num_views']}")
    print(f"\n[Reports - single control point for which tables get produced]")
    for k, v in cfg["reports"].items():
        print(f"  {k:<34s}: {v}")
    for section, keys in [
        ("Run / Device", ["seed", "device"]),
        ("Pre-declared Seed List", ["seeds"]),
        ("Dataset paths", ["data_root"]),
        ("Multi-View Construction (Stage 1)", ["num_views", "knn_k", "ppr_alpha", "ppr_topk"]),
        ("Adaptive Augmentation Selector (Stage 2)", ["augmentation_mode"]),
        ("Augmentation - rule_based params", ["edge_drop_p", "node_drop_p", "edge_add_ratio",
                           "feature_mask_p", "feature_noise_std"]),
        ("Augmentation - learnable controller params", ["controller_hidden_dim", "controller_stat_dim",
                           "controller_num_params", "learnable_param_bounds", "edge_add_pool_ratio"]),
        ("Shared GCN Encoder (Stage 4)", ["hidden_dim", "embed_dim", "encoder_dropout"]),
        ("Projection Head (Stage 5)", ["proj_hidden_dim", "proj_out_dim"]),
        ("Contrastive Loss (Stage 6)", ["temperature", "alpha_intra", "beta_cross", "gamma_hybrid"]),
        ("Optimization", ["epochs", "lr", "lr_schedule_enabled", "min_lr", "warmup_epochs",
                           "weight_decay", "log_every", "eval_every", "cl_batch_size"]),
        ("Early Stopping", ["early_stopping_enabled", "early_stop_patience"]),
        ("Fusion (Stage 9)", ["fusion_mode", "fusion_align_loss", "fusion_entropy_coef",
                           "fusion_train_steps", "fusion_lr"]),
        ("Cluster Number Estimation (Stage 10)", ["auto_estimate_k", "k_estimation_method",
                           "k_search_range", "eigengap_knn_k"]),
        ("Reporting (read-only, never affects training)", ["enable_readonly_diagnostics", "output_dir",
                           "silhouette_sample_size", "output_artifacts"]),
    ]:
        print(f"\n[{section}]")
        for k in keys:
            print(f"  {k:<20s}: {cfg[k]}")

    print(f"\n[Ablation Study (Table 8 style)]")
    print(f"  enabled            : {cfg['reports'].get('ablation_study', {}).get('enabled', False)}")

    print(f"\n[Augmentation-Category Study (Table 9-12 style)]")
    aug_cat = cfg["reports"].get("augmentation_category", {})
    print(f"  target_view_types  : {aug_cat.get('target_view_types', [])}")
    for view_type in ["sparse", "dense", "perceptual", "structured"]:
        if view_type in aug_cat:
            print(f"  {view_type:<18s}: enabled={aug_cat[view_type].get('enabled', False)}")

    print(f"\n[Rule-Based vs. Learnable Study (Table 13 style)]")
    rvl = cfg["reports"].get("rule_vs_learnable", {})
    print(f"  enabled            : {rvl.get('enabled', False)}")
    print(f"  output_table       : {rvl.get('output_table', False)}")

    print(f"\n[Hyperparameter Sensitivity Study (Table 14 style)]")
    hp = cfg["reports"].get("hyperparameter_sensitivity", {})
    for pname in ("edge_drop_rate", "feature_mask_rate", "knn_k", "subgraph_walk_length",
                  "temperature", "alpha_intra", "beta_cross", "gamma_hybrid",
                  "beta_gamma_joint_heatmap", "hidden_dim", "gcn_layers"):
        if pname in hp:
            print(f"  {pname:<26s}: enabled={hp[pname].get('enabled', False)}")

    print(f"\n[Scalability Analysis (Table 15 style)]")
    sc = cfg["reports"].get("scalability_analysis", {})
    for dname in ("vs_batch_size", "vs_num_views", "vs_num_nodes"):
        if dname in sc:
            print(f"  {dname:<26s}: enabled={sc[dname].get('enabled', False)}")
    print(f"  {'measure_epochs':<26s}: {sc.get('measure_epochs', 10)}")

    print("=" * 78 + "\n")


# ==============================================================================
# LR SCHEDULE: constant, or linear warmup -> cosine annealing
# ==============================================================================

def get_lr_for_epoch(epoch: int, cfg: dict) -> float:
    if not cfg["lr_schedule_enabled"]:
        return cfg["lr"]

    base_lr = cfg["lr"]
    min_lr = cfg["min_lr"]
    warmup = max(cfg["warmup_epochs"], 0)
    total = cfg["epochs"]

    if warmup > 0 and epoch <= warmup:
        return base_lr * epoch / warmup

    progress = (epoch - warmup) / max(total - warmup, 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine_factor


# ==============================================================================
# STAGE 1: MULTI-VIEW GRAPH INPUTS  (dataset-driven, see DATASET_REGISTRY)
# ==============================================================================

def build_knn_view(x: torch.Tensor, k: int):
    """Generic k-NN graph on cosine-normalized features - used by both the
    Planetoid path (content-feature view) and the generic path (term /
    thumbnail / comment feature views)."""
    x_np = F.normalize(x, p=2, dim=1).cpu().numpy()
    adj = kneighbors_graph(x_np, n_neighbors=k, mode="connectivity",
                            include_self=False, n_jobs=-1)

    adj = adj.maximum(adj.T)
    edge_index = torch.tensor(np.vstack(adj.nonzero()), dtype=torch.long)
    return edge_index


def build_ppr_view(edge_index: torch.Tensor, num_nodes: int, alpha: float, top_k: int):
    """Personalized PageRank diffusion graph (planetoid path only)."""
    ei = to_undirected(edge_index, num_nodes=num_nodes)
    row, col = ei
    deg = degree(row, num_nodes=num_nodes).clamp(min=1)
    values = torch.ones(row.size(0))
    A = sp.coo_matrix((values.numpy(), (row.numpy(), col.numpy())),
                       shape=(num_nodes, num_nodes))
    D_inv = sp.diags(1.0 / deg.numpy())
    A_norm = D_inv @ A

    I = sp.identity(num_nodes, format="csc")
    S = I.tocsr()
    part = I.tocsr()
    A_norm = A_norm.tocsr()
    for _ in range(10):
        part = (1 - alpha) * (A_norm @ part)
        S = S + part
    S = alpha * S

    S = S.tocsr()
    rows, cols = [], []
    for i in range(num_nodes):
        start, end = S.indptr[i], S.indptr[i + 1]
        row_vals = S.data[start:end]
        row_idx = S.indices[start:end]
        if len(row_vals) > top_k:
            keep = np.argpartition(-row_vals, top_k)[:top_k]
            row_idx, row_vals = row_idx[keep], row_vals[keep]
        rows.extend([i] * len(row_idx))
        cols.extend(row_idx.tolist())
    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    return edge_index


def load_planetoid_dataset(cfg, knn_k_override=None):
    """Loads Cora or Citeseer via torch_geometric.datasets.Planetoid and
    builds its 4 views exactly like the original single-dataset script:
        1 sparse     -> original citation graph
        2 dense      -> kNN graph on content features
        3 perceptual -> PPR diffusion over the citation graph
        4 structured -> clone of the citation graph (feature-mask-only view)

    knn_k_override: used only by the Table 14 hyperparameter-sensitivity
    "knn_k" sweep - overrides the k used for the dense kNN view for THIS
    call only (never mutates cfg or any global registry).
    """
    spec = DATASET_REGISTRY[cfg["dataset_name"]]
    planetoid_name = spec["planetoid_name"]
    root = os.path.join(cfg["data_root"], cfg["dataset_name"])
    os.makedirs(root, exist_ok=True)
    dataset = Planetoid(root=root, name=planetoid_name)
    data = dataset[0]
    print(f"[Data] {planetoid_name} loaded from '{root}' "
          f"(nodes={data.num_nodes}, edges={data.num_edges}, "
          f"features={data.num_node_features}, classes={dataset.num_classes})")

    num_nodes = data.num_nodes
    x = data.x
    edge_index_orig = to_undirected(data.edge_index, num_nodes=num_nodes)

    effective_knn_k = knn_k_override if knn_k_override is not None else cfg["knn_k"]

    views = []
    for vdef in spec["views"][: cfg["num_views"]]:
        build = vdef["build"]
        if build == "planetoid_citation":
            ei = edge_index_orig
        elif build == "planetoid_knn":
            ei = build_knn_view(x, effective_knn_k)
        elif build == "planetoid_ppr":
            ei = build_ppr_view(edge_index_orig, num_nodes, cfg["ppr_alpha"], cfg["ppr_topk"])
        elif build == "planetoid_structured":
            ei = edge_index_orig.clone()
        else:
            raise ValueError(f"Unsupported planetoid view build {build!r}")
        views.append({"type": vdef["type"], "edge_index": ei})
        print(f"[Views] built view ({vdef['type']:<10s} / {build}): {ei.size(1)} directed edges")

    return x, data.y, views, dataset.num_classes


def load_generic_dataset(cfg, knn_k_override=None):
    """
    Loads ACM / DBLP / Amazon / YouTube from a local folder in the standardized
    format documented at the top of this file:
        <data_root>/<dataset_name>/features.npy
        <data_root>/<dataset_name>/labels.npy
        <data_root>/<dataset_name>/graph_<name>.npz   (scipy sparse, real relations)
        <data_root>/<dataset_name>/feat_<name>.npy    (dense features for kNN views)

    knn_k_override: used only by the Table 14 hyperparameter-sensitivity
    "knn_k" sweep. DATASET_REGISTRY[dataset_name]["views"] is a MODULE-LEVEL
    GLOBAL - we never mutate it in place (that would corrupt every later
    sweep iteration and every other caller). Instead, when an override is
    given, we build a LOCAL deep-copied, patched view-list for this call
    only: the knn_k of whichever view has type=="dense" and
    build=="knn_file" is replaced (falling back to a "perceptual" knn_file
    view if no "dense" one exists), and that patched list - never the
    global registry - is what actually gets used to build the views.
    """
    name = cfg["dataset_name"]
    spec = DATASET_REGISTRY[name]
    root = os.path.join(cfg["data_root"], name)

    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"[Data] Expected a folder at '{root}' with '{spec['features_file']}', "
            f"'{spec['labels_file']}' and the per-view files listed in "
            f"DATASET_REGISTRY['{name}']['views'], but the folder does not exist. "
            f"See the file-format docstring at the top of this script."
        )

    feat_path = os.path.join(root, spec["features_file"])
    label_path = os.path.join(root, spec["labels_file"])
    x_np = np.load(feat_path)
    y_np = np.load(label_path)
    x = torch.tensor(x_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.long)
    num_nodes = x.size(0)

    print(f"[Data] '{name}' loaded from '{root}' "
          f"(nodes={num_nodes}, features={x.size(1)}, classes={spec['num_classes']})")

    # ---- Build a LOCAL, patched view-list when knn_k_override is given -----
    # (DATASET_REGISTRY is never mutated - see docstring above.)
    views_spec = spec["views"][: cfg["num_views"]]
    if knn_k_override is not None:
        views_spec = copy.deepcopy(views_spec)
        patch_idx = None
        for i, vdef in enumerate(views_spec):
            if vdef.get("build") == "knn_file" and vdef.get("type") == "dense":
                patch_idx = i
                break
        if patch_idx is None:
            for i, vdef in enumerate(views_spec):
                if vdef.get("build") == "knn_file" and vdef.get("type") == "perceptual":
                    patch_idx = i
                    break
        if patch_idx is not None:
            views_spec[patch_idx]["knn_k"] = knn_k_override
            print(f"[HyperparamSweep] knn_k override={knn_k_override} applied to LOCAL "
                  f"view #{patch_idx} ('{views_spec[patch_idx]['type']}') only - "
                  f"DATASET_REGISTRY left untouched.")
        else:
            print(f"[HyperparamSweep] WARNING: knn_k override requested but dataset "
                  f"'{name}' has no 'dense' or 'perceptual' knn_file view to patch - "
                  f"override ignored for this dataset.")

    views = []
    for vdef in views_spec:
        build = vdef["build"]
        if build == "graph_file":
            path = os.path.join(root, vdef["file"])
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"[Data] Missing graph file '{path}' required for the "
                    f"'{vdef['type']}' view of dataset '{name}'."
                )
            adj = sp.load_npz(path)
            adj = adj.maximum(adj.T)  # symmetrize, in case it was saved directed
            ei = torch.tensor(np.vstack(adj.nonzero()), dtype=torch.long)
        elif build == "knn_file":
            path = os.path.join(root, vdef["file"])
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"[Data] Missing feature file '{path}' required to build the "
                    f"k-NN '{vdef['type']}' view of dataset '{name}'."
                )
            feat_np = np.load(path)
            feat_t = torch.tensor(feat_np, dtype=torch.float32)
            ei = build_knn_view(feat_t, vdef.get("knn_k", 10))
        else:
            raise ValueError(f"Unsupported generic view build {build!r}")

        ei = to_undirected(ei, num_nodes=num_nodes)
        views.append({"type": vdef["type"], "edge_index": ei})
        print(f"[Views] built view ({vdef['type']:<10s} / {build} <- {vdef.get('file')}): "
              f"{ei.size(1)} directed edges")

    return x, y, views, spec["num_classes"]

# ==============================================================================
# CACHE: view/feature construction is deterministic per dataset (no seed
# dependence) - caching it avoids rebuilding the same k-NN graphs / loading
# the same files from scratch on every single seed of a multi-seed run.
# Only used when knn_k_override is None (the hyperparameter "knn_k" sweep
# still bypasses the cache, since it needs a different k each time).
# ==============================================================================
_DATASET_CACHE = {}


def load_dataset(cfg, knn_k_override=None):
    """
    Single Stage-1 entry point used by the rest of the pipeline. Dispatches to
    the Planetoid loader (cora/citeseer) or the generic local-folder loader
    (acm/dblp/amazon/youtube) based on DATASET_REGISTRY[cfg['dataset_name']].

    knn_k_override: optional int, used ONLY by the Table 14
    hyperparameter-sensitivity "knn_k" sweep (run_hyperparameter_variant).
    For the planetoid path it overrides cfg["knn_k"] for this call only; for
    the generic path it patches a LOCAL copy of the relevant view's knn_k
    (see load_generic_dataset docstring) - the global DATASET_REGISTRY is
    never mutated either way. None (the default) reproduces the original,
    unmodified loading behavior exactly.

    CACHING (new): when knn_k_override is None, the (x, gt_labels, views,
    num_classes) tuple is 100% deterministic for a given dataset_name (no
    randomness anywhere in view construction), so it is built ONCE and
    reused for every subsequent call across all seeds/studies - this avoids
    redundantly rebuilding k-NN graphs (and, for planetoid, the PPR
    diffusion graph) from scratch on every single seed of a multi-seed run.
    Returned tensors are cloned so downstream code can freely mutate them
    (e.g. in-place augmentation) without corrupting the cached copy.

    Returns: (x, gt_labels_readonly, views, num_classes) - exactly the four
    values every downstream stage (train(), fuse_embeddings(), evaluate_*(),
    ...) already expects, unchanged from the original Cora-only script.
    """
    cache_key = cfg["dataset_name"]
    if knn_k_override is None and cache_key in _DATASET_CACHE:
        x_c, y_c, views_c, num_classes_c = _DATASET_CACHE[cache_key]
        x_out = x_c.clone()
        y_out = y_c.clone()
        views_out = [{"type": v["type"], "edge_index": v["edge_index"].clone()} for v in views_c]
        print(f"[Data] '{cache_key}' views reused from cache (deterministic - "
              f"skipped rebuilding k-NN/PPR graphs again for this seed).")
        return x_out, y_out, views_out, num_classes_c

    spec = DATASET_REGISTRY[cfg["dataset_name"]]
    if spec["loader"] == "planetoid":
        result = load_planetoid_dataset(cfg, knn_k_override=knn_k_override)
    elif spec["loader"] == "generic":
        result = load_generic_dataset(cfg, knn_k_override=knn_k_override)
    else:
        raise ValueError(f"Unknown loader type {spec['loader']!r} for dataset {cfg['dataset_name']!r}")

    if knn_k_override is None:
        x_r, y_r, views_r, num_classes_r = result
        _DATASET_CACHE[cache_key] = (
            x_r.clone(), y_r.clone(),
            [{"type": v["type"], "edge_index": v["edge_index"].clone()} for v in views_r],
            num_classes_r,
        )

    return result
# def load_dataset(cfg, knn_k_override=None):
#     """
#     Single Stage-1 entry point used by the rest of the pipeline. Dispatches to
#     the Planetoid loader (cora/citeseer) or the generic local-folder loader
#     (acm/dblp/amazon/youtube) based on DATASET_REGISTRY[cfg['dataset_name']].

#     knn_k_override: optional int, used ONLY by the Table 14
#     hyperparameter-sensitivity "knn_k" sweep (run_hyperparameter_variant).
#     For the planetoid path it overrides cfg["knn_k"] for this call only; for
#     the generic path it patches a LOCAL copy of the relevant view's knn_k
#     (see load_generic_dataset docstring) - the global DATASET_REGISTRY is
#     never mutated either way. None (the default) reproduces the original,
#     unmodified loading behavior exactly.

#     Returns: (x, gt_labels_readonly, views, num_classes) - exactly the four
#     values every downstream stage (train(), fuse_embeddings(), evaluate_*(),
#     ...) already expects, unchanged from the original Cora-only script.
#     """
#     spec = DATASET_REGISTRY[cfg["dataset_name"]]
#     if spec["loader"] == "planetoid":
#         return load_planetoid_dataset(cfg, knn_k_override=knn_k_override)
#     elif spec["loader"] == "generic":
#         return load_generic_dataset(cfg, knn_k_override=knn_k_override)
#     else:
#         raise ValueError(f"Unknown loader type {spec['loader']!r} for dataset {cfg['dataset_name']!r}")


# ==============================================================================
# STAGE 2 & 3: ADAPTIVE AUGMENTATION SELECTION + TWO AUGMENTED GRAPHS PER VIEW
# (UNCHANGED from the original script - dataset-agnostic by design)
# ==============================================================================

def edge_drop(edge_index, p):
    if p <= 0:
        return edge_index
    mask = torch.rand(edge_index.size(1), device=edge_index.device) > p
    return edge_index[:, mask]


def node_drop_features(x, p):
    if p <= 0:
        return x
    x = x.clone()
    n = x.size(0)
    drop_idx = torch.randperm(n, device=x.device)[: int(n * p)]
    x[drop_idx] = 0
    return x


def feature_mask(x, p):
    if p <= 0:
        return x
    mask = (torch.rand_like(x) > p).float()
    return x * mask


def add_feature_noise(x, std):
    if std <= 0:
        return x
    return x + torch.randn_like(x) * std


def random_edge_add(edge_index, num_nodes, ratio):
    if ratio <= 0:
        return edge_index
    n_add = int(edge_index.size(1) * ratio)
    new_src = torch.randint(0, num_nodes, (n_add,), device=edge_index.device)
    new_dst = torch.randint(0, num_nodes, (n_add,), device=edge_index.device)
    extra = torch.stack([new_src, new_dst], dim=0)
    return torch.cat([edge_index, extra], dim=1)


def subgraph_extract(edge_index, num_nodes, walk_length):
    """'Subgraph Extract' augmentation (GraphCL-style): starts a random walk
    from EVERY node in parallel, walks up to `walk_length` steps over the
    (directed, as given) graph, then keeps only the edges induced on the set
    of ALL visited nodes (union over all per-node walks). Vectorized via a
    numpy-built adjacency; the keep-mask is built and moved back onto
    `edge_index`'s original device at the very end (critical device-mismatch
    fix - do not remove the `.to(device)` below).
    Only used by the Table 9-12 augmentation-category study below - it never
    runs inside the normal training loop unless explicitly selected there."""
    if edge_index.size(1) == 0 or walk_length <= 0:
        return edge_index
    device = edge_index.device
    row, col = edge_index[0], edge_index[1]

    row_np = row.detach().cpu().numpy()
    col_np = col.detach().cpu().numpy()
    order = np.argsort(row_np, kind="stable")
    row_sorted = row_np[order]
    col_sorted = col_np[order]
    boundaries = np.searchsorted(row_sorted, np.arange(num_nodes + 1))

    visited = np.zeros(num_nodes, dtype=bool)
    current = np.arange(num_nodes)
    visited[current] = True
    for _ in range(walk_length):
        next_nodes = current.copy()
        for i, node in enumerate(current):
            start, end = boundaries[node], boundaries[node + 1]
            neighbors = col_sorted[start:end]
            if neighbors.size > 0:
                next_nodes[i] = neighbors[np.random.randint(neighbors.size)]
        current = next_nodes
        visited[current] = True

    keep_mask_np = visited[row_np] & visited[col_np]
    keep_mask = torch.from_numpy(keep_mask_np).to(device)  # <-- critical .to(device)
    if keep_mask.sum() == 0:
        return edge_index
    return edge_index[:, keep_mask]


def augment_view(view, x, cfg, num_nodes):
    vtype = view["type"]
    ei = view["edge_index"]
    va = cfg.get("rule_based_view_aug", {}).get(vtype, {})

    if vtype == "sparse":
        edp = va.get("edge_drop_p", (0.2, 0.4))
        ndp = va.get("node_drop_p", (0.1, 0.2))
        ei_a, x_a = edge_drop(ei, edp[0]), node_drop_features(x, ndp[0])
        ei_b, x_b = edge_drop(ei, edp[1]), node_drop_features(x, ndp[1])

    elif vtype == "dense":
        ear = va.get("edge_add_ratio", 0.05)
        fmp = va.get("feature_mask_p", (0.2, 0.4))
        ei_a, x_a = random_edge_add(ei, num_nodes, ear), feature_mask(x, fmp[0])
        ei_b, x_b = random_edge_add(ei, num_nodes, ear), feature_mask(x, fmp[1])

    elif vtype == "perceptual":
        walk_len = va.get("subgraph_walk_length", 50)
        fmp = va.get("feature_mask_p", (0.2, 0.4))
        ei_a, x_a = subgraph_extract(ei, num_nodes, walk_len), feature_mask(x, fmp[0])
        ei_b, x_b = subgraph_extract(ei, num_nodes, walk_len), feature_mask(x, fmp[1])

    else:  # structured
        fmp = va.get("feature_mask_p", (0.2, 0.3))
        ei_a, x_a = ei, feature_mask(x, fmp[0])
        ei_b, x_b = ei, feature_mask(x, fmp[1])

    ei_a, _ = add_self_loops(ei_a, num_nodes=num_nodes)
    ei_b, _ = add_self_loops(ei_b, num_nodes=num_nodes)
    return (ei_a, x_a), (ei_b, x_b)

def compute_view_stats(edge_index, x, num_nodes):
    device = x.device
    num_edges = edge_index.size(1)
    density = num_edges / max(float(num_nodes) ** 2, 1.0)
    deg = degree(edge_index[0], num_nodes=num_nodes)
    avg_degree = math.log1p(deg.mean().item())
    feature_variance = math.log1p(x.var().item())
    stats = torch.tensor([density, avg_degree, feature_variance],
                         dtype=torch.float32, device=device)
    return stats


class AugmentationController(nn.Module):
    def __init__(self, stat_dim, hidden_dim, num_params):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(stat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_params),
        )

    def forward(self, stats):
        raw = self.net(stats)
        return torch.sigmoid(raw)


_LEARNABLE_PARAM_ORDER = [
    "edge_drop_p_a", "edge_drop_p_b",
    "node_drop_p_a", "node_drop_p_b",
    "feature_mask_p_a", "feature_mask_p_b",
    "edge_add_ratio", "feature_noise_std",
]


def unpack_learnable_params(raw_params, cfg):
    bounds = cfg["learnable_param_bounds"]
    out = {}
    for i, key in enumerate(_LEARNABLE_PARAM_ORDER):
        lo, hi = bounds[key]
        out[key] = lo + raw_params[i] * (hi - lo)
    return out


def ste_bernoulli_mask(keep_prob, shape, device):
    keep_prob = keep_prob.clamp(1e-4, 1 - 1e-4)
    soft = keep_prob.expand(shape)
    rand = torch.rand(shape, device=device)
    hard = (rand < soft).float()
    mask = hard.detach() + soft - soft.detach()
    return mask


def edge_drop_learnable(edge_index, keep_prob, device):
    m = edge_index.size(1)
    mask = ste_bernoulli_mask(keep_prob, (m,), device)
    return edge_index, mask


def node_drop_features_learnable(x, keep_prob):
    n = x.size(0)
    mask = ste_bernoulli_mask(keep_prob, (n, 1), x.device)
    return x * mask


def feature_mask_learnable(x, keep_prob):
    mask = ste_bernoulli_mask(keep_prob, x.shape, x.device)
    return x * mask


def add_feature_noise_learnable(x, std):
    return x + torch.randn_like(x) * std


def random_edge_add_learnable(edge_index, num_nodes, ratio, pool_ratio, device):
    pool_size = max(int(edge_index.size(1) * pool_ratio), 1)
    candidates = torch.randint(0, num_nodes, (2, pool_size), device=device)
    mask = ste_bernoulli_mask(ratio, (pool_size,), device)
    keep_idx = mask.detach().bool()
    new_edges = candidates[:, keep_idx]
    new_weight = mask[keep_idx]
    return new_edges, new_weight


def augment_view_learnable(view, x, cfg, num_nodes, controller, device):
    vtype = view["type"]
    ei = view["edge_index"]

    stats = compute_view_stats(ei, x, num_nodes)
    raw_params = controller(stats)
    params = unpack_learnable_params(raw_params, cfg)

    if vtype == "sparse":
        ei_a, ew_a = edge_drop_learnable(ei, 1.0 - params["edge_drop_p_a"], device)
        x_a = node_drop_features_learnable(x, 1.0 - params["node_drop_p_a"])
        ei_b, ew_b = edge_drop_learnable(ei, 1.0 - params["edge_drop_p_b"], device)
        x_b = node_drop_features_learnable(x, 1.0 - params["node_drop_p_b"])

    elif vtype == "dense":
        new_edges_a, new_w_a = random_edge_add_learnable(
            ei, num_nodes, params["edge_add_ratio"], cfg["edge_add_pool_ratio"], device)
        ei_a = torch.cat([ei, new_edges_a], dim=1)
        ew_a = torch.cat([torch.ones(ei.size(1), device=device), new_w_a])
        x_a = feature_mask_learnable(x, 1.0 - params["feature_mask_p_a"])
        ei_b = ei
        ew_b = torch.ones(ei.size(1), device=device)
        x_b = feature_mask_learnable(x, 1.0 - params["feature_mask_p_b"])

    elif vtype == "perceptual":
        ei_a, ew_a = edge_drop_learnable(ei, 1.0 - params["edge_drop_p_a"], device)
        x_a = add_feature_noise_learnable(
            feature_mask_learnable(x, 1.0 - params["feature_mask_p_a"]), params["feature_noise_std"])
        ei_b, ew_b = edge_drop_learnable(ei, 1.0 - params["edge_drop_p_b"], device)
        x_b = add_feature_noise_learnable(
            feature_mask_learnable(x, 1.0 - params["feature_mask_p_b"]), params["feature_noise_std"])

    else:  # "structured"
        ei_a, ei_b = ei, ei
        ew_a = torch.ones(ei.size(1), device=device)
        ew_b = torch.ones(ei.size(1), device=device)
        x_a = feature_mask_learnable(x, 1.0 - params["feature_mask_p_a"])
        x_b = feature_mask_learnable(x, 1.0 - params["feature_mask_p_b"])

    ei_a, ew_a = add_self_loops(ei_a, ew_a, fill_value=1.0, num_nodes=num_nodes)
    ei_b, ew_b = add_self_loops(ei_b, ew_b, fill_value=1.0, num_nodes=num_nodes)
    return (ei_a, x_a, ew_a), (ei_b, x_b, ew_b)


def augment_view_uniform(view, x, cfg, num_nodes):
    ei = view["edge_index"]
    p1, p2 = cfg["feature_mask_p"]
    ei_a = edge_drop(ei, cfg["edge_drop_p"][0])
    x_a = feature_mask(x, p1)
    ei_b = edge_drop(ei, cfg["edge_drop_p"][1])
    x_b = feature_mask(x, p2)
    ei_a, _ = add_self_loops(ei_a, num_nodes=num_nodes)
    ei_b, _ = add_self_loops(ei_b, num_nodes=num_nodes)
    return (ei_a, x_a), (ei_b, x_b)


def augment_view_none(view, x, cfg, num_nodes):
    ei = view["edge_index"]
    ei_sl, _ = add_self_loops(ei, num_nodes=num_nodes)
    return (ei_sl, x), (ei_sl, x)

def augment_view_random(view, x, cfg, num_nodes):
    ei = view["edge_index"]

    def _random_branch():
        edp = random.uniform(0.1, 0.4)
        fmp = random.uniform(0.1, 0.4)
        return edge_drop(ei, edp), feature_mask(x, fmp)   # هر دو عملگر با هم

    ei_a, x_a = _random_branch()
    ei_b, x_b = _random_branch()
    ei_a, _ = add_self_loops(ei_a, num_nodes=num_nodes)
    ei_b, _ = add_self_loops(ei_b, num_nodes=num_nodes)
    return (ei_a, x_a), (ei_b, x_b)
 
 


def augment_view_dispatch(view, x, cfg, num_nodes, controller=None, device=None):
    mode = cfg["augmentation_mode"]
    if mode == "learnable":
        if controller is None:
            raise ValueError("augmentation_mode='learnable' requires a controller instance")
        return augment_view_learnable(view, x, cfg, num_nodes, controller, device)
    elif mode == "rule_based":
        (ei_a, x_a), (ei_b, x_b) = augment_view(view, x, cfg, num_nodes)
    elif mode == "uniform":
        (ei_a, x_a), (ei_b, x_b) = augment_view_uniform(view, x, cfg, num_nodes)
    elif mode == "none":
        (ei_a, x_a), (ei_b, x_b) = augment_view_none(view, x, cfg, num_nodes)
    elif mode == "random":
        (ei_a, x_a), (ei_b, x_b) = augment_view_random(view, x, cfg, num_nodes)
    else:
        raise ValueError(f"Unknown augmentation_mode: {mode!r}")
    return (ei_a, x_a, None), (ei_b, x_b, None)


# ==============================================================================
# AUGMENTATION-CATEGORY ABLATION HELPERS (Tables 9-12 style)
# Isolates ONE augmentation operator, instead of the type-specific combo
# `augment_view()` above uses. Used only by run_augmentation_category_study()
# further down - the normal training path never touches these.
# ==============================================================================

AUGMENTATION_CATEGORY_LABELS = {
    "sparse": {
        "edge_drop": "Recommended", "node_drop": "Recommended",
        "edge_add": "Prohibited", "feature_mask": "Neutral",
        "subgraph_extract": "Neutral",
    },
    "dense": {
        "edge_add": "Recommended", "feature_mask": "Recommended",
        "edge_drop": "Prohibited", "node_drop": "Prohibited",
        "subgraph_extract": "Neutral",
    },
    "perceptual": {
        "subgraph_extract": "Recommended", "feature_mask": "Recommended",
        "edge_drop": "Prohibited", "edge_add": "Prohibited",
        "node_drop": "Prohibited",
    },
    "structured": {
        "feature_mask": "Recommended", "feature_noise": "Allowed",
        "edge_drop": "Prohibited", "node_drop": "Prohibited",
        "edge_add": "Prohibited",
    },
}


def apply_single_op_pair(view, x, cfg, num_nodes, op):
    """Builds the two augmented (edge_index, x) branches InfoNCE needs, using
    ONLY the single named operator `op` on this view - mirrors augment_view()
    but with one isolated operator instead of a view-type-specific combo."""
    ei = view["edge_index"]

    if op == "edge_drop":
        lo, hi = cfg["edge_drop_p"]
        ei_a, ei_b = edge_drop(ei, lo), edge_drop(ei, hi)
        x_a, x_b = x, x
    elif op == "node_drop":
        lo, hi = cfg["node_drop_p"]
        ei_a, ei_b = ei, ei
        x_a, x_b = node_drop_features(x, lo), node_drop_features(x, hi)
    elif op == "edge_add":
        ei_a = random_edge_add(ei, num_nodes, cfg["edge_add_ratio"])
        ei_b = random_edge_add(ei, num_nodes, cfg["edge_add_ratio"])
        x_a, x_b = x, x
    elif op == "feature_mask":
        lo, hi = cfg["feature_mask_p"]
        ei_a, ei_b = ei, ei
        x_a, x_b = feature_mask(x, lo), feature_mask(x, hi)
    elif op == "feature_noise":
        ei_a, ei_b = ei, ei
        x_a = add_feature_noise(x, cfg["feature_noise_std"])
        x_b = add_feature_noise(x, cfg["feature_noise_std"])
    elif op == "subgraph_extract":
        # FIX: this value now lives under cfg["reports"]["augmentation_category"],
        # not under a no-longer-existing top-level "augmentation_category_study" key.
        walk_len = cfg.get("reports", {}).get("augmentation_category", {}).get(
            "subgraph_extract_walk_length", 50)
        num_nodes_local = x.size(0)
        ei_a = subgraph_extract(ei, num_nodes_local, walk_len)
        ei_b = subgraph_extract(ei, num_nodes_local, walk_len)
        x_a, x_b = x, x
    else:
        raise ValueError(f"Unknown single-op augmentation: {op!r}")

    ei_a, _ = add_self_loops(ei_a, num_nodes=num_nodes)
    ei_b, _ = add_self_loops(ei_b, num_nodes=num_nodes)
    return (ei_a, x_a), (ei_b, x_b)


# ==============================================================================
# STAGE 4: SHARED GCN ENCODER
# ==============================================================================

class SharedGCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout, num_layers=2):
        """num_layers: number of stacked GCNConv layers (default 2, exactly
        reproducing the original hardcoded conv1/conv2 architecture - nothing
        changes numerically unless the Table 14 'gcn_layers' sweep overrides
        it). num_layers==1 -> a single in_dim->out_dim layer. num_layers>=2 ->
        first layer in_dim->hidden_dim, any middle layers hidden_dim->
        hidden_dim, last layer hidden_dim->out_dim. Same ReLU+dropout-between
        -layers behavior as before (never applied after the final layer)."""
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = nn.ModuleList()
        if num_layers <= 1:
            self.convs.append(GCNConv(in_dim, out_dim))
        else:
            self.convs.append(GCNConv(in_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.convs.append(GCNConv(hidden_dim, out_dim))

        # Backward-compat aliases: some external code (and earlier versions
        # of this script) may still reference encoder.conv1 / encoder.conv2
        # directly for a 2-layer encoder - keep them pointing at the same
        # underlying modules so nothing breaks.
        if num_layers == 2:
            self.conv1 = self.convs[0]
            self.conv2 = self.convs[1]

    def forward(self, x, edge_index, edge_weight=None):
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index, edge_weight)
            if i < len(self.convs) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h


# ==============================================================================
# STAGE 5: PROJECTION HEAD
# ==============================================================================

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, h):
        return self.net(h)


# ==============================================================================
# STAGE 6: MULTI-TERM CONTRASTIVE LEARNING
# ==============================================================================

def info_nce(z1, z2, temperature):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    n = z1.size(0)

    reps = torch.cat([z1, z2], dim=0)
    sim = torch.mm(reps, reps.t()) / temperature
    sim.fill_diagonal_(-1e9)

    pos_idx = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z1.device)
    loss = F.cross_entropy(sim, pos_idx)
    return loss


def multi_term_contrastive_loss(Z1_list, Z2_list, cfg, anchor_idx=None):
    if anchor_idx is not None:
        Z1_list = [z[anchor_idx] for z in Z1_list]
        Z2_list = [z[anchor_idx] for z in Z2_list]

    K = len(Z1_list)
    temp = cfg["temperature"]

    l_intra = sum(info_nce(Z1_list[v], Z2_list[v], temp) for v in range(K)) / K

    cross_terms = []
    for v in range(K):
        for w in range(v + 1, K):
            cross_terms.append(info_nce(Z1_list[v], Z1_list[w], temp))
            cross_terms.append(info_nce(Z2_list[v], Z2_list[w], temp))
    l_cross = sum(cross_terms) / len(cross_terms) if cross_terms else torch.tensor(0.0)

    hybrid_terms = []
    for v in range(K):
        for w in range(K):
            if v != w:
                hybrid_terms.append(info_nce(Z1_list[v], Z2_list[w], temp))
    l_hybrid = sum(hybrid_terms) / len(hybrid_terms) if hybrid_terms else torch.tensor(0.0)

    total = cfg["alpha_intra"] * l_intra + cfg["beta_cross"] * l_cross + cfg["gamma_hybrid"] * l_hybrid

    if cfg.get("normalize_loss_weights", True):
        # Weighted MEAN instead of weighted SUM: divide by the sum of the
        # weights of the STRUCTURALLY-PRESENT terms (a term whose weight is 0,
        # e.g. beta_cross=0 in the NoCross ablation, simply adds 0 here and so
        # is correctly excluded from the mean). l_intra is always present for
        # K >= 1. This keeps every Table-8 variant on the same loss scale - see
        # the "normalize_loss_weights" note in CONFIG for why the raw sum
        # unfairly penalized AGAMC-Full via gradient clipping.
        active_weight = cfg["alpha_intra"]
        if cross_terms:
            active_weight += cfg["beta_cross"]
        if hybrid_terms:
            active_weight += cfg["gamma_hybrid"]
        if active_weight > 0:
            total = total / active_weight

    return total, l_intra.item(), l_cross.item() if cross_terms else 0.0, l_hybrid.item() if hybrid_terms else 0.0


# ==============================================================================
# STAGE 9: EMBEDDING FUSION MODULE
# ==============================================================================

class AttentionFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, h_list):
        H = torch.stack(h_list, dim=1)
        scores = self.score(H).squeeze(-1)
        weights = F.softmax(scores, dim=1)
        fused = (weights.unsqueeze(-1) * H).sum(dim=1)
        return fused, weights


def fuse_embeddings(h_list, cfg, device):
    if cfg["fusion_mode"] == "average":
        fused = torch.stack(h_list, dim=0).mean(dim=0)
        return fused, None
    elif cfg["fusion_mode"] != "attention":
        raise ValueError(f"Unknown fusion_mode: {cfg['fusion_mode']!r}")

    num_nodes = h_list[0].size(0)
    fusion_net = AttentionFusion(h_list[0].size(1)).to(device)
    opt = torch.optim.Adam(fusion_net.parameters(), lr=cfg["fusion_lr"])
    temp = cfg["temperature"]
    entropy_coef = cfg["fusion_entropy_coef"]
    align_loss_type = cfg["fusion_align_loss"]
    batch_size = min(cfg["cl_batch_size"], num_nodes)

    for _ in range(cfg["fusion_train_steps"]):
        opt.zero_grad()
        idx = torch.randperm(num_nodes, device=device)[:batch_size]
        h_batch = [h[idx] for h in h_list]
        fused_batch, weights_batch = fusion_net(h_batch)
        if align_loss_type == "infonce":
            align_loss = sum(info_nce(fused_batch, h, temp) for h in h_batch) / len(h_batch)
        elif align_loss_type == "mse":
            align_loss = sum(F.mse_loss(fused_batch, h.detach()) for h in h_batch) / len(h_batch)
        else:
            raise ValueError(f"Unknown fusion_align_loss: {align_loss_type!r}")
        entropy = -(weights_batch * torch.log(weights_batch.clamp_min(1e-8))).sum(dim=1).mean()
        loss = align_loss + entropy_coef * entropy
        loss.backward()
        opt.step()

    with torch.no_grad():
        fused, weights = fusion_net(h_list)
    return fused, weights


# ==============================================================================
# STAGE 10: CLUSTER NUMBER ESTIMATION (OPTIONAL)
# ==============================================================================

def fast_silhouette(Z_np, labels, cfg):
    """Silhouette coefficient, subsampled via cfg['silhouette_sample_size']
    when the graph is large. Set cfg['silhouette_sample_size'] to None to
    always use the exact full-N computation."""
    sample_size = cfg.get("silhouette_sample_size")
    if sample_size is not None and Z_np.shape[0] > sample_size:
        return silhouette_score(Z_np, labels, sample_size=sample_size, random_state=cfg["seed"])
    return silhouette_score(Z_np, labels)


def estimate_num_clusters_silhouette(Z, cfg, fallback_k):
    if not cfg["auto_estimate_k"]:
        return fallback_k
    Z_np = Z.detach().cpu().numpy()
    lo, hi = cfg["k_search_range"]
    best_k, best_score = fallback_k, -1
    print(f"[Stage 10] Estimating C* via Silhouette score over k in [{lo}, {hi}] ...")
    for k in range(lo, hi + 1):
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=cfg["seed"])
        labels = km.fit_predict(Z_np)
        score = fast_silhouette(Z_np, labels, cfg)
        print(f"           k={k:2d}  silhouette={score:.4f}")
        if score > best_score:
            best_score, best_k = score, k
    print(f"[Stage 10] Estimated C* (silhouette) = {best_k} (silhouette={best_score:.4f})")
    return best_k


def estimate_num_clusters_eigengap(Z, cfg, fallback_k):
    if not cfg["auto_estimate_k"]:
        return fallback_k
    Z_np = Z.detach().cpu().numpy()
    lo, hi = cfg["k_search_range"]
    n = Z_np.shape[0]
    knn_k = min(cfg.get("eigengap_knn_k", 10), n - 1)

    print(f"[Stage 10] Estimating C* via Eigengap heuristic over k in [{lo}, {hi}] ...")
    affinity = kneighbors_graph(Z_np, n_neighbors=knn_k, mode="connectivity", include_self=False)
    W = affinity.maximum(affinity.T).toarray()
    deg = W.sum(axis=1)
    deg_inv_sqrt = 1.0 / np.sqrt(np.clip(deg, 1e-10, None))
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    L_sym = np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt

    eigvals = np.linalg.eigvalsh(L_sym)
    eigvals_sorted = np.sort(eigvals)[: hi + 1]
    gaps = np.diff(eigvals_sorted)

    search_lo, search_hi = max(lo, 1), min(hi, len(gaps))
    best_k, best_gap = fallback_k, -1.0
    for k in range(search_lo, search_hi + 1):
        gap = gaps[k - 1]
        print(f"           k={k:2d}  eigengap={gap:.4f}")
        if gap > best_gap:
            best_gap, best_k = gap, k
    print(f"[Stage 10] Estimated C* (eigengap) = {best_k} (gap={best_gap:.4f})")
    return best_k


# ==============================================================================
# STAGE 11: K-MEANS++ CLUSTERING + EVALUATION
# ==============================================================================

def hungarian_match_clusters(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    D = max(y_pred.max(), y_true.max()) + 1
    cost = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        cost[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(-cost)
    mapping = {r: c for r, c in zip(row_ind, col_ind)}
    y_mapped = np.array([mapping[p] for p in y_pred])
    return y_mapped


def clustering_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_mapped = hungarian_match_clusters(y_true, y_pred)
    return (y_mapped == y_true).mean()


def clustering_f1(y_true, y_pred, average="macro"):
    y_true = np.asarray(y_true)
    y_mapped = hungarian_match_clusters(y_true, y_pred)
    return f1_score(y_true, y_mapped, average=average)


def unsupervised_quality_score(Z, n_clusters, cfg):
    Z_np = Z.detach().cpu().numpy()
    if not np.all(np.isfinite(Z_np)):
        # Defense-in-depth: the training-loop NaN-guard (see train()) should
        # already prevent non-finite embeddings from reaching here, but if
        # one still slips through (e.g. a NaN produced during eval-mode
        # forward pass itself) we report a NaN score instead of crashing
        # KMeans - the caller (early stopping / model selection) already
        # treats a NaN/low score as "not an improvement" and moves on.
        print("           [Warning] non-finite values in probe embedding - "
              "skipping this Silhouette probe (returning NaN).")
        return float("nan"), np.zeros(Z_np.shape[0], dtype=int)
    km = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, random_state=cfg["seed"])
    pred = km.fit_predict(Z_np)
    try:
        sil = fast_silhouette(Z_np, pred, cfg)
    except ValueError:
        sil = float("nan")
    return sil, pred


def evaluate_clustering(Z, gt_labels_readonly, n_clusters, cfg, tag=""):
    Z_np = Z.detach().cpu().numpy()
    if not np.all(np.isfinite(Z_np)):
        print(f"{tag}[Warning] non-finite values in embedding - clustering metrics are "
              f"undefined (NaN) for this evaluation.")
        n = Z_np.shape[0]
        pred = np.zeros(n, dtype=int)
        return {"acc": float("nan"), "nmi": float("nan"), "ari": float("nan"),
                "f1": float("nan"), "silhouette": float("nan"), "pred": pred}
    km = KMeans(n_clusters=n_clusters, init="k-means++", n_init=20, random_state=cfg["seed"])
    pred = km.fit_predict(Z_np)

    acc = clustering_accuracy(gt_labels_readonly, pred)
    nmi = normalized_mutual_info_score(gt_labels_readonly, pred)
    ari = adjusted_rand_score(gt_labels_readonly, pred)
    f1 = clustering_f1(gt_labels_readonly, pred)
    try:
        sil = fast_silhouette(Z_np, pred, cfg)
    except ValueError:
        sil = float("nan")

    print(f"{tag}ACC={acc*100:5.2f}%  NMI={nmi*100:5.2f}%  ARI={ari*100:5.2f}%  "
          f"F1={f1*100:5.2f}%  Silhouette={sil:.4f}")
    return {"acc": acc, "nmi": nmi, "ari": ari, "f1": f1, "silhouette": sil, "pred": pred}


# ==============================================================================
# TRAINING LOOP
# ==============================================================================

def progress_bar(current, total, width=30):
    frac = current / total
    filled = int(width * frac)
    return "[" + "#" * filled + "-" * (width - filled) + f"] {frac*100:5.1f}%"


def train(views, x, gt_labels_readonly, cfg, single_op_override=None):
    """single_op_override: optional {"view_type": ..., "op": ...} dict used
    ONLY by the Table 9-12 category-ablation study (run_augmentation_category_study).
    When set, every view whose type matches view_type is augmented with ONLY
    that single operator via apply_single_op_pair(); every other view keeps
    its normal augment_view_dispatch() behavior. None (the default) reproduces
    the original, unmodified training behavior exactly.

    ADDITIVE Table-15 (scalability) instrumentation: per-epoch wall-clock time
    is always recorded, and (on CUDA) peak memory is tracked via
    torch.cuda.reset_peak_memory_stats/max_memory_allocated. These are stored
    as a side-channel in cfg["_measured"] = {"epoch_times": [...],
    "peak_memory_gb": ... or "N/A (CPU run)"} rather than as new return
    values, so every existing caller that unpacks
    `encoder, projector, epoch_log, probe_log, best_epoch = train(...)`
    keeps working completely unmodified."""
    device = torch.device(cfg["device"])
    x = x.to(device)
    gt_labels_readonly_np = gt_labels_readonly.cpu().numpy()
    num_nodes = x.size(0)
    views = [{"type": v["type"], "edge_index": v["edge_index"].to(device)} for v in views]

    num_gcn_layers = cfg.get("num_gcn_layers", 2)
    encoder = SharedGCNEncoder(x.size(1), cfg["hidden_dim"], cfg["embed_dim"],
                                cfg["encoder_dropout"], num_layers=num_gcn_layers).to(device)
    projector = ProjectionHead(cfg["embed_dim"], cfg["proj_hidden_dim"],
                                cfg["proj_out_dim"]).to(device)

    controller = None
    trainable_params = list(encoder.parameters()) + list(projector.parameters())
    if cfg["augmentation_mode"] == "learnable":
        controller = AugmentationController(
            stat_dim=cfg["controller_stat_dim"],
            hidden_dim=cfg["controller_hidden_dim"],
            num_params=cfg["controller_num_params"],
        ).to(device)
        trainable_params += list(controller.parameters())
        print("[Stage 2] augmentation_mode='learnable' -> AugmentationController "
              "will be trained jointly with the encoder/projector (label-free).")
    else:
        print(f"[Stage 2] augmentation_mode='{cfg['augmentation_mode']}' -> using a "
              f"fixed (non-learnable) augmentation strategy for every view "
              f"(no AugmentationController involved).")

    optimizer = torch.optim.Adam(trainable_params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    print("=" * 78)
    print(" STAGE 4-6: TRAINING SHARED ENCODER WITH MULTI-TERM CONTRASTIVE LOSS")
    if cfg["lr_schedule_enabled"]:
        print(f" LR schedule: linear warmup for {cfg['warmup_epochs']} epochs up to {cfg['lr']}, "
              f"then cosine annealing down to {cfg['min_lr']}")
    else:
        print(f" LR schedule: DISABLED -> constant lr = {cfg['lr']}")
    if cfg["early_stopping_enabled"]:
        print(f" Early stopping: ENABLED, patience={cfg['early_stop_patience']} probes "
              f"(every {cfg['eval_every']} epochs), monitored on label-free Silhouette")
    else:
        print(f" Early stopping: DISABLED -> will always run the full {cfg['epochs']} epochs "
              f"(the best label-free-Silhouette checkpoint is still tracked and restored at the end)")
    print("=" * 78)

    best_score = -1e9
    best_state = None
    best_epoch = -1
    epochs_without_improvement = 0
    epoch_log = []
    probe_log = []
    consecutive_nan_steps = 0
    max_consecutive_nan_steps = 20  # safety valve - see NaN-guard note below

    nan_guard_enabled = cfg.get("nan_guard_enabled", True)
    grad_clip_norm = cfg.get("grad_clip_norm", None)
    if cfg.get("debug_detect_anomaly", False):
        torch.autograd.set_detect_anomaly(True)
        print("[Debug] torch.autograd.set_detect_anomaly(True) is ON - training will be "
              "much slower, but PyTorch will print the exact op that first produced a "
              "NaN/Inf gradient. Turn 'debug_detect_anomaly' back off for normal runs.")

    # ---- Table-15 scalability instrumentation (ADDITIVE, side-channel only) ----
    is_cuda = (device.type == "cuda")
    epoch_times = []
    peak_memory_gb = None if is_cuda else "N/A (CPU run)"

    start_time = time.time()
    stopped_early = False
    for epoch in range(1, cfg["epochs"] + 1):
        epoch_t0 = time.time()
        if is_cuda:
            torch.cuda.reset_peak_memory_stats(device)

        encoder.train()
        projector.train()
        if controller is not None:
            controller.train()

        current_lr = get_lr_for_epoch(epoch, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        optimizer.zero_grad()

        Z1_list, Z2_list = [], []
        for view in views:
            if single_op_override is not None and view["type"] == single_op_override["view_type"]:
                (ei_a, x_a), (ei_b, x_b) = apply_single_op_pair(
                    view, x, cfg, num_nodes, single_op_override["op"])
                ew_a = ew_b = None
            else:
                (ei_a, x_a, ew_a), (ei_b, x_b, ew_b) = augment_view_dispatch(
                    view, x, cfg, num_nodes, controller=controller, device=device)
            h_a = encoder(x_a, ei_a, ew_a)
            h_b = encoder(x_b, ei_b, ew_b)
            Z1_list.append(projector(h_a))
            Z2_list.append(projector(h_b))

        batch_size = min(cfg["cl_batch_size"], num_nodes)
        anchor_idx = torch.randperm(num_nodes, device=device)[:batch_size]
        loss, l_intra, l_cross, l_hybrid = multi_term_contrastive_loss(
            Z1_list, Z2_list, cfg, anchor_idx=anchor_idx)

        # ---- NaN-guard: this is what actually prevents the crash reported
        # with augmentation_mode="learnable" (loss/encoder weights going NaN
        # a couple of epochs in, then propagating into the Stage-10 probe's
        # KMeans call and crashing the whole run). If this step's loss is
        # non-finite, skip backward()/optimizer.step() entirely for this
        # step (the model keeps its last good weights) instead of letting a
        # NaN gradient corrupt every parameter. A small number of skipped
        # steps is harmless; if it happens on many CONSECUTIVE steps, that
        # is a real divergence and we stop the run with a clear error
        # instead of silently limping along forever.
        if nan_guard_enabled and not torch.isfinite(loss):
            consecutive_nan_steps += 1
            print(f"[NaN-Guard] Epoch {epoch:4d}: non-finite loss detected - skipping this "
                  f"optimizer step ({consecutive_nan_steps}/{max_consecutive_nan_steps} "
                  f"consecutive non-finite step(s)).")
            if consecutive_nan_steps >= max_consecutive_nan_steps:
                raise RuntimeError(
                    f"[NaN-Guard] Training diverged: {max_consecutive_nan_steps} consecutive "
                    f"non-finite loss steps (augmentation_mode='{cfg['augmentation_mode']}'). "
                    f"Try lowering 'lr', enabling/lowering 'grad_clip_norm', or switching "
                    f"augmentation_mode away from 'learnable' for this dataset."
                )
            epoch_log.append({"epoch": epoch, "loss": float("nan"), "intra": float("nan"),
                               "cross": float("nan"), "hybrid": float("nan"), "lr": current_lr})
            epoch_times.append(time.time() - epoch_t0)
            continue
        consecutive_nan_steps = 0

        loss.backward()

        # ---- Gradient sanitization: THIS is what actually prevents the
        # single-step full-model corruption seen with augmentation_mode=
        # "learnable" (loss finite at epoch 1, then NaN forever after just
        # one optimizer step). torch.nn.utils.clip_grad_norm_ computes ONE
        # scaling factor from the TOTAL gradient norm across every
        # parameter; if even one parameter's gradient contains a NaN/Inf
        # (which can happen through GCNConv's degree normalization when
        # edge_weight itself is differentiable, as in the learnable path),
        # that total norm becomes NaN too, and the clip step then
        # multiplies EVERY parameter's gradient by that NaN scaling factor
        # - silently destroying the whole model in a single step, even
        # though only one small piece of it was actually bad. Replacing any
        # non-finite entries with 0 BEFORE clipping keeps the corruption
        # local (that one gradient entry simply contributes no update this
        # step) instead of letting it propagate to every other parameter.
        num_sanitized = 0
        for p in trainable_params:
            if p.grad is not None and not torch.isfinite(p.grad).all():
                num_sanitized += 1
                torch.nan_to_num_(p.grad, nan=0.0, posinf=1e4, neginf=-1e4)
        if num_sanitized > 0:
            print(f"[NaN-Guard] Epoch {epoch:4d}: sanitized non-finite gradients in "
                  f"{num_sanitized} parameter tensor(s) (replaced with 0) before the "
                  f"optimizer step, instead of letting clip_grad_norm_ spread the "
                  f"corruption to every parameter.")

        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip_norm)
        optimizer.step()

        # epoch_log.append({"epoch": epoch, "loss": loss.item(), "intra": l_intra,
                        #    "cross": l_cross, "hybrid": l_hybrid, "lr": current_lr})
        should_log_this_epoch = (epoch % cfg["log_every"] == 0 or epoch == cfg["epochs"])
        loss_val = loss.item() if should_log_this_epoch else float("nan")
        epoch_log.append({"epoch": epoch, "loss": loss_val, "intra": l_intra,
                                "cross": l_cross, "hybrid": l_hybrid, "lr": current_lr})

        if epoch % cfg["log_every"] == 0 or epoch == cfg["epochs"]:
            elapsed = time.time() - start_time
            eta = elapsed / epoch * (cfg["epochs"] - epoch)
            bar = progress_bar(epoch, cfg["epochs"])
            print(f"Epoch {epoch:4d}/{cfg['epochs']}  {bar}  "
                  f"loss={loss.item():.4f} (intra={l_intra:.4f} cross={l_cross:.4f} hybrid={l_hybrid:.4f})  "
                  f"lr={current_lr:.2e}  "
                  f"elapsed={timedelta(seconds=int(elapsed))}  eta={timedelta(seconds=int(eta))}")

        if epoch % cfg["eval_every"] == 0 or epoch == cfg["epochs"]:
            encoder.eval()
            with torch.no_grad():
                h_list_probe = [encoder(x, v["edge_index"]) for v in views]
                if cfg["fusion_mode"] == "attention":
                    fused_probe, _ = fuse_embeddings(h_list_probe, cfg, device)
                else:
                    fused_probe = torch.stack(h_list_probe, dim=0).mean(dim=0)

            sil, _ = unsupervised_quality_score(fused_probe, NUM_CLUSTERS_KNOWN, cfg)
            print(f"           [probe @ epoch {epoch:4d}] (unsupervised) Silhouette={sil:.4f}")

            probe_entry = {"epoch": epoch, "silhouette": sil,
                           "acc": float("nan"), "nmi": float("nan"), "ari": float("nan")}

            if cfg["enable_readonly_diagnostics"]:
                mon = evaluate_clustering(fused_probe, gt_labels_readonly_np, NUM_CLUSTERS_KNOWN, cfg,
                                           tag=f"           [read-only diagnostic] ")
                probe_entry["acc"], probe_entry["nmi"], probe_entry["ari"] = mon["acc"], mon["nmi"], mon["ari"]
            probe_log.append(probe_entry)

            if sil > best_score:
                best_score = sil
                best_epoch = epoch
                epochs_without_improvement = 0
                best_state = {
                    "encoder": {k: v.clone() for k, v in encoder.state_dict().items()},
                    "projector": {k: v.clone() for k, v in projector.state_dict().items()},
                }
                if controller is not None:
                    best_state["controller"] = {k: v.clone() for k, v in controller.state_dict().items()}
                print(f"           -> new best checkpoint (epoch {epoch}, "
                      f"unsupervised Silhouette={sil:.4f}) saved")
            else:
                epochs_without_improvement += 1
                if cfg["early_stopping_enabled"]:
                    print(f"           -> no improvement in unsupervised Silhouette for "
                          f"{epochs_without_improvement}/{cfg['early_stop_patience']} probe(s)")

            if cfg["early_stopping_enabled"] and epochs_without_improvement >= cfg["early_stop_patience"]:
                print(f"[EarlyStopping] No improvement in label-free Silhouette for "
                      f"{cfg['early_stop_patience']} consecutive probes -> stopping at epoch {epoch}")
                stopped_early = True
                epoch_times.append(time.time() - epoch_t0)
                if is_cuda:
                    peak_this_epoch = torch.cuda.max_memory_allocated(device) / 1e9
                    peak_memory_gb = peak_this_epoch if peak_memory_gb is None else max(peak_memory_gb, peak_this_epoch)
                break

        epoch_times.append(time.time() - epoch_t0)
        if is_cuda:
            peak_this_epoch = torch.cuda.max_memory_allocated(device) / 1e9
            peak_memory_gb = peak_this_epoch if peak_memory_gb is None else max(peak_memory_gb, peak_this_epoch)

    print("=" * 78)
    status = "EARLY-STOPPED" if stopped_early else "completed all scheduled epochs"
    print(f" Training {status} in {timedelta(seconds=int(time.time() - start_time))}")
    if best_state is not None:
        encoder.load_state_dict(best_state["encoder"])
        projector.load_state_dict(best_state["projector"])
        if controller is not None and "controller" in best_state:
            controller.load_state_dict(best_state["controller"])
        print(f" Restored BEST checkpoint from epoch {best_epoch} "
              f"(label-free Silhouette={best_score:.4f}) for downstream stages 8-11")
    print("=" * 78 + "\n")

    # ---- Publish the additive Table-15 side-channel measurements on cfg ----
    cfg["_measured"] = {"epoch_times": epoch_times, "peak_memory_gb": peak_memory_gb}

    return encoder, projector, epoch_log, probe_log, best_epoch


# ==============================================================================
# SCALABILITY UTILITY: subsample a multi-view graph for the vs_num_nodes sweep
# ==============================================================================

def subsample_multiview_graph(x, views, gt_labels, target_n, seed):
    """Randomly selects `target_n` node indices (seeded), builds an
    old-index -> new-index remap, and for every view's edge_index keeps only
    edges where BOTH endpoints survive, remapped into 0..target_n-1.

    NOTE: clustering-quality metrics computed on this reduced graph are NOT
    meaningful (uniform random node subsampling shifts class balance
    arbitrarily) - this utility exists purely to feed the vs_num_nodes
    timing/memory sweep in run_scalability_variant_num_nodes(). Callers
    should only report time/memory for runs that use this subsampled graph.
    """
    n_total = x.size(0)
    target_n = min(target_n, n_total)
    rng = np.random.RandomState(seed)
    keep_idx = np.sort(rng.choice(n_total, size=target_n, replace=False))
    remap = -np.ones(n_total, dtype=np.int64)
    remap[keep_idx] = np.arange(target_n)

    x_sub = x[torch.from_numpy(keep_idx)].clone()
    gt_sub = gt_labels[torch.from_numpy(keep_idx)].clone()

    views_sub = []
    for v in views:
        ei = v["edge_index"].cpu().numpy()
        src, dst = ei[0], ei[1]
        keep_mask = (remap[src] >= 0) & (remap[dst] >= 0)
        new_src = remap[src[keep_mask]]
        new_dst = remap[dst[keep_mask]]
        new_ei = torch.tensor(np.vstack([new_src, new_dst]), dtype=torch.long)
        views_sub.append({"type": v["type"], "edge_index": new_ei})

    return x_sub, views_sub, gt_sub


# ==============================================================================
# PUBLICATION-READY OUTPUTS
# ==============================================================================

def plot_training_curves(epoch_log, probe_log, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    epochs = [r["epoch"] for r in epoch_log]
    total = [r["loss"] for r in epoch_log]
    intra = [r["intra"] for r in epoch_log]
    cross = [r["cross"] for r in epoch_log]
    hybrid = [r["hybrid"] for r in epoch_log]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(epochs, total, color="black")
    axes[0, 0].set_title("Total Multi-Term Contrastive Loss")
    axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(epochs, intra, label="Intra-View")
    axes[0, 1].plot(epochs, cross, label="Cross-View")
    axes[0, 1].plot(epochs, hybrid, label="Hybrid-View")
    axes[0, 1].set_title("Loss Components")
    axes[0, 1].set_xlabel("Epoch"); axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    if probe_log:
        p_epochs = [r["epoch"] for r in probe_log]
        sil = [r["silhouette"] for r in probe_log]
        best_idx = int(np.argmax(sil))
        axes[1, 0].plot(p_epochs, sil, marker="o", color="green")
        axes[1, 0].axvline(p_epochs[best_idx], color="red", linestyle="--", alpha=0.6,
                            label=f"selected checkpoint (epoch {p_epochs[best_idx]})")
        axes[1, 0].set_title("Model-Selection Signal (label-free)\nSilhouette Coefficient")
        axes[1, 0].set_xlabel("Epoch"); axes[1, 0].set_ylabel("Silhouette")
        axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

        if not all(np.isnan(r["acc"]) for r in probe_log):
            acc = [r["acc"] for r in probe_log]
            nmi = [r["nmi"] for r in probe_log]
            ari = [r["ari"] for r in probe_log]
            axes[1, 1].plot(p_epochs, acc, marker="o", label="ACC")
            axes[1, 1].plot(p_epochs, nmi, marker="s", label="NMI")
            axes[1, 1].plot(p_epochs, ari, marker="^", label="ARI")
            axes[1, 1].set_title("Post-hoc Evaluation vs Ground Truth\n"
                                  "(read-only diagnostic - NEVER used for training/selection)")
            axes[1, 1].set_xlabel("Epoch"); axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)
        else:
            axes[1, 1].axis("off")

    plt.tight_layout()
    path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Output] Saved training curves -> {path}")


def plot_tsne(Z, pred, gt_labels_readonly, output_dir, seed, cfg=None):
    """2D t-SNE of the fused embedding, colored by predicted cluster AND by
    ground-truth class. If umap-learn is installed (and cfg allows it via
    output_artifacts.include.tsne_umap), an extra UMAP panel pair is added
    next to the t-SNE panels. Degrades gracefully (t-SNE only) if umap-learn
    is missing."""
    os.makedirs(output_dir, exist_ok=True)
    Z_np = Z.detach().cpu().numpy()
    tsne = TSNE(n_components=2, random_state=seed, init="pca", perplexity=30)
    emb2d_tsne = tsne.fit_transform(Z_np)

    umap_emb = None
    try:
        import umap  # optional dependency: pip install umap-learn
        reducer = umap.UMAP(n_components=2, random_state=seed)
        umap_emb = reducer.fit_transform(Z_np)
    except ImportError:
        print("[Output] umap-learn not installed - skipping UMAP panel (t-SNE only). "
              "Install with: pip install umap-learn --break-system-packages")
    except Exception as e:
        print(f"[Output] UMAP computation failed ({e}) - skipping UMAP panel.")

    ncols = 4 if umap_emb is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6.5 * ncols, 6))

    axes[0].scatter(emb2d_tsne[:, 0], emb2d_tsne[:, 1], c=pred, cmap="tab10", s=10)
    axes[0].set_title("t-SNE colored by PREDICTED cluster\n(fully unsupervised pipeline output)")
    axes[0].set_xticks([]); axes[0].set_yticks([])

    axes[1].scatter(emb2d_tsne[:, 0], emb2d_tsne[:, 1], c=gt_labels_readonly, cmap="tab10", s=10)
    axes[1].set_title("t-SNE colored by GROUND-TRUTH class (read-only)\n(shown for illustration/paper only)")
    axes[1].set_xticks([]); axes[1].set_yticks([])

    if umap_emb is not None:
        axes[2].scatter(umap_emb[:, 0], umap_emb[:, 1], c=pred, cmap="tab10", s=10)
        axes[2].set_title("UMAP colored by PREDICTED cluster")
        axes[2].set_xticks([]); axes[2].set_yticks([])

        axes[3].scatter(umap_emb[:, 0], umap_emb[:, 1], c=gt_labels_readonly, cmap="tab10", s=10)
        axes[3].set_title("UMAP colored by GROUND-TRUTH class (read-only)")
        axes[3].set_xticks([]); axes[3].set_yticks([])

    plt.tight_layout()
    path = os.path.join(output_dir, "tsne_umap_clusters.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Output] Saved t-SNE/UMAP visualization -> {path}")


def plot_per_view_vs_fused(h_list, fused, pred, view_types, output_dir, seed):
    """One t-SNE panel per view (pre-fusion embedding) next to a panel for the
    final fused embedding (post Stage-9 fusion), all colored by predicted
    cluster - shows Stage 9's added value visually."""
    os.makedirs(output_dir, exist_ok=True)
    n_panels = len(h_list) + 1
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 5.5))
    if n_panels == 1:
        axes = [axes]

    for i, (h, vtype) in enumerate(zip(h_list, view_types)):
        h_np = h.detach().cpu().numpy()
        emb = TSNE(n_components=2, random_state=seed, init="pca", perplexity=30).fit_transform(h_np)
        axes[i].scatter(emb[:, 0], emb[:, 1], c=pred, cmap="tab10", s=8)
        axes[i].set_title(f"View {i+1} ({vtype})\npre-fusion embedding")
        axes[i].set_xticks([]); axes[i].set_yticks([])

    fused_np = fused.detach().cpu().numpy()
    emb_fused = TSNE(n_components=2, random_state=seed, init="pca", perplexity=30).fit_transform(fused_np)
    axes[-1].scatter(emb_fused[:, 0], emb_fused[:, 1], c=pred, cmap="tab10", s=8)
    axes[-1].set_title("Fused embedding\n(post Stage-9 fusion)")
    axes[-1].set_xticks([]); axes[-1].set_yticks([])

    plt.tight_layout()
    path = os.path.join(output_dir, "per_view_vs_fused.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Output] Saved per-view-vs-fused comparison -> {path}")


def plot_confusion_matrix(gt_labels_readonly, pred, output_dir):
    """Hungarian-matched predicted-vs-true confusion matrix heatmap."""
    os.makedirs(output_dir, exist_ok=True)
    gt_np = np.asarray(gt_labels_readonly)
    y_mapped = hungarian_match_clusters(gt_np, pred)
    n_classes = int(max(gt_np.max(), y_mapped.max())) + 1
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(gt_np, y_mapped):
        cm[t, p] += 1

    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted cluster (Hungarian-matched)")
    ax.set_ylabel("Ground-truth class (read-only)")
    ax.set_title("Confusion Matrix (Hungarian-matched)")
    plt.colorbar(im, ax=ax)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(n_classes):
        for j in range(n_classes):
            if cm[i, j] > 0:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontsize=7)
    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Output] Saved confusion matrix -> {path}")


def plot_similarity_heatmap(fused, pred, output_dir, cfg):
    """Pairwise cosine-similarity heatmap of the fused embedding, with nodes
    sorted by predicted cluster (block-diagonal = better separation).
    Subsampled for large N, reusing the silhouette_sample_size setting so
    this never blows up to O(N^2) memory on amazon/youtube."""
    os.makedirs(output_dir, exist_ok=True)
    Z_np = fused.detach().cpu().numpy()
    n = Z_np.shape[0]
    sample_size = cfg.get("silhouette_sample_size")
    if sample_size is not None and n > sample_size:
        rng = np.random.RandomState(cfg.get("seed", 0))
        idx = rng.choice(n, size=sample_size, replace=False)
        Z_sample = Z_np[idx]
        pred_sample = np.asarray(pred)[idx]
    else:
        Z_sample = Z_np
        pred_sample = np.asarray(pred)

    order = np.argsort(pred_sample)
    Z_sorted = Z_sample[order]
    norms = np.linalg.norm(Z_sorted, axis=1, keepdims=True)
    Z_norm = Z_sorted / np.clip(norms, 1e-8, None)
    sim = Z_norm @ Z_norm.T

    fig, ax = plt.subplots(figsize=(6.5, 6))
    im = ax.imshow(sim, cmap="viridis", vmin=-1, vmax=1)
    ax.set_title("Fused-Embedding Similarity Heatmap\n(nodes sorted by predicted cluster)")
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    path = os.path.join(output_dir, "similarity_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Output] Saved similarity heatmap -> {path}")


def plot_attention_weights(view_types, weights, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(view_types, weights, color="steelblue")
    ax.set_title("Mean Attention Weight per View (Stage 9 Fusion)")
    ax.set_ylabel("Weight")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(output_dir, "fusion_attention_weights.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Output] Saved attention-weight chart -> {path}")


def plot_graph_structure(views, pred, output_dir, cfg, max_nodes=500):
    """networkx drawing of each view's graph (subsampled to at most
    `max_nodes` nodes on large datasets), colored by predicted cluster.
    Optional dependency (networkx) - degrades gracefully with a warning if
    not installed. Off by default (output_artifacts.include.graph_structure)
    since it is slow/cluttered on large graphs."""
    os.makedirs(output_dir, exist_ok=True)
    try:
        import networkx as nx  # optional dependency: pip install networkx
    except ImportError:
        print("[Output] networkx not installed - skipping graph-structure visualization. "
              "Install with: pip install networkx --break-system-packages")
        return

    pred_np = np.asarray(pred)
    n_total = pred_np.shape[0]
    if n_total > max_nodes:
        rng = np.random.RandomState(cfg.get("seed", 0))
        keep = set(rng.choice(n_total, size=max_nodes, replace=False).tolist())
    else:
        keep = set(range(n_total))

    n_views = len(views)
    fig, axes = plt.subplots(1, n_views, figsize=(6 * n_views, 6))
    if n_views == 1:
        axes = [axes]

    for i, v in enumerate(views):
        ei = v["edge_index"].detach().cpu().numpy()
        G = nx.Graph()
        G.add_nodes_from(keep)
        for s, d in zip(ei[0], ei[1]):
            if s in keep and d in keep:
                G.add_edge(int(s), int(d))
        pos = nx.spring_layout(G, seed=cfg.get("seed", 0))
        node_colors = [pred_np[node] for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_size=15, node_color=node_colors, cmap="tab10", ax=axes[i])
        nx.draw_networkx_edges(G, pos, alpha=0.15, ax=axes[i])
        axes[i].set_title(f"View {i+1} ({v['type']}) - subsampled to {len(keep)} nodes")
        axes[i].axis("off")

    plt.tight_layout()
    path = os.path.join(output_dir, "graph_structure.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Output] Saved multi-view graph-structure visualization -> {path}")


def plot_interactive_3d(Z, pred, gt_labels_readonly, output_dir, seed):
    """Interactive 3D t-SNE/UMAP scatter (Plotly HTML) with rotate/zoom/hover.
    Prefers UMAP for the 3D projection (faster, better global structure);
    falls back to t-SNE if umap-learn is unavailable. Optional dependency
    (plotly) - degrades gracefully with a warning if not installed."""
    os.makedirs(output_dir, exist_ok=True)
    try:
        import plotly.graph_objects as go  # optional dependency: pip install plotly
    except ImportError:
        print("[Output] plotly not installed - skipping interactive 3D visualization. "
              "Install with: pip install plotly --break-system-packages")
        return

    Z_np = Z.detach().cpu().numpy()
    method_name = "t-SNE"
    try:
        import umap
        reducer = umap.UMAP(n_components=3, random_state=seed)
        emb3d = reducer.fit_transform(Z_np)
        method_name = "UMAP"
    except ImportError:
        emb3d = TSNE(n_components=3, random_state=seed, init="pca", perplexity=30).fit_transform(Z_np)
    except Exception as e:
        print(f"[Output] UMAP 3D projection failed ({e}) - falling back to t-SNE.")
        emb3d = TSNE(n_components=3, random_state=seed, init="pca", perplexity=30).fit_transform(Z_np)

    fig = go.Figure(data=[go.Scatter3d(
        x=emb3d[:, 0], y=emb3d[:, 1], z=emb3d[:, 2],
        mode="markers",
        marker=dict(size=3, color=np.asarray(pred), colorscale="Turbo", opacity=0.85),
        text=[f"gt={g}, pred={p}" for g, p in zip(np.asarray(gt_labels_readonly), np.asarray(pred))],
        hoverinfo="text",
    )])
    fig.update_layout(title=f"Interactive 3D {method_name} embedding (colored by predicted cluster)",
                       margin=dict(l=0, r=0, b=0, t=40))
    path = os.path.join(output_dir, "interactive_3d.html")
    fig.write_html(path)
    print(f"[Output] Saved interactive 3D visualization -> {path}")


def plot_study_bar_chart(title, group_labels, results_by_group, output_dir, filename,
                          metrics=("acc", "nmi", "ari", "f1")):
    """Generic grouped bar chart with error bars over seeds, used for Table 8
    (ablation variants), Table 9-12 (augmentation ops per view-type), and
    Table 13 (rule_based vs learnable). `results_by_group` follows the
    aggregate_variant_results() convention: {group_label: {"acc_mean":...,
    "acc_std":..., "nmi_mean":..., ...}}."""
    os.makedirs(output_dir, exist_ok=True)
    x = np.arange(len(group_labels))
    width = 0.8 / max(len(metrics), 1)
    fig, ax = plt.subplots(figsize=(max(8, len(group_labels) * 1.6), 6))
    for mi, m in enumerate(metrics):
        means = [results_by_group[g][f"{m}_mean"] * 100 for g in group_labels]
        stds = [results_by_group[g][f"{m}_std"] * 100 for g in group_labels]
        ax.bar(x + mi * width, means, width, yerr=stds, capsize=3, label=m.upper())
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(group_labels, rotation=30, ha="right")
    ax.set_ylabel("Score (%)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Output] Saved study bar chart -> {path}")


def plot_study_radar_chart(title, group_labels, results_by_group, output_dir, filename,
                            metrics=("acc", "nmi", "ari", "f1", "silhouette")):
    """Multi-metric radar/spider chart - a companion view to plot_study_bar_chart
    for the same Table 8 / 9-12 / 13 studies. Puts every metric on its own
    spoke (each metric independently min-max normalized across the groups
    being compared, purely for a readable shared 0-1 radial scale - the
    normalization is only a plotting device, the underlying CSV/JSON still
    holds the real values) so shape differences between variants/ops/modes
    are easy to read at a glance."""
    os.makedirs(output_dir, exist_ok=True)

    # silhouette can be negative; every other metric is already a 0-1 score.
    # Min-max normalize each metric across the compared groups so all spokes
    # share a comparable 0-1 radial axis.
    raw = {m: np.array([results_by_group[g].get(f"{m}_mean", results_by_group[g].get(m, np.nan))
                         for g in group_labels], dtype=float) for m in metrics}
    norm = {}
    for m, vals in raw.items():
        lo, hi = np.nanmin(vals), np.nanmax(vals)
        norm[m] = (vals - lo) / (hi - lo) if hi > lo else np.zeros_like(vals)

    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    cmap = plt.get_cmap("tab10")
    for gi, g in enumerate(group_labels):
        values = [norm[m][gi] for m in metrics]
        values += values[:1]
        ax.plot(angles, values, linewidth=1.8, label=g, color=cmap(gi % 10))
        ax.fill(angles, values, alpha=0.08, color=cmap(gi % 10))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_yticks([])
    ax.set_title(f"{title}\n(each spoke min-max normalized across groups shown)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Output] Saved study radar chart -> {path}")


def plot_silhouette_diagram(fused, pred, output_dir, cfg):
    """Classic per-sample Silhouette diagram (as in the original Rousseeuw
    1987 style plots widely used in clustering papers): one horizontal
    "blade" per cluster, samples sorted by silhouette value within the
    cluster, with a vertical line at the overall mean Silhouette score.
    Subsampled for large N via cfg['silhouette_sample_size'] to keep the
    underlying pairwise-distance computation tractable."""
    from sklearn.metrics import silhouette_samples
    os.makedirs(output_dir, exist_ok=True)

    Z_np = fused.detach().cpu().numpy()
    pred_np = np.asarray(pred)
    n = Z_np.shape[0]
    sample_size = cfg.get("silhouette_sample_size")
    if sample_size is not None and n > sample_size:
        rng = np.random.RandomState(cfg.get("seed", 0))
        idx = rng.choice(n, size=sample_size, replace=False)
        Z_sample, pred_sample = Z_np[idx], pred_np[idx]
    else:
        Z_sample, pred_sample = Z_np, pred_np

    try:
        sample_values = silhouette_samples(Z_sample, pred_sample)
    except ValueError as e:
        print(f"[Output] Skipping silhouette diagram (silhouette_samples failed: {e}).")
        return
    overall_mean = sample_values.mean()

    clusters = np.unique(pred_sample)
    fig, ax = plt.subplots(figsize=(7, max(4, 0.4 * len(clusters) + 2)))
    cmap = plt.get_cmap("tab10")
    y_lower = 10
    for ci in clusters:
        cluster_values = np.sort(sample_values[pred_sample == ci])
        size = cluster_values.shape[0]
        y_upper = y_lower + size
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_values,
                          facecolor=cmap(int(ci) % 10), edgecolor=cmap(int(ci) % 10), alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size, str(int(ci)), fontsize=9)
        y_lower = y_upper + 10

    ax.axvline(x=overall_mean, color="red", linestyle="--",
               label=f"mean silhouette = {overall_mean:.3f}")
    ax.set_xlabel("Silhouette coefficient")
    ax.set_ylabel("Samples per predicted cluster")
    ax.set_title("Per-Sample Silhouette Diagram (Stage 11)")
    ax.set_yticks([])
    ax.legend(loc="best")
    plt.tight_layout()
    path = os.path.join(output_dir, "silhouette_diagram.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Output] Saved silhouette diagram -> {path}")


def plot_cluster_size_distribution(gt_labels_readonly, pred, output_dir):
    """Side-by-side bar chart comparing predicted-cluster sizes against the
    true class-size distribution (matched via Hungarian assignment so bars
    line up under the same x-axis label) - a quick visual check for whether
    the model is collapsing multiple classes into one cluster or splitting
    a class across several clusters."""
    os.makedirs(output_dir, exist_ok=True)
    gt_np = np.asarray(gt_labels_readonly)
    y_mapped = hungarian_match_clusters(gt_np, pred)
    n_classes = int(max(gt_np.max(), y_mapped.max())) + 1

    true_counts = np.array([(gt_np == c).sum() for c in range(n_classes)])
    pred_counts = np.array([(y_mapped == c).sum() for c in range(n_classes)])

    x = np.arange(n_classes)
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(6, n_classes * 1.2), 5))
    ax.bar(x - width / 2, true_counts, width, label="Ground truth", color="slategray")
    ax.bar(x + width / 2, pred_counts, width, label="Predicted (Hungarian-matched)", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels([f"class {c}" for c in range(n_classes)])
    ax.set_ylabel("Number of nodes")
    ax.set_title("Cluster Size Distribution: Predicted vs. Ground Truth")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(output_dir, "cluster_size_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Output] Saved cluster size distribution chart -> {path}")


def plot_view_degree_distribution(views, output_dir):
    """Log-log node-degree histogram per view - a standard structural
    diagnostic in multi-view graph papers, showing how differently
    sparse/dense/perceptual/structured views are connected (e.g. the PPR
    "perceptual" view is typically far denser than the raw citation
    "sparse" view)."""
    os.makedirs(output_dir, exist_ok=True)
    n_views = len(views)
    fig, axes = plt.subplots(1, n_views, figsize=(5 * n_views, 4.2))
    if n_views == 1:
        axes = [axes]

    for i, v in enumerate(views):
        ei = v["edge_index"].detach().cpu().numpy()
        num_nodes = int(ei.max()) + 1 if ei.size > 0 else 0
        deg = np.bincount(ei[0], minlength=num_nodes) if ei.size > 0 else np.array([0])
        deg = deg[deg > 0]
        if deg.size == 0:
            axes[i].axis("off")
            continue
        bins = np.logspace(0, np.log10(max(deg.max(), 2)), 25)
        axes[i].hist(deg, bins=bins, color="teal", alpha=0.8)
        axes[i].set_xscale("log"); axes[i].set_yscale("log")
        axes[i].set_title(f"View {i+1} ({v['type']})\ndegree distribution")
        axes[i].set_xlabel("Degree (log)"); axes[i].set_ylabel("Count (log)")
        axes[i].grid(alpha=0.3, which="both")

    plt.tight_layout()
    path = os.path.join(output_dir, "view_degree_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Output] Saved multi-view degree-distribution chart -> {path}")


def plot_interactive_3d_explorer(h_list, fused, pred, gt_labels_readonly, view_types,
                                  output_dir, seed):
    """Multi-view interactive 3D explorer (Plotly HTML): a single figure with
    a dropdown menu to switch between each view's pre-fusion 3D embedding and
    the final fused 3D embedding, all colored by predicted cluster, with
    rotate/zoom/hover. This supersedes a single static "interactive_3d" plot
    with a genuinely useful comparison tool across Stage 8 (per-view) vs
    Stage 9 (fused) embeddings. Optional dependency (plotly); degrades
    gracefully with a warning if not installed. Prefers UMAP for the 3D
    projection (falls back to t-SNE per embedding if UMAP is unavailable)."""
    os.makedirs(output_dir, exist_ok=True)
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("[Output] plotly not installed - skipping interactive 3D explorer. "
              "Install with: pip install plotly --break-system-packages")
        return

    def _project_3d(h):
        h_np = h.detach().cpu().numpy()
        try:
            import umap
            return umap.UMAP(n_components=3, random_state=seed).fit_transform(h_np), "UMAP"
        except Exception:
            return (TSNE(n_components=3, random_state=seed, init="pca",
                          perplexity=30).fit_transform(h_np), "t-SNE")

    labels = [f"View {i+1} ({vt})" for i, vt in enumerate(view_types)] + ["Fused (post Stage-9)"]
    embeddings = [h for h in h_list] + [fused]

    pred_np = np.asarray(pred)
    gt_np = np.asarray(gt_labels_readonly)
    hover_text = [f"gt={g}, pred={p}" for g, p in zip(gt_np, pred_np)]

    fig = go.Figure()
    method_used = "t-SNE"
    for i, (emb_tensor, label) in enumerate(zip(embeddings, labels)):
        emb3d, method_used = _project_3d(emb_tensor)
        fig.add_trace(go.Scatter3d(
            x=emb3d[:, 0], y=emb3d[:, 1], z=emb3d[:, 2],
            mode="markers",
            marker=dict(size=3, color=pred_np, colorscale="Turbo", opacity=0.85),
            text=hover_text, hoverinfo="text",
            name=label, visible=(i == len(embeddings) - 1),  # default view: fused embedding
        ))

    buttons = []
    for i, label in enumerate(labels):
        visibility = [j == i for j in range(len(labels))]
        buttons.append(dict(label=label, method="update",
                             args=[{"visible": visibility}, {"title": f"3D {method_used} - {label}"}]))

    fig.update_layout(
        title=f"3D {method_used} - {labels[-1]}",
        updatemenus=[dict(active=len(labels) - 1, buttons=buttons,
                           x=0.02, y=1.08, xanchor="left")],
        margin=dict(l=0, r=0, b=0, t=60),
    )
    path = os.path.join(output_dir, "interactive_3d_explorer.html")
    fig.write_html(path)
    print(f"[Output] Saved multi-view interactive 3D explorer -> {path}")


def plot_multi_seed_comparison(seed_results, output_dir, cfg=None):
    """Combined visualization across ALL seeds of the primary multi-seed run
    - the companion to the existing per-seed plots and the existing
    per-study (Table 8/9-12/13) combined charts. Produces:
      - a box plot of ACC/NMI/ARI/F1 across seeds (each seed's point labeled,
        overall mean marked in red), plus Silhouette on its own (different
        scale, can be negative)
      - a per-seed grouped bar chart (reusing plot_study_bar_chart) with an
        extra "MEAN +/- STD" group carrying real error bars
      - a per-seed radar chart (reusing plot_study_radar_chart) with an
        extra red "MEAN" line
    Reuses the exact same in-memory seed_results dict run_seeds() already
    built - no CSV re-read, no extra files beyond the PNGs themselves."""
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    seeds = list(seed_results.keys())
    metrics = ["acc", "nmi", "ari", "f1"]
    rng = np.random.RandomState(0)

    # ---- Box plot: ACC/NMI/ARI/F1 together, individual seeds overlaid ----
    data = {m: [seed_results[s][m] * 100 for s in seeds] for m in metrics}
    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        box = ax.boxplot([data[m] for m in metrics], tick_labels=[m.upper() for m in metrics],
                          patch_artist=True, widths=0.5, showmeans=True, meanline=True)
    except TypeError:
        box = ax.boxplot([data[m] for m in metrics], labels=[m.upper() for m in metrics],
                          patch_artist=True, widths=0.5, showmeans=True, meanline=True)
    cmap = plt.get_cmap("tab10")
    for patch, color in zip(box["boxes"], [cmap(i) for i in range(len(metrics))]):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
    for i, m in enumerate(metrics, start=1):
        xs = rng.normal(i, 0.04, size=len(seeds))
        ax.scatter(xs, data[m], color="black", zorder=3, s=22)
        for run_idx, (x, y) in enumerate(zip(xs, data[m]), start=1):
            ax.annotate(f"Run {run_idx}", (x, y), textcoords="offset points", xytext=(5, 0), fontsize=7)
        mean_val = float(np.mean(data[m]))
        ax.scatter([i], [mean_val], color="red", marker="D", zorder=4, s=45,
                   label="overall mean" if i == 1 else None)
    ax.set_ylabel("Score (%)")
    ax.set_title(f"Multi-Seed Comparison ({len(seeds)} seeds) - box plot + overall mean")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="best")
    plt.tight_layout()
    path = os.path.join(output_dir, "multi_seed_boxplot.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Output] Saved multi-seed comparison box plot -> {path}")

    sil_vals = [seed_results[s]["silhouette"] for s in seeds]
    fig, ax = plt.subplots(figsize=(5, 5))
    try:
        ax.boxplot(sil_vals, tick_labels=["Silhouette"], patch_artist=True, showmeans=True, meanline=True)
    except TypeError:
        ax.boxplot(sil_vals, labels=["Silhouette"], patch_artist=True, showmeans=True, meanline=True)
    xs = rng.normal(1, 0.04, size=len(seeds))
    ax.scatter(xs, sil_vals, color="black", zorder=3, s=22)
    for run_idx, (x, y) in enumerate(zip(xs, sil_vals), start=1):
        ax.annotate(f"Run {run_idx}", (x, y), textcoords="offset points", xytext=(5, 0), fontsize=7)
    ax.scatter([1], [float(np.mean(sil_vals))], color="red", marker="D", zorder=4, s=45,
               label="overall mean")
    ax.set_title(f"Multi-Seed Silhouette ({len(seeds)} seeds)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="best")
    plt.tight_layout()
    path2 = os.path.join(output_dir, "multi_seed_silhouette_boxplot.png")
    plt.savefig(path2, dpi=150)
    plt.close(fig)
    print(f"[Output] Saved multi-seed silhouette box plot -> {path2}")

    # ---- Per-seed bar chart + radar chart, PLUS an explicit MEAN group,
    # reusing the existing Table 8/9-12/13 study-chart helpers (they just
    # need a {group_label: {metric_mean, metric_std}} dict). ----
    all_metrics = metrics + ["silhouette"]
    per_seed_as_groups = {}
    for run_idx, s in enumerate(seeds, start=1):
        entry = {}
        for m in all_metrics:
            entry[f"{m}_mean"] = seed_results[s][m]
            entry[f"{m}_std"] = 0.0
        per_seed_as_groups[f"Run {run_idx}"] = entry

    mean_entry = {}
    for m in all_metrics:
        vals = np.array([seed_results[s][m] for s in seeds], dtype=float)
        mean_entry[f"{m}_mean"] = float(vals.mean())
        mean_entry[f"{m}_std"] = float(vals.std())
    per_seed_as_groups["MEAN +/- STD"] = mean_entry

    group_labels = [f"Run {i}" for i in range(1, len(seeds) + 1)] + ["MEAN +/- STD"]

    bar_ok = True
    radar_ok = True
    if cfg is not None:
        include = cfg.get("output_artifacts", {}).get("include", {})
        artifacts_enabled = cfg.get("output_artifacts", {}).get("enabled", True)
        bar_ok = artifacts_enabled and include.get("study_bar_charts", True)
        radar_ok = artifacts_enabled and include.get("radar_charts", True)

    if bar_ok:
        plot_study_bar_chart(
            f"Multi-Seed Per-Seed Results + Overall Mean ({len(seeds)} seeds)",
            group_labels, per_seed_as_groups, output_dir,
            "multi_seed_bar_chart.png", metrics=tuple(all_metrics))
    if radar_ok:
        plot_study_radar_chart(
            f"Multi-Seed Per-Seed Results + Overall Mean ({len(seeds)} seeds)",
            group_labels, per_seed_as_groups, output_dir,
            "multi_seed_radar_chart.png", metrics=tuple(all_metrics))


def save_training_logs(epoch_log, probe_log, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "training_log.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "intra", "cross", "hybrid", "lr"])
        for r in epoch_log:
            writer.writerow([r["epoch"], r["loss"], r["intra"], r["cross"], r["hybrid"], r.get("lr", "")])
    print(f"[Output] Saved per-epoch training log CSV -> {csv_path}")

    probe_csv = os.path.join(output_dir, "probe_log.csv")
    with open(probe_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "silhouette_unsupervised_selection_signal",
                          "acc_readonly_diagnostic", "nmi_readonly_diagnostic", "ari_readonly_diagnostic"])
        for r in probe_log:
            writer.writerow([r["epoch"], r["silhouette"], r["acc"], r["nmi"], r["ari"]])
    print(f"[Output] Saved probe log CSV -> {probe_csv}")


def save_summary_json(cfg, results, best_epoch, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    def clean(v):
        if isinstance(v, tuple):
            return list(v)
        if isinstance(v, dict):
            return {k: clean(vv) for k, vv in v.items()}
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        return v

    summary = {
        "dataset_name": cfg["dataset_name"],
        "num_clusters_known": NUM_CLUSTERS_KNOWN,
        "config": {k: clean(v) for k, v in cfg.items()},
        "model_selection_epoch": best_epoch,
        "model_selection_criterion": "silhouette_coefficient (label-free)",
        "final_results": {k: clean(v) for k, v in results.items() if k != "pred"},
    }
    path = os.path.join(output_dir, "results_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[Output] Saved results summary JSON -> {path}")


# ==============================================================================
# SINGLE-SEED PIPELINE
# ==============================================================================

def run_pipeline_single_seed(seed, base_cfg):
    cfg = copy.deepcopy(base_cfg)
    cfg["seed"] = seed
    cfg["output_dir"] = os.path.join(base_cfg["output_dir"], base_cfg["dataset_name"], f"seed_{seed}")

    set_seed(cfg["seed"])
    print_config(cfg)

    # ---- Stage 1: load data + build multi-view graphs (dataset-driven) --------
    x, gt_labels_readonly, views, dataset_num_classes = load_dataset(cfg)
    if dataset_num_classes != NUM_CLUSTERS_KNOWN:
        print(f"[Warning] DATASET_REGISTRY reports {dataset_num_classes} classes for "
              f"'{cfg['dataset_name']}', but NUM_CLUSTERS_KNOWN={NUM_CLUSTERS_KNOWN} was "
              f"computed at import time. This should not normally happen since both come "
              f"from the same registry entry - double check DATASET_REGISTRY if it does.")

    # ---- Stages 2-7: adaptive augmentation + shared encoder + contrastive training
    encoder, projector, epoch_log, probe_log, best_epoch = train(views, x, gt_labels_readonly, cfg)

    # ---- Stage 8: embeddings from the ORIGINAL (un-augmented) multi-view graphs
    device = torch.device(cfg["device"])
    encoder.eval()
    with torch.no_grad():
        h_list = [encoder(x.to(device), v["edge_index"].to(device)) for v in views]

    # ---- Stage 9: fuse embeddings across views ----------------------------------
    fusion_desc = f" STAGE 9: EMBEDDING FUSION  (mode = {cfg['fusion_mode']}"
    if cfg["fusion_mode"] == "attention":
        fusion_desc += f", align_loss = {cfg['fusion_align_loss']}"
    fusion_desc += ")"
    print("=" * 78)
    print(fusion_desc)
    print("=" * 78)
    fused, view_weights = fuse_embeddings(h_list, cfg, device)
    mean_w = None
    if view_weights is not None:
        mean_w = view_weights.mean(dim=0).detach().cpu().numpy()
        for i, w in enumerate(mean_w):
            print(f"  view {i+1} ({views[i]['type']:<10s}) mean attention weight: {w:.4f}")
    print()

    # ---- Stage 10: cluster number estimation (optional) -------------------------
    print("=" * 78)
    print(" STAGE 10: CLUSTER NUMBER ESTIMATION")
    print("=" * 78)
    if cfg["auto_estimate_k"]:
        if cfg["k_estimation_method"] == "eigengap":
            c_star = estimate_num_clusters_eigengap(fused, cfg, fallback_k=NUM_CLUSTERS_KNOWN)
        else:
            c_star = estimate_num_clusters_silhouette(fused, cfg, fallback_k=NUM_CLUSTERS_KNOWN)
        print(f"[Stage 10] auto_estimate_k=True (method='{cfg['k_estimation_method']}') -> C* = {c_star}")
    else:
        c_star = NUM_CLUSTERS_KNOWN
        print(f"[Stage 10] auto_estimate_k=False -> using NUM_CLUSTERS_KNOWN = {c_star}")
    print()

    # ---- Stage 11: K-Means++ clustering + final evaluation ----------------------
    print("=" * 78)
    print(" STAGE 11: K-MEANS++ CLUSTERING - FINAL RESULT")
    print("=" * 78)
    results = evaluate_clustering(fused, gt_labels_readonly.cpu().numpy(), c_star, cfg, tag="  [FINAL] ")
    print("=" * 78 + "\n")

    # ---- Publication-ready outputs (plots, CSV logs, JSON summary) --------------
    if cfg["output_artifacts"]["enabled"]:
        include = cfg["output_artifacts"].get("include", {})
        print("=" * 78)
        print(" GENERATING PUBLICATION-READY OUTPUTS")
        print("=" * 78)

        if include.get("training_curves", True):
            plot_training_curves(epoch_log, probe_log, cfg["output_dir"])

        if include.get("tsne_umap", True):
            plot_tsne(fused, results["pred"], gt_labels_readonly.cpu().numpy(),
                      cfg["output_dir"], cfg["seed"], cfg)

        if include.get("per_view_vs_fused", True):
            view_types_local = [v["type"] for v in views]
            plot_per_view_vs_fused(h_list, fused, results["pred"], view_types_local,
                                    cfg["output_dir"], cfg["seed"])

        if include.get("confusion_matrix", True):
            plot_confusion_matrix(gt_labels_readonly.cpu().numpy(), results["pred"], cfg["output_dir"])

        if include.get("similarity_heatmap", True):
            plot_similarity_heatmap(fused, results["pred"], cfg["output_dir"], cfg)

        if include.get("attention_weights", True) and mean_w is not None:
            view_types_local = [v["type"] for v in views]
            plot_attention_weights(view_types_local, mean_w, cfg["output_dir"])

        if include.get("graph_structure", False):
            plot_graph_structure(views, results["pred"], cfg["output_dir"], cfg)

        if include.get("silhouette_diagram", True):
            plot_silhouette_diagram(fused, results["pred"], cfg["output_dir"], cfg)

        if include.get("cluster_size_distribution", True):
            plot_cluster_size_distribution(gt_labels_readonly.cpu().numpy(), results["pred"], cfg["output_dir"])

        if include.get("view_degree_distribution", True):
            plot_view_degree_distribution(views, cfg["output_dir"])

        if include.get("interactive_3d", True):
            view_types_local = [v["type"] for v in views]
            plot_interactive_3d_explorer(h_list, fused, results["pred"], gt_labels_readonly.cpu().numpy(),
                                          view_types_local, cfg["output_dir"], cfg["seed"])

        save_training_logs(epoch_log, probe_log, cfg["output_dir"])
        save_summary_json(cfg, results, best_epoch, cfg["output_dir"])
        print(f"\nAll outputs for dataset={cfg['dataset_name']} seed={seed} saved under: {cfg['output_dir']}")
        print("=" * 78)
    else:
        print(f"[Output] output_artifacts.enabled=False -> skipped plots/CSV/JSON "
              f"for dataset={cfg['dataset_name']} seed={seed} (final metrics above are unaffected)")

    _progress_tick(f"{cfg['dataset_name']} seed={seed}")
    return results


# ==============================================================================
# MULTI-SEED ORCHESTRATION
# ==============================================================================

def summarize_multi_seed_results(seed_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    seeds = list(seed_results.keys())
    metrics = ["acc", "nmi", "ari", "f1", "silhouette"]

    print("\n" + "=" * 78)
    print(" MULTI-SEED BENCHMARK - PER-SEED RESULTS")
    print("=" * 78)
    header = f"{'seed':>8s} | " + " | ".join(f"{m.upper():>10s}" for m in metrics)
    print(header)
    print("-" * len(header))
    for seed in seeds:
        r = seed_results[seed]
        row = f"{seed:>8d} | " + " | ".join(f"{r[m]*100:9.2f}%" if m != "silhouette"
                                              else f"{r[m]:10.4f}" for m in metrics)
        print(row)

    stats = {}
    for m in metrics:
        values = np.array([seed_results[s][m] for s in seeds], dtype=float)
        stats[m] = {"mean": float(values.mean()), "std": float(values.std()), "values": values.tolist()}

    print("-" * len(header))
    summary_row = f"{'mean':>8s} | " + " | ".join(
        f"{stats[m]['mean']*100:9.2f}%" if m != "silhouette" else f"{stats[m]['mean']:10.4f}" for m in metrics)
    print(summary_row)
    std_row = f"{'+/- std':>8s} | " + " | ".join(
        f"{stats[m]['std']*100:9.2f}%" if m != "silhouette" else f"{stats[m]['std']:10.4f}" for m in metrics)
    print(std_row)
    print("=" * 78 + "\n")

    csv_path = os.path.join(output_dir, "multi_seed_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["seed"] + [m for m in metrics])
        for seed in seeds:
            r = seed_results[seed]
            writer.writerow([seed] + [r[m] for m in metrics])
        writer.writerow(["mean"] + [stats[m]["mean"] for m in metrics])
        writer.writerow(["std"] + [stats[m]["std"] for m in metrics])
    print(f"[Output] Saved multi-seed CSV summary -> {csv_path}")

    json_path = os.path.join(output_dir, "multi_seed_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "seeds": seeds,
            "per_seed": {str(s): {m: seed_results[s][m] for m in metrics} for s in seeds},
            "aggregate_mean_std": stats,
        }, f, indent=2)
    print(f"[Output] Saved multi-seed JSON summary -> {json_path}")

    return stats


def run_seeds(base_cfg, seeds=None):
    if seeds is None:
        seeds = base_cfg["seeds"]

    if len(seeds) == 1:
        return {seeds[0]: run_pipeline_single_seed(seeds[0], base_cfg)}

    print("\n" + "#" * 78)
    print(f"# MULTI-SEED RUN (dataset={base_cfg['dataset_name']}): running the full "
          f"pipeline for {len(seeds)} seed(s): {seeds}")
    print("#" * 78 + "\n")

    seed_results = {}
    for seed in seeds:
        print(f"\n{'#' * 78}\n# SEED {seed}\n{'#' * 78}\n")
        seed_results[seed] = run_pipeline_single_seed(seed, base_cfg)

    multi_seed_output_dir = os.path.join(base_cfg["output_dir"], base_cfg["dataset_name"])
    summarize_multi_seed_results(seed_results, multi_seed_output_dir)
    if (base_cfg.get("output_artifacts", {}).get("enabled", True)
            and base_cfg.get("output_artifacts", {}).get("include", {}).get("multi_seed_comparison", True)):
        plot_multi_seed_comparison(seed_results, multi_seed_output_dir, base_cfg)
    return seed_results


# ==============================================================================
# GENERIC MEAN/STD HELPER (used by Table 4, Table 8, Table 9-12, Table 13)
# ==============================================================================

def _mean_std(values):
    arr = np.array(values, dtype=float)
    return float(arr.mean()), float(arr.std())


def compute_aggregate_stats(seed_results):
    """Given {seed: results_dict} (as returned by run_seeds / run_pipeline_single_seed
    for every seed), returns {metric: {"mean":..., "std":...}} across seeds.
    Used to build Table 4 (main comparison) regardless of whether one seed or
    many seeds were used."""
    metrics = ["acc", "nmi", "ari", "f1", "silhouette"]
    seeds = list(seed_results.keys())
    stats = {}
    for m in metrics:
        mean, std = _mean_std([seed_results[s][m] for s in seeds])
        stats[m] = {"mean": mean, "std": std}
    return stats


def aggregate_variant_results(per_seed_results):
    metrics = ["acc", "nmi", "ari", "f1", "silhouette"]
    seeds = sorted(per_seed_results.keys())
    agg = {
        "seeds": seeds,
        "per_seed": {s: {m: per_seed_results[s][m] for m in metrics} for s in seeds},
    }
    for m in metrics:
        mean, std = _mean_std([per_seed_results[s][m] for s in seeds])
        agg[f"{m}_mean"] = mean
        agg[f"{m}_std"] = std
        agg[m] = mean
    return agg


# ==============================================================================
# TABLE 4: MAIN ACC COMPARISON TABLE
# ==============================================================================
_TABLE4_BASELINE_METHODS = [
    "K-means", "SC", "GAE", "GraphCL", "CoNMF", "RMSC",
    "GMC", "DMVC", "O2MA", "MCGC", "CMVGL", "GCA",
]


def print_and_save_table4(cfg, primary_stats, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    metrics = ["acc", "nmi", "ari", "f1"]

    print("\n" + "=" * 96)
    print(f" TABLE 4 STYLE SUMMARY - Main Clustering Accuracy Comparison (dataset = {cfg['dataset_name']})")
    print(" NOTE: this script only trains/evaluates AGAMC itself. The 12 baseline")
    print(" methods from the paper's Table 4 are NOT implemented here, so only the")
    print(" 'AGAMC (Ours)' row below is populated with real numbers.")
    print("=" * 96)
    header = f"{'Method':<16s} | " + " | ".join(f"{m.upper():>16s}" for m in metrics)
    print(header)
    print("-" * len(header))

    for name in _TABLE4_BASELINE_METHODS:
        print(f"{name:<16s} | " + " | ".join(f"{'-':>16s}" for _ in metrics))

    row = f"{'AGAMC (Ours)':<16s} | " + " | ".join(
        f"{primary_stats[m]['mean']*100:6.2f}+/-{primary_stats[m]['std']*100:4.2f}%" for m in metrics)
    print(row)
    print("=" * 96 + "\n")

    csv_path = os.path.join(output_dir, "table4_main_comparison.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method"] + [f"{m}_mean" for m in metrics] + [f"{m}_std" for m in metrics])
        for name in _TABLE4_BASELINE_METHODS:
            writer.writerow([name] + ["" for _ in metrics] + ["" for _ in metrics])
        writer.writerow(["AGAMC (Ours)"] + [primary_stats[m]["mean"] for m in metrics]
                         + [primary_stats[m]["std"] for m in metrics])
    print(f"[Output] Saved Table 4 CSV -> {csv_path}")


# ==============================================================================
# TABLES 5/6/7 (NMI / ARI / F1 across datasets) + 4.2.2 RADAR CHART
# All three reuse the SAME primary_stats dict Table 4 already computes -
# no extra training runs are needed for any of these.
# ==============================================================================

_METRIC_TABLE_SPEC = {
    "table5_nmi": ("nmi", "NMI", 5),
    "table6_ari": ("ari", "ARI", 6),
    "table7_f1":  ("f1",  "F1",  7),
}


def print_and_save_metric_table(cfg, primary_stats, output_dir, report_key):
    """Generic single-metric comparison table (Table 5/6/7 style) - same
    baseline-method layout as Table 4, but only one metric column, since only
    AGAMC itself is trained by this script."""
    metric_key, metric_label, table_num = _METRIC_TABLE_SPEC[report_key]
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 78)
    print(f" TABLE {table_num} STYLE SUMMARY - {metric_label} comparison "
          f"(dataset = {cfg['dataset_name']})")
    print(" NOTE: only the 'AGAMC (Ours)' row is populated with real numbers -")
    print(" the baseline methods below are not implemented in this script.")
    print("=" * 78)
    header = f"{'Method':<16s} | {metric_label:>16s}"
    print(header)
    print("-" * len(header))
    for name in _TABLE4_BASELINE_METHODS:
        print(f"{name:<16s} | {'-':>16s}")
    row = (f"{'AGAMC (Ours)':<16s} | "
           f"{primary_stats[metric_key]['mean']*100:6.2f}+/-{primary_stats[metric_key]['std']*100:4.2f}%")
    print(row)
    print("=" * 78 + "\n")

    csv_path = os.path.join(output_dir, f"table{table_num}_{metric_key}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", f"{metric_key}_mean", f"{metric_key}_std"])
        for name in _TABLE4_BASELINE_METHODS:
            writer.writerow([name, "", ""])
        writer.writerow(["AGAMC (Ours)", primary_stats[metric_key]["mean"], primary_stats[metric_key]["std"]])
    print(f"[Output] Saved Table {table_num} CSV -> {csv_path}")


def plot_main_comparison_radar(cfg, primary_stats, output_dir):
    """4.2.2-style radar/spider chart across ACC/NMI/ARI/F1. Only the
    'AGAMC (Ours)' spoke is populated with real numbers (baselines are not
    implemented here) - if you later add baseline numbers, just extend
    `results_by_group` below with one more group per baseline method and
    they will render as additional spokes automatically."""
    results_by_group = {
        "AGAMC (Ours)": {
            "acc_mean": primary_stats["acc"]["mean"], "acc_std": primary_stats["acc"]["std"],
            "nmi_mean": primary_stats["nmi"]["mean"], "nmi_std": primary_stats["nmi"]["std"],
            "ari_mean": primary_stats["ari"]["mean"], "ari_std": primary_stats["ari"]["std"],
            "f1_mean": primary_stats["f1"]["mean"], "f1_std": primary_stats["f1"]["std"],
        }
    }
    plot_study_radar_chart(
        f"Table 4-7 Main Comparison Radar ({cfg['dataset_name']})",
        list(results_by_group.keys()), results_by_group, output_dir,
        "main_comparison_radar_chart.png", metrics=("acc", "nmi", "ari", "f1"))


# ==============================================================================
# ABLATION STUDY (Table 8 style)
# ==============================================================================

def build_ablation_variants():
    return [
        ("AGAMC-Full", {}),
        ("AGAMC-Uniform", {"augmentation_mode": "uniform"}),
        ("AGAMC-NoAug", {"augmentation_mode": "none"}),
        ("AGAMC-RandomAug", {"augmentation_mode": "random"}),
        ("AGAMC-NoHybrid (\u03b3=0)", {"gamma_hybrid": 0.0}),
        ("AGAMC-NoCross (\u03b2=0)", {"beta_cross": 0.0}),
        ("AGAMC-IntraOnly", {"beta_cross": 0.0, "gamma_hybrid": 0.0}),
    ]


def print_and_save_ablation_table(variant_results, variants, output_dir, multi_seed_ablation=False):
    os.makedirs(output_dir, exist_ok=True)
    full_name = variants[0][0]
    full_per_seed_acc = variant_results[full_name]["per_seed"]

    print("\n" + "=" * 100)
    title_suffix = " (mean +/- std over seeds)" if multi_seed_ablation else ""
    print(f" ABLATION STUDY - TABLE 8 STYLE SUMMARY{title_suffix}")
    print(" (Variant | ACC | NMI | ARI | F1 | Delta vs. Full)")
    print("=" * 100)
    col_w = 16 if multi_seed_ablation else 9
    header = (f"{'Variant':<28s} | {'ACC':>{col_w}s} | {'NMI':>{col_w}s} | "
              f"{'ARI':>{col_w}s} | {'F1':>{col_w}s} | {'Delta vs Full':>{col_w}s}")
    print(header)
    print("-" * len(header))

    def fmt_pct(mean, std):
        if multi_seed_ablation:
            return f"{mean*100:6.2f}+/-{std*100:4.2f}%"
        return f"{mean*100:7.2f}%"

    rows_for_csv = []
    for name, _ in variants:
        r = variant_results[name]

        if name == full_name:
            delta_mean, delta_std, delta_str = None, None, "-"
        else:
            paired_deltas = []
            for s in r["seeds"]:
                if s in full_per_seed_acc:
                    acc_full_s = full_per_seed_acc[s]["acc"]
                    acc_var_s = r["per_seed"][s]["acc"]
                    if acc_full_s:
                        paired_deltas.append((acc_full_s - acc_var_s) / acc_full_s * 100)
            if paired_deltas:
                delta_mean, delta_std = _mean_std(paired_deltas)
                delta_str = (f"{delta_mean:+.2f}+/-{delta_std:.2f}%" if multi_seed_ablation
                             else f"{delta_mean:+.2f}%")
            else:
                delta_mean, delta_std, delta_str = float("nan"), float("nan"), "n/a"

        print(f"{name:<28s} | {fmt_pct(r['acc_mean'], r['acc_std']):>{col_w}s} | "
              f"{fmt_pct(r['nmi_mean'], r['nmi_std']):>{col_w}s} | "
              f"{fmt_pct(r['ari_mean'], r['ari_std']):>{col_w}s} | "
              f"{fmt_pct(r['f1_mean'], r['f1_std']):>{col_w}s} | "
              f"{delta_str:>{col_w}s}")

        rows_for_csv.append([
            name, r["acc_mean"], r["acc_std"], r["nmi_mean"], r["nmi_std"],
            r["ari_mean"], r["ari_std"], r["f1_mean"], r["f1_std"],
            delta_mean, delta_std, ";".join(str(s) for s in r["seeds"]),
        ])
    print("=" * 100 + "\n")

    csv_path = os.path.join(output_dir, "table8_ablation.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "acc_mean", "acc_std", "nmi_mean", "nmi_std",
                          "ari_mean", "ari_std", "f1_mean", "f1_std",
                          "delta_acc_vs_full_pct_mean", "delta_acc_vs_full_pct_std", "seeds"])
        writer.writerows(rows_for_csv)
    print(f"[Output] Saved Table 8 (ablation) CSV -> {csv_path}")

    json_path = os.path.join(output_dir, "table8_ablation.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "full_reference_variant": full_name,
            "multi_seed_ablation": multi_seed_ablation,
            "rows": [
                {"variant": row[0], "acc_mean": row[1], "acc_std": row[2],
                 "nmi_mean": row[3], "nmi_std": row[4], "ari_mean": row[5], "ari_std": row[6],
                 "f1_mean": row[7], "f1_std": row[8],
                 "delta_acc_vs_full_pct_mean": row[9], "delta_acc_vs_full_pct_std": row[10],
                 "seeds": row[11]}
                for row in rows_for_csv
            ],
        }, f, indent=2)
    print(f"[Output] Saved Table 8 (ablation) JSON -> {json_path}")

    detail_csv = os.path.join(output_dir, "table8_ablation_per_seed_detail.csv")
    with open(detail_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "seed", "acc", "nmi", "ari", "f1", "silhouette"])
        for name, _ in variants:
            for seed, vals in variant_results[name]["per_seed"].items():
                writer.writerow([name, seed, vals["acc"], vals["nmi"], vals["ari"],
                                  vals["f1"], vals["silhouette"]])
    print(f"[Output] Saved Table 8 per-seed detail CSV -> {detail_csv}")


def run_ablation_study(base_cfg, full_seed_results):
    seeds = base_cfg["seeds"]
    multi_seed_ablation = len(seeds) > 1

    variants = build_ablation_variants()
    full_name = variants[0][0]
    fresh_variants = variants[1:]

    print("\n" + "#" * 78)
    print(f"# ABLATION STUDY / TABLE 8 (dataset={base_cfg['dataset_name']}): {len(variants)} variant(s) "
          f"x {len(seeds)} seed(s); '{full_name}' reused from the primary run")
    print(f"# seeds = {seeds}")
    print("#" * 78 + "\n")

    variant_results = {full_name: aggregate_variant_results(full_seed_results)}
    for name, overrides in fresh_variants:
        variant_slug = name.split(" ")[0].replace("AGAMC-", "").lower()
        per_seed_results = {}
        for seed in seeds:
            cfg = copy.deepcopy(base_cfg)
            cfg.update(overrides)
            cfg["seeds"] = [seed]

            # FIX: this used to write cfg["ablation_study"] = {...}, a stray
            # top-level dict nobody reads any more. The only thing that
            # matters here is disabling the (recursive) ablation-study
            # sub-report inside cfg["reports"] so a fresh single-seed run
            # doesn't try to re-trigger another ablation study on itself.
            cfg["reports"] = copy.deepcopy(cfg["reports"])
            cfg["reports"]["ablation_study"] = {"enabled": False}

            cfg["output_dir"] = os.path.join(base_cfg["output_dir"], base_cfg["dataset_name"],
                                              "ablation", variant_slug)

            print(f"\n{'=' * 78}\n ABLATION VARIANT: {name}  |  seed={seed}\n{'=' * 78}\n")
            per_seed_results[seed] = run_pipeline_single_seed(seed, cfg)

        variant_results[name] = aggregate_variant_results(per_seed_results)

    output_dir = os.path.join(base_cfg["output_dir"], base_cfg["dataset_name"])
    print_and_save_ablation_table(variant_results, variants, output_dir, multi_seed_ablation)

    if base_cfg["output_artifacts"]["enabled"]:
        include = base_cfg["output_artifacts"].get("include", {})
        if include.get("study_bar_charts", True):
            plot_study_bar_chart(
                f"Table 8 Ablation Study ({base_cfg['dataset_name']})",
                [name for name, _ in variants], variant_results, output_dir,
                "table8_ablation_bar_chart.png")
        if include.get("radar_charts", True):
            plot_study_radar_chart(
                f"Table 8 Ablation Study ({base_cfg['dataset_name']})",
                [name for name, _ in variants], variant_results, output_dir,
                "table8_ablation_radar_chart.png")

    return variant_results


# ==============================================================================
# AUGMENTATION-CATEGORY STUDY (Tables 9-12 style, one table per view-type)
# ==============================================================================

def run_category_study_variant(base_cfg, view_type, op, seed):
    """Runs ONE (view_type, op, seed) combination: trains with every view on
    its normal default augmentation EXCEPT the view(s) of type `view_type`,
    which are augmented using ONLY the single operator `op`."""
    cfg = copy.deepcopy(base_cfg)
    cfg["seed"] = seed
    # FIX: disable the nested ablation_study sub-report (not a stray
    # top-level "ablation_study" key, which nothing reads any more).
    cfg["reports"] = copy.deepcopy(cfg["reports"])
    cfg["reports"]["ablation_study"] = {"enabled": False}
    cfg["output_dir"] = os.path.join(base_cfg["output_dir"], base_cfg["dataset_name"],
                                      "category_study", view_type, op, f"seed_{seed}")

    set_seed(seed)
    x, gt_labels_readonly, views, _ = load_dataset(cfg)
    encoder, projector, _, _, _ = train(
        views, x, gt_labels_readonly, cfg,
        single_op_override={"view_type": view_type, "op": op})

    device = torch.device(cfg["device"])
    encoder.eval()
    with torch.no_grad():
        h_list = [encoder(x.to(device), v["edge_index"].to(device)) for v in views]
    fused, _ = fuse_embeddings(h_list, cfg, device)

    result = evaluate_clustering(fused, gt_labels_readonly.cpu().numpy(), NUM_CLUSTERS_KNOWN, cfg,
                                  tag=f"  [category_study view={view_type} op={op} seed={seed}] ")
    _progress_tick(f"category_study view={view_type} op={op} seed={seed}")
    return result


_VIEW_TYPE_TO_TABLE_NUMBER = {
    "sparse": 9,
    "dense": 10,
    "perceptual": 11,
    "structured": 12,
}


def print_and_save_category_table(view_type, op_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    labels = AUGMENTATION_CATEGORY_LABELS.get(view_type, {})
    table_num = _VIEW_TYPE_TO_TABLE_NUMBER.get(view_type, "?")

    print("\n" + "=" * 90)
    print(f" TABLE {table_num} STYLE - view_type = '{view_type}'  "
          f"(Augmentation | ACC | NMI | ARI | F1 | Category)")
    print("=" * 90)
    header = (f"{'Augmentation':<20s} | {'ACC':>8s} | {'NMI':>8s} | "
              f"{'ARI':>8s} | {'F1':>8s} | {'Category':>12s}")
    print(header)
    print("-" * len(header))

    rows = []
    for op, r in sorted(op_results.items(), key=lambda kv: kv[1]["acc_mean"], reverse=True):
        cat = labels.get(op, "n/a")
        print(f"{op:<20s} | {r['acc_mean']*100:7.2f}% | {r['nmi_mean']*100:7.2f}% | "
              f"{r['ari_mean']*100:7.2f}% | {r['f1_mean']*100:7.2f}% | {cat:>12s}")
        rows.append([op, r["acc_mean"], r["acc_std"], r["nmi_mean"], r["nmi_std"],
                     r["ari_mean"], r["ari_std"], r["f1_mean"], r["f1_std"], cat])
    print("=" * 90 + "\n")

    csv_path = os.path.join(output_dir, f"table{table_num}_category_{view_type}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["augmentation", "acc_mean", "acc_std", "nmi_mean", "nmi_std",
                          "ari_mean", "ari_std", "f1_mean", "f1_std", "category"])
        writer.writerows(rows)
    print(f"[Output] Saved Table {table_num} CSV ('{view_type}') -> {csv_path}")


def run_augmentation_category_study(base_cfg):
    study_cfg = base_cfg["reports"]["augmentation_category"]
    # FIX: "augmentation_category" has no single top-level "enabled" flag -
    # each view-type has its own "enabled" flag, already reduced into
    # target_view_types by the sync block after CONFIG is built (and
    # main() already gates the call to this function on that). An empty
    # target_view_types list means nothing to do.
    if not study_cfg.get("target_view_types"):
        return None

    seeds = study_cfg.get("seeds", base_cfg["seeds"])
    dataset_views = DATASET_REGISTRY[base_cfg["dataset_name"]]["views"]
    available_types = {v["type"] for v in dataset_views}

    print("\n" + "#" * 78)
    print(f"# AUGMENTATION-CATEGORY STUDY (Tables 9-12 style), dataset="
          f"{base_cfg['dataset_name']}, seeds={seeds}")
    print(f"# selected view types (from reports): {study_cfg['target_view_types']}")
    print("#" * 78 + "\n")

    output_root = os.path.join(base_cfg["output_dir"], base_cfg["dataset_name"], "category_study")
    bar_chart_enabled = (base_cfg["output_artifacts"]["enabled"]
                          and base_cfg["output_artifacts"].get("include", {}).get("study_bar_charts", True))
    radar_chart_enabled = (base_cfg["output_artifacts"]["enabled"]
                            and base_cfg["output_artifacts"].get("include", {}).get("radar_charts", True))

    all_tables = {}
    for view_type in study_cfg.get("target_view_types", []):
        if view_type not in available_types:
            print(f"[CategoryStudy] Skipping '{view_type}' - dataset "
                  f"'{base_cfg['dataset_name']}' has no view of this type.")
            continue

        ops = study_cfg.get("ops_per_view_type", {}).get(view_type, [])
        table_num = _VIEW_TYPE_TO_TABLE_NUMBER.get(view_type, "?")
        print(f"\n{'=' * 78}\n CATEGORY STUDY: TABLE {table_num} / view_type='{view_type}'  "
              f"({len(ops)} op(s) x {len(seeds)} seed(s))\n{'=' * 78}\n")

        op_results = {}
        for op in ops:
            per_seed = {seed: run_category_study_variant(base_cfg, view_type, op, seed)
                        for seed in seeds}
            op_results[op] = aggregate_variant_results(per_seed)

        all_tables[view_type] = op_results
        if study_cfg.get("output_tables", True):
            print_and_save_category_table(view_type, op_results, output_root)
            if bar_chart_enabled:
                plot_study_bar_chart(
                    f"Table {table_num} Augmentation-Category Study "
                    f"(view_type='{view_type}', {base_cfg['dataset_name']})",
                    list(op_results.keys()), op_results, output_root,
                    f"table{table_num}_category_{view_type}_bar_chart.png")
            if radar_chart_enabled:
                plot_study_radar_chart(
                    f"Table {table_num} Augmentation-Category Study "
                    f"(view_type='{view_type}', {base_cfg['dataset_name']})",
                    list(op_results.keys()), op_results, output_root,
                    f"table{table_num}_category_{view_type}_radar_chart.png")
        else:
            print(f"[CategoryStudy] output_tables=False -> computed results for "
                  f"'{view_type}' but skipped printing/saving its table.")

    return all_tables


# ==============================================================================
# 4.3.2 COMBINED FIGURE: Tables 9-12 side-by-side in one 2x2 layout, bars
# colored by their Recommended / Prohibited / Neutral / Allowed category.
# ==============================================================================

_CATEGORY_COLORS = {
    "Recommended": "#2ca02c",
    "Prohibited":  "#d62728",
    "Neutral":     "#7f7f7f",
    "Allowed":     "#1f77b4",
    "n/a":         "#bbbbbb",
}


def plot_augmentation_category_combined(all_tables, output_dir, cfg):
    """One Figure with up to 4 subplots (sparse/dense/perceptual/structured),
    each a bar chart of ACC per augmentation operator for that view type,
    bars colored by AUGMENTATION_CATEGORY_LABELS. This is the "core validation
    of the taxonomy" figure requested for section 4.3.2, as a single Figure
    rather than 4 separate tables."""
    if not all_tables:
        return
    os.makedirs(output_dir, exist_ok=True)
    view_types = [vt for vt in ("sparse", "dense", "perceptual", "structured") if vt in all_tables]
    if not view_types:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    table_num_map = _VIEW_TYPE_TO_TABLE_NUMBER

    for ax, vtype in zip(axes, view_types):
        op_results = all_tables[vtype]
        labels_map = AUGMENTATION_CATEGORY_LABELS.get(vtype, {})
        ops = list(op_results.keys())
        accs = [op_results[op]["acc_mean"] * 100 for op in ops]
        stds = [op_results[op]["acc_std"] * 100 for op in ops]
        colors = [_CATEGORY_COLORS.get(labels_map.get(op, "n/a"), "#bbbbbb") for op in ops]
        ax.bar(ops, accs, yerr=stds, capsize=3, color=colors)
        ax.set_title(f"Table {table_num_map.get(vtype, '?')} - view_type='{vtype}'")
        ax.set_ylabel("ACC (%)")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.3, axis="y")

    for ax in axes[len(view_types):]:
        ax.axis("off")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in _CATEGORY_COLORS.values()]
    fig.legend(handles, list(_CATEGORY_COLORS.keys()), loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Augmentation-Category Study - Combined View (Tables 9-12, {cfg['dataset_name']})")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    path = os.path.join(output_dir, "tables9_12_combined_figure.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Output] Saved combined Tables 9-12 figure -> {path}")


# ==============================================================================
# RULE-BASED vs. LEARNABLE AUGMENTATION STUDY (Table 13 style)
# ==============================================================================

def print_and_save_rule_vs_learnable_table(dataset_name, mode_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    rb, lr = mode_results["rule_based"], mode_results["learnable"]
    diff_pct = (lr["acc_mean"] - rb["acc_mean"]) * 100

    print("\n" + "=" * 78)
    print(" RULE-BASED vs. LEARNABLE AUGMENTATION - TABLE 13 STYLE SUMMARY")
    print("=" * 78)
    print(f"{'Dataset':<12s} | {'Rule-Based ACC':>16s} | {'Learnable ACC':>16s} | {'Difference':>10s}")
    print(f"{dataset_name:<12s} | {rb['acc_mean']*100:15.2f}% | {lr['acc_mean']*100:15.2f}% | "
          f"{diff_pct:+9.2f}%")
    print("=" * 78 + "\n")

    csv_path = os.path.join(output_dir, "table13_rule_vs_learnable.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "rule_based_acc_mean", "learnable_acc_mean", "diff_pct"])
        writer.writerow([dataset_name, rb["acc_mean"], lr["acc_mean"], diff_pct])
    print(f"[Output] Saved Table 13 CSV -> {csv_path}")


def plot_rule_vs_learnable_cost_chart(dataset_name, mode_results, output_dir, learnable_cost_pct):
    """4.5-style paired bar chart: ACC for Rule-Based vs Learnable, with the
    Learnable bar annotated with its extra training-time cost (e.g. '+30%
    training time'). The cost percentage is an informational annotation you
    set in CONFIG (learnable_cost_pct) - this script does not automatically
    measure the wall-clock cost difference between the two modes."""
    os.makedirs(output_dir, exist_ok=True)
    modes = list(mode_results.keys())
    accs = [mode_results[m]["acc_mean"] * 100 for m in modes]
    stds = [mode_results[m]["acc_std"] * 100 for m in modes]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(modes, accs, yerr=stds, capsize=4, color=["#1f77b4", "#ff7f0e"])
    ax.set_ylabel("ACC (%)")
    ax.set_title(f"Table 13: Rule-Based vs. Learnable ({dataset_name})\n"
                 f"accuracy / computational-cost trade-off")
    ax.grid(alpha=0.3, axis="y")

    if "learnable" in mode_results:
        idx = modes.index("learnable")
        bar = bars[idx]
        ax.annotate(f"+{learnable_cost_pct:.0f}% training time",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 12), textcoords="offset points",
                    ha="center", fontsize=9, color="darkred",
                    arrowprops=dict(arrowstyle="-", color="darkred"))

    plt.tight_layout()
    path = os.path.join(output_dir, "table13_rule_vs_learnable_cost_chart.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Output] Saved Table 13 cost-annotated bar chart -> {path}")


def run_rule_vs_learnable_study(base_cfg):
    study_cfg = base_cfg["reports"]["rule_vs_learnable"]
    if not study_cfg.get("enabled", False):
        return None
    seeds = study_cfg.get("seeds", base_cfg["seeds"])

    print("\n" + "#" * 78)
    print(f"# RULE-BASED vs. LEARNABLE STUDY / TABLE 13 (dataset="
          f"{base_cfg['dataset_name']}, seeds={seeds}")
    print("#" * 78 + "\n")

    mode_results = {}
    for mode in ["rule_based", "learnable"]:
        per_seed = {}
        for seed in seeds:
            cfg = copy.deepcopy(base_cfg)
            cfg["augmentation_mode"] = mode
            cfg["seeds"] = [seed]

            # FIX: disable the nested ablation_study sub-report instead of
            # writing a stray top-level "ablation_study" key.
            cfg["reports"] = copy.deepcopy(cfg["reports"])
            cfg["reports"]["ablation_study"] = {"enabled": False}

            cfg["output_dir"] = os.path.join(base_cfg["output_dir"], base_cfg["dataset_name"],
                                              "rule_vs_learnable", mode)
            print(f"\n{'=' * 78}\n RULE-VS-LEARNABLE: mode='{mode}'  seed={seed}\n{'=' * 78}\n")
            per_seed[seed] = run_pipeline_single_seed(seed, cfg)
        mode_results[mode] = aggregate_variant_results(per_seed)

    output_dir = os.path.join(base_cfg["output_dir"], base_cfg["dataset_name"])
    if study_cfg.get("output_table", True):
        print_and_save_rule_vs_learnable_table(base_cfg["dataset_name"], mode_results, output_dir)
        if base_cfg["output_artifacts"]["enabled"]:
            include = base_cfg["output_artifacts"].get("include", {})
            if include.get("study_bar_charts", True):
                plot_study_bar_chart(
                    f"Table 13 Rule-Based vs. Learnable Augmentation ({base_cfg['dataset_name']})",
                    list(mode_results.keys()), mode_results, output_dir,
                    "table13_rule_vs_learnable_bar_chart.png")
                if study_cfg.get("annotate_cost", False):
                    plot_rule_vs_learnable_cost_chart(
                        base_cfg["dataset_name"], mode_results, output_dir,
                        study_cfg.get("learnable_cost_pct", 30.0))
            if include.get("radar_charts", True):
                plot_study_radar_chart(
                    f"Table 13 Rule-Based vs. Learnable Augmentation ({base_cfg['dataset_name']})",
                    list(mode_results.keys()), mode_results, output_dir,
                    "table13_rule_vs_learnable_radar_chart.png")
    else:
        print("[RuleVsLearnable] output_table=False -> computed results but skipped "
              "printing/saving the table.")

    return mode_results


# ==============================================================================
# TASK A: HYPERPARAMETER SENSITIVITY ANALYSIS (Table 14 style)
# ==============================================================================
# Sweeps ONE hyperparameter at a time (holding everything else at the base
# CONFIG) and reports ACC/NMI/ARI/F1 (mean +/- std over seeds) at every
# declared value. See CONFIG["reports"]["hyperparameter_sensitivity"] and the
# module docstring at the top of this file for the full list of supported
# parameters and the special-case handling notes below.
# ==============================================================================

# Maps a Table-14 sweep param_name -> the plain top-level cfg key it writes
# to (for the "plain direct override" parameters only - the others need
# special handling, see run_hyperparameter_variant below).
_HP_DIRECT_CFG_KEY = {
    "temperature": "temperature",
    "alpha_intra": "alpha_intra",
    "beta_cross": "beta_cross",
    "gamma_hybrid": "gamma_hybrid",
    "hidden_dim": "hidden_dim",
    "gcn_layers": "num_gcn_layers",
}


def _dataset_has_view_type(dataset_name, vtype):
    return any(v["type"] == vtype for v in DATASET_REGISTRY[dataset_name]["views"])


def _dataset_supports_knn_k_sweep(dataset_name):
    """knn_k structurally applies if the dataset has a kNN-built view: the
    Planetoid 'dense' view (build='planetoid_knn') or any generic
    'knn_file'-built view."""
    spec = DATASET_REGISTRY[dataset_name]
    if spec["loader"] == "planetoid":
        return True  # every planetoid dataset has a planetoid_knn dense view
    return any(v.get("build") == "knn_file" for v in spec["views"])


def run_hyperparameter_variant(base_cfg, param_name, value, seed):
    """Runs ONE (param_name, value, seed) combination of the Table 14
    hyperparameter-sensitivity sweep. Mirrors run_category_study_variant's
    structure exactly: deep-copy cfg, apply the override for this specific
    parameter, disable the nested ablation_study sub-report, set a
    namespaced output_dir, set_seed, load_dataset, train, fuse_embeddings,
    evaluate_clustering, _progress_tick, return the result dict."""
    cfg = copy.deepcopy(base_cfg)
    cfg["seed"] = seed
    cfg["reports"] = copy.deepcopy(cfg["reports"])
    cfg["reports"]["ablation_study"] = {"enabled": False}
    cfg["output_dir"] = os.path.join(base_cfg["output_dir"], base_cfg["dataset_name"],
                                      "hyperparameter_sensitivity", param_name, str(value),
                                      f"seed_{seed}")

    single_op_override = None
    knn_k_override = None

    if param_name == "edge_drop_rate":
        cfg["edge_drop_p"] = (value, value)
    elif param_name == "feature_mask_rate":
        cfg["feature_mask_p"] = (value, value)
    elif param_name == "knn_k":
        knn_k_override = value
        cfg["knn_k"] = value  # keep printed config consistent for the planetoid path
    elif param_name == "subgraph_walk_length":
        cfg["reports"]["augmentation_category"] = copy.deepcopy(cfg["reports"]["augmentation_category"])
        cfg["reports"]["augmentation_category"]["subgraph_extract_walk_length"] = value
        single_op_override = {"view_type": "perceptual", "op": "subgraph_extract"}
    elif param_name in _HP_DIRECT_CFG_KEY:
        cfg[_HP_DIRECT_CFG_KEY[param_name]] = value
    else:
        raise ValueError(f"Unknown hyperparameter_sensitivity param_name: {param_name!r}")

    set_seed(seed)
    x, gt_labels_readonly, views, _ = load_dataset(cfg, knn_k_override=knn_k_override)
    encoder, projector, _, _, _ = train(
        views, x, gt_labels_readonly, cfg, single_op_override=single_op_override)

    device = torch.device(cfg["device"])
    encoder.eval()
    with torch.no_grad():
        h_list = [encoder(x.to(device), v["edge_index"].to(device)) for v in views]
    fused, _ = fuse_embeddings(h_list, cfg, device)

    result = evaluate_clustering(
        fused, gt_labels_readonly.cpu().numpy(), NUM_CLUSTERS_KNOWN, cfg,
        tag=f"  [hyperparameter_sensitivity param={param_name} value={value} seed={seed}] ")
    _progress_tick(f"hyperparameter_sensitivity param={param_name} value={value} seed={seed}")
    return result


def print_and_save_hyperparameter_table(param_name, value_results, output_dir):
    """Mirrors print_and_save_category_table, but sorted by VALUE ascending
    (this is a sweep, not a ranking)."""
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 90)
    print(f" TABLE 14 STYLE - Hyperparameter Sensitivity: '{param_name}'  "
          f"(Value | ACC | NMI | ARI | F1)")
    print("=" * 90)
    header = f"{'Value':<16s} | {'ACC':>8s} | {'NMI':>8s} | {'ARI':>8s} | {'F1':>8s}"
    print(header)
    print("-" * len(header))

    rows = []
    for value in sorted(value_results.keys(), key=lambda v: float(v)):
        r = value_results[value]
        print(f"{str(value):<16s} | {r['acc_mean']*100:7.2f}% | {r['nmi_mean']*100:7.2f}% | "
              f"{r['ari_mean']*100:7.2f}% | {r['f1_mean']*100:7.2f}%")
        rows.append([value, r["acc_mean"], r["acc_std"], r["nmi_mean"], r["nmi_std"],
                     r["ari_mean"], r["ari_std"], r["f1_mean"], r["f1_std"]])
    print("=" * 90 + "\n")

    csv_path = os.path.join(output_dir, f"table14_hyperparameter_sensitivity_{param_name}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["value", "acc_mean", "acc_std", "nmi_mean", "nmi_std",
                          "ari_mean", "ari_std", "f1_mean", "f1_std"])
        writer.writerows(rows)
    print(f"[Output] Saved Table 14 CSV ('{param_name}') -> {csv_path}")


def plot_hyperparameter_sensitivity_curve(param_name, value_results, output_dir):
    """x-axis = sorted numeric swept values; one subplot per metric
    (ACC/NMI/ARI/F1) with a shaded +/-1 std band across seeds via
    fill_between."""
    os.makedirs(output_dir, exist_ok=True)
    values_sorted = sorted(value_results.keys(), key=lambda v: float(v))
    metrics = ["acc", "nmi", "ari", "f1"]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for ax, m in zip(axes, metrics):
        means = np.array([value_results[v][f"{m}_mean"] * 100 for v in values_sorted])
        stds = np.array([value_results[v][f"{m}_std"] * 100 for v in values_sorted])
        xs = np.array([float(v) for v in values_sorted])
        ax.plot(xs, means, marker="o", color="steelblue")
        ax.fill_between(xs, means - stds, means + stds, alpha=0.2, color="steelblue")
        ax.set_title(m.upper())
        ax.set_xlabel(param_name)
        ax.set_ylabel("Score (%)")
        ax.grid(alpha=0.3)

    plt.suptitle(f"Table 14 Hyperparameter Sensitivity: '{param_name}'")
    plt.tight_layout()
    path = os.path.join(output_dir, f"table14_hyperparameter_sensitivity_{param_name}_curve.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Output] Saved hyperparameter-sensitivity curve ('{param_name}') -> {path}")


def plot_loss_weight_joint_heatmap(row_values, col_values, acc_grid, output_dir,
                                    row_name="beta_cross", col_name="gamma_hybrid",
                                    filename="table14_loss_weight_joint_heatmap.png",
                                    title_prefix="Table 14"):
    """2D grid over row_name.values x col_name.values, ACC as color. Used for
    both the beta_cross x gamma_hybrid heatmap and the 4.4.2 alpha_intra x
    beta_cross heatmap (same function, different param names/labels)."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(acc_grid * 100, cmap="viridis", origin="lower", aspect="auto")
    ax.set_xticks(range(len(col_values)))
    ax.set_xticklabels([str(v) for v in col_values])
    ax.set_yticks(range(len(row_values)))
    ax.set_yticklabels([str(v) for v in row_values])
    ax.set_xlabel(col_name)
    ax.set_ylabel(row_name)
    ax.set_title(f"{title_prefix} - Joint {row_name} x {col_name} Sensitivity (ACC %)")
    for i in range(len(row_values)):
        for j in range(len(col_values)):
            if not np.isnan(acc_grid[i, j]):
                ax.text(j, i, f"{acc_grid[i, j]*100:.1f}", ha="center", va="center",
                        color="white", fontsize=8)
    plt.colorbar(im, ax=ax, label="ACC (%)")
    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Output] Saved {row_name} x {col_name} joint heatmap -> {path}")


def run_hyperparameter_sensitivity_study(base_cfg):
    """Orchestrator mirroring run_augmentation_category_study: for every
    enabled sub-key, loop values x seeds, aggregate with
    aggregate_variant_results, collect into
    {param_name: {value: aggregated_result}}. Skips a sub-analysis
    gracefully (clear printed message) when it doesn't structurally apply
    to the current dataset."""
    study_cfg = base_cfg["reports"]["hyperparameter_sensitivity"]
    dataset_name = base_cfg["dataset_name"]
    seeds = study_cfg.get("seeds") or base_cfg["seeds"]

    param_order = ["edge_drop_rate", "feature_mask_rate", "knn_k", "subgraph_walk_length",
                   "temperature", "alpha_intra", "beta_cross", "gamma_hybrid",
                   "hidden_dim", "gcn_layers"]
    enabled_params = [p for p in param_order if study_cfg.get(p, {}).get("enabled", False)]
    joint_enabled = study_cfg.get("beta_gamma_joint_heatmap", {}).get("enabled", False)

    if not enabled_params and not joint_enabled:
        return None

    print("\n" + "#" * 78)
    print(f"# HYPERPARAMETER SENSITIVITY STUDY (Table 14 style), dataset={dataset_name}, "
          f"seeds={seeds}")
    print(f"# selected params: {enabled_params}  (joint heatmap: {joint_enabled})")
    print("#" * 78 + "\n")

    output_root = os.path.join(base_cfg["output_dir"], dataset_name, "hyperparameter_sensitivity")
    curves_enabled = (base_cfg["output_artifacts"]["enabled"]
                       and base_cfg["output_artifacts"].get("include", {}).get(
                           "hyperparameter_sensitivity_curves", True))
    heatmap_enabled = (base_cfg["output_artifacts"]["enabled"]
                        and base_cfg["output_artifacts"].get("include", {}).get(
                            "loss_weight_heatmap", True))

    all_results = {}
    cached_beta_grid = {}  # (beta, gamma) -> aggregated result, reused for the joint heatmap

    for param_name in enabled_params:
        # ---- Graceful skip when the sweep doesn't structurally apply -------
        if param_name == "subgraph_walk_length" and not _dataset_has_view_type(dataset_name, "perceptual"):
            print(f"[HyperparamSweep] Skipping 'subgraph_walk_length' - dataset "
                  f"'{dataset_name}' has no 'perceptual' view to drive the "
                  f"subgraph_extract single-op override through.")
            continue
        if param_name == "knn_k" and not _dataset_supports_knn_k_sweep(dataset_name):
            print(f"[HyperparamSweep] Skipping 'knn_k' - dataset '{dataset_name}' has no "
                  f"kNN-built view to sweep.")
            continue

        sub = study_cfg[param_name]
        values = sub["values"]
        print(f"\n{'=' * 78}\n TABLE 14 SWEEP: '{param_name}'  ({len(values)} value(s) x "
              f"{len(seeds)} seed(s))\n{'=' * 78}\n")

        value_results = {}
        for value in values:
            per_seed = {seed: run_hyperparameter_variant(base_cfg, param_name, value, seed)
                        for seed in seeds}
            agg = aggregate_variant_results(per_seed)
            value_results[value] = agg
            if param_name == "beta_cross":
                for g in study_cfg.get("gamma_hybrid", {}).get("values", []):
                    if abs(g - base_cfg["gamma_hybrid"]) < 1e-12:
                        cached_beta_grid[(value, g)] = agg
            if param_name == "gamma_hybrid":
                for b in study_cfg.get("beta_cross", {}).get("values", []):
                    if abs(b - base_cfg["beta_cross"]) < 1e-12:
                        cached_beta_grid[(b, value)] = agg

        all_results[param_name] = value_results
        if study_cfg.get("output_tables", True):
            print_and_save_hyperparameter_table(param_name, value_results, output_root)
            if curves_enabled:
                plot_hyperparameter_sensitivity_curve(param_name, value_results, output_root)
        else:
            print(f"[HyperparamSweep] output_tables=False -> computed results for "
                  f"'{param_name}' but skipped printing/saving its table.")

    # ---- Optional bonus: beta_cross x gamma_hybrid joint heatmap -----------
    if joint_enabled:
        beta_values = study_cfg.get("beta_cross", {}).get("values", [])
        gamma_values = study_cfg.get("gamma_hybrid", {}).get("values", [])
        if not beta_values or not gamma_values:
            print("[HyperparamSweep] beta_gamma_joint_heatmap requested but 'beta_cross' "
                  "and/or 'gamma_hybrid' have no declared 'values' - skipping.")
        else:
            print(f"\n{'=' * 78}\n TABLE 14 JOINT HEATMAP: beta_cross x gamma_hybrid "
                  f"({len(beta_values)}x{len(gamma_values)} grid x {len(seeds)} seed(s))"
                  f"\n{'=' * 78}\n")
            acc_grid = np.full((len(beta_values), len(gamma_values)), np.nan)
            for bi, b in enumerate(beta_values):
                for gi, g in enumerate(gamma_values):
                    if (b, g) in cached_beta_grid:
                        agg = cached_beta_grid[(b, g)]
                    else:
                        cfg_overrides_cache_key = (b, g)
                        per_seed = {}
                        for seed in seeds:
                            cfg = copy.deepcopy(base_cfg)
                            cfg["seed"] = seed
                            cfg["beta_cross"] = b
                            cfg["gamma_hybrid"] = g
                            cfg["reports"] = copy.deepcopy(cfg["reports"])
                            cfg["reports"]["ablation_study"] = {"enabled": False}
                            cfg["output_dir"] = os.path.join(
                                base_cfg["output_dir"], base_cfg["dataset_name"],
                                "hyperparameter_sensitivity", "beta_gamma_joint",
                                f"beta_{b}_gamma_{g}", f"seed_{seed}")
                            set_seed(seed)
                            x, gt_labels_readonly, views, _ = load_dataset(cfg)
                            encoder, projector, _, _, _ = train(views, x, gt_labels_readonly, cfg)
                            device = torch.device(cfg["device"])
                            encoder.eval()
                            with torch.no_grad():
                                h_list = [encoder(x.to(device), v["edge_index"].to(device)) for v in views]
                            fused, _ = fuse_embeddings(h_list, cfg, device)
                            per_seed[seed] = evaluate_clustering(
                                fused, gt_labels_readonly.cpu().numpy(), NUM_CLUSTERS_KNOWN, cfg,
                                tag=f"  [joint_heatmap beta={b} gamma={g} seed={seed}] ")
                            _progress_tick(f"joint_heatmap beta={b} gamma={g} seed={seed}")
                        agg = aggregate_variant_results(per_seed)
                        cached_beta_grid[(b, g)] = agg
                    acc_grid[bi, gi] = agg["acc_mean"]

            if heatmap_enabled:
                plot_loss_weight_joint_heatmap(beta_values, gamma_values, acc_grid, output_root,
                                                row_name="beta_cross", col_name="gamma_hybrid",
                                                filename="table14_loss_weight_joint_heatmap.png",
                                                title_prefix="Table 14")
            all_results["beta_gamma_joint_heatmap"] = {
                "beta_values": beta_values, "gamma_values": gamma_values,
                "acc_grid": acc_grid.tolist(),
            }

    # ---- 4.4.2 bonus: alpha_intra x beta_cross joint heatmap (same pattern) --
    alpha_beta_enabled = study_cfg.get("alpha_beta_joint_heatmap", {}).get("enabled", False)
    if alpha_beta_enabled:
        alpha_values = study_cfg.get("alpha_intra", {}).get("values", [])
        beta_values_ab = study_cfg.get("beta_cross", {}).get("values", [])
        if not alpha_values or not beta_values_ab:
            print("[HyperparamSweep] alpha_beta_joint_heatmap requested but 'alpha_intra' "
                  "and/or 'beta_cross' have no declared 'values' - skipping.")
        else:
            print(f"\n{'=' * 78}\n TABLE 14 JOINT HEATMAP: alpha_intra x beta_cross "
                  f"({len(alpha_values)}x{len(beta_values_ab)} grid x {len(seeds)} seed(s))"
                  f"\n{'=' * 78}\n")
            acc_grid_ab = np.full((len(alpha_values), len(beta_values_ab)), np.nan)
            for ai, a in enumerate(alpha_values):
                for bi, b in enumerate(beta_values_ab):
                    per_seed = {}
                    for seed in seeds:
                        cfg = copy.deepcopy(base_cfg)
                        cfg["seed"] = seed
                        cfg["alpha_intra"] = a
                        cfg["beta_cross"] = b
                        cfg["reports"] = copy.deepcopy(cfg["reports"])
                        cfg["reports"]["ablation_study"] = {"enabled": False}
                        cfg["output_dir"] = os.path.join(
                            base_cfg["output_dir"], base_cfg["dataset_name"],
                            "hyperparameter_sensitivity", "alpha_beta_joint",
                            f"alpha_{a}_beta_{b}", f"seed_{seed}")
                        set_seed(seed)
                        x, gt_labels_readonly, views, _ = load_dataset(cfg)
                        encoder, projector, _, _, _ = train(views, x, gt_labels_readonly, cfg)
                        device = torch.device(cfg["device"])
                        encoder.eval()
                        with torch.no_grad():
                            h_list = [encoder(x.to(device), v["edge_index"].to(device)) for v in views]
                        fused, _ = fuse_embeddings(h_list, cfg, device)
                        per_seed[seed] = evaluate_clustering(
                            fused, gt_labels_readonly.cpu().numpy(), NUM_CLUSTERS_KNOWN, cfg,
                            tag=f"  [alpha_beta_heatmap alpha={a} beta={b} seed={seed}] ")
                        _progress_tick(f"alpha_beta_heatmap alpha={a} beta={b} seed={seed}")
                    agg_ab = aggregate_variant_results(per_seed)
                    acc_grid_ab[ai, bi] = agg_ab["acc_mean"]

            if heatmap_enabled:
                plot_loss_weight_joint_heatmap(alpha_values, beta_values_ab, acc_grid_ab, output_root,
                                                row_name="alpha_intra", col_name="beta_cross",
                                                filename="table14_alpha_beta_joint_heatmap.png",
                                                title_prefix="Table 14 (4.4.2)")
            all_results["alpha_beta_joint_heatmap"] = {
                "alpha_values": alpha_values, "beta_values": beta_values_ab,
                "acc_grid": acc_grid_ab.tolist(),
            }

    return all_results if all_results else None


# ==============================================================================
# TASK B: SCALABILITY ANALYSIS (Table 15 style)
# ==============================================================================
# Measures wall-clock per-epoch time and peak GPU memory (CUDA only) as
# batch size, number of views, or number of nodes changes. These sweeps use
# a SHORT fixed-length run (scalability_analysis["measure_epochs"], early
# stopping disabled) purely for measurement - NOT meant to produce
# converged clustering-quality numbers. See CONFIG["reports"]["scalability_analysis"].
# ==============================================================================

def _aggregate_scalability_results(per_seed_results, include_quality=True):
    """Aggregates the per-seed scalability dicts (each already averaged over
    its own epochs) across seeds: mean/std of epoch_time_mean, mean of
    peak_memory_gb (skipping non-numeric CPU placeholders), and, when
    include_quality=True, mean/std of acc/nmi/ari/f1 as an informational
    (non-converged) column."""
    seeds = sorted(per_seed_results.keys())
    epoch_time_means = [per_seed_results[s]["epoch_time_mean"] for s in seeds]
    mean_et, std_et = _mean_std(epoch_time_means)

    mem_values = [per_seed_results[s]["peak_memory_gb"] for s in seeds]
    numeric_mem = [m for m in mem_values if isinstance(m, (int, float))]
    if numeric_mem:
        mean_mem, std_mem = _mean_std(numeric_mem)
    else:
        mean_mem, std_mem = "N/A (CPU run)", "N/A (CPU run)"

    agg = {
        "seeds": seeds,
        "epoch_time_mean": mean_et,
        "epoch_time_std": std_et,
        "peak_memory_gb_mean": mean_mem,
        "peak_memory_gb_std": std_mem,
        "per_seed": per_seed_results,
    }
    if include_quality and all("acc" in per_seed_results[s] for s in seeds):
        for m in ["acc", "nmi", "ari", "f1"]:
            mean_m, std_m = _mean_std([per_seed_results[s][m] for s in seeds])
            agg[f"{m}_mean"] = mean_m
            agg[f"{m}_std"] = std_m
    return agg


def run_scalability_variant_batch_size(base_cfg, batch_size, seed):
    """Short-run measurement of per-epoch time/memory as cl_batch_size
    varies. Clustering metrics are reported as an informational column
    only (from a short, non-converged run)."""
    study_cfg = base_cfg["reports"]["scalability_analysis"]
    cfg = copy.deepcopy(base_cfg)
    cfg["seed"] = seed
    cfg["cl_batch_size"] = batch_size
    cfg["epochs"] = study_cfg.get("measure_epochs", 10)
    cfg["early_stopping_enabled"] = False  # fixed epoch count for measurement runs
    cfg["reports"] = copy.deepcopy(cfg["reports"])
    cfg["reports"]["ablation_study"] = {"enabled": False}
    cfg["output_dir"] = os.path.join(base_cfg["output_dir"], base_cfg["dataset_name"],
                                      "scalability_analysis", "vs_batch_size", str(batch_size),
                                      f"seed_{seed}")

    set_seed(seed)
    x, gt_labels_readonly, views, _ = load_dataset(cfg)
    encoder, projector, _, _, _ = train(views, x, gt_labels_readonly, cfg)
    measured = cfg["_measured"]
    epoch_times = measured["epoch_times"]
    mean_et, std_et = _mean_std(epoch_times) if epoch_times else (float("nan"), float("nan"))

    device = torch.device(cfg["device"])
    encoder.eval()
    with torch.no_grad():
        h_list = [encoder(x.to(device), v["edge_index"].to(device)) for v in views]
    fused, _ = fuse_embeddings(h_list, cfg, device)
    quality = evaluate_clustering(
        fused, gt_labels_readonly.cpu().numpy(), NUM_CLUSTERS_KNOWN, cfg,
        tag=f"  [scalability vs_batch_size={batch_size} seed={seed} (SHORT/non-converged)] ")

    _progress_tick(f"scalability vs_batch_size={batch_size} seed={seed}")
    return {"epoch_time_mean": mean_et, "epoch_time_std": std_et,
            "peak_memory_gb": measured["peak_memory_gb"],
            "acc": quality["acc"], "nmi": quality["nmi"], "ari": quality["ari"], "f1": quality["f1"]}


def run_scalability_variant_num_views(base_cfg, k, seed):
    """Short-run measurement as the number of views K = 1..len(real views) of
    the CURRENT dataset varies (prefix-truncated - no synthetic/duplicated
    views are invented). Clustering metrics reported as informational only."""
    study_cfg = base_cfg["reports"]["scalability_analysis"]
    cfg = copy.deepcopy(base_cfg)
    cfg["seed"] = seed
    cfg["num_views"] = k
    cfg["epochs"] = study_cfg.get("measure_epochs", 10)
    cfg["early_stopping_enabled"] = False
    cfg["reports"] = copy.deepcopy(cfg["reports"])
    cfg["reports"]["ablation_study"] = {"enabled": False}
    cfg["output_dir"] = os.path.join(base_cfg["output_dir"], base_cfg["dataset_name"],
                                      "scalability_analysis", "vs_num_views", str(k), f"seed_{seed}")

    set_seed(seed)
    x, gt_labels_readonly, views, _ = load_dataset(cfg)  # load_dataset slices spec["views"][:cfg["num_views"]]
    encoder, projector, _, _, _ = train(views, x, gt_labels_readonly, cfg)
    measured = cfg["_measured"]
    epoch_times = measured["epoch_times"]
    mean_et, std_et = _mean_std(epoch_times) if epoch_times else (float("nan"), float("nan"))

    device = torch.device(cfg["device"])
    encoder.eval()
    with torch.no_grad():
        h_list = [encoder(x.to(device), v["edge_index"].to(device)) for v in views]
    fused, _ = fuse_embeddings(h_list, cfg, device)
    quality = evaluate_clustering(
        fused, gt_labels_readonly.cpu().numpy(), NUM_CLUSTERS_KNOWN, cfg,
        tag=f"  [scalability vs_num_views=K{k} seed={seed} (SHORT/non-converged)] ")

    _progress_tick(f"scalability vs_num_views=K{k} seed={seed}")
    return {"epoch_time_mean": mean_et, "epoch_time_std": std_et,
            "peak_memory_gb": measured["peak_memory_gb"],
            "acc": quality["acc"], "nmi": quality["nmi"], "ari": quality["ari"], "f1": quality["f1"]}


def run_scalability_variant_num_nodes(base_cfg, target_n, seed):
    """Short-run measurement as the graph is subsampled down to `target_n`
    nodes (or kept full when target_n=='full') via subsample_multiview_graph.
    Clustering-quality metrics are OMITTED entirely here - uniform random
    node subsampling shifts class balance arbitrarily, so ACC/NMI/ARI/F1 on
    the reduced graph would not be meaningful. Only time/memory are
    reported."""
    study_cfg = base_cfg["reports"]["scalability_analysis"]
    cfg = copy.deepcopy(base_cfg)
    cfg["seed"] = seed
    cfg["epochs"] = study_cfg.get("measure_epochs", 10)
    cfg["early_stopping_enabled"] = False
    cfg["reports"] = copy.deepcopy(cfg["reports"])
    cfg["reports"]["ablation_study"] = {"enabled": False}
    cfg["output_dir"] = os.path.join(base_cfg["output_dir"], base_cfg["dataset_name"],
                                      "scalability_analysis", "vs_num_nodes", str(target_n),
                                      f"seed_{seed}")

    set_seed(seed)
    x, gt_labels_readonly, views, _ = load_dataset(cfg)
    if target_n != "full":
        x, views, gt_labels_readonly = subsample_multiview_graph(
            x, views, gt_labels_readonly, int(target_n), seed)

    encoder, projector, _, _, _ = train(views, x, gt_labels_readonly, cfg)
    measured = cfg["_measured"]
    epoch_times = measured["epoch_times"]
    mean_et, std_et = _mean_std(epoch_times) if epoch_times else (float("nan"), float("nan"))

    print(f"  [scalability vs_num_nodes={target_n} seed={seed}] epoch_time_mean="
          f"{mean_et:.4f}s  peak_memory_gb={measured['peak_memory_gb']}  "
          f"(clustering-quality metrics OMITTED - subsampled graph, class balance not meaningful)")

    _progress_tick(f"scalability vs_num_nodes={target_n} seed={seed}")
    return {"epoch_time_mean": mean_et, "epoch_time_std": std_et,
            "peak_memory_gb": measured["peak_memory_gb"]}


def print_and_save_scalability_table(dimension_name, value_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    has_quality = any("acc_mean" in r for r in value_results.values())

    print("\n" + "=" * 100)
    print(f" TABLE 15 STYLE - Scalability Analysis: '{dimension_name}'  "
          f"(Value | Epoch Time (s) | Peak Mem (GB)" + (" | ACC (informational)" if has_quality else "") + ")")
    print("=" * 100)
    header = f"{'Value':<10s} | {'Epoch Time (s)':>18s} | {'Peak Mem (GB)':>16s}"
    if has_quality:
        header += f" | {'ACC (info.)':>12s}"
    print(header)
    print("-" * len(header))

    rows = []
    for value in value_results.keys():
        r = value_results[value]
        mem_str = (f"{r['peak_memory_gb_mean']:.3f}" if isinstance(r["peak_memory_gb_mean"], (int, float))
                   else str(r["peak_memory_gb_mean"]))
        line = f"{str(value):<10s} | {r['epoch_time_mean']:15.4f}+/-{r['epoch_time_std']:.4f} | {mem_str:>16s}"
        if has_quality and "acc_mean" in r:
            line += f" | {r['acc_mean']*100:11.2f}%"
        print(line)
        rows.append([value, r["epoch_time_mean"], r["epoch_time_std"],
                     r["peak_memory_gb_mean"], r["peak_memory_gb_std"],
                     r.get("acc_mean", ""), r.get("acc_std", "")])
    print("=" * 100 + "\n")

    csv_path = os.path.join(output_dir, f"table15_scalability_{dimension_name}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["value", "epoch_time_mean_s", "epoch_time_std_s",
                          "peak_memory_gb_mean", "peak_memory_gb_std",
                          "acc_mean_informational", "acc_std_informational"])
        writer.writerows(rows)
    print(f"[Output] Saved Table 15 CSV ('{dimension_name}') -> {csv_path}")


def plot_scalability_curve(dimension_name, value_results, output_dir):
    """2-panel figure (epoch time | peak memory) vs the swept dimension."""
    os.makedirs(output_dir, exist_ok=True)
    values = list(value_results.keys())
    x_labels = [str(v) for v in values]
    x_pos = np.arange(len(values))

    epoch_times = [value_results[v]["epoch_time_mean"] for v in values]
    epoch_time_stds = [value_results[v]["epoch_time_std"] for v in values]
    mem_values = [value_results[v]["peak_memory_gb_mean"] for v in values]
    mem_numeric = [m if isinstance(m, (int, float)) else np.nan for m in mem_values]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].errorbar(x_pos, epoch_times, yerr=epoch_time_stds, marker="o", capsize=3, color="steelblue")
    axes[0].set_xticks(x_pos); axes[0].set_xticklabels(x_labels)
    axes[0].set_xlabel(dimension_name); axes[0].set_ylabel("Epoch time (s)")
    axes[0].set_title("Per-Epoch Wall-Clock Time")
    axes[0].grid(alpha=0.3)

    if any(not np.isnan(m) for m in mem_numeric):
        axes[1].plot(x_pos, mem_numeric, marker="s", color="darkorange")
        axes[1].set_ylabel("Peak memory (GB)")
    else:
        axes[1].text(0.5, 0.5, "N/A (CPU run)", ha="center", va="center", transform=axes[1].transAxes)
    axes[1].set_xticks(x_pos); axes[1].set_xticklabels(x_labels)
    axes[1].set_xlabel(dimension_name)
    axes[1].set_title("Peak GPU Memory")
    axes[1].grid(alpha=0.3)

    plt.suptitle(f"Table 15 Scalability Analysis: '{dimension_name}'")
    plt.tight_layout()
    path = os.path.join(output_dir, f"table15_scalability_{dimension_name}_curve.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Output] Saved scalability curve ('{dimension_name}') -> {path}")


def run_scalability_study(base_cfg):
    """Orchestrator: runs whichever of the three scalability dimensions are
    enabled, across seeds, aggregating mean/std of per-epoch time (and, for
    vs_batch_size / vs_num_views, informational clustering-quality means)."""
    study_cfg = base_cfg["reports"]["scalability_analysis"]
    dataset_name = base_cfg["dataset_name"]
    seeds = study_cfg.get("seeds") or base_cfg["seeds"]

    vs_batch_enabled = study_cfg.get("vs_batch_size", {}).get("enabled", False)
    vs_views_enabled = study_cfg.get("vs_num_views", {}).get("enabled", False)
    vs_nodes_enabled = study_cfg.get("vs_num_nodes", {}).get("enabled", False)
    if not (vs_batch_enabled or vs_views_enabled or vs_nodes_enabled):
        return None

    print("\n" + "#" * 78)
    print(f"# SCALABILITY ANALYSIS (Table 15 style), dataset={dataset_name}, seeds={seeds}, "
          f"measure_epochs={study_cfg.get('measure_epochs', 10)}")
    print("#" * 78 + "\n")

    output_root = os.path.join(base_cfg["output_dir"], dataset_name, "scalability_analysis")
    curves_enabled = (base_cfg["output_artifacts"]["enabled"]
                       and base_cfg["output_artifacts"].get("include", {}).get("scalability_curves", True))

    all_results = {}

    if vs_batch_enabled:
        values = study_cfg["vs_batch_size"]["values"]
        print(f"\n{'=' * 78}\n TABLE 15 SWEEP: vs_batch_size  ({len(values)} value(s) x "
              f"{len(seeds)} seed(s))\n{'=' * 78}\n")
        value_results = {}
        for v in values:
            per_seed = {seed: run_scalability_variant_batch_size(base_cfg, v, seed) for seed in seeds}
            value_results[v] = _aggregate_scalability_results(per_seed)
        all_results["vs_batch_size"] = value_results
        if study_cfg.get("output_tables", True):
            print_and_save_scalability_table("vs_batch_size", value_results, output_root)
            if curves_enabled:
                plot_scalability_curve("vs_batch_size", value_results, output_root)

    if vs_views_enabled:
        max_k = len(DATASET_REGISTRY[dataset_name]["views"])
        values = list(range(1, max_k + 1))
        print(f"\n{'=' * 78}\n TABLE 15 SWEEP: vs_num_views  (K=1..{max_k}, dataset '{dataset_name}' "
              f"has {max_k} real view(s) - no synthetic views invented) x {len(seeds)} seed(s)"
              f"\n{'=' * 78}\n")
        value_results = {}
        for k in values:
            per_seed = {seed: run_scalability_variant_num_views(base_cfg, k, seed) for seed in seeds}
            value_results[k] = _aggregate_scalability_results(per_seed)
        all_results["vs_num_views"] = value_results
        if study_cfg.get("output_tables", True):
            print_and_save_scalability_table("vs_num_views", value_results, output_root)
            if curves_enabled:
                plot_scalability_curve("vs_num_views", value_results, output_root)

    if vs_nodes_enabled:
        values = study_cfg["vs_num_nodes"]["values"]
        print(f"\n{'=' * 78}\n TABLE 15 SWEEP: vs_num_nodes  ({len(values)} value(s) x "
              f"{len(seeds)} seed(s)) - clustering-quality metrics OMITTED (subsampled graph)"
              f"\n{'=' * 78}\n")
        value_results = {}
        for v in values:
            per_seed = {seed: run_scalability_variant_num_nodes(base_cfg, v, seed) for seed in seeds}
            value_results[v] = _aggregate_scalability_results(per_seed, include_quality=False)
        all_results["vs_num_nodes"] = value_results
        if study_cfg.get("output_tables", True):
            print_and_save_scalability_table("vs_num_nodes", value_results, output_root)
            if curves_enabled:
                plot_scalability_curve("vs_num_nodes", value_results, output_root)

    return all_results if all_results else None


# ==============================================================================
# 4.6 COMBINED FIGURE: vs_batch_size (time), vs_batch_size (memory),
# vs_num_views (time, dual-axis with memory) and vs_num_nodes (log-log) all
# in one 2x2 Figure, when all three scalability dimensions were run.
# ==============================================================================

def plot_scalability_combined(sc_tables, output_dir):
    if not sc_tables:
        return
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    if "vs_batch_size" in sc_tables:
        vr = sc_tables["vs_batch_size"]
        values = list(vr.keys())
        xs = np.arange(len(values))
        times = [vr[v]["epoch_time_mean"] for v in values]
        axes[0, 0].bar(xs, times, color="steelblue")
        axes[0, 0].set_xticks(xs); axes[0, 0].set_xticklabels([str(v) for v in values])
        axes[0, 0].set_title("Training Time vs. Batch Size")
        axes[0, 0].set_xlabel("batch size"); axes[0, 0].set_ylabel("epoch time (s)")
        axes[0, 0].grid(alpha=0.3, axis="y")

        mem = [vr[v]["peak_memory_gb_mean"] for v in values]
        mem_numeric = [m if isinstance(m, (int, float)) else np.nan for m in mem]
        axes[0, 1].bar(xs, mem_numeric, color="darkorange")
        axes[0, 1].set_xticks(xs); axes[0, 1].set_xticklabels([str(v) for v in values])
        axes[0, 1].set_title("Peak GPU Memory vs. Batch Size")
        axes[0, 1].set_xlabel("batch size"); axes[0, 1].set_ylabel("peak memory (GB)")
        axes[0, 1].grid(alpha=0.3, axis="y")
    else:
        axes[0, 0].axis("off"); axes[0, 1].axis("off")

    if "vs_num_views" in sc_tables:
        vr = sc_tables["vs_num_views"]
        values = list(vr.keys())
        xs = np.arange(len(values))
        times = [vr[v]["epoch_time_mean"] for v in values]
        mem = [vr[v]["peak_memory_gb_mean"] for v in values]
        mem_numeric = [m if isinstance(m, (int, float)) else np.nan for m in mem]
        ax2 = axes[1, 0]
        ax2b = ax2.twinx()
        ax2.plot(xs, times, marker="o", color="steelblue", label="epoch time (s)")
        ax2b.plot(xs, mem_numeric, marker="s", color="darkorange", label="peak memory (GB)")
        ax2.set_xticks(xs); ax2.set_xticklabels([str(v) for v in values])
        ax2.set_xlabel("number of views (K)")
        ax2.set_ylabel("epoch time (s)", color="steelblue")
        ax2b.set_ylabel("peak memory (GB)", color="darkorange")
        ax2.set_title("Scaling with Number of Views")
        ax2.grid(alpha=0.3)
    else:
        axes[1, 0].axis("off")

    if "vs_num_nodes" in sc_tables:
        vr = sc_tables["vs_num_nodes"]
        values = [v for v in vr.keys() if v != "full"]
        if values:
            xs_numeric = [float(v) for v in values]
            times = [vr[v]["epoch_time_mean"] for v in values]
            axes[1, 1].plot(xs_numeric, times, marker="o", color="seagreen")
            axes[1, 1].set_xscale("log"); axes[1, 1].set_yscale("log")
            axes[1, 1].set_xlabel("number of nodes (log)")
            axes[1, 1].set_ylabel("epoch time, s (log)")
            axes[1, 1].set_title("Scaling with Number of Nodes (log-log)")
            axes[1, 1].grid(alpha=0.3, which="both")
        else:
            axes[1, 1].axis("off")
    else:
        axes[1, 1].axis("off")

    plt.tight_layout()
    path = os.path.join(output_dir, "table15_scalability_combined_figure.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Output] Saved combined 2x2 scalability figure -> {path}")


# ==============================================================================
# FINAL CONSOLIDATED REPORT
# ==============================================================================
# Re-prints, together in ONE place, every table that was enabled in
# CONFIG["reports"] for this run. Reuses the already-computed data collected
# during main() (no retraining) - just a formatting/consolidation pass.
# ==============================================================================

def print_final_report(cfg, tables):
    sel = cfg["reports"]
    output_dir = os.path.join(cfg["output_dir"], cfg["dataset_name"])

    # FIX: every one of these is now a nested dict ({"enabled": bool, ...}),
    # not a plain bool, so we must read the "enabled" sub-key explicitly -
    # `if sel.get("ablation_study")` would always be truthy (a non-empty
    # dict is always truthy) regardless of its actual enabled value.
    main_enabled = sel.get("main_comparison", {}).get("enabled", False)
    ablation_enabled = sel.get("ablation_study", {}).get("enabled", False)
    aug_cat_cfg = sel.get("augmentation_category", {})
    rule_vs_learnable_enabled = sel.get("rule_vs_learnable", {}).get("enabled", False)
    hp_cfg = sel.get("hyperparameter_sensitivity", {})
    hp_param_order = ["edge_drop_rate", "feature_mask_rate", "knn_k", "subgraph_walk_length",
                       "temperature", "alpha_intra", "beta_cross", "gamma_hybrid",
                       "hidden_dim", "gcn_layers"]
    hp_enabled_params = [p for p in hp_param_order if hp_cfg.get(p, {}).get("enabled", False)]
    hp_joint_enabled = hp_cfg.get("beta_gamma_joint_heatmap", {}).get("enabled", False)
    sc_cfg = sel.get("scalability_analysis", {})
    sc_enabled_dims = [d for d in ("vs_batch_size", "vs_num_views", "vs_num_nodes")
                       if sc_cfg.get(d, {}).get("enabled", False)]

    print("\n" + "#" * 100)
    print("# FINAL REPORT - ALL TABLES ENABLED FOR THIS RUN")
    print(f"# dataset = {cfg['dataset_name']}")
    enabled_list = []
    if main_enabled:
        enabled_list.append("main_comparison")
    if ablation_enabled:
        enabled_list.append("ablation_study")
    for vtype in ("sparse", "dense", "perceptual", "structured"):
        if aug_cat_cfg.get(vtype, {}).get("enabled", False):
            enabled_list.append(f"augmentation_category.{vtype}")
    if rule_vs_learnable_enabled:
        enabled_list.append("rule_vs_learnable")
    for p in hp_enabled_params:
        enabled_list.append(f"hyperparameter_sensitivity.{p}")
    if hp_joint_enabled:
        enabled_list.append("hyperparameter_sensitivity.beta_gamma_joint_heatmap")
    for d in sc_enabled_dims:
        enabled_list.append(f"scalability_analysis.{d}")
    print(f"# enabled tables = {enabled_list if enabled_list else 'none'}")
    print("#" * 100)

    if main_enabled and "main_comparison" in tables:
        print_and_save_table4(cfg, tables["main_comparison"], output_dir)

    if ablation_enabled:
        if "ablation_study" in tables:
            print_and_save_ablation_table(
                tables["ablation_study"]["variant_results"], tables["ablation_study"]["variants"],
                output_dir, tables["ablation_study"]["multi_seed_ablation"])
        else:
            print("\n[FinalReport] ablation_study was requested but no results were computed "
                  "(check the logs above for errors during the ablation study).")

    for vtype in ("sparse", "dense", "perceptual", "structured"):
        if aug_cat_cfg.get(vtype, {}).get("enabled", False):
            cat_tables = tables.get("category_study", {})
            if vtype in cat_tables:
                print_and_save_category_table(
                    vtype, cat_tables[vtype],
                    os.path.join(output_dir, "category_study"))
            else:
                print(f"\n[FinalReport] augmentation_category.{vtype} was requested but was skipped "
                      f"(dataset '{cfg['dataset_name']}' has no '{vtype}' view, or "
                      f"the study did not produce results - check the logs above).")

    if rule_vs_learnable_enabled:
        if "rule_vs_learnable" in tables:
            print_and_save_rule_vs_learnable_table(
                cfg["dataset_name"], tables["rule_vs_learnable"], output_dir)
        else:
            print("\n[FinalReport] rule_vs_learnable was requested but no results were computed "
                  "(check the logs above for errors during the rule-vs-learnable study).")

    hp_tables = tables.get("hyperparameter_sensitivity", {})
    hp_output_root = os.path.join(output_dir, "hyperparameter_sensitivity")
    for p in hp_enabled_params:
        if p in hp_tables:
            print_and_save_hyperparameter_table(p, hp_tables[p], hp_output_root)
        else:
            print(f"\n[FinalReport] hyperparameter_sensitivity.{p} was requested but was skipped "
                  f"(likely does not structurally apply to dataset '{cfg['dataset_name']}' - "
                  f"check the logs above).")
    if hp_joint_enabled:
        if "beta_gamma_joint_heatmap" not in hp_tables:
            print("\n[FinalReport] hyperparameter_sensitivity.beta_gamma_joint_heatmap was "
                  "requested but no results were computed (check the logs above).")

    sc_tables = tables.get("scalability_analysis", {})
    sc_output_root = os.path.join(output_dir, "scalability_analysis")
    for d in sc_enabled_dims:
        if d in sc_tables:
            print_and_save_scalability_table(d, sc_tables[d], sc_output_root)
        else:
            print(f"\n[FinalReport] scalability_analysis.{d} was requested but no results were "
                  f"computed (check the logs above for errors during the scalability study).")

    print("\n" + "#" * 100)
    print("# END OF FINAL REPORT")
    print("#" * 100 + "\n")


# ==============================================================================
# MAIN PIPELINE  (edit CONFIG["dataset_name"] / CONFIG["reports"], then rerun)
# ==============================================================================

def main():
    # ---- Give this run its own timestamped output folder ----------------------
    # Every downstream function (run_pipeline_single_seed, run_ablation_study,
    # run_category_study_variant, run_rule_vs_learnable_study, ...) builds its
    # own sub-path from CONFIG["output_dir"], so updating this ONE value here,
    # before anything else runs, automatically namespaces every artifact of
    # this run under one dated folder instead of overwriting the last run's
    # outputs on every rerun.
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    CONFIG["output_dir"] = os.path.join(CONFIG["output_dir"], f"{run_timestamp}_{CONFIG['dataset_name']}_{CONFIG['augmentation_mode']}")
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    print(f"[Run] All outputs for this run will be saved under: {CONFIG['output_dir']}")

    # ---- Compute the TOTAL number of individual training runs this ------------
    # invocation will do, across every currently-enabled report/study, so the
    # overall progress/ETA line (see _progress_tick, printed after every
    # individual run finishes) has an accurate denominator from the start.
    ablation_enabled = CONFIG["reports"].get("ablation_study", {}).get("enabled", False)
    aug_cat_cfg = CONFIG["reports"].get("augmentation_category", {})
    aug_cat_enabled = any(
        aug_cat_cfg.get(vt, {}).get("enabled", False)
        for vt in ("sparse", "dense", "perceptual", "structured")
    )
    rule_vs_learnable_enabled = CONFIG["reports"].get("rule_vs_learnable", {}).get("enabled", False)

    hp_cfg = CONFIG["reports"].get("hyperparameter_sensitivity", {})
    hp_param_order = ["edge_drop_rate", "feature_mask_rate", "knn_k", "subgraph_walk_length",
                       "temperature", "alpha_intra", "beta_cross", "gamma_hybrid",
                       "hidden_dim", "gcn_layers"]
    hp_enabled_params = [p for p in hp_param_order if hp_cfg.get(p, {}).get("enabled", False)]
    hp_joint_enabled = hp_cfg.get("beta_gamma_joint_heatmap", {}).get("enabled", False)
    hp_seeds = hp_cfg.get("seeds") or CONFIG["seeds"]

    sc_cfg = CONFIG["reports"].get("scalability_analysis", {})
    sc_seeds = sc_cfg.get("seeds") or CONFIG["seeds"]
    sc_vs_batch_enabled = sc_cfg.get("vs_batch_size", {}).get("enabled", False)
    sc_vs_views_enabled = sc_cfg.get("vs_num_views", {}).get("enabled", False)
    sc_vs_nodes_enabled = sc_cfg.get("vs_num_nodes", {}).get("enabled", False)

    # ---- Primary run gate: ONLY needed if something actually reuses -----------
    # primary_results (Table 4 itself, Tables 5/6/7, the 4.2.2 radar chart, or
    # Table 8's "AGAMC-Full" row). augmentation_category / rule_vs_learnable /
    # hyperparameter_sensitivity / scalability_analysis are all self-contained
    # and never touch primary_results, so they must NOT force this run.
    # Defined here (BEFORE total_runs) so both the run-count estimate below
    # and the actual primary-run block further down can use it.
    needs_primary_run = (
        CONFIG["reports"].get("main_comparison", {}).get("enabled", False)
        or CONFIG["reports"].get("table5_nmi", {}).get("enabled", False)
        or CONFIG["reports"].get("table6_ari", {}).get("enabled", False)
        or CONFIG["reports"].get("table7_f1", {}).get("enabled", False)
        or CONFIG["reports"].get("main_comparison_radar", {}).get("enabled", False)
        or ablation_enabled
    )

    total_runs = len(CONFIG["seeds"]) if needs_primary_run else 0  # primary run (only if actually needed)
    if ablation_enabled:
        num_fresh_variants = len(build_ablation_variants()) - 1  # "Full" reuses the primary run
        total_runs += num_fresh_variants * len(CONFIG["seeds"])
    if aug_cat_enabled:
        for vtype in aug_cat_cfg.get("target_view_types", []):
            ops = aug_cat_cfg.get("ops_per_view_type", {}).get(vtype, [])
            seeds_for_cat = aug_cat_cfg.get("seeds", CONFIG["seeds"])
            total_runs += len(ops) * len(seeds_for_cat)
    if rule_vs_learnable_enabled:
        seeds_for_rvl = CONFIG["reports"].get("rule_vs_learnable", {}).get("seeds", CONFIG["seeds"])
        total_runs += 2 * len(seeds_for_rvl)  # rule_based + learnable
    for p in hp_enabled_params:
        # knn_k / subgraph_walk_length may be skipped later if inapplicable to
        # the dataset - this is a best-effort estimate, harmless if slightly high.
        total_runs += len(hp_cfg.get(p, {}).get("values", [])) * len(hp_seeds)
    if hp_joint_enabled:
        beta_values = hp_cfg.get("beta_cross", {}).get("values", [])
        gamma_values = hp_cfg.get("gamma_hybrid", {}).get("values", [])
        total_runs += len(beta_values) * len(gamma_values) * len(hp_seeds)
    if sc_vs_batch_enabled:
        total_runs += len(sc_cfg.get("vs_batch_size", {}).get("values", [])) * len(sc_seeds)
    if sc_vs_views_enabled:
        total_runs += len(DATASET_REGISTRY[CONFIG["dataset_name"]]["views"]) * len(sc_seeds)
    if sc_vs_nodes_enabled:
        total_runs += len(sc_cfg.get("vs_num_nodes", {}).get("values", [])) * len(sc_seeds)
    _progress_init(total_runs)

    # ---- Primary run: ONLY executed if something actually needs it --------
    # (Table 4 itself, Tables 5/6/7, the 4.2.2 radar chart, or Table 8's
    # "AGAMC-Full" row, which reuses primary_results instead of retraining).
    tables = {}
    primary_results = None

    if needs_primary_run:
        _section_start("Main run (primary multi-seed)")
        primary_results = run_seeds(CONFIG)
        _section_end("Main run (primary multi-seed)")

        tables["main_comparison"] = compute_aggregate_stats(primary_results)

        # ---- Tables 5/6/7 (NMI/ARI/F1) + 4.2.2 radar chart: all reuse the same
        # primary_results the Table-4 row already used - no extra training runs.
        _primary_output_dir = os.path.join(CONFIG["output_dir"], CONFIG["dataset_name"])
        for _report_key in ("table5_nmi", "table6_ari", "table7_f1"):
            if CONFIG["reports"].get(_report_key, {}).get("enabled", False):
                print_and_save_metric_table(CONFIG, tables["main_comparison"], _primary_output_dir, _report_key)
        if CONFIG["reports"].get("main_comparison_radar", {}).get("enabled", False):
            plot_main_comparison_radar(CONFIG, tables["main_comparison"], _primary_output_dir)
    else:
        print("[Run] Skipping primary multi-seed run - nothing enabled in CONFIG['reports'] "
              "needs it (main_comparison / table5_nmi / table6_ari / table7_f1 / "
              "main_comparison_radar / ablation_study are all disabled).")

    if ablation_enabled:
        _section_start("Ablation study (Table 8)")
        variant_results = run_ablation_study(CONFIG, primary_results)
        _section_end("Ablation study (Table 8)")
        tables["ablation_study"] = {
            "variant_results": variant_results,
            "variants": build_ablation_variants(),
            "multi_seed_ablation": len(CONFIG["seeds"]) > 1,
        }

    if aug_cat_enabled:
        _section_start("Augmentation-category study (Tables 9-12)")
        category_tables = run_augmentation_category_study(CONFIG)
        _section_end("Augmentation-category study (Tables 9-12)")
        if category_tables:
            tables["category_study"] = category_tables
            if CONFIG["reports"]["augmentation_category"].get("combined_figure", False):
                plot_augmentation_category_combined(
                    category_tables,
                    os.path.join(CONFIG["output_dir"], CONFIG["dataset_name"], "category_study"),
                    CONFIG)

    if rule_vs_learnable_enabled:
        _section_start("Rule-vs-Learnable study (Table 13)")
        mode_results = run_rule_vs_learnable_study(CONFIG)
        _section_end("Rule-vs-Learnable study (Table 13)")
        if mode_results:
            tables["rule_vs_learnable"] = mode_results

    if hp_enabled_params or hp_joint_enabled:
        _section_start("Hyperparameter sensitivity study (Table 14)")
        hp_tables = run_hyperparameter_sensitivity_study(CONFIG)
        _section_end("Hyperparameter sensitivity study (Table 14)")
        if hp_tables:
            tables["hyperparameter_sensitivity"] = hp_tables

    if sc_vs_batch_enabled or sc_vs_views_enabled or sc_vs_nodes_enabled:
        _section_start("Scalability analysis (Table 15)")
        sc_tables = run_scalability_study(CONFIG)
        _section_end("Scalability analysis (Table 15)")
        if sc_tables:
            tables["scalability_analysis"] = sc_tables
            if CONFIG["reports"]["scalability_analysis"].get("combined_figure", False):
                plot_scalability_combined(
                    sc_tables,
                    os.path.join(CONFIG["output_dir"], CONFIG["dataset_name"], "scalability_analysis"))

    # ---- Show every table enabled in CONFIG["reports"] together, ------
    # one more time, in a single consolidated section at the very end.
    print_final_report(CONFIG, tables)
    _print_section_timing_summary()

    return primary_results


if __name__ == "__main__":
    main()