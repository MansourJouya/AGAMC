"""
================================================================================
download_data.py  -  auto-downloader/preprocessor for acm / dblp / amazon / youtube
================================================================================
Run this ONCE, in the same folder as agamc.py:

    py -3.12 download_data.py

It builds data/acm/, data/dblp/, data/amazon/, data/youtube/ in EXACTLY the
file format agamc.py's DATASET_REGISTRY["generic"] loader expects:

    features.npy, labels.npy, graph_<name>.npz, feat_<name>.npy

so that after this runs once, `py -3.12 agamc.py` works for those four
datasets the same way it already works for cora/citeseer (no manual .mat
download, no Baidu Pan links, no Google Drive).

WHERE EACH DATASET ACTUALLY COMES FROM
---------------------------------------------------------------------------
acm     -> torch_geometric.datasets.HGBDataset(name="ACM")   [real, auto-download]
           paper/author/subject heterogeneous graph -> co-author + co-subject
           meta-path adjacency matrices, paper features used as-is.
dblp    -> torch_geometric.datasets.DBLP()                    [real, auto-download]
           author/paper/term/conference heterogeneous graph (this is the
           SAME preprocessed DBLP used by HAN/HeCo-style papers) -> co-author
           meta-path graph + per-author term-usage features for the kNN view.
amazon  -> torch_geometric.datasets.Amazon(name="Computers")  [real, auto-download]
           real co-purchase graph + real product features (PCA'd to 256-d
           for the second view, as the paper describes).
youtube -> SNAP com-youtube FRIENDSHIP GRAPH + SNAP ground-truth community
           labels (both real, auto-downloaded straight from
           snap.stanford.edu - no login/manual step). BUT: per-node
           thumbnail-CNN / comment-TF-IDF content features for this graph
           are NOT a standard public release (agamc.py's own docstring
           already says this). So the two content-feature views are
           structural proxies (graph-spectral embedding, and
           degree/clustering/triangle-count descriptors) - clearly labeled
           below. Swap in real scraped features later if you get them; just
           overwrite feat_thumbnail_cnn.npy / feat_comment_tfidf.npy.

Everything here is standalone - it does NOT modify agamc.py or its
DATASET_REGISTRY/CONFIG structure in any way.
================================================================================
"""

import os
import gzip
import urllib.request

import numpy as np
import scipy.sparse as sp

DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RAW_ROOT = os.path.join(DATA_ROOT, "_raw")


# ==============================================================================
# shared helpers
# ==============================================================================

def _save_dataset(name, features, labels, graphs: dict, feats: dict):
    root = os.path.join(DATA_ROOT, name)
    os.makedirs(root, exist_ok=True)
    np.save(os.path.join(root, "features.npy"), np.asarray(features, dtype=np.float32))
    np.save(os.path.join(root, "labels.npy"), np.asarray(labels, dtype=np.int64))
    for fname, adj in graphs.items():
        sp.save_npz(os.path.join(root, fname), sp.csr_matrix(adj))
    for fname, arr in feats.items():
        np.save(os.path.join(root, fname), np.asarray(arr, dtype=np.float32))
    print(f"[{name}] saved -> {root}  "
          f"(N={features.shape[0]}, D={features.shape[1]}, "
          f"classes={len(np.unique(labels))}, "
          f"graphs={list(graphs)}, extra_feats={list(feats)})")


def _metapath_adj(bi_adj: sp.spmatrix) -> sp.spmatrix:
    """bi_adj: (N, M) node-to-other-type biadjacency -> (N, N) meta-path graph
    via A A^T, binarized, self-loops removed."""
    m = (bi_adj @ bi_adj.T).tocsr()
    m.setdiag(0)
    m.eliminate_zeros()
    m.data[:] = 1.0
    return m


def _download(url, dst_path):
    if os.path.isfile(dst_path):
        return dst_path
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    print(f"    downloading {url}")
    urllib.request.urlretrieve(url, dst_path)
    return dst_path


# ==============================================================================
# ACM  (torch_geometric.datasets.HGBDataset)
# ==============================================================================

def build_acm():
    from torch_geometric.datasets import HGBDataset
    print("[acm] loading via torch_geometric.datasets.HGBDataset('ACM') ...")
    ds = HGBDataset(root=os.path.join(RAW_ROOT, "acm"), name="ACM")
    data = ds[0]

    paper_x = data["paper"].x.numpy()
    paper_y = data["paper"].y.numpy()
    n_paper = paper_x.shape[0]

    pa = data[("paper", "to", "author")].edge_index.numpy()
    n_author = int(pa[1].max()) + 1
    PA = sp.coo_matrix((np.ones(pa.shape[1]), (pa[0], pa[1])), shape=(n_paper, n_author))
    coauthor = _metapath_adj(PA)

    ps = data[("paper", "to", "subject")].edge_index.numpy()
    n_subject = int(ps[1].max()) + 1
    PS = sp.coo_matrix((np.ones(ps.shape[1]), (ps[0], ps[1])), shape=(n_paper, n_subject))
    subject_graph = _metapath_adj(PS)

    _save_dataset(
        "acm",
        features=paper_x,
        labels=paper_y,
        graphs={"graph_coauthor.npz": coauthor, "graph_subject.npz": subject_graph},
        feats={"feat_term.npy": paper_x},  # paper features are already bag-of-words/term-based
    )


# ==============================================================================
# DBLP  (torch_geometric.datasets.DBLP)
# ==============================================================================

def build_dblp():
    from torch_geometric.datasets import DBLP
    print("[dblp] loading via torch_geometric.datasets.DBLP() ...")
    ds = DBLP(root=os.path.join(RAW_ROOT, "dblp"))
    data = ds[0]

    author_x = data["author"].x.numpy()
    author_y = data["author"].y.numpy()
    n_author = author_x.shape[0]

    ap = data[("author", "to", "paper")].edge_index.numpy()
    n_paper = int(ap[1].max()) + 1
    AP = sp.coo_matrix((np.ones(ap.shape[1]), (ap[0], ap[1])), shape=(n_author, n_paper))
    coauthor = _metapath_adj(AP)

    pt = data[("paper", "to", "term")].edge_index.numpy()
    n_term = int(pt[1].max()) + 1
    PT = sp.coo_matrix((np.ones(pt.shape[1]), (pt[0], pt[1])), shape=(n_paper, n_term))
    author_term = (AP.tocsr() @ PT.tocsr()).toarray()  # per-author term-usage counts

    _save_dataset(
        "dblp",
        features=author_x,
        labels=author_y,
        graphs={"graph_coauthor.npz": coauthor},
        feats={"feat_term.npy": author_term},
    )


# ==============================================================================
# AMAZON  (torch_geometric.datasets.Amazon, "Computers")
# ==============================================================================

def build_amazon():
    from torch_geometric.datasets import Amazon
    from sklearn.decomposition import PCA
    print("[amazon] loading via torch_geometric.datasets.Amazon('Computers') ...")
    ds = Amazon(root=os.path.join(RAW_ROOT, "amazon"), name="Computers")
    data = ds[0]

    x = data.x.numpy()
    y = data.y.numpy()
    n = x.shape[0]

    ei = data.edge_index.numpy()
    copurchase = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)).tocsr()
    copurchase = copurchase.maximum(copurchase.T)

    n_components = min(256, x.shape[1])
    x_pca = PCA(n_components=n_components, random_state=0).fit_transform(x)
    if n_components < 256:
        x_pca = np.pad(x_pca, ((0, 0), (0, 256 - n_components)))

    _save_dataset(
        "amazon",
        features=x,
        labels=y,
        graphs={"graph_copurchase.npz": copurchase},
        feats={"feat_text_pca256.npy": x_pca},
    )


# ==============================================================================
# YOUTUBE  (SNAP com-youtube: real graph + real ground-truth communities,
#           synthetic content-feature proxies - see module docstring)
# ==============================================================================

def build_youtube(n_clusters=5, target_nodes=5000, seed=0):
    import networkx as nx
    from sklearn.decomposition import TruncatedSVD

    raw_dir = os.path.join(RAW_ROOT, "youtube")
    # NOTE: SNAP moved the com-youtube files under /data/bigdata/communities/
    # (the old /data/com-youtube.*.gz paths now 404). Using the current
    # working URLs below.
    graph_gz = _download(
        "https://snap.stanford.edu/data/bigdata/communities/com-youtube.ungraph.txt.gz",
        os.path.join(raw_dir, "com-youtube.ungraph.txt.gz"),
    )
    cmty_gz = _download(
        "https://snap.stanford.edu/data/bigdata/communities/com-youtube.top5000.cmty.txt.gz",
        os.path.join(raw_dir, "com-youtube.top5000.cmty.txt.gz"),
    )

    print("[youtube] parsing top-quality ground-truth communities ...")
    communities = []
    with gzip.open(cmty_gz, "rt") as f:
        for line in f:
            nodes = [int(t) for t in line.split()]
            if len(nodes) >= 20:  # skip tiny communities
                communities.append(nodes)

    # Greedily pick n_clusters near-disjoint communities (largest-quality first,
    # since top5000.cmty.txt.gz is already sorted by quality) until we have
    # ~target_nodes labeled nodes.
    chosen_labels = {}
    cluster_id = 0
    for nodes in communities:
        new_nodes = [n for n in nodes if n not in chosen_labels]
        if len(new_nodes) < 10:
            continue
        for n in new_nodes:
            chosen_labels[n] = cluster_id
        cluster_id = (cluster_id + 1) % n_clusters
        if len(chosen_labels) >= target_nodes and cluster_id == 0:
            break

    node_list = list(chosen_labels.keys())
    if len(node_list) > target_nodes:
        rng = np.random.RandomState(seed)
        node_list = list(rng.choice(node_list, size=target_nodes, replace=False))

    print(f"[youtube] selected {len(node_list)} nodes across {n_clusters} "
          f"ground-truth interest-group communities")

    print("[youtube] parsing friendship edge list (this file is large, please wait) ...")
    keep = set(node_list)
    edges = []
    with gzip.open(graph_gz, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            u, v = line.split()
            u, v = int(u), int(v)
            if u in keep and v in keep:
                edges.append((u, v))

    remap = {old: i for i, old in enumerate(node_list)}
    n = len(node_list)
    labels = np.array([chosen_labels[old] for old in node_list], dtype=np.int64)

    rows = [remap[u] for u, v in edges] + [remap[v] for u, v in edges]
    cols = [remap[v] for u, v in edges] + [remap[u] for u, v in edges]
    friendship = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    friendship.data[:] = 1.0

    G = nx.from_scipy_sparse_array(friendship)
    print(f"[youtube] induced subgraph: {n} nodes, {G.number_of_edges()} edges")

    # --- content-feature PROXIES (see module docstring: no real public release exists)
    svd = TruncatedSVD(n_components=64, random_state=seed)
    spectral_feat = svd.fit_transform(friendship.astype(np.float64))  # "thumbnail_cnn" proxy

    deg = np.array([d for _, d in G.degree()], dtype=np.float64)
    clustering = np.array(list(nx.clustering(G).values()), dtype=np.float64)
    triangles = np.array(list(nx.triangles(G).values()), dtype=np.float64)
    struct_feat = np.stack([deg, clustering, triangles], axis=1)
    struct_feat = np.tile(struct_feat, (1, 20))  # "comment_tfidf" proxy, padded to 60-d
    struct_feat = struct_feat + np.random.RandomState(seed).normal(0, 0.01, size=struct_feat.shape)

    main_feat = np.concatenate([spectral_feat, struct_feat], axis=1)

    _save_dataset(
        "youtube",
        features=main_feat,
        labels=labels,
        graphs={"graph_friendship.npz": friendship},
        feats={"feat_thumbnail_cnn.npy": spectral_feat, "feat_comment_tfidf.npy": struct_feat},
    )


# ==============================================================================
# main
# ==============================================================================

if __name__ == "__main__":
    os.makedirs(DATA_ROOT, exist_ok=True)
    print("=" * 78)
    print(" DOWNLOADING + PREPROCESSING: acm, dblp, amazon, youtube")
    print(" (cora / citeseer already auto-download inside agamc.py - untouched)")
    print("=" * 78)

    steps = [("acm", build_acm), ("dblp", build_dblp),
             ("amazon", build_amazon), ("youtube", build_youtube)]
    for name, fn in steps:
        print(f"\n--- {name} " + "-" * (70 - len(name)))
        try:
            fn()
        except Exception as e:
            print(f"[{name}] FAILED: {e}")
            print(f"[{name}] skipping - fix the error above and re-run this script; "
                  f"the other datasets are unaffected.")

    print("\nDone. In agamc.py, set CONFIG['dataset_name'] to 'acm', 'dblp', "
          "'amazon', or 'youtube' and run it as usual.")