#!/usr/bin/env python3
"""
One-click heterophilic few-shot node-classification benchmark for QEF-HGNN.

This is the heterophilic companion of the attached homophily script and keeps
its code organization, tune-once/freeze protocol, split-level reporting, and
paper-report outputs.  It evaluates:

    MLP, GCN, GAT, GraphSAGE,
    H2GCN, MixHop,
    Heat-GCN, Frac-Heat, Mixed-Graph-Frac-Heat,
    QE-HGNN, QFK-HGNN, QEF-HGNN

on Texas, Wisconsin, Cornell, and Chameleon. All four datasets are
explicitly enforced as C=5 classification problems.

Hyperparameters are selected only on dedicated tuning splits and are then
frozen for all ten matched evaluation seeds.  Neural/control methods keep the
original tune-once protocol.  QEF-family operator methods use a two-stage
hierarchical search (operator first, downstream second) with the same total
nominal budget, evaluated on one dedicated tuning seed.  Every split prints
test accuracy, macro-F1, validation accuracy, and runtime.

For QE-HGNN and QEF-HGNN, the exact two-register Hamiltonian has dimension
n^2.  Because the manuscript does not yet provide a fully specified scalable
matrix-free exact implementation, epsilon > 0 is evaluated with the same
first-order Duhamel retained-eigenspace approximation used by the companion
script.  For QEF-HGNN, the classifier affinity is constructed residually from
the fractional kernel, a label-free semantic affinity, and the signed Duhamel
perturbation.  The signed perturbation is ranked before a monotone positive
weight transform; it is not entrywise clipped to zero.  QFK-HGNN uses the
epsilon=0 factorized fractional-diffusion branch.  These results must not be
described as an exact full n^2 two-register exponential.

PennyLane remains an explicit required dependency and is used to validate the
Bell/entangling primitive underlying H_sem.

Outputs include split-level results, validation trials, frozen
hyperparameters, heterophily summary/wide CSV files, a complete LaTeX table,
per-dataset winners, QEF-HGNN versus the strongest executable baseline, and an
overall ranking by average test accuracy and average rank.

INSTALL
-------
pip install numpy scipy pandas scikit-learn torch torch-geometric optuna
pip install pennylane pennylane-lightning
pip install pynndescent  # optional; useful especially for Chameleon

RUN
---
python qef_hgnn_heterophily_residual_hierarchical.py
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import os
import random
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.special import softmax
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F

# PennyLane is an explicit dependency of the quantum-derived part of this
# benchmark.  The scalable QEF/QE affinity used for the paper remains the
# first-order Duhamel approximation described below; PennyLane is used here
# to instantiate and validate the Bell/entangling circuit primitive and to
# keep a device-facing quantum implementation path in the executable code.
import pennylane as qml
from pennylane import numpy as pnp

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_ROOT = Path("./pyg_data")
OUTPUT_ROOT = Path("./qef_hgnn_heterophily_output_residual")
CACHE_ROOT = OUTPUT_ROOT / "affinity_cache"
GRAPH_CACHE_ROOT = OUTPUT_ROOT / "graph_cache"

# PennyLane backend configuration.  lightning.qubit is preferred when
# pennylane-lightning is installed; default.qubit is used as a safe fallback.
PENNYLANE_DEVICE = "lightning.qubit"
PENNYLANE_FALLBACK_DEVICE = "default.qubit"
RUN_PENNYLANE_SANITY_CHECK = True

DATASETS = ["Texas", "Wisconsin", "Cornell", "Chameleon"]
DATASET_CLASSES = {
    "Texas": 5,
    "Wisconsin": 5,
    "Cornell": 5,
    "Chameleon": 5,
}

BENCHMARK_METHODS = [
    "MLP", "GCN", "GAT", "GraphSAGE",
    "H2GCN", "MixHop",
    "Heat-GCN", "Frac-Heat",
]
MATCHED_CONTROL_METHODS = ["Mixed-Graph-Frac-Heat"]
QEF_FAMILY_METHODS = ["QE-HGNN", "QFK-HGNN", "QEF-HGNN"]
OPERATOR_METHODS = MATCHED_CONTROL_METHODS + QEF_FAMILY_METHODS
RUN_METHODS = BENCHMARK_METHODS + OPERATOR_METHODS

METHOD_GROUPS: List[Tuple[str, List[str]]] = [
    ("Feature-only baseline", ["MLP"]),
    ("Classical message-passing methods", ["GCN", "GAT", "GraphSAGE"]),
    ("Heterophily-aware and higher-order methods", ["H2GCN", "MixHop"]),
    ("Spectral and diffusion methods", ["Heat-GCN", "Frac-Heat"]),
    ("Matched semantic control", ["Mixed-Graph-Frac-Heat"]),
    ("Operator variants and full model", ["QE-HGNN", "QFK-HGNN", "QEF-HGNN"]),
]

SEEDS = list(range(10))
# Hyperparameters are selected only on these fixed dedicated tuning splits,
# then frozen and reused for all final evaluation seeds.
# Hyperparameters are selected once on one dedicated tuning split and then
# frozen across the ten reported evaluation seeds 0,...,9.  This keeps tuning
# leakage-free while avoiding the 3x training multiplier that made MixHop on
# Chameleon appear to hang.
TUNING_SEEDS = [1000]
TRAIN_PER_CLASS = 4
# Explicit reproducibility choice: validation is class-balanced and disjoint.
# Five validation nodes/class is used consistently with the fast companion
# benchmark; all remaining non-training/non-validation nodes form the test set.
VAL_PER_CLASS = 5

# Low-rank approximation used by diffusion/operator kernels.
SPECTRAL_RANK = {
    "Texas": 128,
    "Wisconsin": 128,
    "Cornell": 128,
    "Chameleon": 256,
}
SEMANTIC_DIM = 128
SEMANTIC_MAX_K = 30
KERNEL_TOPK = 64
KERNEL_BLOCK_ROWS = 256

# Paper protocol: Algorithm 1 uses the same nominal Bayesian-optimisation
# cap B=100 for every method.  FAST_DEBUG_MODE is only for smoke tests and
# must not be used for reported paper results.
PAPER_TUNING_BUDGET = 100
FAST_DEBUG_MODE = False
FAST_DEBUG_TUNING_BUDGET = 8
TUNING_BUDGET = FAST_DEBUG_TUNING_BUDGET if FAST_DEBUG_MODE else PAPER_TUNING_BUDGET
SEARCH_TRIALS = TUNING_BUDGET
NEURAL_SEARCH_TRIALS = TUNING_BUDGET
# QEF-family methods keep the SAME total nominal budget, but split it into two
# easier searches: first the operator, then the downstream learner.
OPERATOR_TUNING_BUDGET = max(1, TUNING_BUDGET // 2)
DOWNSTREAM_TUNING_BUDGET = max(1, TUNING_BUDGET - OPERATOR_TUNING_BUDGET)

NEURAL_MAX_EPOCHS = 250
NEURAL_PATIENCE = 30
NEURAL_HIDDEN_GRID = [64, 128]
NEURAL_DROPOUT_GRID = [0.50, 0.70]
NEURAL_LR_GRID = [0.005, 0.010]
NEURAL_WEIGHT_DECAY_GRID = [5e-4, 5e-3]
GAT_HEADS_GRID = [4, 8]
H2_LAYERS_GRID = [1, 2]
HEAT_TIME_GRID = [0.50, 1.00, 2.00, 5.00]
FRAC_ORDER_GRID = [0.50, 0.70, 0.85]
MIXHOP_POWERS = (0, 1, 2)

# Algorithm-2 downstream grids.
S_GRID = [0.50, 0.70, 0.85, 1.00]
EPS_GRID = [0.00, 0.05, 0.10, 0.20, 0.30]
T_GRID = [0.50, 1.00, 2.00, 5.00]
KSEM_GRID = [10, 15, 20, 30]
MU_GRID = [0.00, 0.10, 0.30]
ALPHA_GRID = [0.80, 0.85, 0.90]
LAMBDA_GRID = [0.30, 0.50, 0.70, 0.90]
RIDGE_GRID = [1e-3, 1e-2, 1e-1]
PROPAGATION_STEPS_GRID = [20, 40]
MATCHED_SELF_TRAINING_ROUNDS = 2
SELF_TRAINING_ROUNDS_GRID = [MATCHED_SELF_TRAINING_ROUNDS]
CONFIDENCE_GRID = [0.85, 0.90, 0.95]
PSEUDO_PER_CLASS_GRID = [3, 5]
VOTE_NEIGHBORS_GRID = [10, 15, 20]
MIXED_BETA_GRID = [0.25, 0.50, 0.75]
# Residual QEF-HGNN weights. gamma=delta=0 exactly recovers the fractional
# affinity branch after the same rank-to-positive post-processing.
QEF_SEMANTIC_RESIDUAL_GRID = [0.00, 0.25, 0.50, 0.75, 1.00]
QEF_CORRECTION_RESIDUAL_GRID = [0.00, 0.50, 1.00, 1.50, 2.00]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Runtime correction for MixHop.  The first MixHop layer repeatedly needs
# X, A X, and A^2 X.  On Chameleon, recomputing A X and A^2 X inside every
# epoch/trial dominates runtime.  These input powers are label-free and fixed,
# so precomputing them once per dataset/device is mathematically exact.
MIXHOP_PRECOMPUTE_INPUT_POWERS = True
_MIXHOP_GRAPH_CACHE: Dict[Tuple[str, str], Dict[str, torch.Tensor]] = {}

# Optuna pruning is used only during hyperparameter selection.  It does not
# change final evaluation: the selected configuration is still retrained from
# scratch on every reported seed with validation-only early stopping.
ENABLE_NEURAL_OPTUNA_PRUNING = True
OPTUNA_PRUNER_STARTUP_TRIALS = 10
NEURAL_TUNING_PROGRESS_EVERY = 10


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass(frozen=True)
class OperatorConfig:
    method: str
    s: float
    epsilon: float
    t: float
    k_sem: int
    spectral_rank: int
    affinity_topk: int
    semantic_residual: float = 0.0
    qef_residual: float = 1.0

    def validate(self) -> None:
        if self.method not in {"QE-HGNN", "QFK-HGNN", "QEF-HGNN"}:
            raise ValueError(f"Unsupported operator method: {self.method}")
        if not (0.0 < self.s <= 1.0):
            raise ValueError("s must be in (0, 1].")
        if self.epsilon < 0.0:
            raise ValueError("epsilon must be nonnegative.")
        if self.t <= 0.0:
            raise ValueError("t must be positive.")
        if self.k_sem < 1:
            raise ValueError("k_sem must be positive.")
        if self.spectral_rank < 2:
            raise ValueError("spectral_rank must be at least 2.")
        if self.affinity_topk < 1:
            raise ValueError("affinity_topk must be positive.")
        if self.semantic_residual < 0.0:
            raise ValueError("semantic_residual must be nonnegative.")
        if self.qef_residual < 0.0:
            raise ValueError("qef_residual must be nonnegative.")
        if self.method != "QEF-HGNN" and self.semantic_residual != 0.0:
            raise ValueError("semantic_residual is only used by QEF-HGNN.")


@dataclass(frozen=True)
class DownstreamConfig:
    mu: float
    alpha: float
    fusion_weight: float
    ridge_alpha: float
    propagation_steps: int
    self_training_rounds: int
    confidence_threshold: float
    pseudo_labels_per_class: int
    vote_neighbors: int

    def validate(self) -> None:
        if self.mu < 0.0:
            raise ValueError("mu must be nonnegative.")
        if not (0.0 <= self.alpha < 1.0):
            raise ValueError("alpha must be in [0,1).")
        if not (0.0 <= self.fusion_weight <= 1.0):
            raise ValueError("fusion_weight must be in [0,1].")
        if self.ridge_alpha <= 0.0:
            raise ValueError("ridge_alpha must be positive.")
        if self.propagation_steps < 1:
            raise ValueError("propagation_steps must be positive.")
        if self.self_training_rounds < 0:
            raise ValueError("self_training_rounds must be nonnegative.")
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError("confidence_threshold must be in [0,1].")
        if self.pseudo_labels_per_class < 0:
            raise ValueError("pseudo_labels_per_class must be nonnegative.")
        if self.vote_neighbors < 1:
            raise ValueError("vote_neighbors must be positive.")


@dataclass
class DatasetBundle:
    name: str
    X: np.ndarray
    y: np.ndarray
    W: sp.csr_matrix
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    semantic_rows: np.ndarray
    semantic_cols: np.ndarray
    semantic_weights: np.ndarray
    semantic_ranks: np.ndarray


# =============================================================================
# PENNYLANE QUANTUM PRIMITIVES
# =============================================================================


def make_pennylane_device(wires: int):
    """Create the preferred PennyLane simulator with a robust fallback."""
    if wires < 1:
        raise ValueError("wires must be positive")
    try:
        return qml.device(PENNYLANE_DEVICE, wires=wires)
    except Exception:
        return qml.device(PENNYLANE_FALLBACK_DEVICE, wires=wires)


def pennylane_bell_pair_sanity_check() -> Dict[str, float | str]:
    """
    Validate the Bell-pair primitive used conceptually by H_sem.

    This check is intentionally small and does NOT replace the scalable
    first-order Duhamel affinity used for QE-HGNN/QEF-HGNN on graph datasets.
    It confirms that the executable PennyLane path prepares a maximally
    entangled two-qubit Bell state with the expected probabilities and
    <Z_0 Z_1> correlation.
    """
    dev = make_pennylane_device(wires=2)

    @qml.qnode(dev)
    def bell_circuit():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.probs(wires=[0, 1]), qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    probs, zz = bell_circuit()
    probs = pnp.asarray(probs)
    fidelity_phi_plus = float(probs[0] + probs[3])
    return {
        "pennylane_version": str(qml.__version__),
        "device": str(dev.short_name if hasattr(dev, "short_name") else dev.name),
        "bell_support_probability": fidelity_phi_plus,
        "zz_correlation": float(zz),
    }


# =============================================================================
# REPRODUCIBILITY AND DATA
# =============================================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_dataset(name: str, root: Path) -> Tuple[np.ndarray, np.ndarray, sp.csr_matrix]:
    """Load Texas, Wisconsin, Cornell, and Chameleon from PyTorch Geometric."""
    try:
        from torch_geometric.datasets import WebKB, WikipediaNetwork
        from torch_geometric.utils import remove_self_loops, to_undirected
    except ImportError as exc:
        raise ImportError(
            "PyTorch Geometric is required. Install torch and torch-geometric."
        ) from exc

    if name in {"Texas", "Wisconsin", "Cornell"}:
        dataset = WebKB(root=str(root / name), name=name)
    elif name == "Chameleon":
        dataset = WikipediaNetwork(
            root=str(root / name),
            name="chameleon",
            geom_gcn_preprocess=True,
        )
    else:
        raise ValueError(f"Unsupported heterophilic dataset: {name}")

    data = dataset[0]
    # Dataset-provided masks are intentionally ignored. Every method receives
    # exactly the same custom few-shot split generated by make_split().
    edge_index, _ = remove_self_loops(data.edge_index)
    edge_index = to_undirected(edge_index, num_nodes=data.num_nodes)

    rows = edge_index[0].cpu().numpy()
    cols = edge_index[1].cpu().numpy()
    W = sp.coo_matrix(
        (np.ones(rows.size, dtype=np.float64), (rows, cols)),
        shape=(data.num_nodes, data.num_nodes),
    ).tocsr()
    W.sum_duplicates()
    W.data[:] = 1.0
    W.setdiag(0.0)
    W.eliminate_zeros()

    X = data.x.cpu().numpy().astype(np.float32, copy=False)
    y = data.y.cpu().numpy().astype(np.int64, copy=False)

    # Defensive remapping to contiguous class indices 0,...,C-1.
    _, y = np.unique(y, return_inverse=True)
    return X, y.astype(np.int64), W


def edge_homophily(W: sp.csr_matrix, y: np.ndarray) -> float:
    """Compute edge homophily for reporting only, never for training."""
    upper = sp.triu(W, k=1).tocoo()
    if upper.nnz == 0:
        return float("nan")
    return float(np.mean(y[upper.row] == y[upper.col]))

def row_l2_normalise(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(norm > 1e-12, norm, 1.0)


def make_split(
    y: np.ndarray,
    seed: int,
    train_per_class: int,
    val_per_class: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct a matched class-balanced few-shot split whenever the class size
    permits it.

    Standard case
    -------------
    For a class with at least ``train_per_class + val_per_class`` vertices,
    use exactly ``train_per_class`` training vertices and
    ``val_per_class`` validation vertices.

    Rare-class fallback
    -------------------
    A class can be smaller than the requested budget (Texas contains a
    singleton class in the standard WebKB/Geom-GCN data).  Duplicating that
    node would create leakage and is therefore forbidden.  In that case:
      * use at most ``train_per_class`` distinct vertices for training;
      * use any remaining distinct vertices, up to ``val_per_class``,
        for validation;
      * all unselected vertices remain test vertices.

    Consequently, a singleton class contributes its sole vertex to training
    and has no validation/test vertex.  The class is retained; it is never
    deleted, duplicated, or merged.
    """
    rng = np.random.default_rng(seed)
    n = y.size
    train = np.zeros(n, dtype=bool)
    val = np.zeros(n, dtype=bool)

    for class_id in np.unique(y):
        idx = np.flatnonzero(y == class_id)
        if idx.size == 0:
            continue

        perm = rng.permutation(idx)

        n_train = min(train_per_class, idx.size)
        remaining = idx.size - n_train
        n_val = min(val_per_class, max(0, remaining))

        train[perm[:n_train]] = True
        val[perm[n_train:n_train + n_val]] = True

    test = ~(train | val)

    if not np.any(train):
        raise ValueError("The split contains no training vertices.")
    if not np.any(val):
        raise ValueError("The split contains no validation vertices.")
    if not np.any(test):
        raise ValueError("The split contains no test vertices.")

    return train, val, test


def normalised_laplacian(W: sp.csr_matrix) -> sp.csr_matrix:
    degree = np.asarray(W.sum(axis=1)).ravel()
    inv_sqrt = np.zeros_like(degree, dtype=np.float64)
    positive = degree > 0
    inv_sqrt[positive] = 1.0 / np.sqrt(degree[positive])
    S = sp.diags(inv_sqrt) @ W @ sp.diags(inv_sqrt)
    L = sp.eye(W.shape[0], format="csr", dtype=np.float64) - S
    return ((L + L.T) * 0.5).tocsr()


def smallest_eigenpairs(L: sp.csr_matrix, rank: int) -> Tuple[np.ndarray, np.ndarray]:
    n = L.shape[0]
    k = min(rank, n - 2)
    if k < 2:
        raise ValueError("Graph is too small for the requested eigenspace.")

    values, vectors = eigsh(L, k=k, which="SM", tol=1e-6, maxiter=20000)
    order = np.argsort(values)
    values = np.clip(values[order], 0.0, None)
    vectors = vectors[:, order]
    return values.astype(np.float64), vectors.astype(np.float64)


def label_free_embeddings(X: np.ndarray, dimension: int, seed: int = 0) -> np.ndarray:
    Xn = row_l2_normalise(X)
    max_dim = min(dimension, Xn.shape[0] - 1, Xn.shape[1] - 1)
    if max_dim < 2:
        return Xn
    if Xn.shape[1] <= max_dim:
        return Xn
    svd = TruncatedSVD(n_components=max_dim, random_state=seed, n_iter=7)
    H = svd.fit_transform(Xn)
    return row_l2_normalise(H)


def semantic_topk_edges(
    embeddings: np.ndarray,
    max_k: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute positive cosine top-k neighbours without labels.

    Returns directed row/column/weight/rank arrays.  Rank is zero-based within
    each source row and lets smaller k_sem values reuse the same neighbour set.
    """
    H = row_l2_normalise(embeddings).astype(np.float32)
    n = H.shape[0]
    query_k = min(max_k + 1, n)

    try:
        from pynndescent import NNDescent

        index = NNDescent(
            H,
            n_neighbors=query_k,
            metric="cosine",
            random_state=seed,
            n_jobs=-1,
            low_memory=True,
        )
        indices, distances = index.neighbor_graph
        similarities = 1.0 - distances
    except ImportError:
        from sklearn.neighbors import NearestNeighbors

        print(
            "  WARNING: pynndescent is not installed; using exact brute-force "
            "cosine neighbours, which is slow on large graphs."
        )
        model = NearestNeighbors(
            n_neighbors=query_k,
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        )
        model.fit(H)
        distances, indices = model.kneighbors(H, return_distance=True)
        similarities = 1.0 - distances

    rows: List[int] = []
    cols: List[int] = []
    weights: List[float] = []
    ranks: List[int] = []

    for i in range(n):
        rank_i = 0
        for j, sim in zip(indices[i], similarities[i]):
            j = int(j)
            sim = float(sim)
            if j == i or sim <= 0.0:
                continue
            rows.append(i)
            cols.append(j)
            weights.append(min(sim, 1.0))
            ranks.append(rank_i)
            rank_i += 1
            if rank_i >= max_k:
                break

    return (
        np.asarray(rows, dtype=np.int64),
        np.asarray(cols, dtype=np.int64),
        np.asarray(weights, dtype=np.float64),
        np.asarray(ranks, dtype=np.int64),
    )


def prepare_dataset(name: str) -> DatasetBundle:
    print(f"\nPreparing {name} ...")
    X, y, W = load_dataset(name, DATA_ROOT)
    # One shared label-free feature normalisation for every method.
    X = row_l2_normalise(X).astype(np.float32, copy=False)
    if len(np.unique(y)) != DATASET_CLASSES[name]:
        raise ValueError(
            f"{name}: expected {DATASET_CLASSES[name]} classes, got {len(np.unique(y))}."
        )

    h_emp = edge_homophily(W, y)
    class_count_map = {
        int(class_id): int(np.count_nonzero(y == class_id))
        for class_id in np.unique(y)
    }
    print(
        f"  nodes={len(y)}, undirected_edges={sp.triu(W, k=1).nnz}, "
        f"features={X.shape[1]}, classes={len(np.unique(y))}, "
        f"edge_homophily={h_emp:.3f}"
    )
    print(f"  class_counts={class_count_map}")

    required = TRAIN_PER_CLASS + VAL_PER_CLASS
    rare = {c: n_c for c, n_c in class_count_map.items() if n_c < required}
    if rare:
        print(
            "  SPLIT NOTE: requested train/validation budget "
            f"{TRAIN_PER_CLASS}/{VAL_PER_CLASS} cannot be met for {rare}. "
            "Using distinct-node rare-class fallback; no duplication or leakage."
        )

    L = normalised_laplacian(W)
    rank = SPECTRAL_RANK[name]
    print(f"  computing {rank} smallest Laplacian eigenpairs")
    eigenvalues, eigenvectors = smallest_eigenpairs(L, rank)

    print(f"  computing label-free semantic embeddings (dim={SEMANTIC_DIM})")
    embeddings = label_free_embeddings(X, SEMANTIC_DIM, seed=0)
    print(f"  computing top-{SEMANTIC_MAX_K} semantic neighbours")
    sr, sc, sw, sk = semantic_topk_edges(embeddings, SEMANTIC_MAX_K, seed=0)

    return DatasetBundle(
        name=name,
        X=X,
        y=y,
        W=W,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        semantic_rows=sr,
        semantic_cols=sc,
        semantic_weights=sw,
        semantic_ranks=sk,
    )


# =============================================================================
# FIRST-ORDER LIFTED HEAT-MARGINAL APPROXIMATION
# =============================================================================


def stable_duhamel_integral(rho: np.ndarray, t: float) -> np.ndarray:
    """
    I_ab = integral_0^t exp(-(t-tau) rho_a) exp(-tau rho_b) d tau.
    """
    ra = rho[:, None]
    rb = rho[None, :]
    delta = rb - ra
    base = np.broadcast_to(np.exp(-t * ra), delta.shape)
    out = np.empty_like(delta, dtype=np.float64)
    close = np.abs(delta) < 1e-10
    out[close] = t * base[close]
    not_close = ~close
    out[not_close] = (
        base[not_close]
        * (-np.expm1(-t * delta[not_close]))
        / delta[not_close]
    )
    return out


def semantic_pair_matrix_in_vertex_space(
    U: np.ndarray,
    heat_weights: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    weights: np.ndarray,
    ranks: np.ndarray,
    k_sem: int,
) -> sp.csr_matrix:
    """
    Build Q from the exact first-order pair formula in the retained eigenspace.

    For each unordered semantic pair (i,j), the partial trace of the Duhamel
    insertion produces the 2x2 block

        1/2 [[H_t(j,j), H_t(i,j)],
             [H_t(i,j), H_t(i,i)]].

    The matrix Q is sparse because only retained semantic pairs are included.
    """
    mask = ranks < k_sem
    r = rows[mask]
    c = cols[mask]
    w = weights[mask]

    # Symmetrise directed semantic neighbours and retain one unordered pair.
    S = sp.coo_matrix((w, (r, c)), shape=(U.shape[0], U.shape[0])).tocsr()
    S = S.maximum(S.T)
    S.setdiag(0.0)
    S.eliminate_zeros()
    upper = sp.triu(S, k=1).tocoo()
    i = upper.row.astype(np.int64)
    j = upper.col.astype(np.int64)
    wij = upper.data.astype(np.float64)

    weighted_U = U * heat_weights[None, :]
    heat_diag = np.einsum("ij,ij->i", weighted_U, U)
    heat_ij = np.einsum("ij,ij->i", weighted_U[i], U[j])

    diag = np.zeros(U.shape[0], dtype=np.float64)
    np.add.at(diag, i, 0.5 * wij * heat_diag[j])
    np.add.at(diag, j, 0.5 * wij * heat_diag[i])

    off = 0.5 * wij * heat_ij
    q_rows = np.concatenate([np.arange(U.shape[0]), i, j])
    q_cols = np.concatenate([np.arange(U.shape[0]), j, i])
    q_data = np.concatenate([diag, off, off])
    Q = sp.coo_matrix(
        (q_data, (q_rows, q_cols)),
        shape=(U.shape[0], U.shape[0]),
    ).tocsr()
    Q.sum_duplicates()
    Q.eliminate_zeros()
    return Q


def rowwise_topk_sparse_from_factors(
    U: np.ndarray,
    heat_weights: np.ndarray,
    correction_coeff: np.ndarray | None,
    epsilon_over_trace: float,
    topk: int,
    block_rows: int,
) -> sp.csr_matrix:
    """Create a nonnegative sparse top-k affinity without forming n x n densely."""
    n, rank = U.shape
    if heat_weights.shape != (rank,):
        raise ValueError("heat_weights has incompatible shape.")

    all_rows: List[np.ndarray] = []
    all_cols: List[np.ndarray] = []
    all_data: List[np.ndarray] = []
    UT = U.T

    for start in range(0, n, block_rows):
        stop = min(start + block_rows, n)
        Ub = U[start:stop]
        scores = (Ub * heat_weights[None, :]) @ UT
        if correction_coeff is not None and epsilon_over_trace != 0.0:
            scores -= epsilon_over_trace * ((Ub @ correction_coeff) @ UT)

        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        scores[scores < 0.0] = 0.0
        local_rows = np.arange(start, stop)
        scores[np.arange(stop - start), local_rows] = 0.0

        k = min(topk, n - 1)
        idx = np.argpartition(scores, -k, axis=1)[:, -k:]
        vals = np.take_along_axis(scores, idx, axis=1)
        order = np.argsort(vals, axis=1)[:, ::-1]
        idx = np.take_along_axis(idx, order, axis=1)
        vals = np.take_along_axis(vals, order, axis=1)

        rr = np.repeat(local_rows, k)
        cc = idx.reshape(-1)
        vv = vals.reshape(-1)
        positive = vv > 0.0
        all_rows.append(rr[positive])
        all_cols.append(cc[positive])
        all_data.append(vv[positive])

    A = sp.coo_matrix(
        (
            np.concatenate(all_data),
            (np.concatenate(all_rows), np.concatenate(all_cols)),
        ),
        shape=(n, n),
    ).tocsr()
    A = A.maximum(A.T)
    A.setdiag(0.0)
    A.eliminate_zeros()
    return A



def _rowwise_unit_scale(values: np.ndarray) -> np.ndarray:
    """Scale each dense row by its maximum absolute off-diagonal magnitude."""
    values = np.asarray(values, dtype=np.float64)
    scale = np.max(np.abs(values), axis=1, keepdims=True)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return values / scale


def rowwise_residual_qef_affinity(
    U: np.ndarray,
    heat_weights: np.ndarray,
    semantic_affinity: sp.csr_matrix,
    correction_coeff: np.ndarray | None,
    heat_trace: float,
    epsilon: float,
    semantic_residual: float,
    qef_residual: float,
    topk: int,
    block_rows: int,
) -> sp.csr_matrix:
    """
    Construct the QEF-HGNN residual affinity without clipping signed QEF scores.

    For each row we combine normalized components

        score = H_frac + gamma S_sem + delta * epsilon * C_QEF,

    where C_QEF is the *signed* normalized first-order Duhamel correction
    (-D/Z).  The raw combined scores are ranked directly.  Only after the top-k
    neighbours are chosen are they mapped monotonically to positive weights by
    an adaptive exponential transform.  Thus a negative QEF perturbation can
    lower a candidate's rank rather than being deleted before ranking.

    Setting gamma=delta=0 gives the fractional branch with exactly the same
    post-processing, providing a safe fallback inside the full model.
    """
    n, rank = U.shape
    if heat_weights.shape != (rank,):
        raise ValueError("heat_weights has incompatible shape")
    if heat_trace <= 0.0 or not np.isfinite(heat_trace):
        raise ValueError("heat_trace must be positive and finite")

    S = semantic_affinity.tocsr().astype(np.float64)
    if S.shape != (n, n):
        raise ValueError("semantic_affinity has incompatible shape")

    all_rows: List[np.ndarray] = []
    all_cols: List[np.ndarray] = []
    all_data: List[np.ndarray] = []
    UT = U.T
    k = min(int(topk), n - 1)
    if k < 1:
        raise ValueError("topk must be positive")

    for start in range(0, n, block_rows):
        stop = min(start + block_rows, n)
        Ub = U[start:stop]

        # Fractional heat component.
        heat = (Ub * heat_weights[None, :]) @ UT
        heat = _rowwise_unit_scale(heat)

        # Label-free semantic residual.  It is sparse globally but cheap to
        # materialize for one row block.
        semantic = S[start:stop].toarray()
        semantic = _rowwise_unit_scale(semantic)

        # Signed QEF perturbation.  The theoretical first-order term is
        # H - epsilon * D/Z, so the residual component is -D/Z.
        if correction_coeff is not None and epsilon > 0.0 and qef_residual > 0.0:
            correction = -((Ub @ correction_coeff) @ UT) / heat_trace
            correction = _rowwise_unit_scale(correction)
        else:
            correction = np.zeros_like(heat)

        scores = (
            heat
            + float(semantic_residual) * semantic
            + float(qef_residual) * float(epsilon) * correction
        )
        scores = np.nan_to_num(scores, nan=-1e12, posinf=1e12, neginf=-1e12)

        # Exclude self-neighbours BEFORE ranking.  Do not ReLU/clip scores.
        local_rows = np.arange(start, stop)
        scores[np.arange(stop - start), local_rows] = -np.inf

        idx = np.argpartition(scores, -k, axis=1)[:, -k:]
        vals = np.take_along_axis(scores, idx, axis=1)
        order = np.argsort(vals, axis=1)[:, ::-1]
        idx = np.take_along_axis(idx, order, axis=1)
        vals = np.take_along_axis(vals, order, axis=1)

        # Monotone positive transform performed AFTER ranking.  Centering at
        # each row maximum makes the largest retained affinity equal to one;
        # an adaptive temperature prevents numerical collapse.
        vmax = np.max(vals, axis=1, keepdims=True)
        spread = np.std(vals, axis=1, keepdims=True)
        temperature = np.maximum(spread, 5e-2)
        weights = np.exp(np.clip((vals - vmax) / temperature, -30.0, 0.0))

        rr = np.repeat(local_rows, k)
        cc = idx.reshape(-1)
        ww = weights.reshape(-1)
        keep = np.isfinite(ww) & (ww > 1e-8)
        all_rows.append(rr[keep])
        all_cols.append(cc[keep])
        all_data.append(ww[keep])

    if not all_data or sum(arr.size for arr in all_data) == 0:
        raise FloatingPointError("Residual QEF affinity contains no positive edges")

    A = sp.coo_matrix(
        (np.concatenate(all_data), (np.concatenate(all_rows), np.concatenate(all_cols))),
        shape=(n, n),
    ).tocsr()
    A = A.maximum(A.T)
    A.setdiag(0.0)
    A.eliminate_zeros()
    return A


# =============================================================================
# EXECUTABLE NEURAL AND DIFFUSION BENCHMARKS
# =============================================================================


def symmetric_normalised_adjacency(
    W: sp.csr_matrix,
    add_self_loops: bool = True,
) -> sp.csr_matrix:
    A = W.astype(np.float64).tocsr(copy=True)
    if add_self_loops:
        A = A + sp.eye(A.shape[0], format="csr", dtype=np.float64)
    degree = np.asarray(A.sum(axis=1)).ravel()
    inv_sqrt = np.zeros_like(degree)
    positive = degree > 0.0
    inv_sqrt[positive] = 1.0 / np.sqrt(degree[positive])
    return (sp.diags(inv_sqrt) @ A @ sp.diags(inv_sqrt)).tocsr()


def row_normalised_adjacency(A: sp.csr_matrix) -> sp.csr_matrix:
    A = A.astype(np.float64).tocsr(copy=True)
    degree = np.asarray(A.sum(axis=1)).ravel()
    inv = np.zeros_like(degree)
    positive = degree > 0.0
    inv[positive] = 1.0 / degree[positive]
    return (sp.diags(inv) @ A).tocsr()


def strict_one_two_hop_adjacencies(
    W: sp.csr_matrix,
    dataset_name: str,
) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    cache_dir = GRAPH_CACHE_ROOT / dataset_name
    one_path = cache_dir / "h2_a1.npz"
    two_path = cache_dir / "h2_a2.npz"
    if one_path.exists() and two_path.exists():
        return sp.load_npz(one_path).tocsr(), sp.load_npz(two_path).tocsr()

    A = W.astype(np.float64).tocsr(copy=True)
    A.data[:] = 1.0
    A.setdiag(0.0)
    A.eliminate_zeros()

    A2 = (A @ A).tocsr()
    A2.data[:] = 1.0
    # Strict two-hop neighbourhood: remove one-hop edges and self-connections.
    A2 = A2 - A2.multiply(A)
    A2.data[A2.data < 0.0] = 0.0
    A2.setdiag(0.0)
    A2.eliminate_zeros()
    A2.data[:] = 1.0

    A1n = row_normalised_adjacency(A)
    A2n = row_normalised_adjacency(A2)
    cache_dir.mkdir(parents=True, exist_ok=True)
    sp.save_npz(one_path, A1n)
    sp.save_npz(two_path, A2n)
    return A1n, A2n


def scipy_to_torch_sparse(A: sp.csr_matrix, device: torch.device) -> torch.Tensor:
    coo = A.tocoo()
    indices = torch.as_tensor(
        np.vstack([coo.row, coo.col]), dtype=torch.long, device=device
    )
    values = torch.as_tensor(coo.data, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(
        indices, values, size=coo.shape, device=device
    ).coalesce()


def scipy_to_edge_tensors(
    A: sp.csr_matrix,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    coo = A.tocoo()
    edge_index = torch.as_tensor(
        np.vstack([coo.row, coo.col]), dtype=torch.long, device=device
    )
    edge_weight = torch.as_tensor(coo.data, dtype=torch.float32, device=device)
    return edge_index, edge_weight


def heat_affinity_cache_key(dataset: str, s: float, t: float, topk: int) -> str:
    payload = {"dataset": dataset, "s": s, "t": t, "topk": topk, "v": 1}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]


def build_heat_graph_affinity(
    bundle: DatasetBundle,
    s: float,
    t: float,
    topk: int = KERNEL_TOPK,
) -> sp.csr_matrix:
    key = heat_affinity_cache_key(bundle.name, s, t, topk)
    path = CACHE_ROOT / bundle.name / f"heat_{key}.npz"
    if path.exists():
        return sp.load_npz(path).tocsr()

    U = bundle.eigenvectors[:, : SPECTRAL_RANK[bundle.name]]
    lam = bundle.eigenvalues[: SPECTRAL_RANK[bundle.name]]
    rho = np.power(np.clip(lam, 0.0, None), s)
    h = np.exp(-t * rho)
    A = rowwise_topk_sparse_from_factors(
        U=U,
        heat_weights=h,
        correction_coeff=None,
        epsilon_over_trace=0.0,
        topk=topk,
        block_rows=KERNEL_BLOCK_ROWS,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    sp.save_npz(path, A)
    return A


class GCNNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int, classes: int, dropout: float):
        super().__init__()
        try:
            from torch_geometric.nn import GCNConv
        except ImportError as exc:
            raise ImportError("Install torch-geometric to run GCN benchmarks") from exc
        self.conv1 = GCNConv(in_dim, hidden, cached=False, normalize=True)
        self.conv2 = GCNConv(hidden, classes, cached=False, normalize=True)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, graph: Mapping[str, torch.Tensor]) -> torch.Tensor:
        edge_index = graph["edge_index"]
        edge_weight = graph.get("edge_weight")
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index, edge_weight=edge_weight)


class GATNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: int,
        classes: int,
        dropout: float,
        heads: int,
    ):
        super().__init__()
        try:
            from torch_geometric.nn import GATConv
        except ImportError as exc:
            raise ImportError("Install torch-geometric to run GAT") from exc
        per_head = max(4, hidden // heads)
        self.conv1 = GATConv(
            in_dim, per_head, heads=heads, concat=True, dropout=dropout
        )
        self.conv2 = GATConv(
            per_head * heads, classes, heads=1, concat=False, dropout=dropout
        )
        self.dropout = dropout

    def forward(self, x: torch.Tensor, graph: Mapping[str, torch.Tensor]) -> torch.Tensor:
        edge_index = graph["edge_index"]
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


class GraphSAGENet(nn.Module):
    def __init__(self, in_dim: int, hidden: int, classes: int, dropout: float):
        super().__init__()
        try:
            from torch_geometric.nn import SAGEConv
        except ImportError as exc:
            raise ImportError("Install torch-geometric to run GraphSAGE") from exc
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, classes)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, graph: Mapping[str, torch.Tensor]) -> torch.Tensor:
        edge_index = graph["edge_index"]
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


class H2GCNNet(nn.Module):
    """Strict 1-/2-hop H2GCN-style benchmark with ego separation."""

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        classes: int,
        dropout: float,
        layers: int,
    ):
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden)
        representation_dim = hidden * sum(2 ** level for level in range(layers + 1))
        self.classifier = nn.Linear(representation_dim, classes)
        self.dropout = dropout
        self.layers = layers

    def forward(self, x: torch.Tensor, graph: Mapping[str, torch.Tensor]) -> torch.Tensor:
        a1 = graph["a1"]
        a2 = graph["a2"]
        h = F.relu(self.embed(x))
        representations = [h]
        for _ in range(self.layers):
            one_hop = torch.sparse.mm(a1, h)
            two_hop = torch.sparse.mm(a2, h)
            h = torch.cat([one_hop, two_hop], dim=1)
            representations.append(h)
        z = torch.cat(representations, dim=1)
        z = F.dropout(z, p=self.dropout, training=self.training)
        return self.classifier(z)



class MixHopLayer(nn.Module):
    def __init__(self, in_dim: int, out_per_power: int, powers: Sequence[int]):
        super().__init__()
        self.powers = tuple(sorted(set(int(p) for p in powers)))
        self.linears = nn.ModuleDict(
            {str(p): nn.Linear(in_dim, out_per_power) for p in self.powers}
        )

    def forward(
        self,
        x: torch.Tensor,
        a_norm: torch.Tensor,
        precomputed_powers: Mapping[int, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Apply independent linear maps to adjacency powers.

        For the first layer, powers of the fixed input feature matrix may be
        supplied through ``precomputed_powers``.  This is exactly equivalent to
        recomputing A^p X every epoch because X and A are constants and the
        learnable linear map is applied only after propagation.
        """
        outputs: List[torch.Tensor] = []

        if precomputed_powers is not None:
            for power in self.powers:
                propagated = precomputed_powers.get(power)
                if propagated is None:
                    propagated = x
                    for _ in range(power):
                        propagated = torch.sparse.mm(a_norm, propagated)
                outputs.append(self.linears[str(power)](propagated))
            return torch.cat(outputs, dim=1)

        propagated = x
        max_power = max(self.powers)
        for power in range(max_power + 1):
            if power in self.powers:
                outputs.append(self.linears[str(power)](propagated))
            if power < max_power:
                propagated = torch.sparse.mm(a_norm, propagated)
        return torch.cat(outputs, dim=1)


class MixHopNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int, classes: int, dropout: float):
        super().__init__()
        width = max(8, hidden // len(MIXHOP_POWERS))
        first_dim = width * len(MIXHOP_POWERS)
        self.layer1 = MixHopLayer(in_dim, width, MIXHOP_POWERS)
        self.layer2 = MixHopLayer(first_dim, width, MIXHOP_POWERS)
        self.classifier = nn.Linear(first_dim, classes)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, graph: Mapping[str, torch.Tensor]) -> torch.Tensor:
        a_norm = graph["a_norm"]
        input_power_cache: Dict[int, torch.Tensor] | None = None
        if "x_power_1" in graph and "x_power_2" in graph:
            input_power_cache = {
                0: x,
                1: graph["x_power_1"],
                2: graph["x_power_2"],
            }

        x = F.relu(
            self.layer1(
                x,
                a_norm,
                precomputed_powers=input_power_cache,
            )
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.layer2(x, a_norm))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)


def neural_config_candidates(method: str, dataset_name: str) -> List[NeuralConfig]:
    common = itertools.product(
        NEURAL_HIDDEN_GRID,
        NEURAL_DROPOUT_GRID,
        NEURAL_LR_GRID,
        NEURAL_WEIGHT_DECAY_GRID,
    )
    configs: List[NeuralConfig] = []

    for hidden, dropout, lr, wd in common:
        if method == "GAT":
            for heads in GAT_HEADS_GRID:
                configs.append(
                    NeuralConfig(method, hidden, dropout, lr, wd, heads=heads)
                )
        elif method == "H2GCN":
            for layers in H2_LAYERS_GRID:
                configs.append(
                    NeuralConfig(method, hidden, dropout, lr, wd, h2_layers=layers)
                )
        elif method == "Heat-GCN":
            for heat_t in HEAT_TIME_GRID:
                configs.append(
                    NeuralConfig(
                        method, hidden, dropout, lr, wd, heat_t=heat_t, heat_s=1.0
                    )
                )
        elif method == "Frac-Heat":
            for heat_s, heat_t in itertools.product(
                FRAC_ORDER_GRID, HEAT_TIME_GRID
            ):
                configs.append(
                    NeuralConfig(
                        method,
                        hidden,
                        dropout,
                        lr,
                        wd,
                        heat_t=heat_t,
                        heat_s=heat_s,
                    )
                )
        else:
            configs.append(NeuralConfig(method, hidden, dropout, lr, wd))

    # Deterministic, dataset-specific subset of the full grid.
    rng = np.random.default_rng(
        20260722 + sum(ord(c) for c in f"{method}-{dataset_name}")
    )
    count = min(NEURAL_SEARCH_TRIALS, len(configs))
    selected = rng.choice(len(configs), size=count, replace=False)
    return [configs[int(i)] for i in selected]


def train_neural_candidate(
    bundle: DatasetBundle,
    config: NeuralConfig,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, float, int]:
    set_seed(seed)
    x = torch.as_tensor(
        bundle.X, dtype=torch.float32, device=DEVICE
    )
    y = torch.as_tensor(bundle.y, dtype=torch.long, device=DEVICE)
    train_t = torch.as_tensor(train_mask, dtype=torch.bool, device=DEVICE)
    val_t = torch.as_tensor(val_mask, dtype=torch.bool, device=DEVICE)
    graph = prepare_neural_graph(bundle, config, DEVICE)

    model = build_neural_model(
        config,
        in_dim=x.shape[1],
        classes=int(bundle.y.max()) + 1,
    ).to(DEVICE)
    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_val = -np.inf
    best_epoch = 0
    best_state: Dict[str, torch.Tensor] | None = None
    stale = 0

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        optimiser.zero_grad(set_to_none=True)
        logits = model(x, graph)
        loss = F.cross_entropy(logits[train_t], y[train_t])
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss for {config.method} on {bundle.name}"
            )
        loss.backward()
        optimiser.step()

        model.eval()
        with torch.no_grad():
            logits = model(x, graph)
            val_pred = logits[val_t].argmax(dim=1)
            val_acc = float((val_pred == y[val_t]).float().mean().item())

        if val_acc > best_val + 1e-12:
            best_val = val_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break

    if best_state is None:
        raise RuntimeError(f"No valid checkpoint for {config.method}")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction = model(x, graph).argmax(dim=1).cpu().numpy()

    del model, optimiser, x, y, train_t, val_t, graph
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return prediction, float(best_val), int(best_epoch)


# =============================================================================
# ALGORITHM 2 DOWNSTREAM PIPELINE
# =============================================================================


def validate_affinity(A: sp.spmatrix, n: int) -> sp.csr_matrix:
    A = A.tocsr().astype(np.float64)
    if A.shape != (n, n):
        raise ValueError(f"Affinity shape {A.shape}; expected {(n, n)}.")
    if not np.all(np.isfinite(A.data)):
        raise ValueError("Affinity contains non-finite values.")
    if A.data.size and A.data.min() < -1e-10:
        raise ValueError("Affinity contains negative entries.")
    A.data[A.data < 0.0] = 0.0
    A = A.maximum(A.T)
    A.setdiag(0.0)
    A.eliminate_zeros()
    return A


def row_stochastic(A: sp.csr_matrix, mu: float) -> sp.csr_matrix:
    n = A.shape[0]
    B = A + mu * sp.eye(n, format="csr", dtype=np.float64)
    row_sum = np.asarray(B.sum(axis=1)).ravel()
    isolated = row_sum <= 0.0
    if np.any(isolated):
        B = B.tolil(copy=True)
        idx = np.flatnonzero(isolated)
        B[idx, idx] = 1.0
        B = B.tocsr()
        row_sum = np.asarray(B.sum(axis=1)).ravel()
    return (sp.diags(1.0 / row_sum) @ B).tocsr()


def one_hot(labels: np.ndarray, classes: int) -> np.ndarray:
    Y = np.zeros((labels.size, classes), dtype=np.float64)
    Y[np.arange(labels.size), labels] = 1.0
    return Y


def probabilities(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    return softmax(scores, axis=1)


def ridge_predict(
    X_smoothed: np.ndarray,
    labels: np.ndarray,
    labelled_mask: np.ndarray,
    classes: int,
    gamma: float,
) -> np.ndarray:
    """Dual ridge form; stable when feature dimension is large and labels are scarce."""
    idx = np.flatnonzero(labelled_mask)
    Xl = X_smoothed[idx]
    Yl = one_hot(labels[idx], classes)
    gram = Xl @ Xl.T + gamma * np.eye(idx.size)
    coeff = np.linalg.solve(gram, Yl)
    return (X_smoothed @ Xl.T) @ coeff


def kernel_vote(
    A: sp.csr_matrix,
    labels: np.ndarray,
    labelled_mask: np.ndarray,
    classes: int,
    k_vote: int,
) -> np.ndarray:
    n = A.shape[0]
    out = np.zeros((n, classes), dtype=np.float64)
    labelled = np.zeros(n, dtype=bool)
    labelled[np.flatnonzero(labelled_mask)] = True

    for node in range(n):
        start, end = A.indptr[node], A.indptr[node + 1]
        neigh = A.indices[start:end]
        weight = A.data[start:end]
        keep = labelled[neigh]
        if not np.any(keep):
            continue
        neigh = neigh[keep]
        weight = weight[keep]
        if weight.size > k_vote:
            chosen = np.argpartition(weight, -k_vote)[-k_vote:]
            neigh = neigh[chosen]
            weight = weight[chosen]
        np.add.at(out[node], labels[neigh], weight)

    row_sum = out.sum(axis=1, keepdims=True)
    good = row_sum[:, 0] > 0.0
    out[good] /= row_sum[good]
    return out


def label_propagation(
    P: sp.csr_matrix,
    labels: np.ndarray,
    labelled_mask: np.ndarray,
    classes: int,
    alpha: float,
    steps: int,
) -> np.ndarray:
    Z0 = np.zeros((labels.size, classes), dtype=np.float64)
    idx = np.flatnonzero(labelled_mask)
    Z0[idx, labels[idx]] = 1.0
    Z = Z0.copy()
    for _ in range(steps):
        Z = alpha * (P @ Z) + (1.0 - alpha) * Z0
    return Z


# =============================================================================
# HYPERPARAMETER SEARCH
# =============================================================================


def product_configs(method: str, rank: int) -> List[Tuple[OperatorConfig, DownstreamConfig]]:
    if method == "QE-HGNN":
        operator_grid = itertools.product(
            [1.0], [e for e in EPS_GRID if e > 0.0], T_GRID, KSEM_GRID
        )
    elif method == "QFK-HGNN":
        operator_grid = itertools.product(
            [s for s in S_GRID if s < 1.0], [0.0], T_GRID, [SEMANTIC_MAX_K]
        )
    elif method == "QEF-HGNN":
        operator_grid = itertools.product(
            [s for s in S_GRID if s < 1.0],
            [e for e in EPS_GRID if e > 0.0],
            T_GRID,
            KSEM_GRID,
        )
    else:
        raise ValueError(method)

    operators = [
        OperatorConfig(
            method=method,
            s=s,
            epsilon=eps,
            t=t,
            k_sem=ksem,
            spectral_rank=rank,
            affinity_topk=KERNEL_TOPK,
        )
        for s, eps, t, ksem in operator_grid
    ]

    downstream = [
        DownstreamConfig(
            mu=mu,
            alpha=alpha,
            fusion_weight=lam,
            ridge_alpha=gamma,
            propagation_steps=steps,
            self_training_rounds=rounds,
            confidence_threshold=theta,
            pseudo_labels_per_class=b,
            vote_neighbors=kvote,
        )
        for mu, alpha, lam, gamma, steps, rounds, theta, b, kvote in itertools.product(
            MU_GRID,
            ALPHA_GRID,
            LAMBDA_GRID,
            RIDGE_GRID,
            PROPAGATION_STEPS_GRID,
            SELF_TRAINING_ROUNDS_GRID,
            CONFIDENCE_GRID,
            PSEUDO_PER_CLASS_GRID,
            VOTE_NEIGHBORS_GRID,
        )
    ]

    # Sampling from the Cartesian product avoids materialising millions of pairs.
    rng = np.random.default_rng(20260722 + sum(ord(c) for c in method))
    pairs: List[Tuple[OperatorConfig, DownstreamConfig]] = []
    for _ in range(SEARCH_TRIALS):
        op = operators[int(rng.integers(0, len(operators)))]
        ds = downstream[int(rng.integers(0, len(downstream)))]
        pairs.append((op, ds))

    # Remove exact duplicates while preserving order.
    unique: List[Tuple[OperatorConfig, DownstreamConfig]] = []
    seen = set()
    for op, ds in pairs:
        key = (op, ds)
        if key not in seen:
            seen.add(key)
            unique.append((op, ds))
    return unique


# =============================================================================
# TABLES, BEST-ACCURACY COMPARISON, AND REPORTS
# =============================================================================


def build_detailed_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Create the left-style mean±SD report without changing the right-code protocol."""
    summary = (
        results.groupby(["dataset", "method"], sort=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_macro_f1=("macro_f1", "mean"),
            std_macro_f1=("macro_f1", "std"),
            mean_validation_accuracy=("validation_accuracy", "mean"),
            std_validation_accuracy=("validation_accuracy", "std"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
            std_runtime_seconds=("runtime_seconds", "std"),
            n_splits=("seed", "count"),
        )
        .reset_index()
    )
    for column in [
        "std_accuracy", "std_macro_f1", "std_validation_accuracy",
        "std_runtime_seconds"
    ]:
        summary[column] = summary[column].fillna(0.0)

    summary["accuracy_percent"] = 100.0 * summary["mean_accuracy"]
    summary["accuracy_std_percent"] = 100.0 * summary["std_accuracy"]
    summary["macro_f1_percent"] = 100.0 * summary["mean_macro_f1"]
    summary["macro_f1_std_percent"] = 100.0 * summary["std_macro_f1"]
    summary["validation_percent"] = 100.0 * summary["mean_validation_accuracy"]
    summary["validation_std_percent"] = 100.0 * summary["std_validation_accuracy"]
    return summary


def save_left_style_reports(results: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Save heterophily-table CSVs and render a console/text summary in the left-code style."""
    summary = build_detailed_summary(results)
    summary.to_csv(OUTPUT_ROOT / "heterophily_summary.csv", index=False)

    accuracy_wide = summary.copy()
    accuracy_wide["formatted"] = accuracy_wide.apply(
        lambda r: f"{r['accuracy_percent']:.1f} ± {r['accuracy_std_percent']:.1f}", axis=1
    )
    accuracy_wide = accuracy_wide.pivot(
        index="method", columns="dataset", values="formatted"
    ).reindex(index=RUN_METHODS, columns=DATASETS)
    accuracy_wide.to_csv(OUTPUT_ROOT / "heterophily_accuracy_wide.csv")

    f1_wide = summary.copy()
    f1_wide["formatted"] = f1_wide.apply(
        lambda r: f"{r['macro_f1_percent']:.1f} ± {r['macro_f1_std_percent']:.1f}", axis=1
    )
    f1_wide = f1_wide.pivot(
        index="method", columns="dataset", values="formatted"
    ).reindex(index=RUN_METHODS, columns=DATASETS)
    f1_wide.to_csv(OUTPUT_ROOT / "heterophily_macro_f1_wide.csv")

    lines: List[str] = [
        "FINAL SUMMARY (frozen hyperparameters; matched evaluation splits)",
        "=================================================================",
    ]
    for dataset in DATASETS:
        ds = summary[summary["dataset"] == dataset]
        if ds.empty:
            continue
        lines.extend(["", f"=== {dataset} ==="])
        for method in RUN_METHODS:
            row = ds[ds["method"] == method]
            if row.empty:
                continue
            r = row.iloc[0]
            lines.append(
                f"{method:10s} | accuracy={r['accuracy_percent']:6.2f} ± "
                f"{r['accuracy_std_percent']:5.2f}% | macro-F1={r['macro_f1_percent']:6.2f} ± "
                f"{r['macro_f1_std_percent']:5.2f}% | val={r['validation_percent']:6.2f} ± "
                f"{r['validation_std_percent']:5.2f}% | runtime={r['mean_runtime_seconds']:7.2f} ± "
                f"{r['std_runtime_seconds']:6.2f}s"
            )

    text = "\n".join(lines) + "\n"
    (OUTPUT_ROOT / "final_summary.txt").write_text(text, encoding="utf-8")
    return summary, text


def computed_summary(results: pd.DataFrame) -> Dict[str, Dict[str, Tuple[float, float]]]:
    out: Dict[str, Dict[str, Tuple[float, float]]] = {}
    grouped = results.groupby(["method", "dataset"])["accuracy"]
    for (method, dataset), values in grouped:
        arr = values.to_numpy(dtype=float) * 100.0
        out.setdefault(method, {})[dataset] = (
            float(arr.mean()),
            float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        )
    return out


def latex_cell(mean: float, std: float, bold: bool) -> str:
    content = rf"{mean:.1f} \\pm {std:.1f}"
    return rf"\\textbf{{\\({content}\\)}}" if bold else rf"\\({content}\\)"


def render_table(results: pd.DataFrame, output_path: Path) -> str:
    values = computed_summary(results)
    best = {
        dataset: max(
            values[method][dataset][0]
            for method in RUN_METHODS
            if method in values and dataset in values[method]
        )
        for dataset in DATASETS
    }

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Few-shot node-classification test accuracy (\%) on four",
        r"heterophilic benchmarks targeting four labelled vertices per class.",
        r"For a class containing fewer than four vertices, all available distinct",
        r"vertices are used for training; no vertex is duplicated. Every displayed row",
        r"is recomputed on the same ten matched evaluation",
        r"splits. Hyperparameters are selected using only one dedicated tuning",
        r"split (seed 1000) and then frozen across all ten evaluation seeds.",
        r"For the QEF-family methods, the same total budget is split between",
        r"operator selection and downstream selection; neural early stopping remains",
        r"validation-only within each evaluation seed.",
        r"Bold identifies the largest sample mean in each column and is",
        r"descriptive rather than a significance claim. The shaded row denotes",
        r"the proposed method.}",
        r"\label{tab:heterophily-full-pipeline}",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Method}",
        r"& \textbf{Texas}",
        r"& \textbf{Wisconsin}",
        r"& \textbf{Cornell}",
        r"& \textbf{Chameleon} \\",
        r"& (\(C=5\))",
        r"& (\(C=5\))",
        r"& (\(C=5\))",
        r"& (\(C=5\)) \\",
        r"\midrule",
    ]

    for group, methods in METHOD_GROUPS:
        lines.append(rf"\multicolumn{{5}}{{l}}{{\emph{{{group}}}}} \\")
        for method in methods:
            method_label = {
                "QE-HGNN": r"\textsc{QE-HGNN} (\(s=1\))",
                "QFK-HGNN": r"\textsc{QFK-HGNN} (\(\varepsilon=0\))",
                "QEF-HGNN": r"\textsc{QEF-HGNN}",
            }.get(method, method)
            if method == "QEF-HGNN":
                lines.append(r"\rowcolor{gray!12}")

            cells: List[str] = []
            for dataset in DATASETS:
                if method not in values or dataset not in values[method]:
                    cells.append(r"--")
                    continue
                mean, std = values[method][dataset]
                cells.append(
                    latex_cell(
                        mean,
                        std,
                        math.isclose(mean, best[dataset], abs_tol=1e-12),
                    )
                )
            lines.append(method_label + " & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines.extend([r"\end{tabular}", r"\end{table}"])
    table = "\n".join(lines) + "\n"
    output_path.write_text(table, encoding="utf-8")
    return table


def render_best_accuracy_report(
    winners: pd.DataFrame,
    qef_comparison: pd.DataFrame,
    overall: pd.DataFrame,
    output_path: Path,
) -> str:
    lines = [
        "BEST TEST-ACCURACY COMPARISON",
        "================================",
        "Selection is based on the largest sample mean over matched test splits.",
        "This comparison is descriptive and is not a statistical-significance test.",
        "",
        "Per-dataset winners:",
    ]
    for _, row in winners.iterrows():
        lines.append(
            f"- {row['dataset']}: {row['best_method']} "
            f"{row['best_mean_accuracy_pct']:.2f} ± "
            f"{row['best_std_accuracy_pct']:.2f}%; runner-up "
            f"{row['runner_up_method']} "
            f"{row['runner_up_mean_accuracy_pct']:.2f}%; "
            f"margin {row['margin_pp']:+.2f} pp."
        )

    lines.extend(["", "QEF-HGNN versus the strongest executable baseline:"])
    for _, row in qef_comparison.iterrows():
        lines.append(
            f"- {row['dataset']}: QEF-HGNN "
            f"{row['qef_mean_accuracy_pct']:.2f}% versus "
            f"{row['best_executable_baseline']} "
            f"{row['baseline_mean_accuracy_pct']:.2f}% "
            f"({row['qef_minus_best_baseline_pp']:+.2f} pp; "
            f"rank {int(row['qef_rank_by_sample_mean'])})."
        )

    lines.extend(["", "Overall ranking across datasets:"])
    for _, row in overall.iterrows():
        lines.append(
            f"{int(row['overall_position'])}. {row['method']}: "
            f"average accuracy {row['average_accuracy_pct']:.2f}%, "
            f"average rank {row['average_rank']:.2f}, "
            f"dataset wins {int(row['dataset_wins'])}."
        )
    report = "\n".join(lines) + "\n"
    output_path.write_text(report, encoding="utf-8")
    return report


def save_incremental(
    all_results: List[Dict[str, object]],
    all_tuning: List[Dict[str, object]],
) -> None:
    pd.DataFrame(all_results).to_csv(
        OUTPUT_ROOT / "seedwise_results.csv", index=False
    )
    pd.DataFrame(all_tuning).to_csv(
        OUTPUT_ROOT / "validation_trials.csv", index=False
    )



# =============================================================================
# PAPER-ALIGNED IMPLEMENTATION (v3)
# =============================================================================
#
# The definitions below provide the paper-aligned executable protocol.
# These corrections align the implementation with the manuscript more closely:
#   1) one dedicated tuning split, then frozen hyperparameters;
#   2) the same nominal Bayesian-optimization cap for all methods;
#   3) matched confidence-based self-training for neural and QEF-family models;
#   4) validation nodes are never admitted as pseudo-labels;
#   5) QEF-family code receives only the labels of the initial training set;
#   6) the first-order retained-eigenspace Duhamel kernel is explicitly treated
#      as an approximation before nonlinear affinity post-processing;
#   7) the paper's Mixed-Graph-Frac-Heat matched semantic control is executable;
#   8) the paper's feature-only MLP baseline is executable.
#
# Publication mode follows Algorithm 1's nominal cap B=100.  FAST_DEBUG_MODE
# can be enabled only for debugging; results from that mode must not be reported
# as the B=100 paper protocol.

try:
    import optuna
except ImportError as exc:
    raise ImportError(
        "Optuna is required for the paper-aligned Bayesian tuning protocol. "
        "Install it with: pip install optuna"
    ) from exc

optuna.logging.set_verbosity(optuna.logging.WARNING)

@dataclass(frozen=True)
class NeuralConfig:
    method: str
    hidden: int
    dropout: float
    learning_rate: float
    weight_decay: float
    max_epochs: int = NEURAL_MAX_EPOCHS
    patience: int = NEURAL_PATIENCE
    heads: int = 4
    h2_layers: int = 2
    gpr_k: int = 10
    gpr_alpha: float = 0.10
    heat_t: float = 1.0
    heat_s: float = 1.0
    self_training_rounds: int = 2
    confidence_threshold: float = 0.90
    pseudo_labels_per_class: int = 3

    def validate(self) -> None:
        if self.method not in BENCHMARK_METHODS:
            raise ValueError(f"Unsupported benchmark method: {self.method}")
        if self.hidden < 2:
            raise ValueError("hidden must be at least 2")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must be in [0,1)")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("Invalid optimiser parameters")
        if self.max_epochs < 1 or self.patience < 1:
            raise ValueError("Training budget must be positive")
        if self.heads < 1 or self.h2_layers < 1 or self.gpr_k < 1:
            raise ValueError("Architecture depth/order must be positive")
        if not (0.0 < self.gpr_alpha < 1.0):
            raise ValueError("gpr_alpha must be in (0,1)")
        if self.heat_t <= 0.0 or not (0.0 < self.heat_s <= 1.0):
            raise ValueError("Invalid heat-kernel parameters")
        if self.self_training_rounds < 0:
            raise ValueError("self_training_rounds must be nonnegative")
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError("confidence_threshold must be in [0,1]")
        if self.pseudo_labels_per_class < 0:
            raise ValueError("pseudo_labels_per_class must be nonnegative")


@dataclass(frozen=True)
class MixedGraphConfig:
    beta: float
    s: float
    t: float
    k_sem: int
    spectral_rank: int
    affinity_topk: int

    def validate(self) -> None:
        if not (0.0 <= self.beta <= 1.0):
            raise ValueError("beta must be in [0,1]")
        if not (0.0 < self.s <= 1.0):
            raise ValueError("s must be in (0,1]")
        if self.t <= 0.0:
            raise ValueError("t must be positive")
        if self.k_sem < 1:
            raise ValueError("k_sem must be positive")
        if self.spectral_rank < 2:
            raise ValueError("spectral_rank must be at least 2")
        if self.affinity_topk < 1:
            raise ValueError("affinity_topk must be positive")


class MLPNet(nn.Module):
    """Feature-only two-layer MLP baseline."""

    def __init__(self, in_dim: int, hidden: int, classes: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, classes)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        graph: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        del graph
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc2(x)


def build_neural_model(
    config: NeuralConfig,
    in_dim: int,
    classes: int,
) -> nn.Module:
    config.validate()
    if config.method == "MLP":
        return MLPNet(in_dim, config.hidden, classes, config.dropout)
    if config.method in {"GCN", "Heat-GCN", "Frac-Heat"}:
        return GCNNet(in_dim, config.hidden, classes, config.dropout)
    if config.method == "GAT":
        return GATNet(in_dim, config.hidden, classes, config.dropout, config.heads)
    if config.method == "GraphSAGE":
        return GraphSAGENet(in_dim, config.hidden, classes, config.dropout)
    if config.method == "H2GCN":
        return H2GCNNet(
            in_dim, config.hidden, classes, config.dropout, config.h2_layers
        )
    if config.method == "MixHop":
        return MixHopNet(in_dim, config.hidden, classes, config.dropout)
    raise ValueError(config.method)


def prepare_mixhop_graph_cached(
    bundle: DatasetBundle,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Build MixHop graph tensors once and cache fixed input powers A X, A^2 X."""
    key = (bundle.name, str(device))
    cached = _MIXHOP_GRAPH_CACHE.get(key)
    if cached is not None:
        return cached

    a_norm_sp = symmetric_normalised_adjacency(bundle.W, add_self_loops=True)
    a_norm = scipy_to_torch_sparse(a_norm_sp, device)
    graph: Dict[str, torch.Tensor] = {"a_norm": a_norm}

    if MIXHOP_PRECOMPUTE_INPUT_POWERS:
        print(
            f"    MixHop cache: precomputing A X and A^2 X once for {bundle.name} "
            f"on {device} ..."
        )
        with torch.no_grad():
            x0 = torch.as_tensor(bundle.X, dtype=torch.float32, device=device)
            x1 = torch.sparse.mm(a_norm, x0).detach()
            x2 = torch.sparse.mm(a_norm, x1).detach()
        # X itself is already provided to model.forward; cache only the two
        # expensive propagated feature blocks to keep memory bounded.
        graph["x_power_1"] = x1
        graph["x_power_2"] = x2

    _MIXHOP_GRAPH_CACHE[key] = graph
    return graph


def prepare_neural_graph(
    bundle: DatasetBundle,
    config: NeuralConfig,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    method = config.method
    if method == "MLP":
        return {}
    if method in {"GCN", "GAT", "GraphSAGE"}:
        edge_index, edge_weight = scipy_to_edge_tensors(bundle.W, device)
        graph = {"edge_index": edge_index}
        if method == "GCN":
            graph["edge_weight"] = edge_weight
        return graph
    if method in {"Heat-GCN", "Frac-Heat"}:
        affinity = build_heat_graph_affinity(
            bundle, s=config.heat_s, t=config.heat_t, topk=KERNEL_TOPK
        )
        edge_index, edge_weight = scipy_to_edge_tensors(affinity, device)
        return {"edge_index": edge_index, "edge_weight": edge_weight}
    if method == "H2GCN":
        a1, a2 = strict_one_two_hop_adjacencies(bundle.W, bundle.name)
        return {
            "a1": scipy_to_torch_sparse(a1, device),
            "a2": scipy_to_torch_sparse(a2, device),
        }
    if method == "MixHop":
        return prepare_mixhop_graph_cached(bundle, device)
    raise ValueError(method)


def benchmark_implementation_label(method: str) -> str:
    labels = {
        "MLP": "feature-only two-layer MLP",
        "GCN": "two-layer PyG GCNConv",
        "GAT": "two-layer multi-head PyG GATConv",
        "GraphSAGE": "two-layer mean-aggregation PyG SAGEConv",
        "H2GCN": "ego-separated strict 1-hop/2-hop H2GCN-style network",
        "MixHop": "two-layer adjacency-power mixing with powers 0,1,2",
        "Heat-GCN": "weighted GCN on top-k standard heat-kernel affinity",
        "Frac-Heat": "weighted GCN on top-k fractional heat-kernel affinity",
    }
    return labels[method]


def select_pseudo_labels(
    fused_probabilities: np.ndarray,
    labelled_mask: np.ndarray,
    threshold: float,
    per_class_budget: int,
    excluded_mask: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Balanced pseudo-label selection without validation-set contamination."""
    predicted = fused_probabilities.argmax(axis=1)
    confidence = fused_probabilities.max(axis=1)
    candidates = ~labelled_mask
    if excluded_mask is not None:
        if excluded_mask.shape != labelled_mask.shape:
            raise ValueError("excluded_mask has incompatible shape")
        candidates &= ~excluded_mask

    nodes: List[int] = []
    pseudo: List[int] = []
    for class_id in range(fused_probabilities.shape[1]):
        idx = np.flatnonzero(
            candidates
            & (predicted == class_id)
            & (confidence >= threshold)
        )
        if idx.size == 0:
            continue
        order = np.lexsort((idx, -confidence[idx]))
        chosen = idx[order[:per_class_budget]]
        nodes.extend(chosen.tolist())
        pseudo.extend([class_id] * chosen.size)
    return np.asarray(nodes, dtype=np.int64), np.asarray(pseudo, dtype=np.int64)


def make_observed_label_vector(
    y_true: np.ndarray,
    train_mask: np.ndarray,
) -> np.ndarray:
    """Expose only training labels to the learner; -1 denotes unobserved labels."""
    labels = np.full(y_true.shape, -1, dtype=np.int64)
    labels[train_mask] = y_true[train_mask]
    return labels


def run_algorithm2(
    X: np.ndarray,
    A_K: sp.spmatrix,
    initial_train_mask: np.ndarray,
    observed_labels: np.ndarray,
    classes: int,
    config: DownstreamConfig,
    protected_pseudo_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    QEF-family downstream learner.

    Crucially, this function never receives the complete ground-truth label
    vector.  Only labels on ``initial_train_mask`` may be nonnegative in
    ``observed_labels``.  Validation nodes may be protected from pseudo-label
    admission through ``protected_pseudo_mask``.
    """
    config.validate()
    n = X.shape[0]
    if observed_labels.shape != (n,):
        raise ValueError("observed_labels has incompatible shape")
    if np.any(observed_labels[~initial_train_mask] >= 0):
        raise ValueError(
            "Non-training ground-truth labels were supplied to run_algorithm2."
        )
    if np.any(observed_labels[initial_train_mask] < 0):
        raise ValueError("Every initial training node must have an observed label.")

    A = validate_affinity(A_K, n)
    P = row_stochastic(A, config.mu)
    PX = np.asarray(P @ X, dtype=np.float64)

    labelled = initial_train_mask.copy()
    working_labels = observed_labels.copy()
    fused = np.zeros((n, classes), dtype=np.float64)

    for round_id in range(config.self_training_rounds + 1):
        feat = ridge_predict(
            X_smoothed=PX,
            labels=working_labels,
            labelled_mask=labelled,
            classes=classes,
            gamma=config.ridge_alpha,
        )
        vote = kernel_vote(
            A=A,
            labels=working_labels,
            labelled_mask=labelled,
            classes=classes,
            k_vote=config.vote_neighbors,
        )
        prop = label_propagation(
            P=P,
            labels=working_labels,
            labelled_mask=labelled,
            classes=classes,
            alpha=config.alpha,
            steps=config.propagation_steps,
        )

        fused = (
            config.fusion_weight * probabilities(feat)
            + (1.0 - config.fusion_weight)
            * 0.5
            * (probabilities(vote) + probabilities(prop))
        )

        if round_id == config.self_training_rounds:
            break
        nodes, pseudo = select_pseudo_labels(
            fused_probabilities=fused,
            labelled_mask=labelled,
            threshold=config.confidence_threshold,
            per_class_budget=config.pseudo_labels_per_class,
            excluded_mask=protected_pseudo_mask,
        )
        if nodes.size == 0:
            break
        labelled[nodes] = True
        working_labels[nodes] = pseudo

    return fused.argmax(axis=1)


def operator_cache_key(dataset: str, config: OperatorConfig) -> str:
    payload = {
        "dataset": dataset,
        **asdict(config),
        "approx": "qef-residual-signed-ranking-v4",
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


def build_duhamel_affinity(
    bundle: DatasetBundle,
    config: OperatorConfig,
) -> sp.csr_matrix:
    """
    Build the executable affinity for QE-HGNN, QFK-HGNN, or QEF-HGNN.

    The retained-space first-order approximation before classification
    post-processing is

        K_r^(1) / Z_r = H_{t,r} - (epsilon/Z_r) D_{t,r}.

    QEF-HGNN uses the robust residual construction

        score = H_frac + gamma S_sem + delta * epsilon * C_QEF,

    with C_QEF=-D/Z.  The signed score is ranked before conversion to positive
    top-k edge weights; therefore negative perturbations are not destroyed by
    entrywise clipping.  gamma=delta=0 exactly exposes the fractional fallback.

    QE-HGNN and QFK-HGNN retain the original affinity path so that the change
    targeted at QEF-HGNN does not silently redefine its ablation baselines.
    """
    config.validate()
    key = operator_cache_key(bundle.name, config)
    cache_path = CACHE_ROOT / bundle.name / f"{key}.npz"
    metadata_path = CACHE_ROOT / bundle.name / f"{key}.json"
    if cache_path.exists():
        return sp.load_npz(cache_path).tocsr()

    rank = min(config.spectral_rank, bundle.eigenvectors.shape[1])
    U = bundle.eigenvectors[:, :rank]
    lam = bundle.eigenvalues[:rank]
    rho = np.power(np.clip(lam, 0.0, None), config.s)
    h = np.exp(-config.t * rho)
    Z_r = float(np.sum(h))
    if not np.isfinite(Z_r) or Z_r <= 0.0:
        raise FloatingPointError("Invalid retained heat trace Z_r")

    correction_coeff: np.ndarray | None = None
    epsilon_over_trace = 0.0
    if config.epsilon > 0.0:
        Q = semantic_pair_matrix_in_vertex_space(
            U=U,
            heat_weights=h,
            rows=bundle.semantic_rows,
            cols=bundle.semantic_cols,
            weights=bundle.semantic_weights,
            ranks=bundle.semantic_ranks,
            k_sem=config.k_sem,
        )
        M = U.T @ (Q @ U)
        integral = stable_duhamel_integral(rho, config.t)
        correction_coeff = M * integral
        correction_coeff = 0.5 * (correction_coeff + correction_coeff.T)
        if not np.all(np.isfinite(correction_coeff)):
            raise FloatingPointError("Non-finite first-order Duhamel coefficients")
        epsilon_over_trace = config.epsilon / Z_r

    if config.method == "QEF-HGNN":
        semantic = _semantic_affinity_for_k(bundle, config.k_sem)
        A = rowwise_residual_qef_affinity(
            U=U,
            heat_weights=h,
            semantic_affinity=semantic,
            correction_coeff=correction_coeff,
            heat_trace=Z_r,
            epsilon=config.epsilon,
            semantic_residual=config.semantic_residual,
            qef_residual=config.qef_residual,
            topk=config.affinity_topk,
            block_rows=KERNEL_BLOCK_ROWS,
        )
        postprocessing = (
            "signed residual ranking -> row-wise top-k -> adaptive exponential "
            "positive weights -> symmetrization -> zero diagonal"
        )
        numerical_kernel = (
            "fractional + label-free semantic residual + signed first-order "
            "Duhamel residual in retained eigenspace"
        )
    else:
        # Preserve the previous QE/QFK ablation implementation.
        A = rowwise_topk_sparse_from_factors(
            U=U,
            heat_weights=h,
            correction_coeff=correction_coeff,
            epsilon_over_trace=epsilon_over_trace,
            topk=config.affinity_topk,
            block_rows=KERNEL_BLOCK_ROWS,
        )
        postprocessing = (
            "legacy negative-entry clipping -> row-wise top-k -> symmetrization "
            "-> zero diagonal"
        )
        numerical_kernel = "retained-eigenspace normalized first-order Duhamel approximation"

    A = validate_affinity(A, bundle.X.shape[0])

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    sp.save_npz(cache_path, A)
    metadata_path.write_text(
        json.dumps(
            {
                "dataset": bundle.name,
                "operator_config": asdict(config),
                "raw_kernel_target": "Tr_2(exp(-t H_hat_{s,epsilon}))",
                "numerical_kernel": numerical_kernel,
                "retained_heat_trace_Z_r": Z_r,
                "epsilon_positive_status": (
                    "first-order in epsilon; O(epsilon^2) omitted"
                    if config.epsilon > 0.0 else "not applicable"
                ),
                "qef_fractional_fallback": (
                    config.method == "QEF-HGNN"
                    and config.semantic_residual == 0.0
                    and config.qef_residual == 0.0
                ),
                "postprocessing": postprocessing,
                "psd_warning": (
                    "The theoretical raw heat marginal is PSD; the processed "
                    "classification affinity is not claimed to preserve PSD."
                ),
                "exact_full_n2_heat_marginal": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return A


def _semantic_affinity_for_k(bundle: DatasetBundle, k_sem: int) -> sp.csr_matrix:
    mask = bundle.semantic_ranks < int(k_sem)
    S = sp.coo_matrix(
        (
            bundle.semantic_weights[mask],
            (bundle.semantic_rows[mask], bundle.semantic_cols[mask]),
        ),
        shape=bundle.W.shape,
    ).tocsr()
    S = S.maximum(S.T)
    S.setdiag(0.0)
    S.eliminate_zeros()
    return S


def _frobenius_normalise_sparse(A: sp.csr_matrix) -> sp.csr_matrix:
    A = A.astype(np.float64).tocsr(copy=True)
    norm = float(np.sqrt(np.dot(A.data, A.data))) if A.nnz else 0.0
    if norm <= 1e-15:
        return A
    A.data /= norm
    return A


def _mixed_spectrum_cache_key(
    dataset: str,
    beta: float,
    k_sem: int,
    rank: int,
) -> str:
    payload = {
        "dataset": dataset,
        "beta": float(beta),
        "k_sem": int(k_sem),
        "rank": int(rank),
        "kind": "mixed-graph-spectrum-v1",
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


def mixed_graph_spectrum(
    bundle: DatasetBundle,
    beta: float,
    k_sem: int,
    rank: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Spectrum of L((1-beta) A* + beta S*) for the matched semantic control."""
    key = _mixed_spectrum_cache_key(bundle.name, beta, k_sem, rank)
    path = GRAPH_CACHE_ROOT / bundle.name / f"mixed_spectrum_{key}.npz"
    if path.exists():
        data = np.load(path)
        return data["values"], data["vectors"]

    A_star = _frobenius_normalise_sparse(bundle.W)
    S_star = _frobenius_normalise_sparse(_semantic_affinity_for_k(bundle, k_sem))
    B = ((1.0 - beta) * A_star + beta * S_star).tocsr()
    B = B.maximum(B.T)
    B.setdiag(0.0)
    B.eliminate_zeros()
    L_B = normalised_laplacian(B)
    values, vectors = smallest_eigenpairs(L_B, rank)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, values=values, vectors=vectors)
    return values, vectors


def mixed_graph_cache_key(dataset: str, config: MixedGraphConfig) -> str:
    payload = {"dataset": dataset, **asdict(config), "kind": "mixed-fracheat-v1"}
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


def build_mixed_graph_frac_heat_affinity(
    bundle: DatasetBundle,
    config: MixedGraphConfig,
) -> sp.csr_matrix:
    config.validate()
    key = mixed_graph_cache_key(bundle.name, config)
    path = CACHE_ROOT / bundle.name / f"mixed_{key}.npz"
    meta = CACHE_ROOT / bundle.name / f"mixed_{key}.json"
    if path.exists():
        return sp.load_npz(path).tocsr()

    values, vectors = mixed_graph_spectrum(
        bundle,
        beta=config.beta,
        k_sem=config.k_sem,
        rank=config.spectral_rank,
    )
    rho = np.power(np.clip(values, 0.0, None), config.s)
    h = np.exp(-config.t * rho)
    A = rowwise_topk_sparse_from_factors(
        U=vectors,
        heat_weights=h,
        correction_coeff=None,
        epsilon_over_trace=0.0,
        topk=config.affinity_topk,
        block_rows=KERNEL_BLOCK_ROWS,
    )
    A = validate_affinity(A, bundle.X.shape[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    sp.save_npz(path, A)
    meta.write_text(
        json.dumps(
            {
                "dataset": bundle.name,
                "config": asdict(config),
                "construction": (
                    "B_beta=(1-beta)A_star+beta S_star; "
                    "K=exp(-t L(B_beta)^s) in retained eigenspace"
                ),
                "postprocessing": (
                    "negative-entry clipping -> row-wise top-k -> symmetrization -> zero diagonal"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return A


def _optuna_seed(dataset: str, method: str) -> int:
    token = f"{dataset}|{method}|20260726"
    return 100000 + sum((i + 1) * ord(ch) for i, ch in enumerate(token)) % 800000


def suggest_self_training(
    trial: "optuna.trial.Trial",
) -> Tuple[int, float, int]:
    rounds = MATCHED_SELF_TRAINING_ROUNDS
    threshold = trial.suggest_categorical("confidence_threshold", CONFIDENCE_GRID)
    budget = trial.suggest_categorical("pseudo_labels_per_class", PSEUDO_PER_CLASS_GRID)
    return int(rounds), float(threshold), int(budget)


def suggest_neural_config(
    trial: "optuna.trial.Trial",
    method: str,
) -> NeuralConfig:
    hidden = int(trial.suggest_categorical("hidden", NEURAL_HIDDEN_GRID))
    dropout = float(trial.suggest_categorical("dropout", NEURAL_DROPOUT_GRID))
    lr = float(trial.suggest_categorical("learning_rate", NEURAL_LR_GRID))
    wd = float(trial.suggest_categorical("weight_decay", NEURAL_WEIGHT_DECAY_GRID))
    heads = 4
    h2_layers = 2
    heat_t = 1.0
    heat_s = 1.0
    if method == "GAT":
        heads = int(trial.suggest_categorical("heads", GAT_HEADS_GRID))
    if method == "H2GCN":
        h2_layers = int(trial.suggest_categorical("h2_layers", H2_LAYERS_GRID))
    if method == "Heat-GCN":
        heat_t = float(trial.suggest_categorical("heat_t", HEAT_TIME_GRID))
        heat_s = 1.0
    if method == "Frac-Heat":
        heat_t = float(trial.suggest_categorical("heat_t", HEAT_TIME_GRID))
        heat_s = float(trial.suggest_categorical("heat_s", FRAC_ORDER_GRID))
    rounds, threshold, pseudo_budget = suggest_self_training(trial)
    return NeuralConfig(
        method=method,
        hidden=hidden,
        dropout=dropout,
        learning_rate=lr,
        weight_decay=wd,
        heads=heads,
        h2_layers=h2_layers,
        heat_t=heat_t,
        heat_s=heat_s,
        self_training_rounds=rounds,
        confidence_threshold=threshold,
        pseudo_labels_per_class=pseudo_budget,
    )


def suggest_downstream_config(
    trial: "optuna.trial.Trial",
) -> DownstreamConfig:
    rounds, threshold, pseudo_budget = suggest_self_training(trial)
    return DownstreamConfig(
        mu=float(trial.suggest_categorical("mu", MU_GRID)),
        alpha=float(trial.suggest_categorical("alpha", ALPHA_GRID)),
        fusion_weight=float(trial.suggest_categorical("fusion_weight", LAMBDA_GRID)),
        ridge_alpha=float(trial.suggest_categorical("ridge_alpha", RIDGE_GRID)),
        propagation_steps=int(
            trial.suggest_categorical("propagation_steps", PROPAGATION_STEPS_GRID)
        ),
        self_training_rounds=rounds,
        confidence_threshold=threshold,
        pseudo_labels_per_class=pseudo_budget,
        vote_neighbors=int(trial.suggest_categorical("vote_neighbors", VOTE_NEIGHBORS_GRID)),
    )


def operator_selection_downstream_config() -> DownstreamConfig:
    """Fixed low-variance learner used only to select the operator in stage 1."""
    return DownstreamConfig(
        mu=0.10,
        alpha=0.85,
        fusion_weight=0.70,
        ridge_alpha=1e-2,
        propagation_steps=40,
        self_training_rounds=0,
        confidence_threshold=0.90,
        pseudo_labels_per_class=0,
        vote_neighbors=15,
    )


def suggest_operator_config(
    trial: "optuna.trial.Trial",
    method: str,
    rank: int,
) -> OperatorConfig:
    semantic_residual = 0.0
    qef_residual = 1.0

    if method == "QE-HGNN":
        s = 1.0
        epsilon = float(
            trial.suggest_categorical("epsilon", [e for e in EPS_GRID if e > 0.0])
        )
        k_sem = int(trial.suggest_categorical("k_sem", KSEM_GRID))
    elif method == "QFK-HGNN":
        s = float(trial.suggest_categorical("s", [x for x in S_GRID if x < 1.0]))
        epsilon = 0.0
        k_sem = SEMANTIC_MAX_K
        qef_residual = 0.0
    elif method == "QEF-HGNN":
        s = float(trial.suggest_categorical("s", [x for x in S_GRID if x < 1.0]))
        epsilon = float(
            trial.suggest_categorical("epsilon", [e for e in EPS_GRID if e > 0.0])
        )
        k_sem = int(trial.suggest_categorical("k_sem", KSEM_GRID))
        semantic_residual = float(
            trial.suggest_categorical(
                "semantic_residual", QEF_SEMANTIC_RESIDUAL_GRID
            )
        )
        qef_residual = float(
            trial.suggest_categorical(
                "qef_residual", QEF_CORRECTION_RESIDUAL_GRID
            )
        )
    else:
        raise ValueError(method)

    return OperatorConfig(
        method=method,
        s=s,
        epsilon=epsilon,
        t=float(trial.suggest_categorical("t", T_GRID)),
        k_sem=k_sem,
        spectral_rank=rank,
        affinity_topk=KERNEL_TOPK,
        semantic_residual=semantic_residual,
        qef_residual=qef_residual,
    )


def suggest_mixed_graph_config(
    trial: "optuna.trial.Trial",
    rank: int,
) -> MixedGraphConfig:
    return MixedGraphConfig(
        beta=float(trial.suggest_categorical("beta", MIXED_BETA_GRID)),
        s=float(trial.suggest_categorical("mixed_s", FRAC_ORDER_GRID)),
        t=float(trial.suggest_categorical("mixed_t", T_GRID)),
        k_sem=int(trial.suggest_categorical("mixed_k_sem", KSEM_GRID)),
        spectral_rank=rank,
        affinity_topk=KERNEL_TOPK,
    )


def train_neural_once(
    bundle: DatasetBundle,
    config: NeuralConfig,
    train_mask: np.ndarray,
    observed_labels: np.ndarray,
    val_mask: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, float, int]:
    """Train one neural model without exposing validation/test labels to loss."""
    set_seed(seed)
    if observed_labels.shape != bundle.y.shape:
        raise ValueError("observed_labels has incompatible shape")
    if np.any(observed_labels[train_mask] < 0):
        raise ValueError("Training mask contains an unlabelled entry")

    x = torch.as_tensor(bundle.X, dtype=torch.float32, device=DEVICE)
    train_y = torch.as_tensor(observed_labels, dtype=torch.long, device=DEVICE)
    val_y = torch.as_tensor(bundle.y, dtype=torch.long, device=DEVICE)
    train_t = torch.as_tensor(train_mask, dtype=torch.bool, device=DEVICE)
    val_t = torch.as_tensor(val_mask, dtype=torch.bool, device=DEVICE)
    graph = prepare_neural_graph(bundle, config, DEVICE)

    model = build_neural_model(
        config,
        in_dim=x.shape[1],
        classes=int(bundle.y.max()) + 1,
    ).to(DEVICE)
    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_val = -np.inf
    best_epoch = 0
    best_state: Dict[str, torch.Tensor] | None = None
    stale = 0
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        optimiser.zero_grad(set_to_none=True)
        logits = model(x, graph)
        loss = F.cross_entropy(logits[train_t], train_y[train_t])
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss for {config.method} on {bundle.name}"
            )
        loss.backward()
        optimiser.step()

        model.eval()
        with torch.no_grad():
            logits = model(x, graph)
            val_pred = logits[val_t].argmax(dim=1)
            val_acc = float((val_pred == val_y[val_t]).float().mean().item())
        if val_acc > best_val + 1e-12:
            best_val = val_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break

    if best_state is None:
        raise RuntimeError(f"No valid checkpoint for {config.method}")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits_np = model(x, graph).cpu().numpy()

    del model, optimiser, x, train_y, val_y, train_t, val_t, graph
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return logits_np, float(best_val), int(best_epoch)


def train_neural_with_self_training(
    bundle: DatasetBundle,
    config: NeuralConfig,
    initial_train_mask: np.ndarray,
    val_mask: np.ndarray,
    seed: int,
    trial: "optuna.trial.Trial | None" = None,
    trial_step_offset: int = 0,
    enable_pruning: bool = False,
) -> Tuple[np.ndarray, float, int, int]:
    """Matched balanced self-training; validation nodes remain protected."""
    config.validate()
    working_labels = make_observed_label_vector(bundle.y, initial_train_mask)
    labelled = initial_train_mask.copy()
    logits = np.zeros((bundle.y.size, DATASET_CLASSES[bundle.name]), dtype=np.float64)
    final_val = float("nan")
    total_epochs = 0
    rounds_completed = 0

    for round_id in range(config.self_training_rounds + 1):
        logits, final_val, best_epoch = train_neural_once(
            bundle=bundle,
            config=config,
            train_mask=labelled,
            observed_labels=working_labels,
            val_mask=val_mask,
            seed=seed + 1009 * round_id,
        )
        total_epochs += int(best_epoch)
        rounds_completed = round_id

        if enable_pruning and trial is not None:
            trial.report(float(final_val), step=trial_step_offset + round_id)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if round_id == config.self_training_rounds:
            break
        probs = probabilities(logits)
        nodes, pseudo = select_pseudo_labels(
            fused_probabilities=probs,
            labelled_mask=labelled,
            threshold=config.confidence_threshold,
            per_class_budget=config.pseudo_labels_per_class,
            excluded_mask=val_mask,
        )
        if nodes.size == 0:
            break
        labelled[nodes] = True
        working_labels[nodes] = pseudo

    return logits.argmax(axis=1), final_val, total_epochs, rounds_completed


def evaluate_neural_method_on_dataset(
    bundle: DatasetBundle,
    method: str,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Bayesian tune on dedicated tuning splits, freeze, then evaluate 10 seeds."""
    result_rows: List[Dict[str, object]] = []
    tuning_rows: List[Dict[str, object]] = []
    trial_configs: Dict[int, NeuralConfig] = {}

    print(
        f"  {bundle.name:12s} | {method:22s} | Optuna-TPE tune once on "
        f"seed(s)={TUNING_SEEDS} | budget<={TUNING_BUDGET} | "
        f"pruning={'on' if ENABLE_NEURAL_OPTUNA_PRUNING else 'off'}"
    )

    sampler = optuna.samplers.TPESampler(
        seed=_optuna_seed(bundle.name, method),
        multivariate=False,
    )
    pruner = (
        optuna.pruners.MedianPruner(
            n_startup_trials=OPTUNA_PRUNER_STARTUP_TRIALS,
            n_warmup_steps=0,
            interval_steps=1,
        )
        if ENABLE_NEURAL_OPTUNA_PRUNING
        else optuna.pruners.NopPruner()
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    def objective(trial: "optuna.trial.Trial") -> float:
        config = suggest_neural_config(trial, method)
        trial_configs[trial.number] = config
        vals: List[float] = []
        epochs: List[int] = []
        rounds_done: List[int] = []
        for tuning_index, tuning_seed in enumerate(TUNING_SEEDS):
            train, val, _ = make_split(
                bundle.y,
                seed=tuning_seed,
                train_per_class=TRAIN_PER_CLASS,
                val_per_class=VAL_PER_CLASS,
            )
            _, val_acc, used_epochs, rounds_completed = train_neural_with_self_training(
                bundle,
                config,
                train,
                val,
                tuning_seed,
                trial=trial,
                trial_step_offset=tuning_index * (config.self_training_rounds + 1),
                enable_pruning=ENABLE_NEURAL_OPTUNA_PRUNING,
            )
            vals.append(float(val_acc))
            epochs.append(int(used_epochs))
            rounds_done.append(int(rounds_completed))
        score = float(np.mean(vals))
        tuning_rows.append(
            {
                "dataset": bundle.name,
                "method": method,
                "trial": trial.number,
                "tuning_seeds": json.dumps(TUNING_SEEDS),
                "validation_accuracy": score,
                "selection_score": score,
                "best_epoch_total_mean": float(np.mean(epochs)),
                "self_training_rounds_completed_mean": float(np.mean(rounds_done)),
                "model_config": json.dumps(asdict(config), sort_keys=True),
                "implementation": benchmark_implementation_label(method),
                "tuning_scope": "once_per_dataset",
                "search_algorithm": "Optuna TPESampler",
                "budget_cap": TUNING_BUDGET,
            }
        )
        return score

    def progress_callback(
        study_: "optuna.study.Study",
        trial_: "optuna.trial.FrozenTrial",
    ) -> None:
        done = trial_.number + 1
        if done == 1 or done % NEURAL_TUNING_PROGRESS_EVERY == 0 or done == TUNING_BUDGET:
            states = [t.state.name for t in study_.trials]
            completed = states.count("COMPLETE")
            pruned = states.count("PRUNED")
            best_pct = (
                100.0 * float(study_.best_value)
                if completed > 0
                else float("nan")
            )
            print(
                f"    tuning progress {done:3d}/{TUNING_BUDGET}: "
                f"complete={completed}, pruned={pruned}, best_val={best_pct:6.2f}%"
            )

    study.optimize(
        objective,
        n_trials=TUNING_BUDGET,
        show_progress_bar=False,
        callbacks=[progress_callback],
    )
    best_config = trial_configs[study.best_trial.number]
    best_score = float(study.best_value)
    print(f"    frozen config selected at {100.0*best_score:.2f}% tuning accuracy")

    for split_id, seed in enumerate(SEEDS, start=1):
        train, val, test = make_split(
            bundle.y,
            seed=seed,
            train_per_class=TRAIN_PER_CLASS,
            val_per_class=VAL_PER_CLASS,
        )
        seed_start = time.time()
        prediction, val_acc, total_epochs, rounds_completed = train_neural_with_self_training(
            bundle, best_config, train, val, seed
        )
        runtime_seconds = float(time.time() - seed_start)
        test_truth = bundle.y[test]
        test_pred = prediction[test]
        test_acc = float(accuracy_score(test_truth, test_pred))
        test_macro_f1 = float(
            f1_score(test_truth, test_pred, average="macro", zero_division=0)
        )
        result_rows.append(
            {
                "dataset": bundle.name,
                "method": method,
                "split": split_id,
                "seed": seed,
                "accuracy": test_acc,
                "test_accuracy": test_acc,
                "macro_f1": test_macro_f1,
                "test_macro_f1": test_macro_f1,
                "validation_accuracy": float(val_acc),
                "runtime_seconds": runtime_seconds,
                "n_train": int(np.count_nonzero(train)),
                "n_val": int(np.count_nonzero(val)),
                "n_test": int(np.count_nonzero(test)),
                "best_epoch": int(total_epochs),
                "self_training_rounds_completed": int(rounds_completed),
                "model_config": json.dumps(asdict(best_config), sort_keys=True),
                "operator_config": "",
                "downstream_config": "",
                "approximation": benchmark_implementation_label(method),
                "train_per_class": TRAIN_PER_CLASS,
                "val_per_class": VAL_PER_CLASS,
                "hyperparameter_selection": "Optuna_TPE_once_per_dataset_then_frozen",
                "tuning_budget_cap": TUNING_BUDGET,
                "tuning_seeds": json.dumps(TUNING_SEEDS),
                "pseudo_label_validation_protection": True,
            }
        )
        print(
            f"    split={split_id:02d}/{len(SEEDS):02d} seed={seed:2d} | "
            f"{method:22s} acc={100*test_acc:6.2f}%  "
            f"macro-F1={100*test_macro_f1:6.2f}%  "
            f"val={100*val_acc:6.2f}%  {runtime_seconds:7.1f}s"
        )
    return result_rows, tuning_rows


def _evaluate_qef_family_method(
    bundle: DatasetBundle,
    method: str,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """
    Hierarchical QEF-family tuning with a fixed total budget.

    Stage 1 selects only the operator using a fixed, no-self-training downstream
    learner.  Stage 2 freezes that operator and tunes only the downstream
    learner.  Both stages average validation accuracy over one dedicated
    tuning seed.  The final operator/downstream pair is then frozen for all
    ten reported evaluation seeds.
    """
    result_rows: List[Dict[str, object]] = []
    tuning_rows: List[Dict[str, object]] = []
    affinity_memory: MutableMapping[OperatorConfig, sp.csr_matrix] = {}
    stage1_configs: Dict[int, OperatorConfig] = {}
    stage2_configs: Dict[int, DownstreamConfig] = {}
    fixed_ds = operator_selection_downstream_config()

    print(
        f"  {bundle.name:12s} | {method:22s} | hierarchical Optuna-TPE on "
        f"seed(s)={TUNING_SEEDS} | operator={OPERATOR_TUNING_BUDGET}, "
        f"downstream={DOWNSTREAM_TUNING_BUDGET}, total={TUNING_BUDGET}"
    )

    # --------------------------- Stage 1: operator ---------------------------
    sampler_op = optuna.samplers.TPESampler(
        seed=_optuna_seed(bundle.name, method + "-operator"),
        multivariate=False,
    )
    study_op = optuna.create_study(direction="maximize", sampler=sampler_op)

    def operator_objective(trial: "optuna.trial.Trial") -> float:
        op_cfg = suggest_operator_config(trial, method, SPECTRAL_RANK[bundle.name])
        stage1_configs[trial.number] = op_cfg
        if op_cfg not in affinity_memory:
            affinity_memory[op_cfg] = build_duhamel_affinity(bundle, op_cfg)
        A = affinity_memory[op_cfg]

        vals: List[float] = []
        for tuning_seed in TUNING_SEEDS:
            train, val, _ = make_split(
                bundle.y,
                seed=tuning_seed,
                train_per_class=TRAIN_PER_CLASS,
                val_per_class=VAL_PER_CLASS,
            )
            observed = make_observed_label_vector(bundle.y, train)
            pred = run_algorithm2(
                X=bundle.X,
                A_K=A,
                initial_train_mask=train,
                observed_labels=observed,
                classes=DATASET_CLASSES[bundle.name],
                config=fixed_ds,
                protected_pseudo_mask=val,
            )
            vals.append(float(accuracy_score(bundle.y[val], pred[val])))

        score = float(np.mean(vals))
        tuning_rows.append(
            {
                "dataset": bundle.name,
                "method": method,
                "stage": "operator",
                "trial": trial.number,
                "tuning_seeds": json.dumps(TUNING_SEEDS),
                "validation_accuracy": score,
                "validation_std": float(np.std(vals, ddof=0)),
                "selection_score": score,
                "operator_config": json.dumps(asdict(op_cfg), sort_keys=True),
                "downstream_config": json.dumps(asdict(fixed_ds), sort_keys=True),
                "tuning_scope": "hierarchical_stage1_operator",
                "search_algorithm": "Optuna TPESampler",
                "budget_cap": OPERATOR_TUNING_BUDGET,
            }
        )
        return score

    study_op.optimize(
        operator_objective,
        n_trials=OPERATOR_TUNING_BUDGET,
        show_progress_bar=False,
    )
    best_op = stage1_configs[study_op.best_trial.number]
    A_best = affinity_memory.get(best_op)
    if A_best is None:
        A_best = build_duhamel_affinity(bundle, best_op)
    operator_score = float(study_op.best_value)
    print(
        f"    stage 1 operator selected at {100.0*operator_score:.2f}% "
        f"mean tuning accuracy"
    )

    # ------------------------- Stage 2: downstream --------------------------
    sampler_ds = optuna.samplers.TPESampler(
        seed=_optuna_seed(bundle.name, method + "-downstream"),
        multivariate=False,
    )
    study_ds = optuna.create_study(direction="maximize", sampler=sampler_ds)

    def downstream_objective(trial: "optuna.trial.Trial") -> float:
        ds_cfg = suggest_downstream_config(trial)
        stage2_configs[trial.number] = ds_cfg
        vals: List[float] = []
        for tuning_seed in TUNING_SEEDS:
            train, val, _ = make_split(
                bundle.y,
                seed=tuning_seed,
                train_per_class=TRAIN_PER_CLASS,
                val_per_class=VAL_PER_CLASS,
            )
            observed = make_observed_label_vector(bundle.y, train)
            pred = run_algorithm2(
                X=bundle.X,
                A_K=A_best,
                initial_train_mask=train,
                observed_labels=observed,
                classes=DATASET_CLASSES[bundle.name],
                config=ds_cfg,
                protected_pseudo_mask=val,
            )
            vals.append(float(accuracy_score(bundle.y[val], pred[val])))

        score = float(np.mean(vals))
        tuning_rows.append(
            {
                "dataset": bundle.name,
                "method": method,
                "stage": "downstream",
                "trial": trial.number,
                "tuning_seeds": json.dumps(TUNING_SEEDS),
                "validation_accuracy": score,
                "validation_std": float(np.std(vals, ddof=0)),
                "selection_score": score,
                "operator_config": json.dumps(asdict(best_op), sort_keys=True),
                "downstream_config": json.dumps(asdict(ds_cfg), sort_keys=True),
                "tuning_scope": "hierarchical_stage2_downstream",
                "search_algorithm": "Optuna TPESampler",
                "budget_cap": DOWNSTREAM_TUNING_BUDGET,
            }
        )
        return score

    study_ds.optimize(
        downstream_objective,
        n_trials=DOWNSTREAM_TUNING_BUDGET,
        show_progress_bar=False,
    )
    best_ds = stage2_configs[study_ds.best_trial.number]
    downstream_score = float(study_ds.best_value)
    print(
        f"    stage 2 downstream selected at {100.0*downstream_score:.2f}% "
        f"mean tuning accuracy; configuration frozen"
    )

    # -------------------------- Final evaluation ----------------------------
    for split_id, seed in enumerate(SEEDS, start=1):
        train, val, test = make_split(
            bundle.y,
            seed=seed,
            train_per_class=TRAIN_PER_CLASS,
            val_per_class=VAL_PER_CLASS,
        )
        observed = make_observed_label_vector(bundle.y, train)
        seed_start = time.time()
        pred = run_algorithm2(
            X=bundle.X,
            A_K=A_best,
            initial_train_mask=train,
            observed_labels=observed,
            classes=DATASET_CLASSES[bundle.name],
            config=best_ds,
            protected_pseudo_mask=val,
        )
        runtime_seconds = float(time.time() - seed_start)
        val_acc = float(accuracy_score(bundle.y[val], pred[val]))
        test_truth = bundle.y[test]
        test_pred = pred[test]
        test_acc = float(accuracy_score(test_truth, test_pred))
        test_macro_f1 = float(
            f1_score(test_truth, test_pred, average="macro", zero_division=0)
        )

        if method == "QEF-HGNN":
            approximation = (
                "retained-eigenspace fractional + label-free semantic residual + "
                "signed first-order Duhamel residual; rank before positive transform"
            )
        elif best_op.epsilon > 0.0:
            approximation = (
                "retained-eigenspace normalized first-order Duhamel + legacy "
                "nonlinear affinity postprocessing"
            )
        else:
            approximation = (
                "factorized fractional heat kernel in retained eigenspace + "
                "legacy nonlinear affinity postprocessing"
            )

        result_rows.append(
            {
                "dataset": bundle.name,
                "method": method,
                "split": split_id,
                "seed": seed,
                "accuracy": test_acc,
                "test_accuracy": test_acc,
                "macro_f1": test_macro_f1,
                "test_macro_f1": test_macro_f1,
                "validation_accuracy": val_acc,
                "runtime_seconds": runtime_seconds,
                "n_train": int(np.count_nonzero(train)),
                "n_val": int(np.count_nonzero(val)),
                "n_test": int(np.count_nonzero(test)),
                "operator_config": json.dumps(asdict(best_op), sort_keys=True),
                "downstream_config": json.dumps(asdict(best_ds), sort_keys=True),
                "approximation": approximation,
                "train_per_class": TRAIN_PER_CLASS,
                "val_per_class": VAL_PER_CLASS,
                "hyperparameter_selection": (
                    "hierarchical_Optuna_operator_then_downstream_once_per_dataset_then_frozen"
                ),
                "operator_tuning_budget": OPERATOR_TUNING_BUDGET,
                "downstream_tuning_budget": DOWNSTREAM_TUNING_BUDGET,
                "tuning_budget_cap": TUNING_BUDGET,
                "tuning_seeds": json.dumps(TUNING_SEEDS),
                "pseudo_label_validation_protection": True,
                "exact_full_n2_heat_marginal": False,
            }
        )
        print(
            f"    split={split_id:02d}/{len(SEEDS):02d} seed={seed:2d} | "
            f"{method:22s} acc={100*test_acc:6.2f}%  "
            f"macro-F1={100*test_macro_f1:6.2f}%  "
            f"val={100*val_acc:6.2f}%  {runtime_seconds:7.1f}s"
        )

    return result_rows, tuning_rows


def _evaluate_mixed_graph_method(
    bundle: DatasetBundle,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    method = "Mixed-Graph-Frac-Heat"
    result_rows: List[Dict[str, object]] = []
    tuning_rows: List[Dict[str, object]] = []
    affinity_memory: MutableMapping[MixedGraphConfig, sp.csr_matrix] = {}
    trial_configs: Dict[int, Tuple[MixedGraphConfig, DownstreamConfig]] = {}

    print(
        f"  {bundle.name:12s} | {method:22s} | Optuna-TPE tune once on "
        f"seed(s)={TUNING_SEEDS} | budget<={TUNING_BUDGET}"
    )
    sampler = optuna.samplers.TPESampler(
        seed=_optuna_seed(bundle.name, method),
        multivariate=False,
    )
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: "optuna.trial.Trial") -> float:
        kernel_cfg = suggest_mixed_graph_config(trial, SPECTRAL_RANK[bundle.name])
        ds_cfg = suggest_downstream_config(trial)
        trial_configs[trial.number] = (kernel_cfg, ds_cfg)
        if kernel_cfg not in affinity_memory:
            affinity_memory[kernel_cfg] = build_mixed_graph_frac_heat_affinity(
                bundle, kernel_cfg
            )
        A = affinity_memory[kernel_cfg]
        vals: List[float] = []
        for tuning_seed in TUNING_SEEDS:
            train, val, _ = make_split(
                bundle.y,
                seed=tuning_seed,
                train_per_class=TRAIN_PER_CLASS,
                val_per_class=VAL_PER_CLASS,
            )
            observed = make_observed_label_vector(bundle.y, train)
            pred = run_algorithm2(
                X=bundle.X,
                A_K=A,
                initial_train_mask=train,
                observed_labels=observed,
                classes=DATASET_CLASSES[bundle.name],
                config=ds_cfg,
                protected_pseudo_mask=val,
            )
            vals.append(float(accuracy_score(bundle.y[val], pred[val])))
        score = float(np.mean(vals))
        tuning_rows.append(
            {
                "dataset": bundle.name,
                "method": method,
                "trial": trial.number,
                "tuning_seeds": json.dumps(TUNING_SEEDS),
                "validation_accuracy": score,
                "selection_score": score,
                "operator_config": json.dumps(asdict(kernel_cfg), sort_keys=True),
                "downstream_config": json.dumps(asdict(ds_cfg), sort_keys=True),
                "tuning_scope": "once_per_dataset",
                "search_algorithm": "Optuna TPESampler",
                "budget_cap": TUNING_BUDGET,
            }
        )
        return score

    study.optimize(objective, n_trials=TUNING_BUDGET, show_progress_bar=False)
    best_kernel, best_ds = trial_configs[study.best_trial.number]
    best_score = float(study.best_value)
    A_best = affinity_memory.get(best_kernel)
    if A_best is None:
        A_best = build_mixed_graph_frac_heat_affinity(bundle, best_kernel)
    print(f"    frozen config selected at {100.0*best_score:.2f}% tuning accuracy")

    for split_id, seed in enumerate(SEEDS, start=1):
        train, val, test = make_split(
            bundle.y,
            seed=seed,
            train_per_class=TRAIN_PER_CLASS,
            val_per_class=VAL_PER_CLASS,
        )
        observed = make_observed_label_vector(bundle.y, train)
        seed_start = time.time()
        pred = run_algorithm2(
            X=bundle.X,
            A_K=A_best,
            initial_train_mask=train,
            observed_labels=observed,
            classes=DATASET_CLASSES[bundle.name],
            config=best_ds,
            protected_pseudo_mask=val,
        )
        runtime_seconds = float(time.time() - seed_start)
        val_acc = float(accuracy_score(bundle.y[val], pred[val]))
        test_truth = bundle.y[test]
        test_pred = pred[test]
        test_acc = float(accuracy_score(test_truth, test_pred))
        test_macro_f1 = float(
            f1_score(test_truth, test_pred, average="macro", zero_division=0)
        )
        result_rows.append(
            {
                "dataset": bundle.name,
                "method": method,
                "split": split_id,
                "seed": seed,
                "accuracy": test_acc,
                "test_accuracy": test_acc,
                "macro_f1": test_macro_f1,
                "test_macro_f1": test_macro_f1,
                "validation_accuracy": val_acc,
                "runtime_seconds": runtime_seconds,
                "n_train": int(np.count_nonzero(train)),
                "n_val": int(np.count_nonzero(val)),
                "n_test": int(np.count_nonzero(test)),
                "operator_config": json.dumps(asdict(best_kernel), sort_keys=True),
                "downstream_config": json.dumps(asdict(best_ds), sort_keys=True),
                "approximation": "matched one-register mixed-graph fractional heat control",
                "train_per_class": TRAIN_PER_CLASS,
                "val_per_class": VAL_PER_CLASS,
                "hyperparameter_selection": "Optuna_TPE_once_per_dataset_then_frozen",
                "tuning_budget_cap": TUNING_BUDGET,
                "tuning_seeds": json.dumps(TUNING_SEEDS),
                "pseudo_label_validation_protection": True,
            }
        )
        print(
            f"    split={split_id:02d}/{len(SEEDS):02d} seed={seed:2d} | "
            f"{method:22s} acc={100*test_acc:6.2f}%  "
            f"macro-F1={100*test_macro_f1:6.2f}%  "
            f"val={100*val_acc:6.2f}%  {runtime_seconds:7.1f}s"
        )
    return result_rows, tuning_rows


def evaluate_operator_method_on_dataset(
    bundle: DatasetBundle,
    method: str,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    if method == "Mixed-Graph-Frac-Heat":
        return _evaluate_mixed_graph_method(bundle)
    if method in QEF_FAMILY_METHODS:
        return _evaluate_qef_family_method(bundle, method)
    raise ValueError(method)


def build_comparison_frames(
    results: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Comparison frames with Mixed-Graph-Frac-Heat counted as a true baseline."""
    summary = (
        results.groupby(["method", "dataset"])["accuracy"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary["mean_accuracy_pct"] = 100.0 * summary["mean"]
    summary["std_accuracy_pct"] = 100.0 * summary["std"].fillna(0.0)
    summary = summary.drop(columns=["mean", "std"])

    baseline_methods = set(BENCHMARK_METHODS + MATCHED_CONTROL_METHODS)
    winner_rows: List[Dict[str, object]] = []
    qef_rows: List[Dict[str, object]] = []
    for dataset in DATASETS:
        subset = summary[summary["dataset"] == dataset].sort_values(
            ["mean_accuracy_pct", "std_accuracy_pct"], ascending=[False, True]
        )
        if subset.empty:
            continue
        winner = subset.iloc[0]
        runner = subset.iloc[1] if len(subset) > 1 else subset.iloc[0]
        winner_rows.append(
            {
                "dataset": dataset,
                "best_method": winner["method"],
                "best_mean_accuracy_pct": winner["mean_accuracy_pct"],
                "best_std_accuracy_pct": winner["std_accuracy_pct"],
                "runner_up_method": runner["method"],
                "runner_up_mean_accuracy_pct": runner["mean_accuracy_pct"],
                "margin_pp": winner["mean_accuracy_pct"] - runner["mean_accuracy_pct"],
            }
        )
        baseline_subset = subset[subset["method"].isin(baseline_methods)]
        qef_subset = subset[subset["method"] == "QEF-HGNN"]
        if not baseline_subset.empty and not qef_subset.empty:
            baseline = baseline_subset.iloc[0]
            qef = qef_subset.iloc[0]
            qef_rank = int(
                1 + np.sum(subset["mean_accuracy_pct"].to_numpy() > qef["mean_accuracy_pct"])
            )
            qef_rows.append(
                {
                    "dataset": dataset,
                    "best_executable_baseline": baseline["method"],
                    "baseline_mean_accuracy_pct": baseline["mean_accuracy_pct"],
                    "qef_mean_accuracy_pct": qef["mean_accuracy_pct"],
                    "qef_minus_best_baseline_pp": (
                        qef["mean_accuracy_pct"] - baseline["mean_accuracy_pct"]
                    ),
                    "qef_rank_by_sample_mean": qef_rank,
                }
            )

    rank_frame = summary.copy()
    rank_frame["dataset_rank"] = rank_frame.groupby("dataset")["mean_accuracy_pct"].rank(
        ascending=False, method="average"
    )
    overall = (
        rank_frame.groupby("method")
        .agg(
            average_accuracy_pct=("mean_accuracy_pct", "mean"),
            average_rank=("dataset_rank", "mean"),
        )
        .reset_index()
    )
    wins = (
        rank_frame.assign(
            is_win=rank_frame.groupby("dataset")["mean_accuracy_pct"].transform("max")
            == rank_frame["mean_accuracy_pct"]
        )
        .groupby("method")["is_win"]
        .sum()
        .rename("dataset_wins")
        .reset_index()
    )
    overall = overall.merge(wins, on="method", how="left")
    overall = overall.sort_values(
        ["average_rank", "average_accuracy_pct"], ascending=[True, False]
    ).reset_index(drop=True)
    overall["overall_position"] = np.arange(1, len(overall) + 1)
    return pd.DataFrame(winner_rows), pd.DataFrame(qef_rows), overall


# =============================================================================
# MAIN: NO REQUIRED SUBCOMMAND
# =============================================================================


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    GRAPH_CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    pennylane_check = None
    if RUN_PENNYLANE_SANITY_CHECK:
        pennylane_check = pennylane_bell_pair_sanity_check()
        print(
            "PennyLane quantum sanity check: "
            f"device={pennylane_check['device']}, "
            f"Bell support={pennylane_check['bell_support_probability']:.6f}, "
            f"<Z0Z1>={pennylane_check['zz_correlation']:.6f}"
        )

    metadata = {
        "datasets": DATASETS,
        "methods": RUN_METHODS,
        "seeds": SEEDS,
        "tuning_seeds": TUNING_SEEDS,
        "hyperparameter_selection": (
            "one dedicated tuning seed; QEF-family uses hierarchical "
            "operator-then-downstream selection; all selected settings frozen "
            "before seeds 0,...,9"
        ),
        "tuning_budget_cap": TUNING_BUDGET,
        "qef_family_operator_tuning_budget": OPERATOR_TUNING_BUDGET,
        "qef_family_downstream_tuning_budget": DOWNSTREAM_TUNING_BUDGET,
        "publication_budget_B": PAPER_TUNING_BUDGET,
        "fast_debug_mode": FAST_DEBUG_MODE,
        "train_per_class": TRAIN_PER_CLASS,
        "val_per_class": VAL_PER_CLASS,
        "validation_note": (
            "Target: 4 training and 5 validation vertices per class. "
            "If a class is smaller than the requested budget, use distinct "
            "available vertices for training first and validation second; "
            "never duplicate, merge, or drop the class."
        ),
        "rare_class_policy": "distinct_nodes_only_no_duplication_no_label_leakage",
        "class_count_policy": "C=5 fixed for Texas, Wisconsin, Cornell, and Chameleon",
        "mixed_control_search_trials": SEARCH_TRIALS,
        "qef_family_operator_search_trials": OPERATOR_TUNING_BUDGET,
        "qef_family_downstream_search_trials": DOWNSTREAM_TUNING_BUDGET,
        "neural_search_trials": NEURAL_SEARCH_TRIALS,
        "neural_max_epochs": NEURAL_MAX_EPOCHS,
        "neural_patience": NEURAL_PATIENCE,
        "neural_optuna_pruning": ENABLE_NEURAL_OPTUNA_PRUNING,
        "optuna_pruner_startup_trials": OPTUNA_PRUNER_STARTUP_TRIALS,
        "neural_tuning_progress_every": NEURAL_TUNING_PROGRESS_EVERY,
        "mixhop_precompute_input_powers": MIXHOP_PRECOMPUTE_INPUT_POWERS,
        "device": str(DEVICE),
        "pennylane_version": str(qml.__version__),
        "pennylane_preferred_device": PENNYLANE_DEVICE,
        "pennylane_role": (
            "device-facing Bell/entangling primitive validation; scalable "
            "QE/QEF paper evaluation still uses first-order Duhamel approximation"
        ),
        "pennylane_sanity_check": pennylane_check,
        "spectral_rank": SPECTRAL_RANK,
        "semantic_dimension": SEMANTIC_DIM,
        "semantic_max_k": SEMANTIC_MAX_K,
        "kernel_topk": KERNEL_TOPK,
        "operator_approximation": (
            "QE/QFK: retained-eigenspace legacy affinity; QEF: residual fractional "
            "+ label-free semantic + signed first-order Duhamel correction with "
            "ranking before positive edge-weight transform"
        ),
        "matched_self_training": "same rounds/threshold/pseudo-label grids across neural and kernel/operator families",
        "validation_pseudo_label_policy": "validation nodes excluded from pseudo-label admission",
        "matched_semantic_control": "Mixed-Graph-Frac-Heat",
        "scientific_warning": (
            "Across the four heterophilic datasets, epsilon>0 QEF-family results use "
            "a first-order Duhamel retained-eigenspace approximation. For QEF-HGNN, "
            "the signed residual is ranked before a monotone positive transform; no "
            "entrywise clipping is applied to the raw QEF residual. The processed "
            "affinity is not claimed to preserve PSD, and results must not be described "
            "as the exact full n^2 heat marginal."
        ),
    }
    (OUTPUT_ROOT / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    all_results: List[Dict[str, object]] = []
    all_tuning: List[Dict[str, object]] = []
    start = time.time()

    for dataset_name in DATASETS:
        bundle = prepare_dataset(dataset_name)

        for method in BENCHMARK_METHODS:
            rows, tuning = evaluate_neural_method_on_dataset(bundle, method)
            all_results.extend(rows)
            all_tuning.extend(tuning)
            save_incremental(all_results, all_tuning)

        for method in OPERATOR_METHODS:
            rows, tuning = evaluate_operator_method_on_dataset(bundle, method)
            all_results.extend(rows)
            all_tuning.extend(tuning)
            save_incremental(all_results, all_tuning)

    result_df = pd.DataFrame(all_results)
    # Alias retained for compatibility with the left reporting script.
    result_df.to_csv(OUTPUT_ROOT / "all_split_level_results.csv", index=False)
    detailed_summary, final_summary_text = save_left_style_reports(result_df)

    # One frozen configuration per method/dataset.  This file is useful for
    # reproducibility and for reporting the exact configuration used in all
    # ten final evaluation seeds.
    selected_cols = [
        c for c in [
            "dataset", "method", "model_config", "operator_config",
            "downstream_config", "hyperparameter_selection", "tuning_seeds"
        ] if c in result_df.columns
    ]
    selected_hyperparameters = (
        result_df[selected_cols]
        .drop_duplicates(subset=["dataset", "method"])
        .sort_values(["dataset", "method"])
        .reset_index(drop=True)
    )
    selected_hyperparameters.to_csv(
        OUTPUT_ROOT / "selected_hyperparameters.csv", index=False
    )

    table = render_table(result_df, OUTPUT_ROOT / "table_heterophily_full_pipeline.tex")
    winners, qef_comparison, overall = build_comparison_frames(result_df)
    winners.to_csv(OUTPUT_ROOT / "best_accuracy_by_dataset.csv", index=False)
    qef_comparison.to_csv(
        OUTPUT_ROOT / "qef_vs_best_executable_baseline.csv", index=False
    )
    overall.to_csv(OUTPUT_ROOT / "overall_method_ranking.csv", index=False)
    report = render_best_accuracy_report(
        winners,
        qef_comparison,
        overall,
        OUTPUT_ROOT / "best_accuracy_report.txt",
    )

    print("\n" + final_summary_text)
    print("\n" + table)
    print("\n" + report)
    print(f"Saved results to: {OUTPUT_ROOT / 'seedwise_results.csv'}")
    print(f"Saved split-level alias to: {OUTPUT_ROOT / 'all_split_level_results.csv'}")
    print(f"Saved heterophily summary to: {OUTPUT_ROOT / 'heterophily_summary.csv'}")
    print(f"Saved heterophily accuracy-wide table to: {OUTPUT_ROOT / 'heterophily_accuracy_wide.csv'}")
    print(f"Saved heterophily macro-F1-wide table to: {OUTPUT_ROOT / 'heterophily_macro_f1_wide.csv'}")
    print(f"Saved final text summary to: {OUTPUT_ROOT / 'final_summary.txt'}")
    print(f"Saved tuning log to: {OUTPUT_ROOT / 'validation_trials.csv'}")
    print(f"Saved LaTeX table to: {OUTPUT_ROOT / 'table_heterophily_full_pipeline.tex'}")
    print(f"Saved frozen hyperparameters to: {OUTPUT_ROOT / 'selected_hyperparameters.csv'}")
    print(f"Saved winners to: {OUTPUT_ROOT / 'best_accuracy_by_dataset.csv'}")
    print(
        "Saved QEF comparison to: "
        f"{OUTPUT_ROOT / 'qef_vs_best_executable_baseline.csv'}"
    )
    print(f"Saved overall ranking to: {OUTPUT_ROOT / 'overall_method_ranking.csv'}")
    print(f"Total elapsed time: {(time.time() - start) / 60.0:.1f} minutes")


if __name__ == "__main__":
    main()
