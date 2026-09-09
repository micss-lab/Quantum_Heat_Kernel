# QEF-HGNN — Quantum Entanglement Fractional Heat Kernels for Few-Shot Learning on Heterophilic Graphs

Reference implementation and reproduction package for the paper

> **Quantum Entanglement Fractional Heat Kernels for Few-Shot Learning on Heterophilic Graphs**
> Hamed Gholipour, Farid Bozorgnia, João Neves, Moharram Challenger
> *IEEE Transactions on Quantum Engineering* (under review)

QEF-HGNN builds a single Hermitian generator on the two-register space `H_V ⊗ H_V`,

```
H(s,ε) = L^s ⊗ I + I ⊗ L^s  +  ε · H_sem
```

where `L^s` is the fractional graph Laplacian (`s ∈ (0,1]`) and `H_sem` is a semantic
pair-interaction term built from label-free, symmetric Bell-like projectors
`|Ψ⁺_ij⟩ = (|i,j⟩ + |j,i⟩)/√2` weighted by retained cosine similarities. Imaginary-time
evolution followed by a partial trace yields the vertex-marginal heat kernel
`K^{s,ε}_τ = Tr₂(e^{−τH(s,ε)})`, which is real, symmetric, and strictly positive definite.
That kernel drives the downstream few-shot learner (Algorithm 2 of the paper).

**QEF-HGNN is an operator-kernel construction, not a message-passing GNN.** The acronym is
retained from the method's development. No GNN layer is trained inside the model, and no
entanglement measure is used as a learning objective. The paper claims no computational or
hardware-level quantum advantage.

---

## 1. What is in this repository

Two self-contained, one-click benchmark runners reproducing the **full-pipeline predictive
experiments** (Algorithm 1 + Algorithm 2) of the paper:

- **`qef_hgnn_heterophily.py`** — heterophilic regime: Texas, Wisconsin, Cornell, and
  Chameleon, all enforced as `C = 5` problems. Reproduces the heterophilic full-pipeline
  block of the paper.
- **`qef_hgnn_homophily.py`** — predominantly homophilic regime: Cora (`C = 7`),
  CiteSeer (`C = 6`), PubMed (`C = 3`), and Coauthor-CS (`C = 15`), at their native class
  counts. Reproduces the homophilic full-pipeline block.

Both share the same code organization, split construction, tune-once/freeze protocol,
downstream learner, split-level reporting, and paper-ready output artefacts, so their
numbers are directly comparable.

**Not covered by these two scripts:** the finite-shot Born-rule sampling study and resource
estimates, the reduced-budget matched-kernel block (BO = 30), and the one-dimensional `s`
and `ε` sensitivity sweeps on Cora. The heterophily runner does include
`Mixed-Graph-Frac-Heat`, the primary one-register matched semantic control, at the
full-pipeline budget.

---

## 2. Methods evaluated

**Operator family**

- **`QEF-HGNN`** — the complete model: fractional graph diffusion coupled to the semantic
  pair interaction.
- **`QFK-HGNN`** — ablation with `ε = 0`. The semantic interaction is removed and fractional
  diffusion retained. By Proposition 4 of the paper this coincides with the one-register
  fractional heat kernel under an identical downstream pipeline, which serves as a
  zero-coupling implementation check.
- **`QE-HGNN`** — ablation with `s = 1`. The semantic interaction is retained and fractional
  spectral control removed.

**Baselines**

- Classical message passing: `GCN`, `GAT`, `GraphSAGE`.
- Spectral and diffusion: `Heat-GCN` and `Frac-Heat`, each a weighted GCN over a top-k
  standard or fractional heat-kernel affinity.
- Matched semantic control, heterophily runner only: `Mixed-Graph-Frac-Heat`, which mixes
  the normalized adjacency and the normalized semantic matrix as
  `B_β = (1−β)A* + βS*` and diffuses it by `exp[−τ L_sym(B_β)^s]`. It receives the same
  semantic information as QEF-HGNN without the lifted two-register operator.

The homophily runner also contains complete, tuneable `H2GCN` and `MixHop` implementations.
They are not in the default `BENCHMARK_METHODS` list; add their names to that list to
include them in a run.

### Downstream learner (Algorithm 2)

Given the post-processed affinity `A^{s,ε}_τ`, the learner fuses ridge scores on smoothed
features, a top-`k_v` kernel vote, and personalized label propagation, then runs `R` rounds
of confidence-thresholded self-training. The protected validation set enters only through
its membership mask, which excludes those vertices from pseudo-label admission; their labels
are never read. `run_algorithm2` raises if any non-training ground-truth label is passed in,
so label leakage fails loudly rather than silently.

### Role of PennyLane

PennyLane is a required dependency. It instantiates and validates the Bell/entangling
primitive underlying `H_sem` and keeps a device-facing quantum implementation path in the
executable code. A two-qubit Bell-pair check runs at start-up; its device, support
probability, and `⟨Z₀Z₁⟩` are recorded in `run_metadata.json`. It is not used to compute the
reported kernels.

### Numerical realization — please read before quoting numbers

The lifted generator has dimension `n²` (388,760,089 for PubMed; 33,489 × 33,489 even for
Texas), so it is never assembled densely. In these scripts, `ε > 0` is evaluated with a
**first-order Duhamel retained-eigenspace approximation**, not an exact `n²` two-register
exponential. For `QEF-HGNN`, the classifier affinity is built residually from the fractional
kernel, a label-free semantic affinity, and the *signed* Duhamel perturbation; the signed
term is ranked before a monotone positive edge-weight transform rather than entrywise
clipped. The processed affinity is not claimed to be PSD. `QFK-HGNN` uses the `ε = 0`
factorized fractional-diffusion branch. The same statement is written into every run's
`run_metadata.json` under `scientific_warning`, and results from this code must not be
described as the exact `n²` heat marginal.

---

## 3. Installation

Python 3.10+ recommended.

```bash
pip install numpy scipy pandas scikit-learn torch torch-geometric optuna
pip install pennylane pennylane-lightning
pip install pynndescent   # optional; recommended for PubMed, Coauthor-CS, Chameleon
```

- Install `torch` / `torch-geometric` builds matching your CUDA version.
- `pennylane-lightning` provides the preferred `lightning.qubit` device; without it the code
  falls back to `default.qubit` automatically.
- CUDA is used when available; everything also runs on CPU.

Datasets download on first use into `./pyg_data` via PyTorch Geometric: Planetoid (Cora,
CiteSeer, PubMed), Coauthor (CS), WebKB (Texas, Wisconsin, Cornell), and WikipediaNetwork
with `geom_gcn_preprocess=True` for Chameleon. Graphs are symmetrized, duplicate edges
merged, and self-loops removed before operator construction. Edge homophily is computed as a
post hoc diagnostic only and never enters training.

---

## 4. Running

```bash
python qef_hgnn_heterophily.py
python qef_hgnn_homophily.py
```

No arguments needed. Output directories are `./qef_hgnn_heterophily_output_residual/` and
`./qef_hgnn_homophily_output_residual_fast/`.

Set `FAST_DEBUG_MODE = True` for a smoke test; the tuning budget drops from 100 to 8 trials.
**Do not report results from that mode.** Results are checkpointed after every
(dataset, method) pair, so an interrupted run leaves usable partial CSVs. Affinity operators,
mixed-graph spectra, and MixHop input powers are cached under `affinity_cache/` and
`graph_cache/`, keyed by a hash of dataset and configuration.

---

## 5. Experimental protocol

Enforced in code, not left to convention.

**Splits.** Dataset-provided masks are ignored. Every method receives the identical few-shot
split from `make_split()`: 4 labelled vertices per class, 5 class-balanced disjoint
validation vertices per class, all remaining vertices as test. For a class with fewer
vertices than the budget, distinct available vertices are used — training first, validation
second — never duplicated, merged, or dropped. Ten matched evaluation seeds `0, …, 9`.

**Hyperparameter selection.** Bayesian optimization (Optuna) against validation accuracy on
one dedicated tuning split, seed `1000`, after which every setting is frozen for all ten
evaluation seeds and never reselected on test performance. All methods share the nominal
budget `B = 100`. Neural baselines use one flat search with validation-only early stopping
and optional pruning; QEF-family methods split the *same* total budget into a hierarchical
operator-then-downstream search (50 + 50).

**Matched controls.** Self-training rounds, confidence-threshold grid, class-balancing rule,
pseudo-label admission rule, and stopping criterion are shared across families within a
block, so a gain cannot come from a richer self-training schedule. For kernel-based controls
only the kernel construction changes; stabilization, sparsification, normalization, feature
smoothing, kernel voting, label propagation, score fusion, and self-training are unchanged.

**Reporting.** Every split prints test accuracy, macro-F1, validation accuracy, and runtime.
Means and standard deviations are computed over identical matched splits. Bolded winners in
the generated LaTeX tables are descriptive, not significance claims.

---

## 6. Expected results

The full comparison against every baseline is reported in the paper; the scripts regenerate
it directly. The pattern to expect is as follows.

On the heterophilic benchmarks, QEF-HGNN attains the highest sample mean on Texas,
Wisconsin, and Cornell, and does **not** improve on Chameleon, where the strongest classical
baseline and the QE-HGNN ablation are both ahead of it. Against the matched
`Mixed-Graph-Frac-Heat` control, the mean differences are positive on the three WebKB graphs
and negative on Chameleon. On the predominantly homophilic benchmarks the semantic term is
approximately neutral: QEF-HGNN leads only on Cora, and its differences against Frac-Heat,
the closest classical comparator, sit within split-to-split variability.

The evidence is deliberately conditional. Gains concentrate where topology is a weak proxy
for class-relevant semantic similarity; heterophily alone does not predict a gain.
Chameleon is a documented negative boundary case and carries known duplicate-observation
and leakage concerns. The weak Frac-Heat result on CiteSeer is a limitation of the
fractional-diffusion backbone that QEF-HGNN inherits rather than corrects, and is retained
as a negative result.

---

## 7. Outputs

Written to each script's output directory:

- `seedwise_results.csv` — one row per (dataset, method, seed) with accuracy, macro-F1,
  validation accuracy, runtime, split sizes, frozen configurations, and the approximation
  label. `all_split_level_results.csv` is an alias of the same frame.
- `{homophily,heterophily}_summary.csv` — mean and standard deviation per (dataset, method).
- `{homophily,heterophily}_accuracy_wide.csv` and `..._macro_f1_wide.csv` — accuracy and
  macro-F1 matrices, methods by datasets.
- `table_{homophily,heterophily}_full_pipeline.tex` — LaTeX table with grouped method blocks
  and bolded column winners.
- `best_accuracy_by_dataset.csv` — per-dataset winner.
- `qef_vs_best_executable_baseline.csv` — QEF-HGNN against the strongest executable baseline
  on each dataset.
- `overall_method_ranking.csv` — ranking by average test accuracy and average rank.
- `best_accuracy_report.txt` and `final_summary.txt` — human-readable summaries, also
  printed to stdout.
- `selected_hyperparameters.csv` — the single frozen configuration used for all ten seeds,
  per (dataset, method).
- `validation_trials.csv` — full tuning log, every trial on the tuning split.
- `run_metadata.json` — protocol record: seeds, budgets, device, PennyLane version and
  sanity-check result, approximation and scientific-warning statements.

Total elapsed time is printed at the end of each run. Cost is dominated by the tuning budget
and the largest graphs (Coauthor-CS, PubMed, Chameleon); the spectral rank, the affinity
top-k, and the number of Optuna trials are the knobs that move it most.

---

## 8. Configuration reference

All settings are in the `CONFIGURATION` block at the top of each script.

- `S_GRID` — fractional order `s ∈ (0,1]`.
- `EPS_GRID` — semantic coupling strength `ε ≥ 0`.
- `T_GRID` — imaginary time `τ > 0`.
- `KSEM_GRID` — semantic degree `k_sem`, the retained top-k cosine neighbours per vertex,
  union-symmetrized.
- `MU_GRID`, `ALPHA_GRID`, `LAMBDA_GRID`, `RIDGE_GRID`, `PROPAGATION_STEPS_GRID`,
  `VOTE_NEIGHBORS_GRID` — Algorithm-2 parameters `μ, α, λ, γ, T, k_v`.
- `SELF_TRAINING_ROUNDS_GRID`, `CONFIDENCE_GRID`, `PSEUDO_PER_CLASS_GRID` — Algorithm-2
  self-training parameters `R, θ, b`.
- `QEF_SEMANTIC_RESIDUAL_GRID`, `QEF_CORRECTION_RESIDUAL_GRID` — residual weights; both set
  to zero recovers the fractional branch.
- `MIXED_BETA_GRID` — mixing weight `β` of the Mixed-Graph-Frac-Heat control.
- `SPECTRAL_RANK`, `SEMANTIC_DIM`, `KERNEL_TOPK`, `KERNEL_BLOCK_ROWS` — retained eigenspace,
  semantic embedding dimension, affinity sparsity, and row blocking.
- `TRAIN_PER_CLASS`, `VAL_PER_CLASS`, `SEEDS`, `TUNING_SEEDS` — few-shot budget and split
  protocol.
- `PAPER_TUNING_BUDGET`, `FAST_DEBUG_MODE` — Optuna budget `B`, and debug mode for smoke
  tests only.
- `PENNYLANE_DEVICE`, `RUN_PENNYLANE_SANITY_CHECK` — quantum simulator backend and start-up
  check.

---

## 9. Reproducibility

- Seeds are set for Python, NumPy, and PyTorch in `set_seed()`; splits are derived
  deterministically from the seed, and ties in semantic neighbour selection are resolved by
  a fixed ascending vertex order.
- Labels are defensively remapped to contiguous indices; graphs are made undirected,
  unweighted, and self-loop free before operator construction.
- The exact frozen configuration for every reported seed is in
  `selected_hyperparameters.csv`; the full protocol record is in `run_metadata.json`.
- Delete the output directory, or just `affinity_cache/` and `graph_cache/`, to force a cold
  rebuild of all cached operators.

---

## 10. License and contact

License: add a `LICENSE` file before making the repository public.

Corresponding author: Hamed Gholipour — University of Antwerp (MICSS Lab), Flanders Make,
and University of Beira Interior. Issues and questions via the repository issue tracker.
