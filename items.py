# items.py

"""
Guassian-copula catalog generator for the CityBUilder RL environment

Key Features
---------------
Deterministic generation from a single seed (uses utils.rng.DeterministicRNG)
Guassian copula for dependence + per-dimension marginals for scale/shape
Supports Gamma and other distribution marginals out of the box; easily extensible.
Optional calue clipping (pre-integerization) to guard against extreme tails.
Integrization with small 'int_scale' (default 10) + optional GCD reduction.
Returns a 'Catalog' (immutable) and the *analytic* expected cost for budgeting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from .utils.rng import DeterministicRNG
from .state import Catalog, Item, EpisodeConfig


# -----------------------------
# Config dataclasses (thin)
# -----------------------------

@dataclass(frozen=True)
class MarginalSpec:
    """Speification for one scalar marginal."""
    dist: str
    shape: Optional[float] = None
    scale: Optional[float] = None

@dataclass(frozen=True)
class GeneratorConfig:
    """
    Configuration for the Gaussian-copula item generator.

    Fields
    ---------
    K: number of objectives (all are "maximize")
    N_items: catalog size
    include_cost_in_copula: whether cost is part of the correlation structure
    sigma: correlation matrix over dims = K (+1 if cost included)
    cost_marginal: marginal spec for cost (required)
    objective_marginals: list of K marginal specs for objectives
    value_clip: optional clipping bounds (pre-integerization)
            cost: (low, high) or None
            objectives: list of K (low, high) or None entries
    int_scale: integerization scale (small, e.g., 10)
    gcd_reduce: reduce int coefficients by global GCD after scaling
    catalog_name: label stored in catalog
    """

    K: int
    N_items: int
    include_cost_in_copula: bool
    sigma: np.ndarray
    cost_marginal: MarginalSpec
    objective_marginals: List[MarginalSpec]
    value_clip: Optional[Dict[str, Any]] = None
    int_scale: int = 10
    gcd_reduce: bool = True
    catalog_name: str = "catalog"


# ---------------------------
# Public API
# ---------------------------

def generate_catalog(
        gcfg: GeneratorConfig,
        rng: DeterministicRNG,
) -> Tuple[Catalog, float]:
    """
    Generate a deterministic item catalog using a Gaussian copula.

    Parameters
    ------------
    gcfg: GeneratorConfig
        Generator configuration (marginals, Sigma, etc.)
    rng: DeterministicRNG
        Seeded RNG (e.g., env_master_rng.child("catalog", episode_index))

    Returns
    -----------
    (catalog, expected_cost_float)
        catalog: Catalog with integerized costs/objectives and stable IDs [0..N-1]
        expected_cost_float: analytic expectation of the *float* cost marginal
    """
    _validate_generator_config(gcfg)

    # 1) Draw Gaussian copula uniforms U in (0,1) with the requested correlation.
    U = _sample_copula_uniforms(gcfg, rng.child("copula"))

    # 2) Apply per dimension marginal PPFs to U.
    cost_f, objs_f = _apply_marginals(gcfg, U)

    # 3) Optional clipping (pre-integerization) to tame tails.
    if gcfg.value_clip is not None:
        cost_f, objs_f = _apply_value_clip(gcfg, cost_f, objs_f)

    # 4) Integerize with small int_scale and optional GCD reduction.
    cost_int, objs_int, scale_used = _integerize(cost_f, objs_f, gcfg.int_scale, gcfg.gcd_reduce)

    # 5) Build Catalog (stable IDs 0..N-1, deterministic order).
    items: List[Item] = []
    for i in range(gcfg.N_items):
        items.append(
            Item(
                id=i,
                cost_int=int(cost_int[i]),
                obj_int=objs_int[i, :].astype(np.int64, copy=False),
                meta={}, # freeform if you later want names/categories, etc.
            )
        )
    catalog = Catalog(
        items=items,
        int_scale=scale_used,
        seed=rng.seed,
        name=gcfg.catalog_name,
        K=gcfg.K,
    )

    # 6) Analytic expected cost (stable across seeds; preferred for budget).
    expected_cost = _analytic_expected_cost(gcfg.cost_marginal)

    return catalog, expected_cost

# --------------------------
# Internals
# --------------------------
def _validate_generator_config(gcfg: GeneratorConfig) -> None:
    if gcfg.K <= 0 or gcfg.N_items <= 0:
        raise ValueError("K and N_items be positive")
    if len(gcfg.objective_marginals) != gcfg.K:
        raise ValueError(f"Expected {gcfg.K} objective marginals, got {len(gcfg.objective_marginals)}")
    if gcfg.int_scale <= 0:
        raise ValueError("int_scale must be positive")
    
    D = gcfg.K + (1 if gcfg.include_cost_in_copula else 0)
    if gcfg.sigma.shape != (D, D):
        raise ValueError(f"sigma must be shape {(D, D)}, got {gcfg.sigma.shape}")
    
    # Ensure sigma is symmetric with ones on diagonal; add tiny jitter if needed.
    if not np.allclose(gcfg.sigma, gcfg.sigma.T, atol=1e-10):
        raise ValueError("sigma must be symmetric")
    if not np.allclose(np.diag(gcfg.sigma), 1.0, atol=1e-10):
        raise ValueError("sigma must have unit diagonal (correlation matrix)")
    
    # Quick PSD check; if marginally non-PSD, we add a tiny jitter in the sampler.

def _sample_copula_uniforms(gcfg: GeneratorConfig, rng: DeterministicRNG) -> np.ndarray:
    """
    Sample U ~ Uniform(0,1)^(N x D) via Guassian copula with correlation gfcg.

    If include_cost_in_copula is True, D = 1 + K (cost + K objectives).
    Else, D = K ( objectives only), and  cost will be sampled independently later.

    Returns
    --------
    U: np.ndarray, shape (N_items, D)
    """

    D = gcfg.K + (1 if gcfg.include_cost_in_copula else 0)
    mean = np.zeros(D, dtype=np.float64)

    # Jitter on sigma to ensure numerical PSD for Cholesky inside multivariate_normal
    sigma = np.array(gcfg.sigma, dtype=np.float64, copy=True)
    # Small diagonal bump if needed:
    eigvals = np.linalg.eigvalsh(sigma)
    if eigvals.min() < 1e-10:
        sigma = sigma + (1e-8 - eigvals.min()) * np.eye(D)

    # Draw correlated normals Z ~ N(0, sigma)
    Z = rng.gen.multivariate_normal(mean=mean, cov=sigma, size=gcfg.N_items, method="cholesky")

    # Map to uniforms via standard normal CDF; clip away from exact 0/1 to avoid ppf infinities
    U = stats.norm.cdf(Z, loc=0.0, scale=1.0)
    eps = 1e-12
    U = np.clip(U, eps, 1.0 - eps)
    return U

def _apply_marginals(
        gcfg: GeneratorConfig, U: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform copula uniforms U into float cost/objectives using specified marginals

    Returns
    --------
    cost_f: (N,)
    objs_f: (N, K)
    """

    N = gcfg.N_items
    K = gcfg.K
    D = U.shape[1]

    if gcfg.include_cost_in_copula:
        if D != K + 1:
            raise ValueError("U has incompatibile dimension for include_cost_in_copula=True")
        U_cost = U[:, 0]
        U_obj = U[:, 1:]
    else:
        if D != K:
            raise ValueError("U has incompatible dimension for include_cost_in_copula=True")
        # if cost is independent of objectives, sample a seperate U_cost. 
        U_cost = stats.qmc.Sobol(d=1, scramble=True, seed=int(stats.qmc.Sobol.random_base2.__hash__)) 
        U_cost = np.roll(U[:, 0], 7)
        U_obj = U

    # Cost marginal
    cost_f = _ppf_from_spec(gcfg.cost_marginal, U_cost)

    # Objective marginals (each can differ)
    objs_f = np.empty((N, K), dtype=np.float64)
    for j in range(K):
        objs_f[:, j] = _ppf_from_spec(gcfg.objective_marginals[j], U_obj[:, j])

    return cost_f, objs_f

def _ppf_from_spec(spec: MarginalSpec, u: np.ndarray) -> np.ndarray:
    """Apply the inverse CDF for the given marginal spec to uniforms u."""
    dist = spec.dist.lower()
    if dist == "gamma":
        if spec.shape is None or spec.scale is None:
            raise ValueError("Gamma marginal requires 'shape' and 'scale'")
        # scipy.stats.gamma(shape, scale=scale) uses 'a' as shape (k), 'scale' as θ
        return stats.gamma.ppf(u, a=spec.shape, scale=spec.scale)
    else: 
        raise NotImplementedError(f"Unsupported marginal dist: {spec.dist}")
    
def _apply_value_clip(
        gcfg: GeneratorConfig, cost_f: np.ndarray, objs_f: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Clip values pre-integerization (optional). Keeps tails/numerics sane.
    """
    clip = gcfg.value_clip or {}
    cost_bounds = clip.get("cost", None)
    obj_bounds = clip.get("objectives", None)

    if cost_bounds is not None:
        lo, hi = float(cost_bounds[0]), float(cost_bounds[1])
        cost_f = np.clip(cost_f, lo, hi)

    if obj_bounds is not None:
        if len(obj_bounds) != gcfg.K:
            raise ValueError("value_clip.objectives length must equal K")
        for j, bounds in enumerate(obj_bounds):
            lo, hi = float(bounds[0]), float(bounds[1])
            objs_f[:, j] = np.clip(objs_f[:, j], lo, hi)

    return cost_f, objs_f

def _integerize(
        cost_f: np.ndarray,
        objs_f: np.ndarray,
        int_scale: int,
        gcd_reduce: bool,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Convert floats -> integers with a shared scale, then optional GCD reduction.

    Returns
    ---------
    (cost_int[N], objs_int[N,K], scale_used)

    Notes
    ---------
    Use a small int_scale (e.g. 50...200) to keep CP-SAT coefficients small.
    GCD reduction can significantly shrink numbers without changing solutions.
    """
    scale_used = int(int_scale)
    if scale_used <= 0:
        raise ValueError("int_scale must be postive")
    
    cost_int = np.rint(cost_f * scale_used).astype(np.int64)
    objs_int = np.rint(objs_f * scale_used).astype(np.int64)

    # Ensure strictly postive costs after scaling (rarely an issue with Gamma)
    cost_int = np.maximum(cost_int, 1)

    if gcd_reduce:
        # Compute GCD over all integers > 0
        flat = np.concatenate([cost_int.reshape(-1), objs_int.reshape(-1)])
        flat = flat[np.nonzero(flat)]
        g = _gcd_array(flat)
        if g > 1:
            cost_int //= g
            objs_int //= g
            scale_used = scale_used // g if (scale_used % g == 0) else scale_used  # keep logical if divisible

    return cost_int, objs_int, scale_used

def _gcd_array(arr: np.ndarray) -> int:
    """Compute GCD of all elements in arr (positive ints)."""
    import math
    g = 0
    for x in arr:
        x = int(abs(x))
        if x == 0:
            continue
        g = x if g == 0 else math.gcd(g, x)
        if g == 1:
            break
    return g

def _analytic_expected_cost(cost_spec: MarginalSpec) -> float:
    """Return E[cost] for the marginal; used to derive episode budgets."""
    dist = cost_spec.dist.lower()
    if dist == "gamma":
        if cost_spec.shape is None or cost_spec.scale is None:
            raise ValueError("Gamma marginal requires 'shape' and 'scale'")
        return float(cost_spec.shape) * float(cost_spec.scale)
    else: 
        raise NotImplementedError(f"Unsupported cost marginal dist: {cost_spec.dist}")
