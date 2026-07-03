"""Core tools for geometric information decomposition (GID) on S^1 and S^2.

The module is intentionally small and pedagogical. It provides reusable
functions for the examples in the notebooks, rather than a full software
package for arbitrary spherical harmonics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize, brentq
from scipy.special import logsumexp, i0
from scipy.stats import chi2


# -----------------------------------------------------------------------------
# Basic weighted empirical utilities
# -----------------------------------------------------------------------------

def normalize_weights(w: ArrayLike) -> np.ndarray:
    """Return nonnegative weights normalized to sum to one."""
    w = np.asarray(w, dtype=float)
    if w.ndim != 1:
        raise ValueError("weights must be one-dimensional")
    if np.any(w < 0):
        raise ValueError("weights must be nonnegative")
    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        raise ValueError("weights must have positive finite sum")
    return w / s


def effective_sample_size(w: ArrayLike) -> float:
    """Return the usual inverse-Simpson effective sample size."""
    w = normalize_weights(w)
    return float(1.0 / np.sum(w ** 2))


def normalize_rows(X: ArrayLike, eps: float = 1e-15) -> np.ndarray:
    """Normalize rows of an array to unit Euclidean norm."""
    X = np.asarray(X, dtype=float)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return X / norms


# -----------------------------------------------------------------------------
# Quadrature and sampling on S^1 and S^2
# -----------------------------------------------------------------------------

def uniform_grid_s1(num_grid: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Equally spaced quadrature grid on S^1 with normalized weights."""
    theta = np.linspace(0.0, 2.0 * np.pi, num_grid, endpoint=False)
    q = np.full(num_grid, 1.0 / num_grid)
    return theta, q


def fibonacci_sphere(num_points: int = 6000) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic approximately uniform points and equal weights on S^2."""
    i = np.arange(num_points, dtype=float)
    z = 1.0 - 2.0 * (i + 0.5) / num_points
    phi = (np.pi * (3.0 - np.sqrt(5.0))) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    q = np.full(num_points, 1.0 / num_points)
    return X, q


def sample_vonmises_s1(n: int, mu: float, kappa: float, rng: np.random.Generator) -> np.ndarray:
    """Sample angles from a von Mises distribution on S^1."""
    return rng.vonmises(mu=mu, kappa=kappa, size=n) % (2.0 * np.pi)


def sample_mixture_vonmises_s1(
    n: int,
    mus: Sequence[float],
    kappas: Sequence[float] | float,
    probs: Optional[Sequence[float]],
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a finite mixture of von Mises distributions on S^1."""
    mus = np.asarray(mus, dtype=float)
    if np.isscalar(kappas):
        kappas = np.full(len(mus), float(kappas))
    else:
        kappas = np.asarray(kappas, dtype=float)
    if probs is None:
        probs = np.full(len(mus), 1.0 / len(mus))
    else:
        probs = normalize_weights(probs)
    z = rng.choice(len(mus), size=n, p=probs)
    theta = np.empty(n)
    for j in range(len(mus)):
        idx = np.where(z == j)[0]
        if idx.size:
            theta[idx] = rng.vonmises(mus[j], kappas[j], size=idx.size)
    return theta % (2.0 * np.pi)


def _basis_from_mu(mu: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return an orthonormal basis (e1, e2, mu) in R^3."""
    mu = np.asarray(mu, dtype=float)
    mu = mu / np.linalg.norm(mu)
    if abs(mu[2]) < 0.9:
        a = np.array([0.0, 0.0, 1.0])
    else:
        a = np.array([1.0, 0.0, 0.0])
    e1 = a - np.dot(a, mu) * mu
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(mu, e1)
    return e1, e2, mu


def sample_vmf_s2(n: int, mu: ArrayLike, kappa: float, rng: np.random.Generator) -> np.ndarray:
    """Sample from a vMF distribution on S^2.

    The density is with respect to normalized uniform surface measure and is
    proportional to exp(kappa * mu^T x).
    """
    mu = np.asarray(mu, dtype=float)
    mu = mu / np.linalg.norm(mu)
    if kappa < 1e-12:
        return normalize_rows(rng.normal(size=(n, 3)))
    u = rng.uniform(size=n)
    # Stable inverse-CDF for z = cos(angle from mu) on [-1, 1].
    z = -1.0 + np.log1p(u * np.expm1(2.0 * kappa)) / kappa
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    radial = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    e1, e2, mu = _basis_from_mu(mu)
    X = z[:, None] * mu[None, :] + radial[:, None] * (
        np.cos(phi)[:, None] * e1[None, :] + np.sin(phi)[:, None] * e2[None, :]
    )
    return normalize_rows(X)


def sample_antipodal_s2(n: int, mu: ArrayLike, kappa: float, rng: np.random.Generator) -> np.ndarray:
    """Equal mixture of two vMF components centered at mu and -mu."""
    mu = np.asarray(mu, dtype=float)
    z = rng.integers(0, 2, size=n)
    X = np.empty((n, 3))
    idx = z == 0
    X[idx] = sample_vmf_s2(idx.sum(), mu, kappa, rng)
    X[~idx] = sample_vmf_s2((~idx).sum(), -mu, kappa, rng)
    return X


def sample_girdle_s2(n: int, axis: ArrayLike, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Approximate girdle distribution concentrated near the equator orthogonal to axis."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    e1, e2, axis = _basis_from_mu(axis)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    c = sigma * rng.normal(size=n)
    X = a[:, None] * e1[None, :] + b[:, None] * e2[None, :] + c[:, None] * axis[None, :]
    return normalize_rows(X)


def sample_tetrahedral_s2(n: int, kappa: float, rng: np.random.Generator) -> np.ndarray:
    """Mixture of four vMF components centered at tetrahedral directions."""
    centers = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
    ], dtype=float)
    centers = normalize_rows(centers)
    z = rng.integers(0, 4, size=n)
    X = np.empty((n, 3))
    for j in range(4):
        idx = z == j
        X[idx] = sample_vmf_s2(idx.sum(), centers[j], kappa, rng)
    return X


# -----------------------------------------------------------------------------
# Feature maps
# -----------------------------------------------------------------------------

def features_s1_fourier(theta: ArrayLike, L: int) -> np.ndarray:
    """Fourier features cos(k theta), sin(k theta), k=1,...,L."""
    theta = np.asarray(theta, dtype=float).reshape(-1)
    cols = []
    for k in range(1, L + 1):
        cols.append(np.cos(k * theta))
        cols.append(np.sin(k * theta))
    return np.column_stack(cols) if cols else np.zeros((theta.size, 0))


def features_s2_linear(X: ArrayLike) -> np.ndarray:
    """Linear coordinate features on S^2."""
    return np.asarray(X, dtype=float)


def features_s2_quadratic_traceless(X: ArrayLike) -> np.ndarray:
    """A five-dimensional basis for traceless quadratic features on S^2."""
    X = np.asarray(X, dtype=float)
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    rt2 = np.sqrt(2.0)
    return np.column_stack([
        x * x - z * z,
        y * y - z * z,
        rt2 * x * y,
        rt2 * x * z,
        rt2 * y * z,
    ])


def features_s2_level12(X: ArrayLike) -> np.ndarray:
    """Concatenate level-1 linear and level-2 traceless quadratic features."""
    return np.column_stack([features_s2_linear(X), features_s2_quadratic_traceless(X)])


# -----------------------------------------------------------------------------
# Maximum-entropy fitting by quadrature
# -----------------------------------------------------------------------------

@dataclass
class MaxEntFit:
    lam: np.ndarray
    psi: float
    target_moment: np.ndarray
    fitted_moment: np.ndarray
    D: float
    success: bool
    message: str
    probabilities_quad: np.ndarray
    density_quad: np.ndarray


def _log_partition(phi_quad: np.ndarray, quad_weights: np.ndarray, lam: np.ndarray):
    logits = phi_quad @ lam
    log_terms = np.log(quad_weights) + logits
    psi = logsumexp(log_terms)
    probs = np.exp(log_terms - psi)
    return psi, probs


def fit_maxent_quadrature(
    phi_data: ArrayLike,
    weights: ArrayLike,
    phi_quad: ArrayLike,
    quad_weights: Optional[ArrayLike] = None,
    ridge: float = 0.0,
    x0: Optional[ArrayLike] = None,
    tol: float = 1e-9,
    max_iter: int = 1000,
) -> MaxEntFit:
    """Fit a finite-dimensional maximum-entropy model by quadrature.

    Parameters
    ----------
    phi_data:
        Feature matrix evaluated at the observed support points.
    weights:
        Empirical probability weights.
    phi_quad:
        Feature matrix evaluated at quadrature points from the reference measure.
    quad_weights:
        Quadrature weights approximating normalized reference volume.
    ridge:
        Optional ridge penalty on natural parameters. Entropy is computed for the
        fitted density, using its fitted moment rather than the target moment.
    """
    phi_data = np.asarray(phi_data, dtype=float)
    phi_quad = np.asarray(phi_quad, dtype=float)
    if phi_data.ndim != 2 or phi_quad.ndim != 2:
        raise ValueError("feature arrays must be two-dimensional")
    if phi_data.shape[1] != phi_quad.shape[1]:
        raise ValueError("data and quadrature features must have the same number of columns")
    w = normalize_weights(weights)
    q = np.full(phi_quad.shape[0], 1.0 / phi_quad.shape[0]) if quad_weights is None else normalize_weights(quad_weights)
    target = w @ phi_data
    p = phi_quad.shape[1]
    if x0 is None:
        x0 = np.zeros(p)
    else:
        x0 = np.asarray(x0, dtype=float)

    def objective(lam):
        psi, _ = _log_partition(phi_quad, q, lam)
        return psi - lam @ target + 0.5 * ridge * np.dot(lam, lam)

    def gradient(lam):
        _, probs = _log_partition(phi_quad, q, lam)
        fit_moment = probs @ phi_quad
        return fit_moment - target + ridge * lam

    res = minimize(
        objective,
        x0,
        jac=gradient,
        method="BFGS",
        options={"gtol": tol, "maxiter": max_iter},
    )
    # In finite quadrature, BFGS can return precision-loss warnings even with a
    # negligible moment residual. Treat the returned point as useful; expose the
    # raw success/message for transparency.
    lam = res.x
    psi, probs = _log_partition(phi_quad, q, lam)
    fit_moment = probs @ phi_quad
    density_quad = np.exp(phi_quad @ lam - psi)  # density relative to reference measure
    D = float(lam @ fit_moment - psi)
    return MaxEntFit(
        lam=lam,
        psi=float(psi),
        target_moment=target,
        fitted_moment=fit_moment,
        D=D,
        success=bool(res.success),
        message=str(res.message),
        probabilities_quad=probs,
        density_quad=density_quad,
    )


def information_gaps(D_values: Sequence[float]) -> np.ndarray:
    """Return successive differences of cumulative deficits with D_0=0."""
    D = np.asarray(D_values, dtype=float)
    if D[0] != 0:
        D = np.r_[0.0, D]
    return np.diff(D)


def effective_uncertainty(D: float | ArrayLike) -> np.ndarray:
    """Return exp(-D), the effective occupied volume fraction."""
    return np.exp(-np.asarray(D, dtype=float))


def fit_fourier_hierarchy_s1(theta, weights, max_L=4, grid_size=2048):
    """Fit Fourier maximum-entropy hierarchy on S^1."""
    theta_grid, q = uniform_grid_s1(grid_size)
    D = [0.0]
    fits = []
    for L in range(1, max_L + 1):
        fit = fit_maxent_quadrature(
            features_s1_fourier(theta, L),
            weights,
            features_s1_fourier(theta_grid, L),
            q,
        )
        fits.append(fit)
        D.append(fit.D)
    I = np.diff(np.array(D))
    return {"D": np.array(D), "I": I, "fits": fits, "grid": theta_grid, "quad_weights": q}


def fit_s2_level12(X, weights, quad_X, quad_weights=None):
    """Fit level 1 and level 2 low-order hierarchy on S^2."""
    fit1 = fit_maxent_quadrature(
        features_s2_linear(X), weights, features_s2_linear(quad_X), quad_weights
    )
    fit2 = fit_maxent_quadrature(
        features_s2_level12(X), weights, features_s2_level12(quad_X), quad_weights
    )
    D = np.array([0.0, fit1.D, fit2.D])
    I = np.diff(D)
    return {"D": D, "I": I, "fits": [fit1, fit2]}


# -----------------------------------------------------------------------------
# Baseline summaries
# -----------------------------------------------------------------------------

def mean_resultant(X: ArrayLike, weights: ArrayLike) -> tuple[np.ndarray, float]:
    """Weighted mean resultant vector and length."""
    X = np.asarray(X, dtype=float)
    w = normalize_weights(weights)
    r = w @ X
    return r, float(np.linalg.norm(r))


def A3(kappa: float) -> float:
    """Mean resultant function for vMF on S^2: coth(kappa) - 1/kappa."""
    kappa = float(kappa)
    if abs(kappa) < 1e-6:
        return kappa / 3.0 - kappa ** 3 / 45.0
    return 1.0 / np.tanh(kappa) - 1.0 / kappa


def estimate_kappa_s2(R: float) -> float:
    """Invert the S^2 vMF mean resultant function."""
    R = float(np.clip(R, 0.0, 0.999999))
    if R < 1e-8:
        return 0.0
    return float(brentq(lambda k: A3(k) - R, 1e-8, 1e4, maxiter=200))


def vmf_entropy_deficit_s2(kappa: float) -> float:
    """KL divergence from uniform for vMF on S^2, with normalized surface measure."""
    kappa = float(kappa)
    if kappa < 1e-8:
        return 0.0
    psi = np.log(np.sinh(kappa) / kappa)
    return float(kappa * A3(kappa) - psi)


def frobenius_anisotropy_s2(X: ArrayLike, weights: ArrayLike) -> float:
    """|| sum_i w_i x_i x_i^T - I/3 ||_F on S^2."""
    X = np.asarray(X, dtype=float)
    w = normalize_weights(weights)
    Q = (X * w[:, None]).T @ X
    return float(np.linalg.norm(Q - np.eye(3) / 3.0, ord="fro"))


# -----------------------------------------------------------------------------
# Density helpers for visualization
# -----------------------------------------------------------------------------

def von_mises_kernel_density_s1(theta_data, weights, theta_grid, kappa=14.0):
    """Weighted von Mises kernel density relative to normalized uniform measure."""
    theta_data = np.asarray(theta_data, dtype=float)
    theta_grid = np.asarray(theta_grid, dtype=float)
    w = normalize_weights(weights)
    C = i0(kappa)  # normalized-uniform density denominator
    dens = np.zeros(theta_grid.size)
    for wi, ti in zip(w, theta_data):
        dens += wi * np.exp(kappa * np.cos(theta_grid - ti)) / C
    return dens


def orthographic_project_s2(X: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Simple orthographic projection using x-y coordinates."""
    X = np.asarray(X, dtype=float)
    return X[:, 0], X[:, 1]


# -----------------------------------------------------------------------------
# Null calibration utilities
# -----------------------------------------------------------------------------

def covariance_under_quadrature(phi_quad: ArrayLike, quad_weights: Optional[ArrayLike] = None) -> np.ndarray:
    """Covariance of features under a quadrature rule."""
    Phi = np.asarray(phi_quad, dtype=float)
    q = np.full(Phi.shape[0], 1.0 / Phi.shape[0]) if quad_weights is None else normalize_weights(quad_weights)
    mu = q @ Phi
    C = ((Phi - mu) * q[:, None]).T @ (Phi - mu)
    return C


def null_gap_quadratic(phi_added_data: ArrayLike, weights: ArrayLike, S: ArrayLike) -> float:
    """Second-order approximation to a null information gap for added features.

    This is 0.5 * m_v^T S^{-1} m_v for a reduced null where the residualized
    added-feature covariance is S. For the uniform S^2 examples with linear
    reduced features, the cross-covariance vanishes and S is the covariance of
    the five traceless quadratic features.
    """
    Phi = np.asarray(phi_added_data, dtype=float)
    w = normalize_weights(weights)
    m = w @ Phi
    Sinv_m = np.linalg.solve(np.asarray(S, dtype=float), m)
    return float(0.5 * m @ Sinv_m)


def moment_covariance_deterministic_weights(phi_data: ArrayLike, weights: ArrayLike) -> np.ndarray:
    """Estimate Sigma for a deterministic-weight CLT with a_n=(sum w_i^2)^-1/2."""
    Phi = np.asarray(phi_data, dtype=float)
    w = normalize_weights(weights)
    mu = w @ Phi
    denom = np.sum(w ** 2)
    return ((Phi - mu) * (w ** 2)[:, None]).T @ (Phi - mu) / denom


def moment_covariance_self_normalized_is(
    phi_data: ArrayLike, weights: ArrayLike, null_mean: Optional[ArrayLike] = None
) -> np.ndarray:
    """Null-specific Sigma estimator for self-normalized importance weights.

    For normalized weights w_i and a_n=sqrt(n), the influence multiplier is
    approximately n*w_i. The centering should be the null/reduced-model mean for
    null calibration.
    """
    Phi = np.asarray(phi_data, dtype=float)
    w = normalize_weights(weights)
    n = Phi.shape[0]
    if null_mean is None:
        null_mean = np.zeros(Phi.shape[1])
    null_mean = np.asarray(null_mean, dtype=float)
    A = (n * w)[:, None] * (Phi - null_mean)
    return (A.T @ A) / n


def weighted_chisq_pvalue(
    observed: float,
    omegas: ArrayLike,
    rng: Optional[np.random.Generator] = None,
    num_sim: int = 20000,
) -> float:
    """Monte Carlo upper-tail probability for sum_j omega_j * chi^2_1."""
    omegas = np.asarray(omegas, dtype=float)
    omegas = omegas[omegas > 1e-12]
    if rng is None:
        rng = np.random.default_rng(0)
    if omegas.size == 0:
        return float(observed <= 0)
    draws = rng.chisquare(df=1.0, size=(num_sim, omegas.size)) @ omegas
    return float(np.mean(draws >= observed))


def quadratic_form_omegas(S: ArrayLike, Sigma: ArrayLike) -> np.ndarray:
    """Eigenvalue weights for 0.5 * S^{-1/2} Sigma S^{-1/2}."""
    S = np.asarray(S, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    evals, evecs = np.linalg.eigh(S)
    Sinvhalf = (evecs / np.sqrt(np.maximum(evals, 1e-15))) @ evecs.T
    M = 0.5 * Sinvhalf @ Sigma @ Sinvhalf
    M = 0.5 * (M + M.T)
    vals = np.linalg.eigvalsh(M)
    return np.maximum(vals, 0.0)


def chi_square_pvalue_for_equal_weights(n: int, I_gap: float, df: int) -> float:
    """p-value for 2 n I_gap against chi-square_df."""
    return float(1.0 - chi2.cdf(2.0 * n * I_gap, df=df))
