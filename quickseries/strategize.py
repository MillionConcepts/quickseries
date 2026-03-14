import dataclasses as dc
from functools import partial
from numbers import Real
from typing import Any
import warnings

import numpy as np
import sympy as sp

from quickseries import quickseries
from quickseries.sputils import lambdify


def safe_total_degree(expr: sp.Expr):
    """
    Computes total polynomial degree for a single term,
    without dying when passed monomials like x**3*y**2 or x*y*z.
    """
    if expr.is_Number:
        return 0
    poly = expr.as_poly()
    if poly is None:
        return 0  # Not a polynomial (shouldn't happen here)
    return poly.total_degree()


@dc.dataclass(frozen=True)
class Strategy:
    name: str
    # TODO: implement per-axis basis
    basis: str
    bounds: tuple[tuple[float, float]]
    nterms: int
    point: Real | None = None

    def sub(self, **kwargs):
        return Strategy(**(dc.asdict(self) | kwargs))


def get_candidate_strategies(
    expr: str | sp.Expr,
    bounds=None,
    point=None,
    nterms=20,
):
    if isinstance(expr, str):
        expr = sp.sympify(expr)
    syms = sorted(expr.free_symbols, key=str)
    ndim = len(syms)
    if bounds is None:
        bounds = [(-1, 1)] * ndim
    elif isinstance(bounds[0], (int, float)) and ndim > 1:
        bounds = [bounds] * ndim
    elif isinstance(bounds[0], Real):
        bounds = [bounds]
    lbounds = [(a - abs(a) * 0.1, b + abs(b) * 0.1) for a, b in bounds]

    strategies = [
        Strategy("legendre", "legendre", tuple(bounds), nterms),
        # Strategy("legendre_loose", "legendre", tuple(lbounds), nterms),
        Strategy("chebyshev", "chebyshev", tuple(bounds), nterms),
        Strategy("taylor", "taylor", tuple(bounds), nterms, point),
    ]
    return strategies


def error_metrics(y_true, y_pred, eps_rel=1e-12, eps_abs=1e-8):
    ymax = np.max(np.abs(y_true))
    abs_err = np.abs(y_pred - y_true)
    scale = max(1e-8, 10 ** np.floor(np.log10(ymax + 1)))
    abs_maxerr = np.max(abs_err)
    abs_mederr = np.median(abs_err)
    abs_maxerr_filt = np.percentile(abs_err, 99)
    return {
        "ymax": float(ymax),
        "ymin": float(np.min(np.abs(y_true))),
        "norm_maxerr": float(abs_maxerr / scale),
        "norm_mederr": float(abs_mederr / scale),
        "norm_maxerr_filt": float(abs_maxerr_filt / scale),
        "abs_maxerr": float(abs_maxerr),
        "abs_mederr": float(abs_mederr),
        "abs_maxerr_filt": float(abs_maxerr_filt),
        "mse": float(np.mean(abs_err**2)),
    }


def benchmark_strategy(
    expr: sp.Expr,
    strategy,
    sample_points: int = 10000,
    timeit_cycles: int = 0,
    **quickkwargs,
) -> dict[str, Any]:
    expr = sp.sympify(expr) if isinstance(expr, str) else expr
    syms = sorted(expr.free_symbols, key=str)
    ndim = len(syms)
    f_true = lambdify(expr)
    result_obj = {
        "strategy": strategy,
        "quickkwargs": quickkwargs,
        "error": None,
    }
    kwargs = {
        "basis": strategy.basis,
        "bounds": strategy.bounds,
        "nterms": strategy.nterms,
        "point": strategy.point,
        "cache": False,
        "extended_output": True,
    } | quickkwargs
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Covariance")
        try:
            result = quickseries(expr, **kwargs)
        except Exception as ex:
            return result_obj | {"error": f"{type(ex)}: {ex}"}
    fn_approx, metadata = result
    pts_per_ax = round(sample_points ** (1 / ndim))
    grid = [np.linspace(a, b, pts_per_ax) for a, b in strategy.bounds]
    mesh = np.meshgrid(*grid, indexing="ij")
    coords = np.stack([m.flatten() for m in mesh], axis=1)
    # recomputing true, min, max, etc. on every call is inefficient
    # but probably not to the extent that it matters
    # TODO, probably: we shoulld really be filtering to finite values
    #  here, right? There are functions with poles or undefined regions
    #  we might be able to approximate across under some circumstances
    y_true = f_true(*coords.T)
    y_pred = fn_approx(*coords.T)
    result_obj |= {
        "fn": fn_approx,
        "poly": metadata["expr"],
        "nprune": metadata.get("nprune", 0),
        **error_metrics(y_true, y_pred),
    }
    if timeit_cycles > 0:
        import timeit

        result_obj["time_approx"] = timeit.timeit(
            lambda: fn_approx(*coords.T), number=timeit_cycles
        )
        result_obj["time_true"] = timeit.timeit(
            lambda: f_true(*coords.T), number=timeit_cycles
        )
        result_obj["timeratio"] = (
            result_obj["time_approx"] / result_obj["time_true"]
        )
    return result_obj


def check_error_criteria(metrics, **reqs):
    for k, v in reqs.items():
        if v is None:
            continue
        if metrics[k] > v:
            return False
    return True


class ThresholdSweeper:
    def __init__(
        self,
        expr,
        strategy,
        f_true=None,
        bounds=None,
        sample_points=10000,
        abs_max_tol=None,
        abs_med_tol=None,
        norm_max_tol=1e-4,
        norm_med_tol=1e-6,
        max_nterms=28,
        patience=3,
        rel_plateau=5e-5,
        max_runtime_ratio=3,
        min_nterms=4,
        persistence=1,
        **quickkwargs,
    ):
        expr = sp.sympify(expr) if isinstance(expr, str) else expr
        self.expr = expr
        self.strategy = strategy
        self.f_true = f_true or lambdify(expr)
        self.bounds = bounds or strategy.bounds
        self.sample_points = sample_points
        self.max_nterms = max_nterms
        self.patience = patience
        self.rel_plateau = rel_plateau
        self.max_runtime_ratio = max_runtime_ratio
        self.min_nterms = min_nterms
        self.results = []
        self.quickkwargs = quickkwargs
        self.persistence = persistence
        self.status = "pending"
        self.best_err = None
        self.error_criteria = {
            "abs_maxerr": abs_max_tol,
            "abs_mederr": abs_med_tol,
            "norm_maxerr": norm_max_tol,
            "norm_mederr": norm_med_tol,
        }
        self.error_checker = partial(
            check_error_criteria, **self.error_criteria
        )

    def run(self):
        fail_count = 0
        persist_count = 0
        error_count = 0
        self.best_err = float("inf")
        self.status = "incomplete"
        for n in range(self.min_nterms, self.max_nterms + 1):
            print(f"{n}...", end="")
            result = benchmark_strategy(
                self.expr,
                self.strategy.sub(nterms=n),
                sample_points=self.sample_points,
                timeit_cycles=1,
                **self.quickkwargs,
            )
            self.results.append(result)
            if result["error"] is not None:
                result["status"] = "error"
                error_count += 1
                if error_count > 2:
                    self.status = "err_ceiling"
                    return self.results
                continue
            timeratio = result.get("timeratio", 1.0)
            # begin exit plan if error acceptable
            if self.error_checker(result) and timeratio < 1:
                result["status"] = "success"
                self.status = "success"
            elif not self.error_checker(result):
                result["status"] = "poor_fit"
            else:
                result["status"] = "slow"
            # Plateau detection
            maxerr = result["norm_maxerr"]
            improved = maxerr < self.best_err * (1 + self.rel_plateau)
            if improved:
                self.best_err = maxerr
            if improved and (timeratio < 1):
                failcount = 0
            else:
                fail_count += 1
            # Bail if it’s getting worse or too slow
            if fail_count >= self.patience and timeratio > 1:
                self.status = "perf_wall"
                break
            elif fail_count >= self.patience:
                self.status = "diverging"
                break
            if timeratio > self.max_runtime_ratio:
                self.status = "perf_floor"
                break
            if self.status == "success" and persist_count >= self.persistence:
                break
            elif self.status == "success":
                persist_count += 1
        return self.results


def build_scanners(fn, strategies, **score_kwargs):
    return [ThresholdSweeper(fn, s, **score_kwargs) for s in strategies]


def scan_strategies(scanners):
    results = []
    for scanner in scanners:
        print(scanner.strategy.name, end=": ")
        results += scanner.run()
        print()
    return results


def score_evaluations(results, weights=(0.5, 0.5), epsilon=1e-16):
    """
    Compute composite scores for evaluation results.

    Parameters:
        results: dict[strategy_name, result_dict]
        weights: (w_err, w_speed), sum to 1
        epsilon: small constant to avoid log(0)

    Returns:
        List of dicts with original result + score metadata, sorted by score ascending.
    """
    filtered = [r for r in results if r.get("status") == "success"]

    if not filtered:
        return []

    # error and speed metrics
    e_max_all = np.array(
        [np.log10(r["norm_maxerr"] + epsilon) for r in filtered]
    )
    e_med_all = np.array(
        [np.log10(r["norm_mederr"] + epsilon) for r in filtered]
    )
    speeds = np.array([r["timeratio"] for r in filtered])

    # normalized error scores
    e_max_norm = (e_max_all - e_max_all.min()) / (
        e_max_all.max() - e_max_all.min() + epsilon
    )
    e_med_norm = (e_med_all - e_med_all.min()) / (
        e_med_all.max() - e_med_all.min() + epsilon
    )
    err_scores = 0.7 * e_max_norm + 0.3 * e_med_norm

    # normalized speed scores
    speed_clamped = np.clip(speeds, 0.1, 10.0)
    speed_scores = (speed_clamped - speed_clamped.min()) / (
        speed_clamped.max() - speed_clamped.min() + epsilon
    )

    # combined score (lower is better)
    w_err, w_speed = weights
    final_scores = w_err * err_scores + w_speed * speed_scores

    scored = []
    for r, score, es, ss in zip(
        filtered, final_scores, err_scores, speed_scores
    ):
        entry = r.copy()
        entry["score"] = score
        entry["err_score"] = es
        entry["speed_score"] = ss
        scored.append(entry)

    return sorted(scored, key=lambda x: x["score"])
