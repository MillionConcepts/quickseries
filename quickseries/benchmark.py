import warnings
from inspect import getfullargspec
from itertools import product
from time import time
import timeit
from typing import Union, Sequence, Optional

import numpy as np
import sympy as sp
from dustgoggles.func import gmap

from quickseries import quickseries
from quickseries.approximate import _makebounds
from quickseries.sputils import lambdify, LmSig


def _offset_check_cycle(
    absdiff: float,
    frange: tuple[float, float],
    lamb: LmSig,
    quick: LmSig,
    vecs: Sequence[np.ndarray],
    worstpoint: Optional[list[float]],
) -> tuple[float, float, float, tuple[float, float], list[float], bool, bool]:
    approx_y = quick(*vecs)
    orig_y = lamb(*vecs)
    frange = (min(orig_y.min(), frange[0]), max(orig_y.max(), frange[1]))
    offset = abs(approx_y - orig_y)
    worstix = np.argmax(offset)
    if (new_absdiff := offset[worstix]) > absdiff:
        absdiff = new_absdiff
        worstpoint = [v[worstix] for v in vecs]
    illposed = not bool(np.isfinite(orig_y).all())
    misfit = not bool(np.isfinite(approx_y).all()) and not illposed
    return (
        absdiff,
        np.median(offset),
        np.mean(offset**2),
        frange,
        worstpoint,
        illposed,
        misfit,
    )


def benchmark(
    func: Union[str, sp.Expr, sp.core.function.FunctionClass],
    offset_resolution: int = 10000,
    n_offset_shuffles: int = 50,
    timeit_cycles: int = 20000,
    testbounds="equal",
    cache: bool = False,
    **quickkwargs,
) -> dict[str, sp.Expr | float | np.ndarray | str | list[float]]:
    lamb = lambdify(func)
    compile_start = time()
    to_fire = lambda: quickseries(
        func, **(quickkwargs | {"extended_output": True, "cache": cache})
    )
    result, gen_warnings = trap_runtime_warnings(to_fire)
    if result is None:  # unhandled exception
        return {"exception": gen_warnings[0]}
    quick, ext = result
    ext["diverged"] = any(
        "overflow" in w or "divide" in w or "zero" in w or "invalid" in w
        for w in gen_warnings
    )
    gentime = time() - compile_start
    if testbounds == "equal":
        testbounds, _ = _makebounds(
            quickkwargs.get("bounds"), len(getfullargspec(lamb).args), None
        )
    vecs = [np.linspace(*b, offset_resolution) for b in testbounds]
    if (pre := quickkwargs.get("precision")) is not None:
        vecs = gmap(lambda arr: arr.astype(getattr(np, f"float{pre}")), vecs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if len(testbounds) > 1:
            absdiff, frange, illposed, mediff, misfit, mse, worstpoint = (
                _check_offsets_nd(lamb, n_offset_shuffles, quick, vecs)
            )
        # no point in shuffling for 1D -- we're doing that for > 1D
        # because it becomes quickly unreasonable in terms of memory
        # to be exhaustive, but this _is_ exhaustive for 1D
        else:
            absdiff, frange, illposed, mediff, misfit, mse, worstpoint = (
                check_offsets_1d(lamb, quick, vecs)
            )
        # TODO: should probably permit specifying numba signatures for jitted
        #  functions -- both here and in primary quickseries().
        approx_time = timeit.timeit(lambda: quick(*vecs), number=timeit_cycles)
        orig_time = timeit.timeit(lambda: lamb(*vecs), number=timeit_cycles)
        orig_s = orig_time / timeit_cycles
        approx_s = approx_time / timeit_cycles
        poly = ext["expr"].as_poly()
        if poly is not None:
            try:
                coeffs = poly.all_coeffs()
                nonzero = sum(1 for c in coeffs if not c.equals(0))
                ext |= {
                    "poly_degree": poly.degree(),
                    "nonzero_coeffs": nonzero,
                    "coeff_sparsity": nonzero / (poly.degree() + 1),
                }
            except sp.polys.polyerrors.PolynomialError:
                # all_coeffs() does not support multivariate polynomials
                pass
        return {
            "absdiff": float(absdiff),
            "reldiff": float(absdiff / np.ptp(frange)),
            "mediff": float(mediff),
            "mse": float(mse),
            "worstpoint": list(map(float, worstpoint)),
            "range": tuple(map(float, frange)),
            "orig_s": orig_s,
            "approx_s": approx_s,
            "timeratio": approx_s / orig_s,
            "gentime": gentime,
            "polyfunc": quick,
            "illposed": illposed,
            "misfit": misfit,
        } | ext


def check_offsets_1d(lamb, quick, vecs):
    approx_y, orig_y = quick(*vecs), lamb(*vecs)
    frange = (orig_y.min(), orig_y.max())
    offset = abs(approx_y - orig_y)
    worstix = np.argmax(offset)
    absdiff = offset[worstix]
    mediff = np.median(offset)
    mse = np.mean(offset**2)
    worstpoint = [vecs[0][worstix]]
    illposed = not bool(np.isfinite(orig_y).all())
    misfit = not bool(np.isfinite(approx_y).all()) and not illposed
    return absdiff, frange, illposed, mediff, misfit, mse, worstpoint


def _check_offsets_nd(lamb, n_offset_shuffles, quick, vecs):
    # always check the extrema of the bounds
    extrema = [[] for _ in vecs]
    for p in product((-1, 1), repeat=len(vecs)):
        for i, side in enumerate(p):
            extrema[i].append(vecs[i][side])
    extrema = [np.array(e) for e in extrema]
    absdiff, _, __, frange, worstpoint, illposed, misfit = _offset_check_cycle(
        0, (np.inf, -np.inf), lamb, quick, extrema, None
    )
    medians, mses = [], []
    for _ in range(n_offset_shuffles):
        gmap(np.random.shuffle, vecs)
        absdiff, mediff, mse, frange, worstpoint, ip, mf = _offset_check_cycle(
            absdiff, frange, lamb, quick, vecs, worstpoint
        )
        illposed = ip or illposed
        misfit = mf or misfit
        medians.append(mediff)
        mses.append(mse)
    mediff, mse = np.median(medians), np.median(mses)
    return absdiff, frange, illposed, mediff, misfit, mse, worstpoint


def trap_runtime_warnings(fn):
    with warnings.catch_warnings(record=True) as wlog:
        warnings.simplefilter("always")
        try:
            result = fn()
        except Exception as e:
            return None, [e]
        messages = [str(w.message) for w in wlog]
        return result, messages


def benchmark_multi(
    func,
    bounds,
    param_grid: dict[str, list],
    fixed: dict = None,
    keyfn = lambda opts: tuple((k, v) for k, v in opts.items())
):
    if "bounds" in param_grid:
        raise TypeError("Do not specify bounds in param_grid")
    fixed = {} if fixed is None else fixed
    results = {}
    for combo in product(*param_grid.values()):
        opts = dict(zip(param_grid.keys(), combo)) | fixed
        key = keyfn(opts)
        try:
            results[key] = benchmark(func, bounds=bounds, **opts)
        except Exception as ex:
            results[key] = ex
    return results
