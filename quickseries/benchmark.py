import timeit
from inspect import getfullargspec
from itertools import product
from typing import Union

import numpy as np
import sympy as sp
from dustgoggles.func import gmap

from quickseries import quickseries
from quickseries.approximate import _makebounds
from quickseries.sputils import lambdify


def _offset_check_cycle(
    absdiff,
    frange,
    lamb,
    quick,
    vecs,
    worstpoint,
):
    approx_y, orig_y = quick(*vecs), lamb(*vecs)
    frange = (min(orig_y.min(), frange[0]), max(orig_y.max(), frange[1]))
    offset = abs(approx_y - orig_y)
    worstix = np.argmax(offset)
    if (new_absdiff := offset[worstix]) > absdiff:
        absdiff = new_absdiff
        worstpoint = [v[worstix] for v in vecs]
    return absdiff, np.median(offset), np.mean(offset ** 2), frange, worstpoint


def benchmark(
    func: Union[str, sp.Expr, sp.core.function.FunctionClass],
    offset_resolution: int = 10000,
    n_offset_shuffles: int = 50,
    timeit_cycles: int = 20000,
    testbounds="equal",
    cache: bool = False,
    **quickkwargs
):
    lamb = lambdify(func)
    quick, ext = quickseries(
        func, **(quickkwargs | {'extended_output': True, 'cache': cache})
    )
    if testbounds == "equal":
        testbounds, _ = _makebounds(
            quickkwargs.get("bounds"), len(getfullargspec(lamb).args), None
        )
    vecs = [np.linspace(*b, offset_resolution) for b in testbounds]
    if (pre := quickkwargs.get("precision")) is not None:
        vecs = gmap(
            lambda arr: arr.astype(getattr(np, f"float{pre}")), vecs
        )
    if len(testbounds) > 1:
        # always check the extrema of the bounds
        extrema = [[] for _ in vecs]
        for p in product((-1, 1), repeat=len(vecs)):
            for i, side in enumerate(p):
                extrema[i].append(vecs[i][side])
        extrema = [np.array(e) for e in extrema]
        absdiff, _, __, frange, worstpoint = _offset_check_cycle(
            0, (np.inf, -np.inf), lamb, quick, extrema, None
        )
        medians, mses = [], []
        for _ in range(n_offset_shuffles):
            gmap(np.random.shuffle, vecs)
            absdiff, mediff, mse, frange, worstpoint = _offset_check_cycle(
                absdiff, frange, lamb, quick, vecs, worstpoint
            )
            medians.append(mediff)
            mses.append(mse)
        mediff, mse = np.median(medians), np.median(mses)
    # no point in shuffling for 1D -- we're doing that for > 1D
    # because it becomes quickly unreasonable in terms of memory
    # to be exhaustive, but this _is_ exhaustive for 1D
    else:
        approx_y, orig_y = quick(*vecs), lamb(*vecs)
        frange = (orig_y.min(), orig_y.max())
        offset = abs(approx_y - orig_y)
        worstix = np.argmax(offset)
        absdiff = offset[worstix]
        mediff = np.median(offset)
        mse = np.mean(offset ** 2)
        worstpoint = [vecs[0][worstix]]
        del offset, orig_y, approx_y
    # TODO: should probably permit specifying dtype for jitted
    #  functions -- both here and in primary quickseries().
    approx_time = timeit.timeit(lambda: quick(*vecs), number=timeit_cycles)
    orig_time = timeit.timeit(lambda: lamb(*vecs), number=timeit_cycles)
    orig_s = orig_time / timeit_cycles
    approx_s = approx_time / timeit_cycles
    return {
        'absdiff': absdiff,
        'reldiff': absdiff / np.ptp(frange),
        'mediff': mediff,
        'mse': mse,
        'worstpoint': worstpoint,
        'range': frange,
        'orig_s': orig_s,
        'approx_s': approx_s,
        'timeratio': approx_s / orig_s,
        'quick': quick
    } | ext
