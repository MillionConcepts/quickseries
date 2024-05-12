from inspect import getfullargspec, signature
from itertools import chain, product
import re
import timeit
from typing import Literal, Optional, Sequence, Union

from cytoolz import groupby
from dustgoggles.dynamic import compile_source, define, getsource
from dustgoggles.func import gmap
import numpy as np
import sympy as sp

from quickseries.expansions import series_lambda, multivariate_taylor
from quickseries.simplefit import fit
from quickseries.sourceutils import lastline
from quickseries.sputils import LmSig, lambdify

"""signature of sympy-lambdified numpy/scipy functions"""

EXP_PATTERN = re.compile(r"\w+ ?\*\* ?(\d+)")
"""what exponentials in sympy-lambdified functions look like"""


def is_simple_poly(expr: sp.Expr) -> bool:
    gens = sp.poly_from_expr(expr)[1]['gens']
    return all(isinstance(g, sp.Symbol) for g in gens)


def regexponents(text: str) -> tuple[int]:
    # noinspection PyTypeChecker
    return tuple(map(int, re.findall(EXP_PATTERN, text)))


def optimize_exponents(
    exps: Sequence[int]
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    # list of tuples like: (factor, [factors to use in decomposition])
    replacements = [(e, [e]) for e in exps]
    # which factors have we already assessed?
    reduced = set()
    # which factors have we used in decompositions?
    extant = tuple(chain(*[r[1] for r in replacements]))
    while True:
        if len(extant) == 1:  # trivial case
            replacements[0][1][:] = [1 for _ in range(replacements[0][0])]
            break
        counts = {
            k: len(v) for k, v in groupby(lambda x: x, extant).items()
            if k not in reduced
        }
        if len(counts) < 2:  # nothing useful left to do
            break
        elif counts[max(counts)] > 1:
            # don't decompose the biggest factor; because it appears more than
            # once, we'd like to precompute it
            reduced.add(max(counts))
            continue
        else:
            # otherwise, do a decomposition pass with the smallest factor
            factor = sorted(counts.keys())[0]
        for k, v in replacements:
            factorization = []
            # "divide out" `factor` from elements of existing decomposition
            for f in v:
                # don't decompose factors we've already evaluated, and don't
                # try to divide `factor` out of smaller factors (nonsensical)
                if f in reduced or f <= factor:
                    factorization.append(f)
                    continue
                factorization.append(factor)
                difference = f - factor
                while difference >= max([e for e in extant if e != f]):
                    factorization.append(factor)
                    difference = difference - factor
                if difference > 0:
                    factorization.append(difference)
            v[:] = factorization
        reduced.add(factor)
        extant = tuple(chain(*[r[1] for r in replacements]))
    # this is a kind of set operation: we no longer care about number of
    # occurrences
    replacements = {k: v for k, v in replacements}
    # figure out which factors we'd like to predefine as variables, and what
    # the "building blocks" of those variables are. 1 is a placeholder: we
    # will never define it, but it's useful in this loop.
    variables = {1: [1]}
    for e in sorted(set(extant)):
        if e == 1:
            continue
        if exps.count(e) == 1:
            if not any(k > e for k, v in replacements.items()):
                continue
        vfactor, remainder = [], e
        while remainder > 0:
            pick = max([v for v in variables.keys() if v <= remainder])
            vfactor.append(pick)
            remainder -= pick
        variables[e] = vfactor
    # remove the placeholder first-order variable
    variables.pop(1)
    return replacements, variables


def force_line_precision(line: str, precision: Literal[16, 32, 64]) -> str:
    constructor_rep = f"numpy.float{precision}"
    constructor = getattr(np, f"float{precision}")
    last, out = 0, ""
    for match in re.finditer(
        r"([+* (-]+|^)([\d.]+)(e[+\-]?\d+)?.*?([+* )]|$)", line
    ):
        out += line[last:match.span()[0]]
        # don't replace exponents
        if match.group(1) == "**":
            out += line[slice(*match.span())]
        else:
            # NOTE: casting number to string within the f-string statement
            # appears to upcast it before generating the representation.
            number = str(constructor(float(match.group(2))))
            out += f"{match.group(1)}{constructor_rep}({number}"
            if match.group(3) is not None:  # scientific notation
                out += match.group(3)
            out += f"){match.group(4)}"
        last = match.span()[1]
    return out + line[last:]


def rewrite(
    poly_lambda: LmSig,
    precompute: bool = True,
    precision: Optional[Literal[16, 32, 64]] = None
) -> tuple[LmSig, str]:
    if precompute is False and precision is None:
        return poly_lambda, getsource(poly_lambda)
    # name of arguments to the lambdified function
    free = getfullargspec(poly_lambda).args
    lines = [f"def _lambdifygenerated({', '.join(free)}):"]
    # sympy will always place this on a single line; it includes
    # the Python expression form of the hornerized polynomial
    # and a return statement; lastline() strips it
    polyexpr = lastline(poly_lambda)
    # remove pointless '1.0' terms
    polyexpr = re.sub(r"(?:\*+)?1\.0\*+", "", polyexpr)
    if precompute is True:
        polyexpr, lines = _rewrite_precomputed(polyexpr, free, lines)
    if precision is not None:
        polyexpr = force_line_precision(polyexpr, precision)
    lines.append(f"    return {polyexpr}")
    source = "\n".join(lines)
    # noinspection PyUnresolvedReferences
    opt = define(compile_source(source), poly_lambda.__globals__)
    opt.__doc__ = ("\n".join(map(str.strip, lines[1:])))
    return opt, source


def _rewrite_precomputed(polyexpr, free, lines):
    # replacements: what factors we will decompose each factor into
    # free: which factors we will define as variables, and their
    # "building blocks"
    for f in free:
        expat = re.compile(rf"{f}+ ?\*\* ?(\d+)")
        replacements, variables = optimize_exponents(
            gmap(int, expat.findall(polyexpr))
        )
        for k, v in variables.items():
            multiplicands = []
            for power in v:
                if power == 1:
                    multiplicands.append(f)
                else:
                    multiplicands.append(f"{f}{power}")
            lines.append(f"    {f}{k} = {'*'.join(multiplicands)}")
        for k, v in replacements.items():
            substitution = '*'.join([f"{f}{r}" if r != 1 else f for r in v])
            polyexpr = polyexpr.replace(f"{f}**{k}", substitution)
    return polyexpr, lines


def _pvec(bounds, offset_resolution):
    axes = [np.linspace(*b, offset_resolution) for b in bounds]
    indices = map(np.ravel, np.indices([offset_resolution for _ in bounds]))
    return [j[i] for j, i in zip(axes, indices)]


def _perform_series_fit(
    func, bounds, nterms, fitres, point, apply_bounds, is_poly
):
    if (len(bounds) == 1) and (is_poly is False):
        approx, expr = series_lambda(func, point[0], nterms, True)
    else:
        approx, expr = multivariate_taylor(func, point, nterms, True)
    lamb, vecs = lambdify(func), _pvec(bounds, fitres)
    try:
        dep = lamb(*vecs)
    except TypeError as err:
        # this is a potentially slow but unavoidable case
        if "converted to Python scalars" not in str(err):
            raise
        dep = np.array([lamb(v) for v in vecs])
    guess = [
        1 for _ in range(len(signature(approx).parameters) - len(vecs))
    ]
    params, _ = fit(
        func=approx,
        vecs=vecs,
        dependent_variable=dep,
        guess=guess,
        bounds=(-5, 5) if apply_bounds is True else None
    )
    # insert coefficients into polynomial
    expr = expr.subs({f'a_{i}': coef for i, coef in enumerate(params)})
    return expr, params


def quickseries(
    func: Union[str, sp.Expr, sp.core.function.FunctionClass],
    bounds: tuple[float, float] = (-1, 1),
    nterms: int = 9,
    point: Optional[float] = None,
    fitres: int = 100,
    prefactor: Optional[bool] = None,
    approx_poly: bool = False,
    jit: bool = False,
    precision: Optional[Literal[16, 32, 64]] = None,
    fit_series_expansion: bool = True,
    bound_series_fit: bool = False,
    extended_output: bool = False
) -> Union[LmSig, tuple[LmSig, dict]]:
    prefactor = prefactor if prefactor is not None else not jit
    expr = func if isinstance(func, sp.Expr) else sp.sympify(func)
    if len(expr.free_symbols) == 0:
        raise ValueError("func must have at least one free variable.")
    free = sorted(expr.free_symbols, key=lambda s: str(s))
    bounds, point = _makebounds(bounds, len(free), point)
    ext, is_poly = {}, is_simple_poly(expr)
    if (approx_poly is True) or (is_poly is False):
        if fit_series_expansion is True:
            expr, ext['params'] = _perform_series_fit(
                func, bounds, nterms, fitres, point, bound_series_fit, is_poly
            )
        elif (len(free) > 1) or (is_poly is True):
            _, expr = multivariate_taylor(func, point, nterms, False)
        else:
            _, expr = series_lambda(func, point[0], nterms, False)
    # rewrite polynomial in horner form for fast evaluation
    ext['expr'] = sp.horner(expr)
    polyfunc = sp.lambdify(free, ext['expr'], ("scipy", "numpy"))
    # optionally, rewrite it to precompute stray powers and force precision
    polyfunc, ext["source"] = rewrite(polyfunc, prefactor, precision)
    # optionally, convert it to a numbafied CPUDispatcher function
    if jit is True:
        import numba
        polyfunc = numba.njit(polyfunc)
    if extended_output is True:
        return polyfunc, ext
    return polyfunc


def _makebounds(bounds, n_free, x0):
    bounds = (-1, 1) if bounds is None else bounds
    if not isinstance(bounds[0], (list, tuple)):
        bounds = [bounds for _ in range(n_free)]
    if x0 is None:
        x0 = [np.mean(b) for b in bounds]
    elif not isinstance(x0, (list, tuple)):
        x0 = [x0 for _ in bounds]
    return bounds, x0


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
    **quickkwargs
):
    lamb = lambdify(func)
    quick, ext = quickseries(func, **(quickkwargs | {'extended_output': True}))
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
