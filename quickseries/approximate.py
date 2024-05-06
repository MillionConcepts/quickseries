from functools import reduce
from inspect import getfullargspec, signature
from itertools import chain, count, product
import re
import timeit
from numbers import Number
from typing import Any, Callable, Literal, Optional, Sequence, Union

from cytoolz import groupby
from dustgoggles.dynamic import compile_source, define, getsource
import numpy as np
import sympy as sp
from dustgoggles.func import gmap
from dustgoggles.structures import listify

from quickseries.simplefit import fit

LmSig = Callable[[Any], Union[np.ndarray, float]]
"""signature of sympy-lambdified numpy/scipy functions"""

EXP_PATTERN = re.compile(r"\w+ ?\*\* ?(\d+)")
"""what exponentials in sympy-lambdified functions look like"""


def is_simple_poly(expr: sp.Expr) -> bool:
    gens = sp.poly_from_expr(expr)[1]['gens']
    return len(gens) == 1 and isinstance(gens[0], sp.Symbol)


def lambdify(
    func: Union[str, sp.Expr],
    modules: Union[str, Sequence[str]] = ("scipy", "numpy")
) -> LmSig:
    """
    Transform a sympy Expr or a string representation of a function into a
    callable with enforced argument order, incorporating code from specified
    modules.
    """
    if isinstance(func, str):
        func = sp.sympify(func)
    # noinspection PyTypeChecker
    return sp.lambdify(
        sorted(func.free_symbols, key=lambda x: str(x)), func, modules
    )


def _rectify_series(series, add_coefficients):
    outargs, coefsyms = [], []
    for a in series.args:
        # NOTE: the Expr.evalf() calls are simply to try to evaluate
        #  anything we can.
        if isinstance(a, sp.Order):
            continue
        elif isinstance(a, (sp.Mul, sp.Symbol, sp.Pow)):
            if add_coefficients is True:
                coefficient = sp.symbols(f"a_{len(coefsyms)}")
                outargs.append((coefficient * a).evalf())
                coefsyms.append(coefficient)
            else:
                outargs.append(a.evalf())
        elif isinstance((number := a.evalf()), sp.Number):
            outargs.append(number)
        else:
            raise ValueError(
                f"don't know how to handle expression element {a} of "
                f"type({type(a)})"
            )
    return sum(outargs), coefsyms


def series_lambda(
    func: Union[str, sp.Expr],
    x0: float = 0,
    order: int = 9,
    add_coefficients: bool = False,
    modules: Union[str, Sequence[str]] = ("scipy", "numpy")
) -> tuple[LmSig, sp.Expr]:
    """
    Construct a power expansion of a sympy Expr or the string expression of a
    function; optionally, add free coefficients to the terms of the resulting
    polynomial to permit optimization by downstream functions.

    Args:
        func: Mathematical function to expand, expressed as a string or a
            sympy Expr.
        x0: Point about which to expand func.
        order: Order of power expansion.
        add_coefficients: If True, add additional arguments/symbols to the
            returned function and Expr corresponding to the polynomial's
            coefficients.
        modules: Modules from which to draw the building blocks of the
            returned function.

    Returns:
        approximant: Python function that implements the power expansion.
        expr: sympy Expr used to construct approximant.
    """
    func = sp.sympify(func) if isinstance(func, str) else func
    # limiting precision of x0 is necessary due to a bug in sp.series
    series = sp.series(func, x0=round(x0, 6), n=order)
    # noinspection PyTypeChecker
    # remove Order (limit behavior) terms, try to split constants from
    # polynomial terms
    expr, coefsyms = _rectify_series(series, add_coefficients)
    syms = sorted(func.free_symbols, key=lambda x: str(x))
    # noinspection PyTypeChecker
    return sp.lambdify(syms + coefsyms, expr, modules), expr


def additive_combinations(n_terms, number):
    if n_terms == 1:
        return [(n,) for n in range(number + 1)]
    combinations = []  # NOTE: this is super gross-looking written as a chain
    for j in range(number + 1):
        combinations += [
            (j, *t)
            for t in additive_combinations(n_terms - 1, number - j)
        ]
    return combinations


def multivariate_taylor(
    func: Union[str, sp.Expr],
    point: Sequence[float],
    order: int,
    add_coefficients: bool = False
) -> tuple[LmSig, sp.Expr]:
    if not isinstance(func, sp.Expr):
        func = sp.sympify(func)
    pointsyms = listify(func.free_symbols)
    dimensionality = len(pointsyms)
    argsyms = listify(
        sp.symbols(",".join([f"x{i}" for i in range(dimensionality)]))
    )
    ixsyms = listify(
        sp.symbols(",".join(f"i{i}" for i in range(dimensionality)))
    )
    deriv = sp.Derivative(func, *[(p, i) for p, i in zip(pointsyms, ixsyms)])
    fact = reduce(sp.Mul, [sp.factorial(i) for i in ixsyms])
    err = reduce(
        sp.Mul,
        [(x - a) ** i for x, a, i in zip(argsyms, pointsyms, ixsyms)]
    )
    taylor = deriv / fact * err
    decomp = additive_combinations(dimensionality, order)
    built = reduce(
        sp.Add,
        (taylor.subs({i: d for i, d in zip(ixsyms, d)}) for d in decomp)
    ).doit()
    evaluated = built.subs({s: p for s, p in zip(pointsyms, point)}).evalf()
    # this next line is kind of aesthetic -- we just want the argument names
    # to be consistent with the input
    evaluated = evaluated.subs({a: p for a, p in zip(argsyms, pointsyms)})
    evaluated, coefsyms = _rectify_series(evaluated, add_coefficients)
    return sp.lambdify(pointsyms + coefsyms, evaluated), evaluated


def lastline(func: Callable) -> str:
    """try to get the last line of a function, sans return statement"""
    return tuple(
        filter(None, getsource(func).split("\n"))
    )[-1].replace("return", "").strip()


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
        r"([+* (-]+)([\d.]+)(e[+\-]?\d+)?.*?([+* )]|$)", line
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
) -> LmSig:
    if precompute is False and precision is None:
        return poly_lambda
    # name of arguments to the lambdified function
    free = getfullargspec(poly_lambda).args
    lines = [f"def _lambdifygenerated({', '.join(free)}):"]
    # sympy will always place this on a single line; it includes
    # the Python expression form of the hornerized polynomial
    # and a return statement; lastline() strips it
    polyexpr = lastline(poly_lambda)
    if precompute is True:
        polyexpr, lines = _rewrite_precomputed(polyexpr, free, lines)
    if precision is not None:
        polyexpr = force_line_precision(polyexpr, precision)
    lines.append(f"    return {polyexpr}")
    # noinspection PyUnresolvedReferences
    opt = define(compile_source("\n".join(lines)), poly_lambda.__globals__)
    opt.__doc__ = ("\n".join(map(str.strip, lines[1:])))
    return opt


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


def _perform_series_fit(func, bounds, order, resolution, x0, apply_bounds):
    if len(bounds) == 1:
        approx, expr = series_lambda(func, x0[0], order, True)
    else:
        approx, expr = multivariate_taylor(func, x0, order, True)
    lamb, vecs = lambdify(func), _pvec(bounds, resolution)
    try:
        dep = lamb(*vecs)
    except TypeError as err:
        # this is a potentially slow but unavoidable case
        if "converted to Python scalars" not in str(err):
            raise
        dep = np.array([lamb(v) for v in vecs])
    kw = {}
    if apply_bounds is True:
        kw['bounds'] = (-5, 5)
    guess = [
        1 for _ in range(len(signature(approx).parameters) - len(vecs))
    ]
    params, _ = fit(
        func=approx, vecs=vecs, dependent_variable=dep, guess=guess, **kw
    )
    # insert coefficients into polynomial
    expr = expr.subs({f'a_{i}': coef for i, coef in enumerate(params)})
    return expr


def quickseries(
    func: Union[str, sp.Expr, sp.core.function.FunctionClass],
    bounds: tuple[float, float] = (-1, 1),
    order: int = 9,
    point: Optional[float] = None,
    resolution: int = 100,
    prefactor: Optional[bool] = None,
    approx_poly: bool = False,
    jit: bool = False,
    precision: Optional[Literal[16, 32, 64]] = None,
    fit_series_expansion: bool = True,
    bound_series_fit: bool = False
) -> LmSig:
    prefactor = prefactor if prefactor is not None else not jit
    expr = func if isinstance(func, sp.Expr) else sp.sympify(func)
    if len(expr.free_symbols) == 0:
        raise ValueError("func must have at least one free variable.")
    free = tuple(expr.free_symbols)
    bounds, point = _makebounds(bounds, len(free), point)
    if (approx_poly is True) or (not is_simple_poly(expr)):
        if fit_series_expansion is True:
            expr = _perform_series_fit(
                func, bounds, order, resolution, point, bound_series_fit
            )
        elif len(free) > 1:
            approx, expr = multivariate_taylor(func, point, order, False)
        else:
            _, expr = series_lambda(func, point[0], order, False)
    # rewrite polynomial in horner form for fast evaluation
    expr = sp.polys.polyfuncs.horner(expr)
    polyfunc = sp.lambdify(free, expr, ("scipy", "numpy"))
    # optionally, rewrite it to precompute stray powers and force precision
    polyfunc = rewrite(polyfunc, prefactor, precision)
    # optionally, convert it to a numbafied CPUDispatcher function
    if jit is True:
        import numba
        polyfunc = numba.njit(polyfunc)
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


def benchmark(
    func: Union[str, sp.Expr, sp.core.function.FunctionClass],
    offset_resolution: int = 10000,
    n_offset_shuffles: int = 50,
    timeit_cycles: int = 10000,
    testbounds="equal",
    **quickkwargs
):
    lamb = lambdify(sp.sympify(func))
    quick = quickseries(func, **quickkwargs)
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
        medians, stds = [], []
        for _ in range(n_offset_shuffles):
            gmap(np.random.shuffle, vecs)
            absdiff, mediff, stdiff, frange, worstpoint = _offset_check_cycle(
                absdiff, frange, lamb, quick, vecs, worstpoint
            )
            medians.append(mediff)
            stds.append(stdiff)
        mediff, stdiff = np.median(medians), np.median(stds)
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
        stdiff = np.std(offset)
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
        'stdiff': stdiff,
        'worstpoint': worstpoint,
        'range': frange,
        'orig_s': orig_s, 
        'approx_s': approx_s,
        'timeratio': approx_s / orig_s,
        'quick': quick
    }


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
    return absdiff, np.median(offset), np.std(offset), frange, worstpoint
