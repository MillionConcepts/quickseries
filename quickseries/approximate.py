from inspect import getfullargspec
from itertools import chain
import re
from typing import Callable, Union, Any, Optional, Sequence

from cytoolz import groupby
from dustgoggles.dynamic import compile_source, define, getsource
from dustgoggles.structures import listify
import numpy as np
import sympy as sp

from quickseries.simplefit import fit

LmSig = Callable[[Any], Union[np.ndarray, float]]
"""signature of sympy-lambdified numpy/scipy functions"""

EXP_PATTERN = re.compile(r"\w+ ?\*\* ?(\d+)")
"""what exponentials in sympy-lambdified functions look like"""

SIMPLE_POLY_ELEMENTS = (
    sp.Add, sp.Mul, sp.Number, sp.Symbol, sp.Pow, sp.Integer
)


def is_simple_poly(expr: sp.Expr) -> bool:
    for arg in sp.preorder_traversal(expr):
        if not isinstance(arg, SIMPLE_POLY_ELEMENTS):
            return False
    return True


def lambdify(
    func: Union[str, sp.Expr],
    modules: Union[str, Sequence[str]] = ("scipy", "numpy")
) -> LmSig:
    if isinstance(func, str):
        func = sp.sympify(func)
    # noinspection PyTypeChecker
    return sp.lambdify(sorted(func.free_symbols), func, modules)


def series_lambda(
    func: Union[str, sp.Expr],
    x0: float = 0,
    n_terms: int = 9,
    add_coefficients: bool = False,
    module: str = "numpy",
) -> tuple[LmSig, sp.Expr]:
    if isinstance(func, str):
        func = sp.sympify(func)
    series = sp.series(func, x0=x0, n=n_terms)
    # [:-1] is to remove O(x) term
    # noinspection PyTypeChecker
    terms, args = series.args[:-1], sorted(func.free_symbols)
    if add_coefficients is True:
        c_syms = listify(
            sp.symbols(",".join([f"a_{n}" for n in range(len(terms))]))
        )
        terms = [t * sym for t, sym in zip(terms, c_syms)]
        args += c_syms
    expr = sum(terms)
    # noinspection PyTypeChecker
    return sp.lambdify(args, expr, module), expr


def lastline(func: Callable) -> str:
    return tuple(
        filter(None, getsource(func).split("\n"))
    )[-1].replace("return", "").strip()


def regexponents(text: str) -> tuple[int]:
    # noinspection PyTypeChecker
    return tuple(map(int, re.findall(EXP_PATTERN, text)))


def optimize_exponents(
    exps: Sequence[int]
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    replacements = [(e, [e]) for e in exps]
    reduced = set()
    extant = tuple(chain(*[r[1] for r in replacements]))
    while True:
        if len(extant) == 1:
            replacements[0][1][:] = [1 for _ in range(replacements[0][0])]
            break
        counts = {
            k: len(v)
            for k, v in groupby(lambda x: x, extant).items()
            if k not in reduced
        }
        if len(counts) == 0:
            break
        if len(counts) == 1:
            break
        elif counts[max(counts)] > 1:
            reduced.add(max(counts))
            continue
        else:
            factor = sorted(counts.keys())[0]
        for k, v in replacements:
            factorization = []
            for f in v:
                if f in reduced or f <= factor:
                    factorization.append(f)
                    continue
                factorization.append(factor)
                difference = f - factor
                while difference >= max(
                    [e for e in extant if e != f]
                ):
                    factorization.append(factor)
                    difference = difference - factor
                if difference > 0:
                    factorization.append(difference)
            v[:] = factorization
        reduced.add(factor)
        extant = tuple(chain(*[r[1] for r in replacements]))
    replacements = {k: v for k, v in replacements}
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
    variables.pop(1)
    return replacements, variables


def rewrite_precomputed(poly_lambda: LmSig) -> LmSig:
    # name of single argument to the lambdified function
    varname = getfullargspec(poly_lambda).args[0]
    # sympy will always place this on a single line; it includes
    # the Python expression form of the hornerized polynomial
    # and a return statement; lastline() strips it
    polyexpr = lastline(poly_lambda)
    replacements, variables = optimize_exponents(regexponents(polyexpr))
    lines = [f"def _lambdifygenerated({varname}):"]
    for k, v in variables.items():
        multiplicands = []
        for power in v:
            if power == 1:
                multiplicands.append(varname)
            else:
                multiplicands.append(f"{varname}{power}")
        lines.append(f"    {varname}{k} = {'*'.join(multiplicands)}")
    for k, v in replacements.items():
        substitution = '*'.join(
            [f"{varname}{r}" if r != 1 else varname for r in v]
        )
        polyexpr = polyexpr.replace(f"{varname}**{k}", substitution)
    lines.append(f"    return {polyexpr}")
    # noinspection PyUnresolvedReferences
    opt = define(compile_source("\n".join(lines)), poly_lambda.__globals__)
    opt.__doc__ = ("\n".join(map(str.strip, lines[1:])))
    return opt


def quickseries(
    func: Union[str, sp.Expr, sp.core.function.FunctionClass],
    bounds: tuple[float, float] = (-1, 1),
    n_terms: int = 9,
    x0: Optional[float] = None,
    resolution: int = 100,
    prefactor: Optional[bool] = None,
    approx_poly: bool = False,
    jit: bool = False
) -> LmSig:
    prefactor = prefactor if prefactor is not None else not jit
    expr = func if isinstance(func, sp.Expr) else sp.sympify(func)
    if len(expr.free_symbols) != 1:
        raise ValueError("This function only supports univariate functions.")
    if approx_poly is False and is_simple_poly(expr):
        polyfunc = lambdify(sp.polys.polyfuncs.horner(expr))
    else:
        x0 = x0 if x0 is not None else np.mean(bounds)
        approx, expr = series_lambda(func, x0, n_terms, True)
        vec, lamb = np.linspace(*bounds, resolution), lambdify(func)
        try:
            dep = lamb(vec)
        except TypeError as err:
            # this is a potentially slow but unavoidable case
            if "converted to Python scalars" not in str(err):
                raise
            dep = np.array([lamb(v) for v in vec])
        params, _ = fit(approx, 1, vec, dep, bounds=bounds)
        # insert coefficients into polynomial
        substituted = expr.subs(
            {f'a_{i}': coef for i, coef in enumerate(params)}
        )
        # rewrite polynomial in horner form for fast evaluation and convert
        # it to a numpy function
        polyfunc = sp.lambdify(
            getfullargspec(lamb).args[0],
            sp.polys.polyfuncs.horner(substituted),
             ("scipy", "numpy")
        )
    # optionally, rewrite it to precompute stray powers
    if prefactor is True:
        polyfunc = rewrite_precomputed(polyfunc)
    # optionally, jit it with numba
    if jit is True:
        import numba
        polyfunc = numba.njit(polyfunc)
    return polyfunc
