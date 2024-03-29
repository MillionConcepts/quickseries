from inspect import getfullargspec
from itertools import chain, count
import re
from typing import Any, Callable, Literal, Optional, Sequence, Union

from cytoolz import groupby
from dustgoggles.dynamic import compile_source, define, getsource
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
"""sympy classes that may appear in a 'simple' (by our standards) polynomial"""


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
    order: int = 9,
    add_coefficients: bool = False,
    module: str = "numpy",
) -> tuple[LmSig, sp.Expr]:
    func = sp.sympify(func) if isinstance(func, str) else func
    # limiting precision of x0 is necessary due to a bug in sp.series
    series = sp.series(func, x0=round(x0, 6), n=order)
    # noinspection PyTypeChecker
    args, syms = series.args, sorted(func.free_symbols)
    # remove Order (limit behavior) terms, try to split constants from
    # polynomial terms
    outargs, coefs = [], count()
    for a in args:
        if isinstance(a, sp.Order):
            continue
        elif isinstance(a, (sp.Mul, sp.Symbol, sp.Pow)):
            if add_coefficients is True:
                coefficient = sp.symbols(f"a_{next(coefs)}")
                outargs.append(coefficient * a)
                syms.append(coefficient)
            else:
                outargs.append(a)
        # check for things like 'cos(2)' and try to turn them into numbers
        elif isinstance((number := a.evalf()), sp.Number):
            outargs.append(number)
        else:
            raise ValueError(
                f"don't know how to handle expression element {a} of "
                f"type({type(a)})"
            )
    expr = sum(outargs)
    # noinspection PyTypeChecker
    return sp.lambdify(syms, expr, module), expr


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
    constructor = f"numpy.float{precision}"
    last, out = 0, ""
    for match in re.finditer(
        r"([+* (-]+)([\d.]+)(e[+\-]?\d+)?.*?([+* )]|$)", line
    ):
        out += line[last:match.span()[0]]
        # don't replace exponents
        if match.group(1) == "**":
            out += line[slice(*match.span())]
        else:
            out += f"{match.group(1)}{constructor}({match.group(2)}"
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
    # name of single argument to the lambdified function
    varname = getfullargspec(poly_lambda).args[0]
    lines = [f"def _lambdifygenerated({varname}):"]
    # sympy will always place this on a single line; it includes
    # the Python expression form of the hornerized polynomial
    # and a return statement; lastline() strips it
    polyexpr = lastline(poly_lambda)
    if precompute is True:
        polyexpr, lines = _rewrite_precomputed(polyexpr, varname, lines)
    if precision is not None:
        polyexpr = force_line_precision(polyexpr, precision)
    lines.append(f"    return {polyexpr}")
    # noinspection PyUnresolvedReferences
    opt = define(compile_source("\n".join(lines)), poly_lambda.__globals__)
    opt.__doc__ = ("\n".join(map(str.strip, lines[1:])))
    return opt


def _rewrite_precomputed(polyexpr, varname, lines):
    # replacements: what factors we will decompose each factor into
    # variables: which factors we will define as variables, and their
    # "building blocks"
    replacements, variables = optimize_exponents(regexponents(polyexpr))
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
    return polyexpr, lines


def quickseries(
    func: Union[str, sp.Expr, sp.core.function.FunctionClass],
    bounds: tuple[float, float] = (-1, 1),
    order: int = 9,
    x0: Optional[float] = None,
    resolution: int = 100,
    prefactor: Optional[bool] = None,
    approx_poly: bool = False,
    jit: bool = False,
    bit_depth: Optional[Literal[16, 32, 64]] = None,
) -> LmSig:
    prefactor = prefactor if prefactor is not None else not jit
    expr = func if isinstance(func, sp.Expr) else sp.sympify(func)
    if len(expr.free_symbols) != 1:
        raise ValueError("This function only supports univariate functions.")
    free = tuple(expr.free_symbols)[0]
    if approx_poly is True or not is_simple_poly(expr):
        x0 = x0 if x0 is not None else np.mean(bounds)
        approx, expr = series_lambda(func, x0, order, True)
        vec, lamb = np.linspace(*bounds, resolution), lambdify(func)
        try:
            dep = lamb(vec)
        except TypeError as err:
            # this is a potentially slow but unavoidable case
            if "converted to Python scalars" not in str(err):
                raise
            dep = np.array([lamb(v) for v in vec])
        params, _ = fit(approx, 1, vec, dep)
        # insert coefficients into polynomial
        expr = expr.subs({f'a_{i}': coef for i, coef in enumerate(params)})
    # rewrite polynomial in horner form for fast evaluation
    expr = sp.polys.polyfuncs.horner(expr)
    polyfunc = sp.lambdify(free, expr, ("scipy", "numpy"))
    # optionally, rewrite it to precompute stray powers and force precision
    polyfunc = rewrite(polyfunc, prefactor, bit_depth)
    # optionally, convert it to a numbafied CPUDispatcher function
    if jit is True:
        import numba
        polyfunc = numba.njit(polyfunc)
    return polyfunc
