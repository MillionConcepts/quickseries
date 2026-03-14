from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import sympy as sp
from sympy import Basic, Expr, Symbol, expand
from sympy.integrals.quadrature import gauss_legendre

def s1(expr: Expr) -> Basic:
    """
    Return arbitrarily 'first' free symbol in expression. Do not use this on
    expressions with more than one free symbol.
    """
    return tuple(expr.free_symbols)[0]

@dataclass(frozen=True)
class Orthobasis:
    name: str
    orthopoly: Callable[[int, sp.Symbol], sp.Expr]
    domain: tuple[float, float]
    weight_fn: Callable[[sp.Symbol], sp.Expr | int]
    transform_input: Callable[[sp.Expr, float, float, sp.Symbol], sp.Expr]
    transform_output: Callable[[sp.Expr, float, float, sp.Symbol], sp.Expr]
    quadrature: Optional[Callable[[int], tuple]] = None
    norm: Callable[[int], float] = lambda x: 1


# TODO: transforms won't work with > 1 free variable
def make_legendre_basis():

    def transform_input(expr, a, b, sym):
        mapped = ((b - a) / 2) * sym + (a + b) / 2
        return expr.subs(sym, mapped)

    def transform_output(expr, a, b, sym):
        # degenerate case
        if isinstance(expr, int):
            return expr
        mapped = (2 * sym - (a + b)) / (b - a)
        return sp.expand(expr.subs(sym, mapped))

    def quadrature(n):
        # TODO: configurable precision
        pts, wts = gauss_legendre(n, 7)
        return pts, wts

    def norm(n):
        return 2 / (2 * n + 1)

    return Orthobasis(
        name="legendre",
        orthopoly=lambda n, v: sp.legendre(n, v),
        domain=(-1, 1),
        weight_fn=lambda v: 1,
        transform_input=transform_input,
        transform_output=transform_output,
        quadrature=quadrature,
        norm=norm
    )


def make_chebyshev_basis():

    def transform_input(expr, a, b, sym):
        mapped = ((b - a) / 2) * sym + (a + b) / 2
        return expr.subs(sym, mapped)

    def transform_output(expr, a, b, sym):
        # degenerate case
        if isinstance(expr, int):
            return expr
        mapped = (2 * sym - (a + b)) / (b - a)
        return sp.expand(expr.subs(sym, mapped))


    def quadrature(n):
        # Gauss-Chebyshev quadrature of the first kind
        k = np.arange(1, n + 1)
        pts = np.cos((2 * k - 1) * np.pi / (2 * n))
        wts = np.pi / n * np.ones(n)
        return pts.tolist(), wts.tolist()

    def norm(n):
        return np.pi if n == 0 else np.pi / 2

    def weight(sym: sp.Symbol):
        return sym / sp.sqrt(1 - sym ** 2)

    return Orthobasis(
        name="chebyshev",
        orthopoly=lambda n, v: sp.chebyshevt(n, v),
        domain=(-1, 1),
        weight_fn = weight,
        transform_input=transform_input,
        transform_output=transform_output,
        quadrature=quadrature,
        norm=norm
    )


BASIS_REGISTRY = {
    "legendre": make_legendre_basis(),
    "chebyshev": make_chebyshev_basis()
}
