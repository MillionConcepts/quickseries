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
    orthopoly: Callable[[int, sp.Basic], sp.Expr]
    domain: tuple[float, float]
    weight_fn: Callable[[sp.Symbol], sp.Expr | int]
    transform_input: Callable[[sp.Expr, float, float], sp.Expr]
    transform_output: Callable[[sp.Expr, float, float], sp.Expr]
    quadrature: Optional[Callable[[int], tuple]] = None


# TODO: transforms won't work with > 1 free variable
def make_legendre_basis():
    xi = Symbol('xi')

    def transform_input(expr, a, b):
        mapped = ((b - a) / 2) * xi + (a + b) / 2
        return expr.subs(xi, mapped)

    def transform_output(poly_expr, a, b):
        # noinspection PyTypeChecker
        inv = (2 * s1(poly_expr) - (a + b)) / (b - a)
        return expand(poly_expr.subs(xi, inv))

    def quadrature(n):
        # TODO: configurable precision
        pts, wts = gauss_legendre(n, 7)
        return pts, wts

    return Orthobasis(
        name="legendre",
        orthopoly=lambda n, v: sp.legendre(n, v),
        domain=(-1, 1),
        weight_fn=lambda v: 1,
        transform_input=transform_input,
        transform_output=transform_output,
        quadrature=quadrature,
    )


def make_chebyshev_basis():
    xi = sp.Symbol('xi')

    def transform_input(expr, a, b):
        # Map from [a, b] to [-1, 1]
        mapped = ((b - a) / 2) * xi + (a + b) / 2
        return expr.subs(xi, mapped)

    def transform_output(poly_expr, a, b):
        # Invert the above mapping
        inv = (2 * sp.Symbol(str(poly_expr.free_symbols.pop())) - (a + b)) / (b - a)
        return sp.expand(poly_expr.subs(xi, inv))

    def quadrature(n):
        # Gauss-Chebyshev quadrature of the first kind
        k = np.arange(1, n + 1)
        pts = np.cos((2 * k - 1) * np.pi / (2 * n))
        wts = np.pi / n * np.ones(n)
        return pts.tolist(), wts.tolist()

    return Orthobasis(
        name="chebyshev",
        orthopoly=lambda n, v: sp.chebyshevt(n, v),
        domain=(-1, 1),
        weight_fn=lambda x: 1 / sp.sqrt(1 - x**2),
        transform_input=transform_input,
        transform_output=transform_output,
        quadrature=quadrature,
    )


BASIS_REGISTRY = {
    "legendre": make_legendre_basis(),
    "chebyshev": make_chebyshev_basis()
}
