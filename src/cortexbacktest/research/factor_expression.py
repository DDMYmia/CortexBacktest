"""Bounded factor expressions over native builders; never execute source code."""
from __future__ import annotations

import ast
import operator
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

import polars as pl

from .indicators import momentum, risk_adjusted_momentum, rolling_volatility, rolling_zscore, sma_ratio
from .preprocessing import rank_cross_section

BUILDERS = {"momentum": momentum, "sma_ratio": sma_ratio,
            "rolling_volatility": rolling_volatility, "rolling_zscore": rolling_zscore,
            "risk_adjusted_momentum": risk_adjusted_momentum}


@dataclass(frozen=True)
class FactorExpression:
    operator: str
    children: tuple[FactorExpression, ...] = ()

    @property
    def name(self) -> str:
        return f"{self.operator}({','.join(child.name for child in self.children)})" if self.children else self.operator

    @property
    def leaves(self) -> frozenset[str]:
        return frozenset().union(*(child.leaves for child in self.children)) if self.children else frozenset((self.operator,))

    def build(self, leaf: Callable[[str], pl.DataFrame]) -> pl.DataFrame:
        cache: dict[str, pl.DataFrame] = {}

        def evaluate(node: FactorExpression) -> pl.DataFrame:
            if node.name in cache:
                return cache[node.name]
            if not node.children:
                result = leaf(node.operator)
            elif node.operator == "rank":
                result = rank_cross_section(evaluate(node.children[0]))
            else:
                left, right = (evaluate(child) for child in node.children)
                symbols = left.columns[1:]
                operation = {"add": operator.add, "sub": operator.sub,
                             "mul": operator.mul, "div": operator.truediv}[node.operator]
                values = operation(left.select(symbols), right.select(symbols))
                result = values.select(pl.when(pl.all().is_finite()).then(pl.all()).otherwise(None))
                result.insert_column(0, left["date"])
            cache[node.name] = result
            return result

        return evaluate(self)


@lru_cache(maxsize=256)
def parse_expression(source: str) -> FactorExpression:
    """Validate at most 31 nodes / six levels and canonicalize commutative pairs."""
    if not isinstance(source, str) or len(source) > 512:
        raise ValueError("Factor expression must be a string of at most 512 characters")
    count = 0

    def visit(node: ast.AST, depth: int) -> FactorExpression:
        nonlocal count
        count += 1
        if count > 31 or depth > 6:
            raise ValueError("Factor expression exceeds node or depth budget")
        if isinstance(node, ast.Name) and node.id in BUILDERS:
            return FactorExpression(node.id)
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.keywords:
            raise ValueError("Only native factor names and supported operator calls are allowed")
        op = node.func.id
        if op not in {"rank", "add", "sub", "mul", "div"} or len(node.args) != (1 if op == "rank" else 2):
            raise ValueError("Unsupported factor operator or arity")
        children = tuple(visit(child, depth + 1) for child in node.args)
        if op in {"add", "mul"}:
            children = tuple(sorted(children, key=lambda child: child.name))
        if op in {"sub", "div"} and children[0] == children[1]:
            raise ValueError("Constant self-combinations are not factor hypotheses")
        return FactorExpression(op, children)

    try:
        return visit(ast.parse(source.strip(), mode="eval").body, 1)
    except (SyntaxError, RecursionError) as exc:
        raise ValueError("Invalid factor expression") from exc
