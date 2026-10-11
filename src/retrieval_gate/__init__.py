"""Fail CI when any retrieval query gets worse."""
from .compare import GateResult, QueryResult, compare
from .metrics import METRICS, InputError, score_query

__version__ = "0.1.0"
__all__ = ["compare", "score_query", "GateResult", "QueryResult", "InputError", "METRICS", "__version__"]

