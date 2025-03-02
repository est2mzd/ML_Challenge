from _typeshed import Incomplete
from collections.abc import Generator

from networkx.utils.backends import _dispatchable

@_dispatchable
def is_eulerian(G): ...
@_dispatchable
def is_semieulerian(G): ...
@_dispatchable
def eulerian_circuit(G, source: Incomplete | None = None, keys: bool = False) -> Generator[Incomplete, Incomplete, None]: ...
@_dispatchable
def has_eulerian_path(G, source: Incomplete | None = None): ...
@_dispatchable
def eulerian_path(G, source: Incomplete | None = None, keys: bool = False) -> Generator[Incomplete, Incomplete, None]: ...
@_dispatchable
def eulerize(G): ...
