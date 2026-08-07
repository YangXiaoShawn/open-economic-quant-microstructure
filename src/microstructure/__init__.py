"""Event-driven market-microstructure research package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("event-driven-microstructure")
except PackageNotFoundError:  # pragma: no cover - editable source without metadata
    __version__ = "0+unknown"

__all__ = ["__version__"]
