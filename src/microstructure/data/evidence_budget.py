"""Fail-closed accounting for retained raw-evidence bytes."""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class EvidenceBudgetError(RuntimeError):
    """Base error for retained-evidence budget failures."""


class EvidenceBudgetExceeded(EvidenceBudgetError):
    """Raised before a retained artifact would exceed its byte budget."""


class EvidenceBudgetStateError(EvidenceBudgetError):
    """Raised when a reservation is finalized more than once."""


def _scan_regular_file_bytes(root: Path) -> int:
    """Return logical bytes below *root* without following symbolic links."""
    if root.is_symlink():
        raise EvidenceBudgetError(f"evidence-budget root must not be a symlink: {root}")
    if not root.exists():
        return 0
    if not root.is_dir():
        raise EvidenceBudgetError(f"evidence-budget root is not a directory: {root}")

    total = 0
    pending = [root]
    try:
        while pending:
            directory = pending.pop()
            with os.scandir(directory) as entries:
                for entry in entries:
                    if entry.is_symlink():
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False):
                        total += entry.stat(follow_symlinks=False).st_size
    except OSError as exc:
        raise EvidenceBudgetError(f"cannot scan retained evidence below {root}") from exc
    return total


class EvidenceReservation:
    """One exclusive byte reservation, committed only after durable retention."""

    __slots__ = ("_budget", "_bytes_reserved", "_label", "_state")

    def __init__(
        self,
        budget: RetainedEvidenceBudget,
        bytes_reserved: int,
        label: str,
    ) -> None:
        self._budget = budget
        self._bytes_reserved = bytes_reserved
        self._label = label
        self._state = "active"

    @property
    def bytes_reserved(self) -> int:
        return self._bytes_reserved

    @property
    def label(self) -> str:
        return self._label

    @property
    def active(self) -> bool:
        return self._state == "active"

    def commit(self) -> None:
        """Charge the reservation after its bytes have been retained."""
        if self._state != "active":
            raise EvidenceBudgetStateError(f"reservation is already {self._state}")
        self._budget._commit(self._bytes_reserved)
        self._state = "committed"

    def release(self) -> None:
        """Return an unused reservation to the available budget."""
        if self._state != "active":
            raise EvidenceBudgetStateError(f"reservation is already {self._state}")
        self._budget._release(self._bytes_reserved)
        self._state = "released"

    def __enter__(self) -> EvidenceReservation:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> None:
        if self.active:
            self.release()


class RetainedEvidenceBudget:
    """Thread-safe accounting for a bounded directory of immutable evidence.

    Existing regular files are counted once at construction. Symbolic links
    are neither counted nor traversed. Callers must share one instance across
    writers targeting the same root so outstanding reservations cannot
    oversubscribe the limit.
    """

    __slots__ = ("_limit_bytes", "_lock", "_reserved_bytes", "_root", "_used_bytes")

    def __init__(self, root: str | Path, limit_bytes: int) -> None:
        if isinstance(limit_bytes, bool) or limit_bytes < 0:
            raise ValueError("limit_bytes must be a nonnegative integer")
        self._root = Path(root).expanduser().absolute()
        self._limit_bytes = limit_bytes
        self._lock = threading.RLock()
        self._reserved_bytes = 0
        self._used_bytes = _scan_regular_file_bytes(self._root)
        if self._used_bytes > self._limit_bytes:
            raise EvidenceBudgetExceeded(
                "preexisting retained evidence exceeds the configured byte budget: "
                f"used={self._used_bytes}, limit={self._limit_bytes}, root={self._root}"
            )

    @property
    def root(self) -> Path:
        return self._root

    @property
    def limit_bytes(self) -> int:
        return self._limit_bytes

    @property
    def used_bytes(self) -> int:
        with self._lock:
            return self._used_bytes

    @property
    def reserved_bytes(self) -> int:
        with self._lock:
            return self._reserved_bytes

    @property
    def remaining_bytes(self) -> int:
        with self._lock:
            return self._limit_bytes - self._used_bytes - self._reserved_bytes

    def assert_contains(self, path: str | Path) -> None:
        """Reject targets outside the budget root or below an in-root symlink."""
        target = Path(os.path.abspath(Path(path).expanduser()))
        if not target.is_relative_to(self._root):
            raise EvidenceBudgetError(
                f"retained-evidence target is outside budget root: target={target}, "
                f"root={self._root}"
            )
        current = self._root
        for component in target.relative_to(self._root).parts:
            current /= component
            if current.is_symlink():
                raise EvidenceBudgetError(
                    f"retained-evidence target traverses a symlink: {current}"
                )

    @contextmanager
    def write_transaction(self) -> Iterator[None]:
        """Serialize deduplication checks, reservations, and artifact commits."""
        with self._lock:
            yield

    def reserve(
        self,
        bytes_to_add: int,
        *,
        label: str = "retained evidence",
    ) -> EvidenceReservation:
        """Atomically reserve bytes or fail before the caller writes them."""
        if isinstance(bytes_to_add, bool) or bytes_to_add < 0:
            raise ValueError("bytes_to_add must be a nonnegative integer")
        with self._lock:
            projected = self._used_bytes + self._reserved_bytes + bytes_to_add
            if projected > self._limit_bytes:
                raise EvidenceBudgetExceeded(
                    f"{label} would exceed retained-evidence budget: "
                    f"requested={bytes_to_add}, used={self._used_bytes}, "
                    f"reserved={self._reserved_bytes}, limit={self._limit_bytes}, "
                    f"root={self._root}"
                )
            self._reserved_bytes += bytes_to_add
        return EvidenceReservation(self, bytes_to_add, label)

    def _commit(self, bytes_reserved: int) -> None:
        with self._lock:
            if bytes_reserved > self._reserved_bytes:
                raise EvidenceBudgetStateError("reservation accounting underflow on commit")
            self._reserved_bytes -= bytes_reserved
            self._used_bytes += bytes_reserved

    def _release(self, bytes_reserved: int) -> None:
        with self._lock:
            if bytes_reserved > self._reserved_bytes:
                raise EvidenceBudgetStateError("reservation accounting underflow on release")
            self._reserved_bytes -= bytes_reserved
