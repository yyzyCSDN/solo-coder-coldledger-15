from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib, json
from typing import Iterable

GENESIS = "GENESIS-RELEASE"


def _utc(at: datetime) -> datetime:
    if at.tzinfo is None:
        raise ValueError("timezone-aware datetime required")
    return at.astimezone(timezone.utc)


@dataclass(frozen=True)
class ReleaseOrder:
    case_reference: str
    order_seq: int
    decision_seq: int  # board decision this order executes
    lot_ids: tuple[str, ...]
    released_by: str
    at: datetime
    note: str
    supersedes_order_seq: int | None
    previous_digest: str
    digest: str


def order_digest(
    case_reference: str,
    order_seq: int,
    decision_seq: int,
    lot_ids: tuple[str, ...],
    released_by: str,
    at: datetime,
    note: str,
    supersedes_order_seq: int | None,
    previous_digest: str,
) -> str:
    body = [
        case_reference,
        order_seq,
        decision_seq,
        list(lot_ids),
        released_by,
        _utc(at).isoformat(),
        note,
        supersedes_order_seq,
        previous_digest,
    ]
    raw = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha512(raw).hexdigest()


def issue_order(
    case_reference: str,
    decision_seq: int,
    lot_ids: Iterable[str],
    released_by: str,
    at: datetime,
    prior: Iterable[ReleaseOrder] = (),
    supersedes_order_seq: int | None = None,
    expected_seq: int | None = None,
    note: str = "",
) -> ReleaseOrder:
    lots = tuple(sorted(lot_ids))
    if not lots:
        raise ValueError("release order must name at least one lot")
    rows = sorted(prior, key=lambda o: o.order_seq)
    if any(o.order_seq != i for i, o in enumerate(rows, start=1)):
        raise ValueError("prior orders are not a contiguous chain")
    seq = len(rows) + 1
    if expected_seq is not None and expected_seq != len(rows):
        raise RuntimeError("release chain concurrency conflict")
    prev_digest = rows[-1].digest if rows else GENESIS
    if seq == 1:
        if supersedes_order_seq is not None:
            raise ValueError("first release order cannot supersede")
    elif supersedes_order_seq is None:
        raise ValueError("new release order must explicitly supersede the prior order")
    when = _utc(at)
    if rows and when < _utc(rows[-1].at):
        raise ValueError("release time moved backward")
    digest = order_digest(
        case_reference, seq, decision_seq, lots, released_by, when, note,
        supersedes_order_seq, prev_digest,
    )
    return ReleaseOrder(
        case_reference, seq, decision_seq, lots, released_by, when, note,
        supersedes_order_seq, prev_digest, digest,
    )


def verify_chain(rows: Iterable[ReleaseOrder]) -> bool:
    prev = GENESIS
    for expected_seq, o in enumerate(sorted(rows, key=lambda x: x.order_seq), start=1):
        if o.order_seq != expected_seq:
            return False
        if o.previous_digest != prev:
            return False
        if order_digest(
            o.case_reference, o.order_seq, o.decision_seq, o.lot_ids,
            o.released_by, o.at, o.note, o.supersedes_order_seq, o.previous_digest,
        ) != o.digest:
            return False
        prev = o.digest
    return True


def current_order(rows: Iterable[ReleaseOrder]) -> ReleaseOrder | None:
    rows = sorted(rows, key=lambda o: o.order_seq)
    return rows[-1] if rows else None


def released_lot_scope(rows: Iterable[ReleaseOrder]) -> tuple[str, ...]:
    """Lots covered by the newest non-superseded order.
    Superseded orders stay in the chain but no longer define the scope."""
    order = current_order(rows)
    return order.lot_ids if order else ()
