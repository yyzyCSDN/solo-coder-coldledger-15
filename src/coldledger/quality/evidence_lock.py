from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib, json
from typing import Iterable

from ..quality.evidence_graph import CaseEdge, reachable

GENESIS = "GENESIS-EVIDENCE"


def _utc(at: datetime) -> datetime:
    if at.tzinfo is None:
        raise ValueError("timezone-aware datetime required")
    return at.astimezone(timezone.utc)


@dataclass(frozen=True)
class EvidenceItem:
    item_id: str
    kind: str  # telemetry | calibration | photo | lab | statement | document
    source: str
    captured_at: datetime
    content_hash: str

    def validate(self) -> None:
        if not self.item_id:
            raise ValueError("item_id required")
        if not self.kind:
            raise ValueError("kind required")
        if not self.content_hash:
            raise ValueError("content_hash required")
        _utc(self.captured_at)


@dataclass(frozen=True)
class SealedEvidence:
    case_reference: str
    seal_seq: int
    sealed_by: str
    sealed_at: datetime
    items: tuple[EvidenceItem, ...]
    lot_ids: tuple[str, ...]
    previous_seal_digest: str
    digest: str

    def item_ids(self) -> tuple[str, ...]:
        return tuple(i.item_id for i in self.items)


def seal_digest(
    case_reference: str,
    seal_seq: int,
    sealed_by: str,
    sealed_at: datetime,
    items: tuple[EvidenceItem, ...],
    lot_ids: tuple[str, ...],
    previous_seal_digest: str,
) -> str:
    body = [
        case_reference,
        seal_seq,
        sealed_by,
        _utc(sealed_at).isoformat(),
        [
            [i.item_id, i.kind, i.source, _utc(i.captured_at).isoformat(), i.content_hash]
            for i in items
        ],
        list(lot_ids),
        previous_seal_digest,
    ]
    raw = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha512(raw).hexdigest()


def freeze_bundle(
    case_reference: str,
    sealed_by: str,
    sealed_at: datetime,
    items: Iterable[EvidenceItem],
    lot_ids: Iterable[str] = (),
    previous: SealedEvidence | None = None,
) -> SealedEvidence:
    """Freeze evidence once. A frozen bundle is content-addressed by its digest;
    any later mutation changes the digest and breaks seal verification."""
    rows = tuple(items)
    if not rows:
        raise ValueError("cannot seal an empty evidence bundle")
    seen: set[str] = set()
    for item in rows:
        item.validate()
        if item.item_id in seen:
            raise ValueError(f"duplicate evidence item: {item.item_id}")
        seen.add(item.item_id)
    at = _utc(sealed_at)
    if previous is not None:
        if previous.case_reference != case_reference:
            raise ValueError("previous seal belongs to another case")
        if at < _utc(previous.sealed_at):
            raise ValueError("seal time moved backward")
        prev_digest = previous.digest
        seal_seq = previous.seal_seq + 1
    else:
        prev_digest = GENESIS
        seal_seq = 1
    digest = seal_digest(
        case_reference, seal_seq, sealed_by, at, rows, tuple(lot_ids), prev_digest
    )
    return SealedEvidence(
        case_reference,
        seal_seq,
        sealed_by,
        at,
        rows,
        tuple(lot_ids),
        prev_digest,
        digest,
    )


def verify_seal(seal: SealedEvidence) -> bool:
    expected = seal_digest(
        seal.case_reference,
        seal.seal_seq,
        seal.sealed_by,
        seal.sealed_at,
        seal.items,
        seal.lot_ids,
        seal.previous_seal_digest,
    )
    return expected == seal.digest


def verify_chain(seals: Iterable[SealedEvidence]) -> bool:
    expected_seq = 1
    prev = GENESIS
    for seal in sorted(seals, key=lambda s: s.seal_seq):
        if seal.seal_seq != expected_seq:
            return False
        if seal.previous_seal_digest != prev:
            return False
        if not verify_seal(seal):
            return False
        prev = seal.digest
        expected_seq += 1
    return True


def find_item(seal: SealedEvidence, item_id: str) -> EvidenceItem | None:
    for item in seal.items:
        if item.item_id == item_id:
            return item
    return None


def blast_radius(seal: SealedEvidence, edges: Iterable[CaseEdge]) -> frozenset[str]:
    """Lots reachable from the sealed lots over the case evidence graph.
    Because the digest binds lot_ids and items, a tampered scope cannot match."""
    if not verify_seal(seal):
        raise ValueError("seal verification failed")
    return frozenset(reachable(seal.lot_ids, edges))
