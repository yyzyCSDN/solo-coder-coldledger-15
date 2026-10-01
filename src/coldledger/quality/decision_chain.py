from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib, json
from typing import Iterable

GENESIS = "GENESIS-DECISION"
OUTCOMES = ("release", "reject")


def _utc(at: datetime) -> datetime:
    if at.tzinfo is None:
        raise ValueError("timezone-aware datetime required")
    return at.astimezone(timezone.utc)


@dataclass(frozen=True)
class BoardDecision:
    case_reference: str
    seq: int
    outcome: str  # release | reject
    decided_by: str  # panel chair or escalation officer
    panel_revision: int
    vote_tally: tuple[tuple[str, str], ...]  # (staff_id, position)
    evidence_digest: str  # the sealed evidence bundle this decision rests on
    reason: str
    at: datetime
    supersedes_seq: int | None  # None -> first decision; never overwrite
    previous_digest: str
    digest: str


def decision_digest(
    case_reference: str,
    seq: int,
    outcome: str,
    decided_by: str,
    panel_revision: int,
    vote_tally: tuple[tuple[str, str], ...],
    evidence_digest: str,
    reason: str,
    at: datetime,
    supersedes_seq: int | None,
    previous_digest: str,
) -> str:
    body = [
        case_reference,
        seq,
        outcome,
        decided_by,
        panel_revision,
        [list(v) for v in vote_tally],
        evidence_digest,
        reason,
        _utc(at).isoformat(),
        supersedes_seq,
        previous_digest,
    ]
    raw = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha512(raw).hexdigest()


def issue(
    case_reference: str,
    outcome: str,
    decided_by: str,
    panel_revision: int,
    vote_tally: Iterable[tuple[str, str]],
    evidence_digest: str,
    reason: str,
    at: datetime,
    prior: Iterable[BoardDecision] = (),
    supersedes_seq: int | None = None,
    expected_seq: int | None = None,
) -> BoardDecision:
    """Issue a decision onto the append-only chain.

    A new decision never modifies an old one. The first decision has
    supersedes_seq=None; every later decision must explicitly supersede an
    existing one. Concurrent issuance is rejected via expected_seq."""
    if outcome not in OUTCOMES:
        raise ValueError("outcome must be release or reject")
    if not reason:
        raise ValueError("decision reason required")
    if not evidence_digest:
        raise ValueError("decision must reference sealed evidence")
    rows = sorted(prior, key=lambda d: d.seq)
    chain_seqs = [d.seq for d in rows]
    if any(a != b for a, b in enumerate(chain_seqs, start=1)):
        raise ValueError("prior decisions are not a contiguous chain")
    seq = len(rows) + 1
    if expected_seq is not None and expected_seq != len(rows):
        raise RuntimeError("decision chain concurrency conflict")
    prev_digest = rows[-1].digest if rows else GENESIS
    if seq == 1:
        if supersedes_seq is not None:
            raise ValueError("first decision cannot supersede")
    else:
        if supersedes_seq is None:
            raise ValueError("new decision must explicitly supersede a prior one")
        if supersedes_seq not in chain_seqs:
            raise ValueError("superseded decision does not exist")
        superseded = rows[supersedes_seq - 1]
        if superseded.outcome == outcome:
            raise ValueError("superseding decision must change the outcome")
    when = _utc(at)
    if rows and when < _utc(rows[-1].at):
        raise ValueError("decision time moved backward")
    tally_rows = tuple(sorted(vote_tally))
    digest = decision_digest(
        case_reference, seq, outcome, decided_by, panel_revision, tally_rows,
        evidence_digest, reason, when, supersedes_seq, prev_digest,
    )
    return BoardDecision(
        case_reference, seq, outcome, decided_by, panel_revision, tally_rows,
        evidence_digest, reason, when, supersedes_seq, prev_digest, digest,
    )


def verify_chain(rows: Iterable[BoardDecision]) -> bool:
    prev = GENESIS
    for expected_seq, d in enumerate(sorted(rows, key=lambda x: x.seq), start=1):
        if d.seq != expected_seq:
            return False
        if d.previous_digest != prev:
            return False
        if decision_digest(
            d.case_reference, d.seq, d.outcome, d.decided_by, d.panel_revision,
            d.vote_tally, d.evidence_digest, d.reason, d.at,
            d.supersedes_seq, d.previous_digest,
        ) != d.digest:
            return False
        prev = d.digest
    return True


def current(rows: Iterable[BoardDecision]) -> BoardDecision | None:
    rows = sorted(rows, key=lambda d: d.seq)
    return rows[-1] if rows else None


def superseded_seqs(rows: Iterable[BoardDecision]) -> frozenset[int]:
    return frozenset(d.supersedes_seq for d in rows if d.supersedes_seq is not None)


def lineage(rows: Iterable[BoardDecision], seq: int) -> tuple[BoardDecision, ...]:
    """Walk the supersede links from a decision back to the original."""
    by_seq = {d.seq: d for d in rows}
    if seq not in by_seq:
        raise ValueError("unknown decision seq")
    out: list[BoardDecision] = []
    cur: int | None = seq
    while cur is not None:
        d = by_seq[cur]
        out.append(d)
        cur = d.supersedes_seq
    return tuple(out)
