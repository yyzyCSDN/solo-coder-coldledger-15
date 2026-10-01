from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

POSITIONS = ("release", "reject", "abstain")
VOTING = POSITIONS[:2]


def _utc(at: datetime) -> datetime:
    if at.tzinfo is None:
        raise ValueError("timezone-aware datetime required")
    return at.astimezone(timezone.utc)


@dataclass(frozen=True)
class Reviewer:
    staff_id: str
    name: str
    department: str
    roles: frozenset[str] = frozenset()  # chair | member
    active: bool = True


@dataclass(frozen=True)
class CaseContext:
    reference: str
    owning_departments: frozenset[str]
    involved_parties: frozenset[str]  # staff / suppliers / carriers tied to the event
    severity: int = 1


@dataclass(frozen=True)
class Conflict:
    staff_id: str
    reason: str  # same_department | involved_party | inactive | duplicate
    detail: str = ""


@dataclass(frozen=True)
class PanelSeat:
    staff_id: str
    role: str  # chair | member
    revision: int = 1


@dataclass(frozen=True)
class Panel:
    reference: str
    size: int
    seats: tuple[PanelSeat, ...]
    revision: int = 1
    formed_by: str = ""
    formed_at: datetime | None = None
    exclusions: tuple[Conflict, ...] = ()

    def staff(self) -> frozenset[str]:
        return frozenset(s.staff_id for s in self.seats)

    def chair(self) -> str:
        return next(s.staff_id for s in self.seats if s.role == "chair")

    def seat_of(self, staff_id: str) -> PanelSeat | None:
        for seat in self.seats:
            if seat.staff_id == staff_id:
                return seat
        return None


def screen_conflicts(
    reviewers: Iterable[Reviewer], ctx: CaseContext
) -> tuple[Reviewer, ...]:
    """Independent review: nobody from the department that owns the deviation,
    nobody named as an involved party, nobody inactive. Structured rejection
    reasons are available via conflicts_for() so they land in the audit chain."""
    rejected = {c.staff_id for c in conflicts_for(list(reviewers), ctx)}
    return tuple(r for r in reviewers if r.staff_id not in rejected)


def conflicts_for(
    reviewers: Iterable[Reviewer], ctx: CaseContext
) -> tuple[Conflict, ...]:
    out: list[Conflict] = []
    seen: set[str] = set()
    for reviewer in reviewers:
        if reviewer.staff_id in seen:
            out.append(Conflict(reviewer.staff_id, "duplicate", "listed twice"))
            continue
        seen.add(reviewer.staff_id)
        if not reviewer.active:
            out.append(Conflict(reviewer.staff_id, "inactive", reviewer.department))
        elif reviewer.staff_id in ctx.involved_parties:
            out.append(Conflict(reviewer.staff_id, "involved_party", reviewer.department))
        elif reviewer.department in ctx.owning_departments:
            out.append(Conflict(reviewer.staff_id, "same_department", reviewer.department))
    return tuple(out)


def form_panel(
    ctx: CaseContext,
    reviewers: Iterable[Reviewer],
    size: int,
    formed_by: str,
    at: datetime,
    chair_id: str | None = None,
) -> Panel:
    if size < 3 or size % 2 == 0:
        raise ValueError("panel size must be an odd number >= 3")
    eligible = screen_conflicts(reviewers, ctx)
    if len(eligible) < size:
        raise ValueError("not enough independent reviewers")
    chosen = tuple(sorted(eligible, key=lambda r: r.staff_id)[:size])
    ids = {r.staff_id: r for r in chosen}
    if chair_id is None:
        chairs = [r.staff_id for r in chosen if "chair" in r.roles]
        chair_id = chairs[0] if chairs else chosen[0].staff_id
    if chair_id not in ids:
        raise ValueError("chair must be an eligible panel member")
    seats: list[PanelSeat] = []
    for reviewer in chosen:
        role = "chair" if reviewer.staff_id == chair_id else "member"
        seats.append(PanelSeat(reviewer.staff_id, role, 1))
    return Panel(
        ctx.reference,
        size,
        tuple(seats),
        revision=1,
        formed_by=formed_by,
        formed_at=_utc(at),
        exclusions=conflicts_for(reviewers, ctx),
    )


def replace_member(
    panel: Panel,
    leaving: str,
    replacement: Reviewer,
    ctx: CaseContext,
    by: str,
    at: datetime,
) -> Panel:
    seat = panel.seat_of(leaving)
    if seat is None:
        raise ValueError("not a current panel member")
    if replacement.staff_id in panel.staff() - {leaving}:
        raise ValueError("replacement already sits on the panel")
    if not screen_conflicts([replacement], ctx):
        raise ValueError("replacement reviewer has a conflict of interest")
    new_seats = tuple(
        PanelSeat(
            replacement.staff_id,
            seat.role,
            panel.revision + 1,
        )
        if s.staff_id == leaving
        else PanelSeat(s.staff_id, s.role, s.revision)
        for s in panel.seats
    )
    return Panel(
        panel.reference,
        panel.size,
        new_seats,
        revision=panel.revision + 1,
        formed_by=by,
        formed_at=_utc(at),
        exclusions=panel.exclusions,
    )


@dataclass(frozen=True)
class Ballot:
    staff_id: str
    position: str
    at: datetime
    panel_revision: int
    rationale: str = ""
    evidence_digest: str = ""


@dataclass(frozen=True)
class Tally:
    panel_revision: int
    votes: tuple[Ballot, ...]
    release: int
    reject: int
    abstain: int
    quorum: int

    def is_complete(self) -> bool:
        return len(self.votes) >= self.quorum


@dataclass(frozen=True)
class VoteResult:
    tally: Tally
    settled: bool
    outcome: str | None  # release | reject | None (tie / pending)
    tied: bool = False


def cast_ballot(
    panel: Panel,
    ballots: Iterable[Ballot],
    staff_id: str,
    position: str,
    at: datetime,
    rationale: str = "",
    evidence_digest: str = "",
) -> tuple[tuple[Ballot, ...], Ballot]:
    seat = panel.seat_of(staff_id)
    if seat is None:
        raise PermissionError("reviewer is not on the panel")
    if position not in POSITIONS:
        raise ValueError("position must be release, reject or abstain")
    rows = tuple(ballots)
    current = [b for b in rows if b.panel_revision == panel.revision]
    if any(b.staff_id == staff_id for b in current):
        raise ValueError("reviewer has already voted on this panel revision")
    ballot = Ballot(
        staff_id, position, _utc(at), panel.revision, rationale, evidence_digest
    )
    return rows + (ballot,), ballot


def tally(panel: Panel, ballots: Iterable[Ballot]) -> Tally:
    rows = tuple(b for b in ballots if b.panel_revision == panel.revision)
    panel_ids = panel.staff()
    release = sum(1 for b in rows if b.position == "release")
    reject = sum(1 for b in rows if b.position == "reject")
    abstain = sum(1 for b in rows if b.position == "abstain")
    if any(b.staff_id not in panel_ids for b in rows):
        raise ValueError("ballot from outside the panel")
    return Tally(panel.revision, tuple(sorted(rows, key=lambda b: b.staff_id)),
                 release, reject, abstain, quorum=panel.size)


def decide(panel: Panel, ballots: Iterable[Ballot]) -> VoteResult:
    t = tally(panel, ballots)
    if not t.is_complete():
        return VoteResult(t, False, None)
    majority = panel.size // 2 + 1
    if t.release >= majority:
        return VoteResult(t, True, "release")
    if t.reject >= majority:
        return VoteResult(t, True, "reject")
    # full panel split or blocked by abstentions -> tie, escalate, never default
    return VoteResult(t, False, None, tied=True)


@dataclass(frozen=True)
class TieBreak:
    decided_by: str
    position: str
    at: datetime
    rationale: str
    evidence_digest: str


def break_tie(
    panel: Panel,
    t: Tally,
    tie_breaker: str,
    position: str,
    at: datetime,
    rationale: str,
    evidence_digest: str = "",
) -> TieBreak:
    """A tie is never silently resolved. An independent escalation officer
    (external to the panel) decides on the record."""
    if tie_breaker in panel.staff():
        raise PermissionError("tie-breaker must be independent of the panel")
    if position not in VOTING:
        raise ValueError("tie must be broken as release or reject")
    if not rationale:
        raise ValueError("tie-break rationale required")
    return TieBreak(tie_breaker, position, _utc(at), rationale, evidence_digest)
