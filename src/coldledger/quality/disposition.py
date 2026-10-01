"""Temperature-excursion disposition workflow.

Chains case state, a frozen evidence graph, conflict-checked independent
review panels, tie-broken ballots and final release into one hash-chained
audit trail. Decisions are append-only: a later decision may supersede an
earlier one, never overwrite it.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from threading import RLock
import hashlib
import json

STATES = ('opened', 'evidence_frozen', 'deliberating', 'decided', 'closed')
VOTES = ('approve', 'reject', 'abstain')
CONFLICT_SCOPES = ('case', 'product', 'lot')


def _utc(v: datetime) -> datetime:
    if v.tzinfo is None or v.utcoffset() is None:
        raise ValueError('timezone-aware datetime required')
    return v.astimezone(timezone.utc)


@dataclass(frozen=True)
class EvidenceItem:
    key: str
    kind: str
    submitted_by: str
    at: datetime
    digest: str

    def validate(self) -> None:
        if not self.key or not self.kind:
            raise ValueError('evidence key and kind required')
        if not self.submitted_by:
            raise ValueError('evidence submitter required')
        if not self.digest:
            raise ValueError('evidence content digest required')


@dataclass(frozen=True)
class EvidenceLink:
    source: str
    target: str
    kind: str = 'supports'


@dataclass(frozen=True)
class Reviewer:
    key: str
    team: str
    roles: frozenset = frozenset()


@dataclass(frozen=True)
class Conflict:
    reviewer: str
    scope: str
    value: str
    reason: str
    at: datetime


@dataclass(frozen=True)
class Ballot:
    reviewer: str
    vote: str
    rationale: str
    round: int
    at: datetime


@dataclass(frozen=True)
class Decision:
    key: str
    case: str
    round: int
    outcome: str
    rule: str
    ballots: tuple
    tie_break: Ballot | None
    evidence_digest: str
    supersedes: str | None
    rationale: str
    at: datetime


@dataclass(frozen=True)
class Release:
    case: str
    decision: str
    outcome: str
    evidence_digest: str
    actor: str
    at: datetime


@dataclass(frozen=True)
class DispositionPolicy:
    panel_size: int = 3
    quorum: int = 2
    tie_break_roles: frozenset = frozenset({'quality_head'})

    def validate(self) -> None:
        if self.panel_size < 3 or self.panel_size % 2 == 0:
            raise ValueError('panel size must be odd and at least 3')
        if not 1 <= self.quorum <= self.panel_size:
            raise ValueError('quorum outside 1..panel_size')
        if not self.tie_break_roles:
            raise ValueError('tie-break role required')


@dataclass(frozen=True)
class DispositionCase:
    key: str
    product: str
    lot: str
    owner: str
    opened_at: datetime
    state: str = 'opened'
    revision: int = 0
    round: int = 1
    evidence: tuple = ()
    links: tuple = ()
    evidence_digest: str | None = None
    panel: tuple = ()
    ballots: tuple = ()
    decisions: tuple = ()
    releases: tuple = ()
    updated_at: datetime | None = None


@dataclass(frozen=True)
class AuditRecord:
    index: int
    case: str
    actor: str
    action: str
    at: datetime
    payload: tuple
    previous: str
    digest: str


def _evidence_digest(items: tuple, links: tuple) -> str:
    payload = {
        'items': sorted((i.key, i.kind, i.submitted_by, _utc(i.at).isoformat(), i.digest) for i in items),
        'links': sorted((l.source, l.target, l.kind) for l in links),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(raw).hexdigest()


def _audit_digest(index, case, actor, action, at, payload, previous) -> str:
    raw = json.dumps([index, case, actor, action, _utc(at).isoformat(), list(payload), previous],
                     separators=(',', ':')).encode()
    return hashlib.sha256(raw).hexdigest()


def tally(case: DispositionCase, policy: DispositionPolicy) -> str:
    """One of 'pending', 'approved', 'rejected', 'tie' for the current round."""
    if len(case.ballots) < len(case.panel):
        return 'pending'
    votes = [b for b in case.ballots if b.vote != 'abstain']
    approve = sum(1 for b in votes if b.vote == 'approve')
    reject = sum(1 for b in votes if b.vote == 'reject')
    if len(votes) < policy.quorum or approve == reject:
        return 'tie'
    return 'approved' if approve > reject else 'rejected'


class DispositionBoard:
    def __init__(self, policy: DispositionPolicy | None = None, roster: tuple = ()):
        self._policy = policy or DispositionPolicy()
        self._policy.validate()
        self._roster = {r.key: r for r in roster}
        if len(self._roster) != len(tuple(roster)):
            raise ValueError('duplicate reviewer in roster')
        self._cases: dict[str, DispositionCase] = {}
        self._conflicts: list[Conflict] = []
        self._audit: list[AuditRecord] = []
        self._lock = RLock()

    # -- audit chain -----------------------------------------------------
    def _record(self, case: str, actor: str, action: str, at: datetime, payload: tuple) -> AuditRecord:
        previous = self._audit[-1].digest if self._audit else 'GENESIS'
        index = len(self._audit) + 1
        record = AuditRecord(index, case, actor, action, _utc(at), payload, previous,
                             _audit_digest(index, case, actor, action, at, payload, previous))
        self._audit.append(record)
        return record

    def audit_trail(self) -> tuple:
        return tuple(self._audit)

    def audit_for(self, key: str) -> tuple:
        return tuple(r for r in self._audit if r.case == key)

    def verify(self) -> bool:
        previous = 'GENESIS'
        for r in self._audit:
            if r.previous != previous:
                return False
            if r.digest != _audit_digest(r.index, r.case, r.actor, r.action, r.at, r.payload, previous):
                return False
            previous = r.digest
        return True

    # -- case helpers ----------------------------------------------------
    def get(self, key: str) -> DispositionCase:
        with self._lock:
            return self._cases[key]

    def _case(self, key: str) -> DispositionCase:
        try:
            return self._cases[key]
        except KeyError:
            raise ValueError(f'unknown case: {key}') from None

    @staticmethod
    def _check_revision(case: DispositionCase, expected_revision: int) -> None:
        if expected_revision != case.revision:
            raise RuntimeError('optimistic concurrency conflict')

    @staticmethod
    def _check_clock(case: DispositionCase, at: datetime) -> datetime:
        at = _utc(at)
        if case.updated_at is not None and at < case.updated_at:
            raise ValueError('event time moved backward')
        return at

    def _advance(self, case: DispositionCase, at: datetime, **changes) -> DispositionCase:
        nxt = replace(case, revision=case.revision + 1, updated_at=at, **changes)
        self._cases[case.key] = nxt
        return nxt

    # -- conflicts of interest -------------------------------------------
    def declare_conflict(self, reviewer: str, scope: str, value: str, reason: str, at: datetime) -> Conflict:
        with self._lock:
            if reviewer not in self._roster:
                raise ValueError('unknown reviewer')
            if scope not in CONFLICT_SCOPES:
                raise ValueError('unknown conflict scope')
            if not value or not reason:
                raise ValueError('conflict value and reason required')
            conflict = Conflict(reviewer, scope, value, reason, _utc(at))
            self._conflicts.append(conflict)
            self._record('*', reviewer, 'conflict_declared', at,
                         (('scope', scope), ('value', value), ('reason', reason)))
            return conflict

    def _conflict_for(self, reviewer: str, case: DispositionCase) -> Conflict | None:
        for c in self._conflicts:
            if c.reviewer != reviewer:
                continue
            if (c.scope == 'case' and c.value == case.key) or \
               (c.scope == 'product' and c.value == case.product) or \
               (c.scope == 'lot' and c.value == case.lot):
                return c
        return None

    def _check_eligible(self, case: DispositionCase, reviewer: str) -> None:
        if reviewer not in self._roster:
            raise ValueError(f'unknown reviewer: {reviewer}')
        if reviewer == case.owner:
            raise ValueError('case owner cannot review own case')
        if reviewer in {e.submitted_by for e in case.evidence}:
            raise ValueError('evidence submitter cannot review')
        conflict = self._conflict_for(reviewer, case)
        if conflict is not None:
            raise ValueError(f'conflict of interest: {conflict.reason}')

    def eligible_reviewers(self, key: str) -> tuple:
        with self._lock:
            case = self._case(key)
            out = []
            for rk in sorted(self._roster):
                try:
                    self._check_eligible(case, rk)
                except ValueError:
                    continue
                out.append(rk)
            return tuple(out)

    # -- lifecycle ---------------------------------------------------------
    def open_case(self, key: str, product: str, lot: str, owner: str, at: datetime) -> DispositionCase:
        with self._lock:
            if key in self._cases:
                raise ValueError('case already exists')
            if not key or not product or not lot or not owner:
                raise ValueError('key, product, lot and owner required')
            case = DispositionCase(key, product, lot, owner, _utc(at), updated_at=_utc(at))
            self._cases[key] = case
            self._record(key, owner, 'case_opened', at,
                         (('product', product), ('lot', lot)))
            return case

    def add_evidence(self, key: str, item: EvidenceItem, actor: str, at: datetime,
                     expected_revision: int) -> DispositionCase:
        with self._lock:
            case = self._case(key)
            self._check_revision(case, expected_revision)
            at = self._check_clock(case, at)
            if case.state != 'opened':
                raise ValueError('evidence frozen: amendment requires reopen')
            item.validate()
            if item.key in {e.key for e in case.evidence}:
                raise ValueError('duplicate evidence key')
            nxt = self._advance(case, at, evidence=case.evidence + (item,))
            self._record(key, actor, 'evidence_added', at,
                         (('evidence', item.key), ('kind', item.kind), ('submitted_by', item.submitted_by)))
            return nxt

    def link_evidence(self, key: str, link: EvidenceLink, actor: str, at: datetime,
                      expected_revision: int) -> DispositionCase:
        with self._lock:
            case = self._case(key)
            self._check_revision(case, expected_revision)
            at = self._check_clock(case, at)
            if case.state != 'opened':
                raise ValueError('evidence frozen: amendment requires reopen')
            keys = {e.key for e in case.evidence}
            if link.source not in keys or link.target not in keys:
                raise ValueError('link references unknown evidence')
            if link.source == link.target:
                raise ValueError('self link')
            if link in case.links:
                raise ValueError('duplicate link')
            nxt = self._advance(case, at, links=case.links + (link,))
            self._record(key, actor, 'evidence_linked', at,
                         (('source', link.source), ('target', link.target), ('kind', link.kind)))
            return nxt

    def freeze_evidence(self, key: str, actor: str, at: datetime, expected_revision: int) -> DispositionCase:
        with self._lock:
            case = self._case(key)
            self._check_revision(case, expected_revision)
            at = self._check_clock(case, at)
            if case.state != 'opened':
                raise ValueError(f'cannot freeze from state {case.state}')
            if not case.evidence:
                raise ValueError('cannot freeze empty evidence')
            digest = _evidence_digest(case.evidence, case.links)
            nxt = self._advance(case, at, state='evidence_frozen', evidence_digest=digest)
            self._record(key, actor, 'evidence_frozen', at,
                         (('digest', digest), ('items', str(len(case.evidence)))))
            return nxt

    def assign_panel(self, key: str, reviewers: tuple, actor: str, at: datetime,
                     expected_revision: int) -> DispositionCase:
        with self._lock:
            case = self._case(key)
            self._check_revision(case, expected_revision)
            at = self._check_clock(case, at)
            first = case.state == 'evidence_frozen'
            reassign = case.state == 'deliberating' and not case.ballots
            if not (first or reassign):
                raise ValueError(f'cannot assign panel from state {case.state}')
            members = tuple(sorted(reviewers))
            if len(members) != self._policy.panel_size:
                raise ValueError(f'panel must have exactly {self._policy.panel_size} members')
            if len(set(members)) != len(members):
                raise ValueError('duplicate panel member')
            for m in members:
                self._check_eligible(case, m)
            nxt = self._advance(case, at, state='deliberating', panel=members)
            self._record(key, actor, 'panel_assigned' if first else 'panel_reassigned', at,
                         (('members', ','.join(members)), ('round', str(case.round))))
            return nxt

    def cast_ballot(self, key: str, reviewer: str, vote: str, rationale: str, at: datetime,
                    expected_revision: int) -> DispositionCase:
        with self._lock:
            case = self._case(key)
            self._check_revision(case, expected_revision)
            at = self._check_clock(case, at)
            if case.state != 'deliberating':
                raise ValueError(f'cannot vote from state {case.state}')
            if reviewer not in case.panel:
                raise ValueError('not a panel member')
            if vote not in VOTES:
                raise ValueError('unknown vote')
            if not rationale:
                raise ValueError('ballot rationale required')
            if reviewer in {b.reviewer for b in case.ballots}:
                raise ValueError('ballot already cast this round')
            ballot = Ballot(reviewer, vote, rationale, case.round, at)
            nxt = self._advance(case, at, ballots=case.ballots + (ballot,))
            self._record(key, reviewer, 'ballot_cast', at,
                         (('vote', vote), ('round', str(case.round)), ('rationale', rationale)))
            return nxt

    def _decide(self, case: DispositionCase, outcome: str, rule: str, tie_break: Ballot | None,
                rationale: str, at: datetime) -> DispositionCase:
        supersedes = case.decisions[-1].key if case.decisions else None
        decision = Decision(f'{case.key}:D{case.round}', case.key, case.round, outcome, rule,
                            case.ballots, tie_break, case.evidence_digest, supersedes, rationale, at)
        nxt = self._advance(case, at, state='decided', decisions=case.decisions + (decision,))
        self._record(case.key, tie_break.reviewer if tie_break else case.owner, 'decision_recorded', at,
                     (('decision', decision.key), ('outcome', outcome), ('rule', rule),
                      ('supersedes', supersedes or ''), ('evidence_digest', case.evidence_digest or '')))
        return nxt

    def record_decision(self, key: str, actor: str, rationale: str, at: datetime,
                        expected_revision: int) -> DispositionCase:
        with self._lock:
            case = self._case(key)
            self._check_revision(case, expected_revision)
            at = self._check_clock(case, at)
            if case.state != 'deliberating':
                raise ValueError(f'cannot decide from state {case.state}')
            outcome = tally(case, self._policy)
            if outcome == 'pending':
                raise ValueError('panel ballots incomplete')
            if outcome == 'tie':
                raise ValueError('tie or quorum miss: resolve via tie-break')
            return self._decide(case, outcome, 'panel_majority', None, rationale, at)

    def resolve_tie(self, key: str, tie_breaker: str, vote: str, rationale: str, at: datetime,
                    expected_revision: int) -> DispositionCase:
        with self._lock:
            case = self._case(key)
            self._check_revision(case, expected_revision)
            at = self._check_clock(case, at)
            if case.state != 'deliberating':
                raise ValueError(f'cannot tie-break from state {case.state}')
            if tally(case, self._policy) != 'tie':
                raise ValueError('no tie to resolve')
            if vote not in ('approve', 'reject'):
                raise ValueError('tie-break vote must be approve or reject')
            if not rationale:
                raise ValueError('tie-break rationale required')
            if tie_breaker in case.panel:
                raise ValueError('tie-breaker must be independent of panel')
            self._check_eligible(case, tie_breaker)
            roles = self._roster[tie_breaker].roles
            if not roles & self._policy.tie_break_roles:
                raise ValueError('tie-breaker lacks required role')
            ballot = Ballot(tie_breaker, vote, rationale, case.round, at)
            outcome = 'approved' if vote == 'approve' else 'rejected'
            nxt = self._decide(case, outcome, 'tie_break', ballot, rationale, at)
            self._record(key, tie_breaker, 'tie_resolved', at,
                         (('vote', vote), ('round', str(case.round))))
            return nxt

    def reopen(self, key: str, actor: str, reason: str, at: datetime, expected_revision: int,
               new_evidence: tuple = (), new_links: tuple = ()) -> DispositionCase:
        """Open a new round whose decision will supersede the current one."""
        with self._lock:
            case = self._case(key)
            self._check_revision(case, expected_revision)
            at = self._check_clock(case, at)
            if case.state != 'decided':
                raise ValueError(f'cannot reopen from state {case.state}')
            if not reason:
                raise ValueError('reopen reason required')
            evidence, links = case.evidence, case.links
            for item in new_evidence:
                item.validate()
                if item.key in {e.key for e in evidence}:
                    raise ValueError('duplicate evidence key')
                evidence = evidence + (item,)
            keys = {e.key for e in evidence}
            for link in new_links:
                if link.source not in keys or link.target not in keys:
                    raise ValueError('link references unknown evidence')
                if link.source == link.target:
                    raise ValueError('self link')
                links = links + (link,)
            for member in case.panel:
                if member in {e.submitted_by for e in new_evidence}:
                    raise ValueError(f'panel member {member} submitted new evidence: reassign first')
            digest = _evidence_digest(evidence, links)
            supersedes = case.decisions[-1].key
            nxt = self._advance(case, at, state='deliberating', round=case.round + 1,
                                evidence=evidence, links=links, evidence_digest=digest, ballots=())
            self._record(key, actor, 'case_reopened', at,
                         (('reason', reason), ('supersedes', supersedes), ('round', str(nxt.round)),
                          ('new_evidence', str(len(new_evidence))), ('evidence_digest', digest)))
            return nxt

    def finalize(self, key: str, actor: str, at: datetime, expected_revision: int) -> Release:
        with self._lock:
            case = self._case(key)
            self._check_revision(case, expected_revision)
            at = self._check_clock(case, at)
            if case.state != 'decided':
                raise ValueError(f'cannot finalize from state {case.state}')
            decision = case.decisions[-1]
            outcome = 'released' if decision.outcome == 'approved' else 'rejected'
            release = Release(key, decision.key, outcome, decision.evidence_digest, actor, at)
            self._advance(case, at, state='closed', releases=case.releases + (release,))
            self._record(key, actor, 'case_finalized', at,
                         (('decision', decision.key), ('outcome', outcome),
                          ('evidence_digest', decision.evidence_digest)))
            return release

    # -- queries -----------------------------------------------------------
    def current_decision(self, key: str) -> Decision | None:
        case = self._case(key)
        return case.decisions[-1] if case.decisions else None

    def decision_chain(self, key: str) -> tuple:
        return self._case(key).decisions

    def release_status(self, key: str) -> str | None:
        case = self._case(key)
        return case.releases[-1].outcome if case.releases else None
