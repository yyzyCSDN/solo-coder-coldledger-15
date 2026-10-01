from __future__ import annotations
from dataclasses import replace
from datetime import datetime
from threading import RLock
from typing import Iterable

from ..audit.trail import AuditTrail
from ..quality.evidence_graph import CaseEdge
from ..quality.evidence_lock import (
    EvidenceItem, SealedEvidence, freeze_bundle, verify_chain as verify_seals,
    blast_radius,
)
from ..quality.review_panel import (
    Reviewer, CaseContext, Panel, Ballot, TieBreak, VoteResult,
    form_panel, replace_member, screen_conflicts, cast_ballot, decide, break_tie,
)
from ..quality.decision_chain import (
    BoardDecision, issue as issue_decision, verify_chain as verify_decisions,
    current as current_decision,
)
from ..quality.release_chain import (
    ReleaseOrder, issue_order, verify_chain as verify_orders,
    released_lot_scope,
)
from ..quality.excursion_state import (
    ExcursionState, Command, apply,
)


class AdjudicationError(ValueError):
    """A rule of the deviation adjudication process was violated."""


class TieEscalation(RuntimeError):
    """The panel finished without a majority; an independent officer must decide."""

    def __init__(self, reference: str, result: VoteResult):
        super().__init__(f"tie on {reference}; independent escalation required")
        self.reference = reference
        self.result = result


class _CaseFile:
    def __init__(self, state: ExcursionState):
        self.state = state
        self.context: CaseContext | None = None
        self.edges: tuple[CaseEdge, ...] = ()
        self.seals: list[SealedEvidence] = []
        self.panel: Panel | None = None
        self.ballots: tuple[Ballot, ...] = ()
        self.tiebreaks: list[TieBreak] = []
        self.decisions: list[BoardDecision] = []
        self.orders: list[ReleaseOrder] = []
        self.last_vote: VoteResult | None = None


class AdjudicationBoard:
    """Stateful orchestration of temperature-excursion deviation cases.

    Every mutation advances an optimistic-concurrency case revision and is
    recorded on a hash-chained audit trail. Evidence bundles, board decisions
    and release orders each have their own append-only hash chain: a new
    conclusion can only supersede a prior one, never overwrite it."""

    def __init__(self, audit: AuditTrail | None = None):
        self._lock = RLock()
        self._files: dict[str, _CaseFile] = {}
        self._reviewers: dict[str, Reviewer] = {}
        self.audit = audit or AuditTrail()

    # ------------------------------------------------------------------ staff
    def register_reviewer(self, reviewer: Reviewer) -> None:
        with self._lock:
            if reviewer.staff_id in self._reviewers:
                raise AdjudicationError("reviewer already registered")
            self._reviewers[reviewer.staff_id] = reviewer
            self.audit.append(reviewer.staff_id, "reviewer.register", reviewer.staff_id)

    def open_case(self, reference: str, owner: str, at: datetime,
                  reason: str = "temperature excursion detected") -> ExcursionState:
        with self._lock:
            if reference in self._files:
                raise AdjudicationError("case already open")
            state = ExcursionState(reference, "detected", 0, owner, at)
            state.validate()
            self._files[reference] = _CaseFile(state)
            self.audit.append(owner, "case.open", f"{reference}:detected", at)
            return state

    def triage(self, reference: str, actor: str, at: datetime,
               owning_departments: Iterable[str], involved_parties: Iterable[str],
               severity: int = 1) -> ExcursionState:
        with self._lock:
            f = self._require(reference)
            nxt, event = apply(f.state, Command(
                reference, f.state.revision, "triaged", actor, at,
                "triage and declare case context"))
            f.state = nxt
            f.context = CaseContext(
                reference, frozenset(owning_departments),
                frozenset(involved_parties), severity)
            self.audit.append(actor, "case.triage",
                              f"{reference}:{event.before}->{event.after}@{nxt.revision}", at)
            return nxt

    def set_case_graph(self, reference: str, edges: Iterable[CaseEdge]) -> None:
        with self._lock:
            f = self._require(reference)
            f.edges = tuple(edges)

    # ---------------------------------------------------------------- evidence
    def freeze_evidence(self, reference: str, by: str, at: datetime,
                        items: Iterable[EvidenceItem],
                        lot_ids: Iterable[str] = ()) -> SealedEvidence:
        with self._lock:
            f = self._require(reference)
            if f.state.state in ("adjudicated", "closed"):
                raise AdjudicationError(
                    "settled case must be reopened before new evidence is sealed")
            previous = f.seals[-1] if f.seals else None
            seal = freeze_bundle(reference, by, at, items, lot_ids, previous)
            f.seals.append(seal)
            self.audit.append(by, "evidence.freeze",
                              f"{reference}:seal#{seal.seal_seq}:{seal.digest[:12]}", at)
            return seal

    def current_seal(self, reference: str) -> SealedEvidence:
        with self._lock:
            f = self._require(reference)
            if not f.seals:
                raise AdjudicationError("no sealed evidence")
            return f.seals[-1]

    # ------------------------------------------------------------------ panel
    def assign_panel(self, reference: str, by: str, at: datetime, size: int,
                     chair_id: str | None = None) -> Panel:
        with self._lock:
            f = self._require(reference)
            if f.context is None:
                raise AdjudicationError("case must be triaged before panel assignment")
            if f.state.state == "detected":
                raise AdjudicationError("case must be triaged before panel assignment")
            if f.state.state in ("adjudicated", "closed"):
                raise AdjudicationError("reopen the case before seating a new panel")
            base_revision = f.panel.revision if f.panel else 0
            panel = form_panel(f.context, self._reviewers.values(), size, by, at, chair_id)
            if base_revision:
                panel = self._bump_panel_revision(panel, base_revision + 1)
            f.panel = panel
            excluded = ",".join(
                f"{c.staff_id}={c.reason}" for c in panel.exclusions) or "-"
            self.audit.append(by, "panel.form",
                              f"{reference}:rev{panel.revision}:excluded[{excluded}]", at)
            if f.state.state == "triaged":
                f.state, event = apply(f.state, Command(
                    reference, f.state.revision, "investigating", by, at,
                    "independent review panel seated"))
                self.audit.append(by, "case.transition",
                                  f"{reference}:{event.before}->{event.after}@{f.state.revision}", at)
            return f.panel

    @staticmethod
    def _bump_panel_revision(panel: Panel, revision: int) -> Panel:
        return replace(
            panel,
            revision=revision,
            seats=tuple(replace(s, revision=revision) for s in panel.seats),
        )

    def recuse_member(self, reference: str, leaving: str, replacement_id: str,
                      by: str, at: datetime) -> Panel:
        """A conflicted reviewer is replaced. The panel revision advances and
        votes from the old revision no longer count; they remain on record."""
        with self._lock:
            f = self._require(reference)
            if f.panel is None or f.context is None:
                raise AdjudicationError("no panel to amend")
            replacement = self._reviewers.get(replacement_id)
            if replacement is None:
                raise AdjudicationError("replacement reviewer is unknown")
            panel = replace_member(
                f.panel, leaving, replacement, f.context, by, at)
            f.panel = panel
            self.audit.append(by, "panel.recuse",
                              f"{reference}:rev{panel.revision}:{leaving}->{replacement_id}", at)
            return panel

    # ------------------------------------------------------------------ votes
    def cast_vote(self, reference: str, staff_id: str, position: str,
                  at: datetime, rationale: str = "") -> Ballot:
        with self._lock:
            f = self._require(reference)
            if f.panel is None:
                raise AdjudicationError("no panel seated")
            if f.state.state != "investigating":
                raise AdjudicationError("votes are only accepted during investigation")
            seal = f.seals[-1] if f.seals else None
            if seal is None:
                raise AdjudicationError("evidence must be frozen before voting")
            f.ballots, ballot = cast_ballot(
                f.panel, f.ballots, staff_id, position, at,
                rationale, seal.digest)
            self.audit.append(staff_id, "vote.cast",
                              f"{reference}:rev{f.panel.revision}:{position}", at)
            f.last_vote = decide(f.panel, f.ballots)
            return ballot

    def tally_status(self, reference: str) -> VoteResult:
        with self._lock:
            f = self._require(reference)
            if f.panel is None:
                raise AdjudicationError("no panel seated")
            return decide(f.panel, f.ballots)

    # ------------------------------------------------------------- adjudication
    def adjudicate(self, reference: str, actor: str, at: datetime,
                   reason: str) -> BoardDecision:
        """Close the panel vote into a board decision. A split panel raises
        TieEscalation instead of defaulting either way."""
        with self._lock:
            f = self._require(reference)
            if f.panel is None:
                raise AdjudicationError("no panel seated")
            if f.state.state != "investigating":
                raise AdjudicationError("case is not under investigation")
            seal = f.seals[-1]
            result = decide(f.panel, f.ballots)
            if not result.settled:
                if result.tied:
                    self.audit.append(actor, "vote.tie",
                                      f"{reference}:rev{f.panel.revision}:release{result.tally.release}/reject{result.tally.reject}/abstain{result.tally.abstain}", at)
                    raise TieEscalation(reference, result)
                raise AdjudicationError("panel vote is incomplete")
            decision = self._record_decision(reference, f, result.outcome,
                                             f.panel.chair(), at, reason)
            self.audit.append(actor, "decision.issue",
                              f"{reference}:seq{decision.seq}:{decision.outcome}", at)
            return decision

    def escalate_tie(self, reference: str, officer_id: str, position: str,
                     at: datetime, reason: str) -> BoardDecision:
        """Independent escalation officer resolves a tied panel on the record.
        The officer must be independent of the panel (and, if known to the
        board, free of the same conflicts the panel was screened for)."""
        with self._lock:
            f = self._require(reference)
            if f.panel is None:
                raise AdjudicationError("no panel seated")
            result = decide(f.panel, f.ballots)
            if not result.tied:
                raise AdjudicationError("panel is not tied")
            if officer_id in f.panel.staff():
                raise PermissionError("tie-breaker must be independent of the panel")
            officer = self._reviewers.get(officer_id)
            if officer is not None and f.context is not None:
                if officer not in screen_conflicts([officer], f.context):
                    raise AdjudicationError("escalation officer carries a conflict of interest")
            tie = break_tie(f.panel, result.tally, officer_id, position, at,
                            reason, f.seals[-1].digest)
            f.tiebreaks.append(tie)
            self.audit.append(officer_id, "tie.break",
                              f"{reference}:rev{f.panel.revision}:{position}", at)
            decision = self._record_decision(reference, f, position, officer_id,
                                             at, reason)
            self.audit.append(officer_id, "decision.escalate",
                              f"{reference}:seq{decision.seq}:{position}", at)
            return decision

    def _record_decision(self, reference: str, f: _CaseFile, outcome: str,
                         decided_by: str, at: datetime, reason: str) -> BoardDecision:
        seal = f.seals[-1]
        if not verify_seals(f.seals):
            raise AdjudicationError("evidence seal chain is broken")
        assert f.panel is not None
        tally_rows = tuple(
            (b.staff_id, b.position)
            for b in f.ballots
            if b.panel_revision == f.panel.revision
        )
        # None on the first decision; afterwards it must explicitly supersede.
        supersedes = f.decisions[-1].seq if f.decisions else None
        decision = issue_decision(
            reference, outcome, decided_by, f.panel.revision, tally_rows,
            seal.digest, reason, at, prior=f.decisions,
            supersedes_seq=supersedes, expected_seq=len(f.decisions))
        f.decisions.append(decision)
        nxt, event = apply(f.state, Command(
            reference, f.state.revision, "adjudicated", decided_by, at,
            f"decision seq {decision.seq}: {outcome}"))
        f.state = nxt
        self.audit.append(decided_by, "case.transition",
                          f"{reference}:{event.before}->{event.after}@{nxt.revision}", at)
        return decision

    # ---------------------------------------------------------------- reopen
    def reopen_for_supersede(self, reference: str, actor: str, at: datetime,
                             reason: str) -> None:
        """Reopen an adjudicated/closed case because new evidence requires a
        superseding decision. This is a forward revision with full history."""
        with self._lock:
            f = self._require(reference)
            if f.state.state not in ("adjudicated", "closed"):
                raise AdjudicationError("only a settled case may be reopened")
            nxt, event = apply(f.state, Command(
                reference, f.state.revision, "investigating", actor, at,
                f"reopen for superseding decision: {reason}"))
            f.state = nxt
            f.last_vote = None
            self.audit.append(actor, "case.reopen",
                              f"{reference}:{event.before}->{event.after}@{nxt.revision}", at)

    # ---------------------------------------------------------------- release
    def release(self, reference: str, released_by: str, at: datetime,
                lot_ids: Iterable[str] | None = None, note: str = "") -> ReleaseOrder:
        with self._lock:
            f = self._require(reference)
            if f.state.state != "adjudicated":
                raise AdjudicationError("case must be adjudicated before release")
            decision = current_decision(f.decisions)
            assert decision is not None
            if decision.outcome != "release":
                raise AdjudicationError("current decision rejects release")
            seal = f.seals[-1]
            scope = blast_radius(seal, f.edges)
            if lot_ids is None:
                chosen = scope
            else:
                chosen = frozenset(lot_ids)
                if not chosen:
                    raise AdjudicationError("release scope is empty")
                if not chosen <= scope:
                    raise AdjudicationError("release scope exceeds frozen evidence graph")
            order = issue_order(
                reference, decision.seq, chosen, released_by, at,
                prior=f.orders,
                supersedes_order_seq=f.orders[-1].order_seq if f.orders else None,
                expected_seq=len(f.orders), note=note)
            f.orders.append(order)
            nxt, event = apply(f.state, Command(
                reference, f.state.revision, "closed", released_by, at,
                f"release order {order.order_seq}"))
            f.state = nxt
            self.audit.append(released_by, "release.issue",
                              f"{reference}:order#{order.order_seq}:lots[{','.join(order.lot_ids) or '-'}]", at)
            self.audit.append(released_by, "case.transition",
                              f"{reference}:{event.before}->{event.after}@{nxt.revision}", at)
            return order

    # ------------------------------------------------------------- verification
    def verify(self, reference: str) -> dict[str, bool]:
        with self._lock:
            f = self._require(reference)
            return {
                "audit": self.audit.verify(),
                "evidence_chain": verify_seals(f.seals),
                "decision_chain": verify_decisions(f.decisions),
                "release_chain": verify_orders(f.orders),
            }

    def released_lots(self, reference: str) -> tuple[str, ...]:
        with self._lock:
            f = self._require(reference)
            return released_lot_scope(f.orders)

    def state(self, reference: str) -> ExcursionState:
        with self._lock:
            return self._require(reference).state

    def panel(self, reference: str) -> Panel:
        with self._lock:
            f = self._require(reference)
            if f.panel is None:
                raise AdjudicationError("no panel seated")
            return f.panel

    def _require(self, reference: str) -> _CaseFile:
        f = self._files.get(reference)
        if f is None:
            raise AdjudicationError("unknown case")
        return f
