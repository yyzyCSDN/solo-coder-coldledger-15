import unittest
from datetime import datetime, timezone
from dataclasses import replace

from coldledger.api import ColdLedgerService
from coldledger.quality.evidence_graph import CaseEdge
from coldledger.quality.evidence_lock import (
    EvidenceItem, freeze_bundle, verify_chain as verify_seals,
)
from coldledger.quality.review_panel import (
    Reviewer, CaseContext, form_panel, conflicts_for, cast_ballot, decide,
    break_tie,
)
from coldledger.quality.decision_chain import (
    issue as issue_decision, verify_chain as verify_decisions, lineage,
)
from coldledger.quality.release_chain import (
    issue_order, verify_chain as verify_orders, released_lot_scope,
)
from coldledger.quality.adjudication import (
    AdjudicationBoard, AdjudicationError, TieEscalation,
)
from coldledger.audit.trail import AuditTrail

UTC = timezone.utc
T = lambda h, d=1: datetime(2026, 10, d, h, tzinfo=UTC)


def reviewers():
    return [
        Reviewer("qa1", "Alice", "quality", frozenset({"chair"})),
        Reviewer("qa2", "Bob", "quality"),
        Reviewer("qa3", "Carol", "quality"),
        Reviewer("wh1", "Dan", "warehouse"),  # same department as owning team
        Reviewer("log1", "Eve", "logistics"),
        Reviewer("sup1", "Finn", "supplier"),
    ]


def evidence_items(*ids):
    return tuple(
        EvidenceItem(i, "telemetry", "sensor-7", T(2), f"hash-{i}") for i in ids
    )


class PanelConflictTests(unittest.TestCase):
    def test_owner_department_and_involved_party_are_excluded(self):
        ctx = CaseContext("C1", frozenset({"warehouse"}), frozenset({"log1"}))
        panel = form_panel(ctx, reviewers(), 3, "reg", T(3))
        self.assertEqual(panel.staff(), {"qa1", "qa2", "qa3"})
        reasons = {c.staff_id: c.reason for c in conflicts_for(reviewers(), ctx)}
        self.assertEqual(reasons["wh1"], "same_department")
        self.assertEqual(reasons["log1"], "involved_party")

    def test_insufficient_independent_reviewers(self):
        ctx = CaseContext("C2", frozenset({"quality", "warehouse",
                                           "logistics", "supplier"}), frozenset())
        with self.assertRaises(ValueError):
            form_panel(ctx, reviewers(), 3, "reg", T(3))

    def test_even_or_small_panel_rejected(self):
        ctx = CaseContext("C3", frozenset({"warehouse"}), frozenset())
        with self.assertRaises(ValueError):
            form_panel(ctx, reviewers(), 2, "reg", T(3))


class EvidenceFreezeTests(unittest.TestCase):
    def test_seal_is_immutable_and_chain_verifies(self):
        s1 = freeze_bundle("C", "qa1", T(3), evidence_items("e1"), ["L1"])
        s2 = freeze_bundle("C", "qa1", T(4), evidence_items("e2"), ["L2"], s1)
        self.assertTrue(verify_seals([s2, s1]))
        tampered = replace(s1, lot_ids=("L9",))
        self.assertFalse(verify_seals([tampered, s2]))

    def test_seal_time_cannot_move_backward(self):
        s1 = freeze_bundle("C", "qa1", T(4), evidence_items("e1"), ["L1"])
        with self.assertRaises(ValueError):
            freeze_bundle("C", "qa1", T(3), evidence_items("e2"), ["L2"], s1)


class VoteTieTests(unittest.TestCase):
    def _panel(self):
        ctx = CaseContext("C4", frozenset({"warehouse"}), frozenset({"log1"}))
        return form_panel(ctx, reviewers(), 3, "reg", T(3))

    def test_majority_release(self):
        panel = self._panel()
        ballots = ()
        for staff, pos in (("qa1", "release"), ("qa2", "release"), ("qa3", "reject")):
            ballots, _ = cast_ballot(panel, ballots, staff, pos, T(5), "ok")
        result = decide(panel, ballots)
        self.assertTrue(result.settled)
        self.assertEqual(result.outcome, "release")

    def test_abstentions_cannot_release_and_split_is_a_tie(self):
        panel = self._panel()
        ballots = ()
        for staff, pos in (("qa1", "release"), ("qa2", "reject"), ("qa3", "abstain")):
            ballots, _ = cast_ballot(panel, ballots, staff, pos, T(5))
        result = decide(panel, ballots)
        self.assertFalse(result.settled)
        self.assertTrue(result.tied)

    def test_double_vote_rejected(self):
        panel = self._panel()
        ballots, _ = cast_ballot(panel, (), "qa1", "release", T(5))
        with self.assertRaises(ValueError):
            cast_ballot(panel, ballots, "qa1", "reject", T(6))

    def test_tie_breaker_must_be_independent(self):
        panel = self._panel()
        ballots = ()
        for staff, pos in (("qa1", "release"), ("qa2", "reject"), ("qa3", "abstain")):
            ballots, _ = cast_ballot(panel, ballots, staff, pos, T(5))
        t = decide(panel, ballots).tally
        with self.assertRaises(PermissionError):
            break_tie(panel, t, "qa1", "release", T(7), "reason")
        tb = break_tie(panel, t, "officer1", "reject", T(7), "additional risk review")
        self.assertEqual(tb.position, "reject")


class DecisionChainTests(unittest.TestCase):
    def test_new_decision_must_supersede_and_change_outcome(self):
        d1 = issue_decision("C", "release", "qa1", 1,
                            (("qa1", "release"), ("qa2", "release"), ("qa3", "reject")),
                            "seal1", "panel majority", T(8))
        with self.assertRaises(ValueError):
            issue_decision("C", "reject", "off1", 2,
                           (("off1", "reject"),), "seal2", "new evidence", T(9),
                           prior=[d1])  # missing supersedes_seq
        d2 = issue_decision("C", "reject", "off1", 2,
                            (("off1", "reject"),), "seal2", "new evidence", T(9),
                            prior=[d1], supersedes_seq=1)
        self.assertTrue(verify_decisions([d2, d1]))
        with self.assertRaises(ValueError):
            issue_decision("C", "reject", "off2", 3, (("off2", "reject"),),
                           "seal3", "same outcome", T(10), prior=[d1, d2],
                           supersedes_seq=2)  # must change outcome
        self.assertEqual([d.seq for d in lineage([d1, d2], 2)], [2, 1])

    def test_concurrent_append_conflicts(self):
        d1 = issue_decision("C", "release", "qa1", 1, (("qa1", "release"),),
                            "seal1", "first", T(8))
        with self.assertRaises(RuntimeError):
            issue_decision("C", "reject", "off1", 2, (("off1", "reject"),),
                           "seal2", "later", T(9), prior=[d1],
                           supersedes_seq=1, expected_seq=2)


class ReleaseChainTests(unittest.TestCase):
    def test_orders_append_only_and_scope_tracks_latest(self):
        o1 = issue_order("C", 1, ["L1", "L2"], "qa1", T(10))
        o2 = issue_order("C", 2, ["L1"], "qa1", T(12), prior=[o1],
                         supersedes_order_seq=1, note="L2 blocked")
        self.assertTrue(verify_orders([o2, o1]))
        self.assertEqual(released_lot_scope([o1, o2]), ("L1",))
        with self.assertRaises(ValueError):
            issue_order("C", 2, ["L9"], "qa1", T(13), prior=[o1, o2])


class BoardEndToEndTests(unittest.TestCase):
    def _board(self, escalation_officer: bool = False):
        board = AdjudicationBoard()
        for r in reviewers():
            board.register_reviewer(r)
        # The independent escalation officer is only registered after the panel
        # is seated in scenarios that need one, so they never land on the panel.
        if escalation_officer:
            board.register_reviewer(Reviewer("esc1", "Grace", "regulatory"))
        return board

    @staticmethod
    def _vote(board, positions, day=1, start_hour=5):
        members = sorted(board.panel("C").staff())
        for staff, pos in zip(members, positions):
            board.cast_vote("C", staff, pos, T(start_hour, day), "recorded")

    def test_full_release_flow_with_audit_chain(self):
        board = self._board()
        board.open_case("C", "wh1", T(1))
        board.triage("C", "qa1", T(2), ["warehouse"], ["wh1", "log1", "sup1"], severity=2)
        board.set_case_graph("C", [
            CaseEdge("L1", "L2"), CaseEdge("L2", "L3"),
        ])
        seal = board.freeze_evidence("C", "qa1", T(3), evidence_items("e1", "e2"),
                                     lot_ids=["L1"])
        panel = board.assign_panel("C", "reg", T(4), 3)
        self.assertEqual(panel.staff(), {"qa1", "qa2", "qa3"})
        self.assertEqual(board.state("C").state, "investigating")
        self._vote(board, ["release", "release", "reject"])
        decision = board.adjudicate("C", "qa1", T(6), "majority accepts")
        self.assertEqual(decision.outcome, "release")
        self.assertEqual(board.state("C").state, "adjudicated")
        order = board.release("C", "qa1", T(7), note="released")
        # scope is the full transitive blast radius of sealed lot L1
        self.assertEqual(set(order.lot_ids), {"L1", "L2", "L3"})
        self.assertEqual(board.state("C").state, "closed")
        self.assertEqual(set(board.released_lots("C")), {"L1", "L2", "L3"})
        checks = board.verify("C")
        self.assertTrue(all(checks.values()), checks)
        self.assertEqual(seal.digest, decision.evidence_digest)

    def test_conflicted_reviewer_cannot_be_assigned_or_vote(self):
        board = self._board()
        board.open_case("C", "wh1", T(1))
        board.triage("C", "qa1", T(2), ["warehouse"], ["wh1", "log1", "sup1"])
        board.freeze_evidence("C", "qa1", T(3), evidence_items("e1"), ["L1"])
        panel = board.assign_panel("C", "reg", T(4), 3)
        self.assertNotIn("wh1", panel.staff())
        with self.assertRaises(PermissionError):
            board.cast_vote("C", "wh1", "release", T(5))

    def test_tie_is_escalated_and_recorded_not_defaulted(self):
        board = self._board()
        board.open_case("C", "wh1", T(1))
        board.triage("C", "qa1", T(2), ["warehouse"], ["wh1", "log1", "sup1"])
        board.freeze_evidence("C", "qa1", T(3), evidence_items("e1"), ["L1"])
        board.assign_panel("C", "reg", T(4), 3)
        self._vote(board, ["release", "reject", "abstain"])
        with self.assertRaises(TieEscalation):
            board.adjudicate("C", "qa1", T(6), "tie")
        # a panel member cannot break their own tie
        with self.assertRaises(PermissionError):
            board.escalate_tie("C", "qa1", "release", T(7), "independent review")
        board.register_reviewer(Reviewer("esc1", "Grace", "regulatory"))
        decision = board.escalate_tie("C", "esc1", "reject", T(7),
                                      "precautionary per SOP")
        self.assertEqual(decision.outcome, "reject")
        with self.assertRaises(AdjudicationError):
            board.release("C", "qa1", T(8))  # reject decision blocks release
        self.assertTrue(all(board.verify("C").values()))

    def test_new_evidence_only_supersedes_never_overwrites(self):
        board = self._board()
        board.open_case("C", "wh1", T(1))
        board.triage("C", "qa1", T(2), ["warehouse"], ["wh1", "log1", "sup1"])
        board.freeze_evidence("C", "qa1", T(3), evidence_items("e1"), ["L1"])
        board.assign_panel("C", "reg", T(4), 3)
        self._vote(board, ["release", "release", "release"])
        d1 = board.adjudicate("C", "qa1", T(6), "first conclusion")
        board.release("C", "qa1", T(7), lot_ids=["L1"])
        self.assertEqual(board.state("C").state, "closed")

        # new lab result arrives: reopen with a forward revision
        board.reopen_for_supersede("C", "qa1", T(8), "lab result e2")
        self.assertEqual(board.state("C").state, "investigating")
        # evidence can only be added after reopen; the old seal stays on the chain
        board.freeze_evidence("C", "lab1", T(9), evidence_items("e2"), ["L1"])
        board.assign_panel("C", "reg", T(10), 3)  # fresh panel revision
        self._vote(board, ["reject", "reject", "reject"], day=1, start_hour=11)
        d2 = board.adjudicate("C", "qa1", T(12), "superseding conclusion")
        self.assertEqual(d2.seq, 2)
        self.assertEqual(d2.supersedes_seq, 1)
        self.assertNotEqual(d2.digest, d1.digest)
        # original decision and release order are still on the chains, untouched
        self.assertTrue(all(board.verify("C").values()))
        with self.assertRaises(AdjudicationError):
            board.release("C", "qa1", T(13))  # current decision rejects
        self.assertEqual(board.state("C").state, "adjudicated")

    def test_recusal_advances_panel_revision_and_voids_old_votes(self):
        board = self._board()
        board.open_case("C", "wh1", T(1))
        board.triage("C", "qa1", T(2), ["warehouse"], ["wh1", "log1", "sup1"])
        board.freeze_evidence("C", "qa1", T(3), evidence_items("e1"), ["L1"])
        board.assign_panel("C", "reg", T(4), 3)
        board.cast_vote("C", "qa1", "release", T(5))
        # a seated member belatedly reveals a conflict; replaced by a fresh,
        # independent reviewer registered only after the panel was formed
        board.register_reviewer(Reviewer("ind1", "Ivan", "regulatory"))
        panel = board.recuse_member("C", "qa2", "ind1", "reg", T(6))
        self.assertEqual(panel.revision, 2)
        self.assertNotIn("qa2", panel.staff())
        self.assertIn("ind1", panel.staff())
        status = board.tally_status("C")
        self.assertEqual(status.tally.release, 0)  # old-revision votes don't count
        with self.assertRaises(AdjudicationError):
            board.adjudicate("C", "qa1", T(7), "incomplete")

    def test_release_scope_cannot_exceed_frozen_evidence(self):
        board = self._board()
        board.open_case("C", "wh1", T(1))
        board.triage("C", "qa1", T(2), ["warehouse"], ["wh1", "log1", "sup1"])
        board.set_case_graph("C", [CaseEdge("L1", "L2")])
        board.freeze_evidence("C", "qa1", T(3), evidence_items("e1"), ["L1"])
        board.assign_panel("C", "reg", T(4), 3)
        self._vote(board, ["release", "release", "release"])
        board.adjudicate("C", "qa1", T(6), "ok")
        with self.assertRaises(AdjudicationError):
            board.release("C", "qa1", T(7), lot_ids=["L9"])

    def test_tampered_evidence_breaks_verification(self):
        board = self._board()
        board.open_case("C", "wh1", T(1))
        board.triage("C", "qa1", T(2), ["warehouse"], ["wh1", "log1", "sup1"])
        board.freeze_evidence("C", "qa1", T(3), evidence_items("e1"), ["L1"])
        seal = board.current_seal("C")
        bad = replace(seal, lot_ids=("L9",))
        board._files["C"].seals[0] = bad
        self.assertFalse(board.verify("C")["evidence_chain"])


class ServiceWiringTests(unittest.TestCase):
    def test_service_exposes_board(self):
        board = ColdLedgerService().adjudication_board()
        self.assertIsInstance(board, AdjudicationBoard)
        self.assertIsInstance(board.audit, AuditTrail)


if __name__ == "__main__":
    unittest.main()
