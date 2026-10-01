import unittest
from dataclasses import replace
from datetime import datetime, timezone, timedelta

from coldledger.quality.disposition import (
    DispositionBoard, DispositionPolicy, EvidenceItem, EvidenceLink,
    Reviewer, tally,
)

UTC = timezone.utc
T0 = datetime(2026, 9, 1, 8, tzinfo=UTC)


def t(hours):
    return T0 + timedelta(hours=hours)


ROSTER = (
    Reviewer('alice', 'qa'),
    Reviewer('bob', 'qa'),
    Reviewer('carol', 'micro'),
    Reviewer('dave', 'micro'),
    Reviewer('erin', 'ops'),
    Reviewer('head', 'qa', frozenset({'quality_head'})),
)


def make_board(policy=None):
    return DispositionBoard(policy or DispositionPolicy(), ROSTER)


def seed_case(board, key='DEV-1', owner='erin'):
    board.open_case(key, 'VAC-01', 'LOT-9', owner, t(0))
    board.add_evidence(key, EvidenceItem('e1', 'telemetry', 'ops-tech', t(1), 'sha256:aaa'), 'ops-tech', t(1), 0)
    board.add_evidence(key, EvidenceItem('e2', 'photo', 'qa-alice', t(2), 'sha256:bbb'), 'qa-alice', t(2), 1)
    board.link_evidence(key, EvidenceLink('e1', 'e2'), 'ops-tech', t(2), 2)
    board.freeze_evidence(key, 'qa-mgr', t(3), 3)
    return board.get(key)


def vote_through(board, key, votes, start_rev, at=None):
    rev = start_rev
    for reviewer, vote in votes:
        board.cast_ballot(key, reviewer, vote, f'{vote} per review', at or t(10), rev)
        rev += 1
    return rev


class DispositionTests(unittest.TestCase):
    def test_happy_path_release(self):
        b = make_board()
        case = seed_case(b)
        self.assertEqual(case.state, 'evidence_frozen')
        b.assign_panel('DEV-1', ('alice', 'bob', 'carol'), 'qa-mgr', t(4), 4)
        rev = vote_through(b, 'DEV-1', [('alice', 'approve'), ('bob', 'approve'), ('carol', 'reject')], 5)
        b.record_decision('DEV-1', 'qa-mgr', 'majority approve', t(11), rev)
        decision = b.current_decision('DEV-1')
        self.assertEqual(decision.outcome, 'approved')
        self.assertEqual(decision.rule, 'panel_majority')
        self.assertIsNone(decision.supersedes)
        release = b.finalize('DEV-1', 'qa-mgr', t(12), rev + 1)
        self.assertEqual(release.outcome, 'released')
        self.assertEqual(release.decision, 'DEV-1:D1')
        self.assertEqual(release.evidence_digest, decision.evidence_digest)
        self.assertEqual(b.release_status('DEV-1'), 'released')
        self.assertTrue(b.verify())

    def test_no_single_person_decision(self):
        with self.assertRaises(ValueError):
            DispositionPolicy(panel_size=1).validate()
        with self.assertRaises(ValueError):
            DispositionPolicy(panel_size=4).validate()
        b = make_board()
        seed_case(b)
        with self.assertRaises(ValueError):
            b.assign_panel('DEV-1', ('alice',), 'qa-mgr', t(4), 4)
        with self.assertRaises(ValueError):
            b.assign_panel('DEV-1', ('alice', 'alice', 'bob'), 'qa-mgr', t(4), 4)

    def test_independence_and_conflict_of_interest(self):
        b = make_board()
        seed_case(b, owner='erin')
        # case owner excluded
        with self.assertRaises(ValueError):
            b.assign_panel('DEV-1', ('erin', 'bob', 'carol'), 'qa-mgr', t(4), 4)
        # evidence submitter excluded (e2 submitted by qa-alice -> key 'qa-alice';
        # roster key 'alice' is a different identity, so declare a real submitter)
        b2 = make_board()
        b2.open_case('DEV-2', 'VAC-01', 'LOT-9', 'erin', t(0))
        b2.add_evidence('DEV-2', EvidenceItem('e1', 'telemetry', 'alice', t(1), 'sha256:aaa'), 'alice', t(1), 0)
        b2.freeze_evidence('DEV-2', 'qa-mgr', t(2), 1)
        with self.assertRaises(ValueError):
            b2.assign_panel('DEV-2', ('alice', 'bob', 'carol'), 'qa-mgr', t(3), 2)
        # declared conflict on the lot excludes the reviewer
        b.declare_conflict('carol', 'lot', 'LOT-9', 'prior investigation involvement', t(3))
        with self.assertRaises(ValueError):
            b.assign_panel('DEV-1', ('alice', 'bob', 'carol'), 'qa-mgr', t(4), 4)
        self.assertNotIn('carol', b.eligible_reviewers('DEV-1'))
        self.assertNotIn('erin', b.eligible_reviewers('DEV-1'))
        self.assertIn('bob', b.eligible_reviewers('DEV-1'))

    def test_evidence_frozen_immutable(self):
        b = make_board()
        seed_case(b)
        with self.assertRaises(ValueError):
            b.add_evidence('DEV-1', EvidenceItem('e3', 'coa', 'lab', t(4), 'sha256:ccc'), 'lab', t(4), 4)
        with self.assertRaises(ValueError):
            b.link_evidence('DEV-1', EvidenceLink('e1', 'e1'), 'lab', t(4), 4)

    def test_tie_resolved_by_independent_head(self):
        b = make_board()
        seed_case(b)
        b.assign_panel('DEV-1', ('alice', 'bob', 'carol'), 'qa-mgr', t(4), 4)
        rev = vote_through(b, 'DEV-1', [('alice', 'approve'), ('bob', 'reject'), ('carol', 'abstain')], 5)
        self.assertEqual(tally(b.get('DEV-1'), b._policy), 'tie')
        with self.assertRaises(ValueError):
            b.record_decision('DEV-1', 'qa-mgr', 'tie attempt', t(11), rev)
        # panel member cannot tie-break
        with self.assertRaises(ValueError):
            b.resolve_tie('DEV-1', 'alice', 'approve', 'self break', t(11), rev)
        # reviewer without quality_head role cannot tie-break
        with self.assertRaises(ValueError):
            b.resolve_tie('DEV-1', 'dave', 'approve', 'no role', t(11), rev)
        b.resolve_tie('DEV-1', 'head', 'reject', 'stability data insufficient', t(11), rev)
        decision = b.current_decision('DEV-1')
        self.assertEqual(decision.outcome, 'rejected')
        self.assertEqual(decision.rule, 'tie_break')
        self.assertEqual(decision.tie_break.reviewer, 'head')
        release = b.finalize('DEV-1', 'qa-mgr', t(12), rev + 1)
        self.assertEqual(release.outcome, 'rejected')

    def test_quorum_miss_goes_to_tie_break(self):
        b = make_board()
        seed_case(b)
        b.assign_panel('DEV-1', ('alice', 'bob', 'carol'), 'qa-mgr', t(4), 4)
        rev = vote_through(b, 'DEV-1', [('alice', 'approve'), ('bob', 'abstain'), ('carol', 'abstain')], 5)
        self.assertEqual(tally(b.get('DEV-1'), b._policy), 'tie')
        b.resolve_tie('DEV-1', 'head', 'approve', 'quorum miss, head decides', t(11), rev)
        self.assertEqual(b.current_decision('DEV-1').outcome, 'approved')

    def test_supersede_only_never_overwrite(self):
        b = make_board()
        seed_case(b)
        b.assign_panel('DEV-1', ('alice', 'bob', 'carol'), 'qa-mgr', t(4), 4)
        rev = vote_through(b, 'DEV-1', [('alice', 'approve'), ('bob', 'approve'), ('carol', 'approve')], 5)
        b.record_decision('DEV-1', 'qa-mgr', 'round 1 approve', t(11), rev)
        first = b.current_decision('DEV-1')
        # cannot decide again without reopening a new round
        with self.assertRaises(ValueError):
            b.record_decision('DEV-1', 'qa-mgr', 'overwrite attempt', t(12), rev + 1)
        # new telemetry arrives -> reopen with amended evidence
        b.reopen('DEV-1', 'qa-mgr', 'new telemetry contradicts round 1', t(13), rev + 1,
                 new_evidence=(EvidenceItem('e3', 'telemetry', 'ops-tech', t(13), 'sha256:ccc'),),
                 new_links=(EvidenceLink('e3', 'e1', 'contradicts'),))
        case = b.get('DEV-1')
        self.assertEqual(case.state, 'deliberating')
        self.assertEqual(case.round, 2)
        self.assertEqual(case.ballots, ())
        self.assertNotEqual(case.evidence_digest, first.evidence_digest)
        rev = case.revision
        rev = vote_through(b, 'DEV-1', [('alice', 'reject'), ('bob', 'reject'), ('carol', 'approve')], rev, at=t(14))
        b.record_decision('DEV-1', 'qa-mgr', 'round 2 reject on new data', t(15), rev)
        chain = b.decision_chain('DEV-1')
        self.assertEqual(len(chain), 2)
        # old conclusion intact, new one points back at it
        self.assertEqual(chain[0], first)
        self.assertEqual(chain[0].outcome, 'approved')
        self.assertEqual(chain[1].supersedes, first.key)
        self.assertEqual(chain[1].outcome, 'rejected')
        self.assertEqual(b.current_decision('DEV-1'), chain[1])
        # final release follows the superseding decision
        release = b.finalize('DEV-1', 'qa-mgr', t(16), rev + 1)
        self.assertEqual(release.outcome, 'rejected')
        self.assertEqual(release.decision, 'DEV-1:D2')
        self.assertTrue(b.verify())

    def test_panel_member_cannot_submit_new_evidence(self):
        b = make_board()
        seed_case(b)
        b.assign_panel('DEV-1', ('alice', 'bob', 'carol'), 'qa-mgr', t(4), 4)
        rev = vote_through(b, 'DEV-1', [('alice', 'approve'), ('bob', 'approve'), ('carol', 'approve')], 5)
        b.record_decision('DEV-1', 'qa-mgr', 'round 1', t(11), rev)
        with self.assertRaises(ValueError):
            b.reopen('DEV-1', 'qa-mgr', 'panel conflict', t(12), rev + 1,
                     new_evidence=(EvidenceItem('e3', 'statement', 'bob', t(12), 'sha256:ddd'),))

    def test_optimistic_concurrency(self):
        b = make_board()
        seed_case(b)
        with self.assertRaises(RuntimeError):
            b.assign_panel('DEV-1', ('alice', 'bob', 'carol'), 'qa-mgr', t(4), 99)

    def test_audit_chain_detects_tamper(self):
        b = make_board()
        seed_case(b)
        b.assign_panel('DEV-1', ('alice', 'bob', 'carol'), 'qa-mgr', t(4), 4)
        self.assertTrue(b.verify())
        actions = [r.action for r in b.audit_for('DEV-1')]
        self.assertEqual(actions, ['case_opened', 'evidence_added', 'evidence_added',
                                   'evidence_linked', 'evidence_frozen', 'panel_assigned'])
        b._audit[2] = replace(b._audit[2], action='evidence_deleted')
        self.assertFalse(b.verify())

    def test_full_audit_sequence(self):
        b = make_board()
        seed_case(b)
        b.assign_panel('DEV-1', ('alice', 'bob', 'carol'), 'qa-mgr', t(4), 4)
        rev = vote_through(b, 'DEV-1', [('alice', 'approve'), ('bob', 'approve'), ('carol', 'reject')], 5)
        b.record_decision('DEV-1', 'qa-mgr', 'ok', t(11), rev)
        b.finalize('DEV-1', 'qa-mgr', t(12), rev + 1)
        trail = b.audit_trail()
        self.assertEqual(trail[0].previous, 'GENESIS')
        for prev, cur in zip(trail, trail[1:]):
            self.assertEqual(cur.previous, prev.digest)
        self.assertEqual(trail[-1].action, 'case_finalized')


if __name__ == '__main__':
    unittest.main()
