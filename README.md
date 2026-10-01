# ColdLedger Pharmaceutical Cold-Chain Operations Lab

Dependency-free Python reference implementation for temperature-sensitive pharmaceutical distribution. It models inventory expiry, chain-of-custody, telemetry correction, packaging qualification, dry-ice depletion, freezer allocation, customs holds, serialization, recalls, release decisions, vehicle compartment compatibility, returns, SLA calendars and cross-dock state.

The `coldledger.quality` package additionally models the temperature-excursion deviation adjudication process:

- `evidence_lock` — evidence bundles are frozen as content-addressed, hash-chained seals; tampering breaks verification.
- `review_panel` — independent panel assignment with structured conflict-of-interest screening, ballots, majority tally, abstention/tie handling and an independent tie-break escalation.
- `decision_chain` — append-only, hash-chained board decisions; a new decision can only `supersede` a prior one (and must change its outcome), never overwrite it.
- `release_chain` — final release orders on their own append-only chain, scoped to the transitive blast radius of the sealed evidence graph; newer orders supersede older ones.
- `excursion_state` — case state machine with optimistic-concurrency revisions; settled cases may be reopened into a forward revision for a superseding decision.
- `adjudication` — `AdjudicationBoard` orchestrates case state, evidence freeze, panel assignment, votes, tie escalation, decisions and release, recording every mutation on a hash-chained `AuditTrail`.

Runtime third-party dependencies: none.
