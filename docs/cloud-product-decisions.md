# Cloud product and operating decisions

Status: **open decision record**. This document supplies options and evidence
fields. It does not choose a provider, region, recovery commitment, payment
provider, price, billing unit or free allowance. No customer agreement is recorded.

The product order remains local MCP, Matters Cloud, then separately operated
private company installations. Local use stays account-free. Standard ChatGPT
web uses a Matters Cloud account; private endpoints are configured by Matters.

## Hosting and recovery

| Decision | Options to compare | Evidence needed before a decision |
| --- | --- | --- |
| Provider | Hetzner or another suitable provider; managed database versus an operated database | Actual compute, disk, traffic, backup and operator costs; availability and support requirements |
| Region | A region acceptable to the intended users and any data-location requirements | User locations, latency measurements and contractual requirements |
| Availability | One recoverable installation versus redundant application/database capacity | Pilot load, outage cost, operator coverage and tested failure behavior |
| Recovery point | A backup schedule and, if required, continuous archive/replay | Acceptable lost work, write rate, retained backups and observed recovery points |
| Recovery time | Restore to replacement capacity versus a pre-provisioned recovery system | Timed drills at representative size, dependencies and staffing |
| Production roles | Separate app, migration, backup and controlled support roles | Reviewed grants and refused-operation tests |
| Account lifecycle | Verified sign-up, recovery, deletion and revocation processes | Abuse model, support responsibilities, retention policy and end-to-end tests |

Record the selected option, decision owner, date, supporting evidence and review
trigger when a decision is actually made. Leave unanswered fields open. A draft
runbook or disposable localhost TLS installation does not satisfy those decisions.

## Pilot measurements and cost inputs

Collect a bounded, consented pilot window. Preserve the unit and source of every
number; leave unavailable values unknown. The example in
`ops/cloud/pilot-cost-inputs.example.json` contains no invented measurements.

| Input | Unit and source | Why it matters |
| --- | --- | --- |
| Active accounts and memberships | Counts per observation window | Adoption and the scope charged to a customer |
| Graphs, matters and conditions | Counts and size distributions | Storage and realistic graph-operation workload |
| Stored graph/database size | Bytes; engine/database measurement | Database, index, retention and backup capacity |
| Operations | Counts by HTTP/MCP, read/write and outcome | Workload; requests must not be double-counted across adapters |
| Operation latency and errors | Median/tail durations and failure counts | Capacity and service quality; health probes are a separate series |
| Compute | CPU time or vCPU-hours and memory usage | Hosting capacity and shared-service cost |
| Network | Outbound bytes by relevant destination class | Traffic cost and privacy boundaries |
| Backups | Encrypted bytes, retained copies, capture/restore duration | Storage and recovery costs |
| Model work funded by Matters | Requests, provider/model, input/output tokens and cache usage when applicable | Optional generation cost; a ChatGPT subscription is not server API funding |
| Human operations | Time spent on support, upgrades and incidents | Ongoing cost beyond the server bill |

Record actual supplier quotes as currency, amount, unit, effective date and
source. Convert them to the same observation window before comparing options.
Show assumptions separately from measurements and avoid a false precision total
when a cost input is missing. Fixture measurements from a recovery test are
engineering evidence, not sufficient evidence for a free allowance or price.

## Free allowance and subscriptions

The free allowance is limited, but its quantities and charging scope remain
undecided. Compare a simple subscription, included resource allowance and any
additional-use model against observed cost and user needs. Decide whether usage
belongs to an individual, a project or a future organization account. Define
whether optional model work is included or separately enabled before offering it.

Record the billing unit, allowance, paid entitlements, cancellation timing,
payment-failure behavior and dispute/support process only after approval. Do not
insert sample prices or numerical limits into application behavior as if they
were selected policy. Policy-dependent billing implementation waits for those
inputs; an endpoint returning invented subscription states would not be progress.

## Proposed billing and entitlement boundary

Billing belongs above graph rules and storage. A provider adapter verifies the
chosen provider's signed event format and maps verified events into internal
entitlement changes. Signature algorithms, event names and payload fields cannot
be finalized before a provider is selected.

The implementation should provide these properties once that decision exists:

- Authenticate events before parsing them into account changes. Keep signing
  secrets outside the repository and support reviewed rotation.
- Store provider identity plus event identity under a uniqueness constraint so
  retries do not duplicate grants, charges or usage. Apply event recording and
  entitlement updates in one database transaction.
- Reject or park unknown events; reconcile out-of-order state with an
  authoritative provider record. Do not infer paid status from a browser redirect.
- Separate graph permission checks from subscription entitlement checks. A paid
  account never gains access to another person's graph.
- Enforce usage atomically on the server across browser HTTP and MCP calls,
  with one usage identity so adapters cannot double-charge or bypass limits.
- Preserve stored data after cancellation, payment failure or an exceeded
  allowance. Continue authorized reads/export and explain the next action;
  destructive cleanup requires a distinct, selected retention policy.
- Record entitlement changes and reconciliation evidence without card data,
  secret values, raw account content or full provider payloads in ordinary logs.

These are acceptance properties for future implementation, not a claim that a
billing provider is connected or subscriptions are available. Keep local engine
operations usable without a Matters account or entitlement service.
