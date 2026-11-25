# Operator monitoring rehearsal

`ops/cloud/monitor.py` checks an HTTPS ingress and an encrypted PostgreSQL
backup. It produces JSON and process exit codes for an operator scheduler or
sidecar. It sends no email, webhook or other notification and exposes no public
administration or metrics endpoint.

This is provider-independent preparation. A hosting provider, region,
availability commitment, backup schedule, recovery targets and alert recipient
have not been selected. The operator must supply the backup-age threshold.

## What a successful sample proves

By default, the health check reuses the verified-CA probe in
`ops/cloud/staging.py`. It connects directly to an exact `https://localhost[:port]` or
`https://127.0.0.1[:port]` origin, requests `/healthz`, requires HTTP 200 with
`status: ok`, verifies the certificate, ignores proxy environment settings and
refuses redirects. Credentials, paths, query strings, fragments and external
origins are refused unless explicitly enabled. Run this beside the TLS ingress
for a local check.

For an operator-selected private or public HTTPS endpoint, add
`--allow-remote-origin` and supply its exact origin and trusted CA file. This
uses direct HTTPS with certificate/hostname verification, the same fixed
`/healthz` request, no ambient proxies or redirects, a 1,024-byte response bound
and the whole-check deadline. DNS resolution is included in that deadline. No
provider or hostname is selected by the tool. A probe measures the network path
from the operator's chosen monitoring location, so local checks alone cannot
establish availability from an external customer's network.

Backup checks reuse `ops/cloud/recovery.py` to authenticate AES-GCM encryption,
validate the manifest and PostgreSQL archive hash, and compare the authenticated
creation time with the supplied age threshold. File modification time does not
make an old backup fresh. A wrong key, modified archive, oversized file, invalid
manifest or implausible future timestamp fails the check. Plaintext data and
manifest database/owner names are never written to the monitor's records.

Successful archive authentication does not prove that PostgreSQL can restore
the dump or that all application behavior survives recovery. Continue the
isolated restore drills described in [Cloud operations](cloud-operations.md).
The monitor does not restore a database or change application state.

## Run one sample

Use the same operator environment as the recovery tooling, with
`ops/cloud/requirements.txt` installed. Prepare the private TLS staging setup
and create an encrypted backup first, following the operations guide. Keep the
certificate, backup, key and monitoring state outside this repository.

For a disposable rehearsal, the following explicitly uses a **300-second test
threshold**. This is not a production backup or recovery commitment:

```sh
export MATTERS_MONITOR_MAX_BACKUP_AGE_SECONDS=300
python ops/cloud/monitor.py \
  --origin https://localhost:19443 \
  --ca-file /operator-private/matters/server.crt \
  --backup-file /operator-private/matters/latest-backup.enc \
  --key-file /operator-private/matters/backup.key \
  --state-directory /operator-private/matters/monitor-state \
  --max-backup-age-seconds "$MATTERS_MONITOR_MAX_BACKUP_AGE_SECONDS"
```

Replace the example paths with the operator's files. Backup and key files must
be regular files owned by the running user, with mode `0600` or stricter; the
key must contain exactly 32 random bytes. The CA certificate may be publicly
readable and owned by the running user or root. Symlinks, hard-linked files, symlinked parent directories and paths
inside the repository are refused.

The monitor creates the state directory with mode `0700` on first use. It binds
that directory to the origin, remote-target setting and artifact paths and writes immutable `0600`
sample records. Reuse the same directory for later samples. It does not adopt
an unrelated existing directory, replace existing operator files, follow
symlinks, or silently repair corrupted or interrupted state. A second monitor
using that directory is refused while the first holds its lock.

The backup can be rotated at the same path. Changing the origin or artifact
paths requires a new state directory so histories cannot silently mix different
installations. Preserve old records according to the eventual operator
retention policy; this tool does not delete them.

## Read the result

Each sample emits one JSON line and writes the same safe record. It contains
the check time, sequence, state/event, latency, authenticated backup age and
encrypted byte count. It does not include URLs, paths, response bodies,
credentials, keys, raw archive contents or graph data.

| Field/value | Meaning |
| --- | --- |
| `state: healthy` | HTTPS health and the authenticated backup-age check passed. |
| `state: failure` | At least one check failed, could not be read safely, or exceeded a bound. |
| `state: stale` | HTTPS passed, but the authenticated backup is older than the supplied threshold. |
| `event: healthy`, `failure`, or `stale` | Initial state, or a transition into that state. |
| `event: recovered` | The previous failure/stale state has returned to healthy. |
| `event: unchanged` | The state matches the previous sample. |

Exit code `0` means the final sample is healthy; `1` means its state is failure
or stale; `2` means configuration or private state could not be used safely.
With repeated samples, inspect all JSON events as well as the final exit code.
Errors use fixed codes; exception text and unexpected request arguments are
not echoed. The operator can connect these results to an independently chosen
alerting system after deciding who receives alerts and how escalation works.

## Time and resource bounds

The default whole-check deadline is five seconds, separately applied to HTTPS
and backup validation. Each runs in a child process which is stopped at the
deadline, including slow response-body reads or decryption. Health reads use
the existing probe's 1,024-byte bound. Backup input is capped at 64 MiB, and
child output/state records are each capped at 8 KiB. A smaller archive bound
can be supplied with `--max-backup-bytes`.

One invocation normally takes one sample. For a finite rehearsal, add explicit
settings such as `--iterations 3 --interval-seconds 5`. The tool permits at most
100 iterations, a 30-second whole-check deadline, a 60-second interval, and a
configured worst-case duration of one hour. It refuses an unbounded loop.
The state directory holds at most 10,000 samples and refuses further writes
when full. An operator scheduler can run single samples and retain/rotate
histories outside this tool.

`--clock-skew-seconds` controls permitted future clock skew, with a technical
default of 60 seconds and a maximum of one hour. Archive size, timeout and clock
bounds protect the tool; they are not product service-level decisions.

## Verification

`ops/cloud/test_monitor.py` starts an owned local TLS server with generated
certificates and encrypted fixture archives. It checks healthy/failure/stale/
recovery transitions, wrong CA and key, tampering, redirect refusal, bounded
response reads, a slow-body deadline, private append-only state, concurrent
state ownership, unsafe paths and safe exit diagnostics. It uses no personal
credentials or external notification service. The explicit remote-target
transport is exercised against the same owned TLS fixture; the tests contact
no external target.

```sh
python -m pytest -q ops/cloud/test_monitor.py
```
