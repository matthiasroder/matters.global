# Cloud staging and recovery preparation

These tools exercise an isolated installation on the operator's machine. They
do not select a hosting provider, publish a service, approve a recovery target,
or establish a company agreement. Production operational acceptance stays open.

Normal ChatGPT web access uses a Matters Cloud account and the hosted endpoint.
Private company endpoints are configured and operated by Matters. These staging
instructions are for operators, not customer local-tunnel onboarding.

## Operator environment

Use a POSIX operator environment, Docker at the managed local socket,
Python 3.10 or newer, OpenSSL, and a
separate Python environment for the recovery tool:

```sh
python -m venv /tmp/matters-operations-venv
/tmp/matters-operations-venv/bin/pip install -r ops/cloud/requirements.txt
```

The extra dependency is for authenticated backup encryption only; the application
does not acquire a new mandatory dependency. The key is an operator-owned file
containing exactly 32 random bytes, with mode 0600 or stricter, outside the
repository. Do not pass encryption keys, database passwords or tokens on command
lines, put them in logs, or commit them.

The scripts require the database container to carry an exact
`global.matters.operations-owner` label. A different or missing label refuses
the operation. They use the local Docker socket and preserve registry/proxy
configuration while ignoring stale Docker client TLS/context selectors. They
never restart the daemon, prune global resources or infer ownership from a
container's name alone.

The authenticated backup manifest also records its installation owner. Restore
refuses another owner's labelled container even when a key can decrypt the
archive. Replacement recovery capacity must carry the same approved installation
identity; moving graphs between different installations is a separate authorized
export/import operation.

## Local HTTPS staging

Build the reviewed application image with the Cloud and MCP extras:

```sh
docker build -f Dockerfile.cloud -t matters-cloud:staging .
python ops/cloud/staging.py prepare \
  --directory /tmp/matters-staging-run \
  --image matters-cloud:staging --port 19443
```

The destination must be new and outside the repository. Preparation creates a
mode-0700 directory containing a private environment file, an encryption key and
a two-day localhost certificate. The certificate key is readable by the
non-root proxy through its read-only mount; the protected host directory limits
host access. No other service receives that key. The helper prints paths and an
operations-owner label, never secret values.

Choose a distinct Compose project name for the run and keep it for recovery:

```sh
docker compose --project-name matters-staging-run \
  --env-file /tmp/matters-staging-run/staging.env \
  -f ops/cloud/staging.compose.yml config --quiet
docker compose --project-name matters-staging-run \
  --env-file /tmp/matters-staging-run/staging.env \
  -f ops/cloud/staging.compose.yml up --build -d
python ops/cloud/staging.py probe \
  --origin https://localhost:19443 \
  --ca-file /tmp/matters-staging-run/server.crt
```

The probe verifies the certificate and hostname; it does not disable TLS checks
or follow redirects. It reports only sample counts and health latency. The
certificate is trusted for that probe alone; no machine-wide trust is installed.

Only the TLS proxy is published, on loopback. Application and PostgreSQL ports
stay inside this project's network. The app receives an exact HTTPS origin and
uses secure cookies. The proxy preserves that host and sets the forwarded scheme.
Its small image copies only the public proxy configuration with explicit read
permissions; TLS private keys are supplied at runtime, never baked into it.
The app and proxy are non-root, have read-only root filesystems, bounded temporary
storage, dropped capabilities and process limits. CPU/memory/process values in
the Compose file are disposable staging bounds, not selected customer quotas or
production sizing.

Proxy request/error logging is disabled so OAuth query values, account identifiers
and cookies are not emitted as ordinary request diagnostics. Use health and
aggregate probes; any temporary detailed diagnostics need a reviewed redaction
and retention procedure. Health checks prove availability,
not authorization. Verify an incorrect browser origin is refused, test account
separation, and check only aggregate metrics before using a real pilot.

The database initialization administrator is a disposable staging role. A
production installation needs separately reviewed application, migration, backup
and support roles. Configure client IP handling only for the trusted proxy and
add deployment-wide abuse controls; process-local attempt limits are not a
shared protection across workers.

If a managed build proxy needs an additional CA, use a temporary Dockerfile with
a BuildKit secret for that build step. Keep certificate verification enabled;
do not put the session CA in image layers or change production trust defaults.

## Encrypted PostgreSQL backup

Resolve the selected database container from the exact project rather than
guessing a name:

```sh
MATTERS_DB_CONTAINER="$(docker compose --project-name matters-staging-run --env-file /tmp/matters-staging-run/staging.env -f ops/cloud/staging.compose.yml ps -q database)"
```

Use the operations-owner value printed by preparation. It is a label, not a
password. Keep all following artifacts outside the checkout:

```sh
/tmp/matters-operations-venv/bin/python ops/cloud/recovery.py backup \
  --container "$MATTERS_DB_CONTAINER" --owner THE_PRINTED_OWNER \
  --username matters_ops --database matters \
  --key-file /tmp/matters-staging-run/backup.key \
  --output /tmp/matters-staging-run/backup.enc
```

For a database that requires password authentication inside its container, mount
an operator-managed protected PostgreSQL password file and pass its container
path with `--passfile /run/secrets/pgpass`. The helper never puts a database
password or connection URL in process arguments.

The tool runs `pg_dump` in custom format, records the database schema version,
and encrypts archive plus manifest with AES-256-GCM. The version header is
authenticated as well. Wrong keys, modifications and truncation are refused
before any restore database is created. The encrypted output is mode 0600 and
is published exclusively: an existing file or symlink is never replaced. Raw
archives are held only in bounded process memory and are not written to disk.

The default 64 MiB byte bound and 120-second command bound protect the operator
machine during this preparation. They are adjustable with `--max-bytes` and
`--timeout`; selecting larger values requires enough memory. They are not a
backup size promise or product allowance. Large production databases may need
a reviewed streaming backup path instead.

Retain encryption keys separately from encrypted backups, with controlled access,
rotation history and a tested recovery path. Losing the key loses the backup.
The tool does not select a secret manager or off-site storage provider.

## Restore to a new isolated database

Restore never replaces an existing database. Its target must start with
`restore_`, differ from the source, and match the explicit confirmation:

```sh
/tmp/matters-operations-venv/bin/python ops/cloud/recovery.py restore \
  --container "$MATTERS_DB_CONTAINER" --owner THE_PRINTED_OWNER \
  --username matters_ops --database restore_drill \
  --confirm-database restore_drill \
  --key-file /tmp/matters-staging-run/backup.key \
  --source /tmp/matters-staging-run/backup.enc
```

The archive is restored in one PostgreSQL transaction. Schema versions 1 and 2
are supported; other versions require a reviewed tool update. The destination
receives a unique ownership comment. Ordinary failures attempt cleanup only when
that exact comment is unchanged. There is no force-drop operation. A process
kill or another active database connection can leave an isolated destination;
inspect ownership and active connections before any manual cleanup.

Restored browser sessions are deleted by default. For schema 2, pending OAuth
requests, grants, codes and access/refresh credentials are removed while client
registrations are retained. An old revoked login or issued token must not become
valid again merely because a backup was restored. Users sign in and authorize
again. Do not route application traffic to the restored database until validation
finishes.

The result reports schema version and aggregate database/account/graph/event
measurements. Check expected graph content and ownership through the same
application release, perform fresh sign-in, verify old credentials are refused,
and demonstrate an authorized edit plus cross-account refusal. Test the result
after process restart. Keep migrations separate from a backup capture; do not
silently upgrade a restored schema or run an older binary against a newer one.

For a drill, quiesce the disposable app before capture so known fixture counts
can be compared. On a live installation, an independently sampled row count may
change while the consistent `pg_dump` snapshot is taken. Record capture time,
restore duration, data verification and the observed lost-work window. A fast
fixture drill does not establish the service's RPO or RTO.

## Measurements and resource evidence

Use aggregate database inputs without account emails, graph titles, source text,
tokens or request URLs:

```sh
/tmp/matters-operations-venv/bin/python ops/cloud/recovery.py sample \
  --container "$MATTERS_DB_CONTAINER" --owner THE_PRINTED_OWNER \
  --username matters_ops --database matters
```

The sample reports database bytes, accounts, graphs and graph-event count.
These are current database totals, not active-account counts for a billing
window. Graph events measure recorded changes, not every HTTP or MCP call. The HTTPS
probe measures synthetic health latency, not user-facing latency guarantees.
Capture CPU/memory/disk/network aggregates through the selected runtime's metrics,
and add request counters labelled by interface, operation and outcome rather
than user content or high-cardinality identifiers.

Use [the product decision record](cloud-product-decisions.md) to collect a real
pilot window and cost inputs. Logs or synthetic fixtures cannot substitute for
an approved commercial policy.

## Stop and clean up

Stop only the exact run's Compose project. Normal `down` retains its database
volume. Deleting a volume or the external secret directory destroys that run's
data or its decryption key and needs an explicit operator decision. Do not
delete other projects, use global prune, or reuse an operations-owner value for
an unrelated installation. Private company procedures are described in
[Private hosted Matters](private-hosted.md).
