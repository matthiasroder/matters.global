# Matters Cloud foundation

This is the first account-based hosting foundation. It adds persistent accounts,
personal and project graphs, an authenticated HTTP API, graph transfer, whole-graph invitations and
a browser interface with reused Overview controls. Optional hosted MCP adds
explicit account consent for selected graphs. Graph validation and changes use the
same engine rules as local JSON graphs.

PostgreSQL is the intended hosted database. SQLite is available for local
development. The local CLI, browser and stdio MCP server continue to work without
a Matters account or subscription.

Account passwords are hashed with Argon2. Browser sessions use opaque cookies
whose server-side records contain token hashes. Graphs and their ownership live
in the database, with graph identifiers and revision checks on operations.

The foundation is not a production deployment or the complete subscription
product. Hosted MCP/OAuth, portable graph transfer, invitations and reused
Overview controls are implemented. Public hosting, a live ChatGPT web connection,
production account lifecycle, billing and quotas remain separate work. No hosting provider, region, prices,
free allowance or recovery commitment is selected by these instructions.

## ChatGPT product route

The standard ChatGPT web offering uses **Matters Cloud with a Matters account**.
Its authenticated MCP connection uses only the hosted graphs the account
holder explicitly selects and currently has permission to access. Normal customers will not be asked to configure a local tunnel or expose
their computer to the internet. The hosted transport and consent flow are implemented; a live ChatGPT web
connection still requires a publicly reachable HTTPS installation and actual
client verification. See [hosted MCP setup](hosted-mcp.md) and the
[compatibility limits](hosted-auth-compatibility.md).

Private company installations will have separately operated, isolated endpoints
configured by us. Local stdio MCP remains an account-free option for clients
that can run the local process. The technical private-connectivity discussion
in the local MCP guide does not describe standard customer onboarding.

## Local development with SQLite

From a checkout, install the optional Cloud dependencies:

```sh
python -m pip install -e '.[cloud]'
matters cloud --database-url sqlite:///./matters-cloud-dev.db --development
```

Open `http://127.0.0.1:8766` on the machine running the process. Use temporary
development accounts with passwords of at least 12 characters. Keep this database
out of source control. The default
listener is loopback; `--development` permits the non-secure cookie needed for
plain local HTTP and refuses a non-loopback listener.

The database URL may instead come from the environment:

```sh
export MATTERS_CLOUD_DATABASE_URL='sqlite:///./matters-cloud-dev.db'
matters cloud --development
```

Stop and restart the command with the same URL to reuse the accounts and graphs.
Using a different path creates a different installation. A relative SQLite URL
is relative to the process's working directory. SQLite in-memory databases are
for isolated checks, not persistent local development or a hosted deployment.

SQLite accepts `sqlite:////absolute/path/to/development.db` for an absolute path.
The parent directory must be available to the application user.

## PostgreSQL configuration

Use a dedicated database and database role for each installation. The SQLAlchemy
URL uses the Psycopg driver:

```text
postgresql+psycopg://USER:URL_ENCODED_PASSWORD@HOST:5432/DATABASE
```

Supply the complete URL through `MATTERS_CLOUD_DATABASE_URL`; do not put a live
database password in a checked-in file, command example, or image layer. Percent
encode characters in URL components when needed. Keep the database private to
the application and the operator's administration path.

The `postgresql://` shorthand is normalized to `postgresql+psycopg://` by the
application. PostgreSQL tools such as `pg_dump` use their own connection format.

An existing local PostgreSQL database can be used for development with the same
loopback application command:

```sh
matters cloud --development
```

Set `MATTERS_CLOUD_DATABASE_URL` to that database first. No external provider or
Matters account is needed to run an isolated development installation.

## Optional container setup

`Dockerfile.cloud` builds a non-root application image. It copies the package
source and packaging metadata, not local graphs, Beads, Git history or credential
files. The application listens on container port `3000`. A non-loopback listener
requires an explicit HTTPS public origin and uses secure cookies.

`compose.cloud.yml` defines a PostgreSQL 17 service with a separate persistent
volume and an application service. It does not publish the database port. The
application is published on host loopback at port `8766`, for use by an
operator-configured TLS reverse proxy. The Compose file supplies neither that
proxy nor a certificate, hostname or hosting provider.

The disposable Compose database uses its initialization administrator as the
application role. Before hosting, create separate least-privilege application,
schema-management and backup roles and review their grants. The database's
creation-time role is not a production permission design.

For a disposable installation, create a password in the shell rather than
writing it to the repository:

```sh
export MATTERS_DB_PASSWORD="$(python -c 'import secrets; print(secrets.token_hex(24))')"
export MATTERS_CLOUD_DATABASE_URL="postgresql+psycopg://matters:${MATTERS_DB_PASSWORD}@database:5432/matters"
export MATTERS_CLOUD_PUBLIC_ORIGIN='https://YOUR-CHOSEN-HOST'
docker compose -f compose.cloud.yml up --build -d
```

Replace the hostname with the actual HTTPS origin served by the reverse proxy.
Production credentials should come from the selected hosting secret-management
mechanism. Keep the password needed to reopen a persistent development volume
in an operator-controlled secret store; generating a new value does not change
an already initialized PostgreSQL volume's password.

Secure cookies are intentional: browsing the published application port over
plain HTTP is not a working sign-in configuration. Use the chosen HTTPS origin,
or run the loopback SQLite development command separately. Do not enable
development mode to bypass the container's network boundary.

To build or run the image without Compose:

```sh
docker build -f Dockerfile.cloud -t matters-cloud:local .
docker run --rm --name matters-cloud-app \
  --publish 127.0.0.1:8766:3000 \
  --env MATTERS_CLOUD_DATABASE_URL \
  matters-cloud:local \
  --host 0.0.0.0 --port 3000 --public-origin https://YOUR-CHOSEN-HOST
```

The standalone example requires a reachable database URL, not the Compose-only
`database` hostname. Shutdown uses normal SIGTERM/SIGINT handling. Stopping
Compose retains the database volume. Removing that volume deletes the stored
accounts and graphs and is a separate destructive operator action.

## HTTPS, cookies and origins

Production operation requires TLS at a reverse proxy and an explicit public
origin:

```sh
matters cloud --host 0.0.0.0 --port 3000 --public-origin https://YOUR-CHOSEN-HOST
```

Set the database URL in the environment first. An origin is just scheme, host
and optional port; it is not a URL with a path, credentials, query or fragment.
The public origin constrains permitted request hosts and browser mutation
origins. Preserve the intended host at the reverse proxy and configure forwarded
headers only for trusted proxy connections. Keep the application listener and
database off direct public ingress.

Default cookies are secure; only explicit loopback development permits plain
HTTP cookies. Sessions are persisted with hashed tokens, and passwords use
Argon2. The browser UI uses the authenticated account's access rather than
accepting an account ID as proof of authorization. Local browser shell and
arbitrary server-file-selection routes are not part of the Cloud application.

`GET /healthz` checks application/database availability and returns a small
status response. It is not proof of a valid session, correct graph permissions,
backup freshness or production readiness. Configure proxy host headers for
internal probes too; the Compose health check uses the selected public host.

The current HTTP boundary provides register/login/logout/me routes, graph
listing and creation, graph reads, derived views, revision-checked graph
operations, portable import/export, visual payloads, and owner-controlled
invitations and role changes. The session supplies the account identity; requests do not choose
an arbitrary acting user. The browser-session API remains separate from the optional OAuth and MCP
endpoints enabled with `--mcp`. Mutation bodies are JSON and are subject to request-size and origin
checks. All assets are served from the installed package.

Authentication and OAuth attempt limits use shared, atomic database counters
across workers and restarts. Recovery and verification have separate security
budgets; these are not subscription allowances. Stored counter keys are hashes,
cardinality and cleanup are bounded, and a storage failure refuses admission.

Forwarded client headers are disabled by default. Behind a proxy, explicitly
repeat `--trusted-proxy IP_OR_CIDR` for the trusted proxy addresses/networks.
The proxy must overwrite incoming forwarding headers and the application port
must remain private. Do not trust a global range or a client-supplied header as
identity. See the operator guide for the isolated staging network configuration.

## Schema changes and deployment boundaries

The database carries schema version 3. Startup initializes a new database or
validates that exact version. An existing unversioned, incomplete or unsupported
Matters schema fails closed rather than being silently upgraded or rewritten.
Version 1 and version 2 installations require an explicit upgrade after a verified backup:

```sh
matters cloud-migrate
```

Set `MATTERS_CLOUD_DATABASE_URL` to the intended installation first. This
transactional, additive upgrade preserves existing accounts, sessions, graphs,
roles, history, citations, invitations and OAuth grants while adding the missing
installation tables, including account-action and shared security-limit records. Partial
or unsupported schemas are refused. Repeating a successful upgrade is a no-op.
Before any schema upgrade, review the migration, test it on a restored copy
and record how to recover. Do not run an older
application against a newer database to attempt a rollback.

Choose immutable application and database image versions for a deployed release.
The image tags in the optional setup are development starting points; pin the
reviewed image digests and dependency set when preparing a production release.
Separate development, staging and production databases, roles, secrets and backup
destinations. Private company installations need their own boundaries as well.

## PostgreSQL backup and restore preparation

The local portable-graph format also moves individual graphs through the
Cloud API: `GET /api/graphs/{graph_id}/export` returns a bundle;
`POST /api/graphs/import` accepts `{title, kind, bundle}` and creates a new
graph owned by the authenticated account. Citations are preserved; permissions
and installation credentials are not transferred. A graph package is not
a complete installation backup: account data, graph ownership and other database
records also need recovery.

Use PostgreSQL's custom dump format for the application database. Give the
operator's PostgreSQL tools a separate `PGSERVICE` entry and protected password
file, or another approved authentication method, outside the repository. A
SQLAlchemy URL containing `+psycopg` is not a PostgreSQL CLI connection string.

With `PGSERVICE=matters-source` pointing to the source installation:

```sh
pg_dump --format=custom --no-owner --no-acl --file matters-cloud.dump
pg_restore --list matters-cloud.dump
```

Keep dumps in encrypted, access-controlled storage outside the application host
with a recorded retention policy. Database password hashes, graph contents and
session records are sensitive even though the database does not store raw
passwords or raw session tokens.

For a restore drill, create a **new, empty, isolated database** and set
`PGSERVICE=matters-restore` to that destination. Stop any test application using
it, then restore:

```sh
pg_restore --exit-on-error --single-transaction --no-owner --no-acl \
  --dbname "service=matters-restore" matters-cloud.dump
```

Do not run this against the source or an existing customer database. Use a
compatible PostgreSQL client version and record the server, client and
application versions used. For a PostgreSQL major-version change, rehearse the
documented upgrade path separately.

Start the same application release against the restored database. Verify account
sign-in, personal/project graph membership, graph counts and content, read/write
behavior and cross-user access refusal. Confirm changes survive a restart.
Record backup time, restore start/end, observed data loss and what was checked.
Also decide whether recovered sessions must be invalidated under the eventual
incident policy; this foundation does not imply a complete incident workflow.

Logical dumps do not cover host configuration, TLS keys, deployment secrets,
database roles or every infrastructure dependency. Record how those are
recovered independently. Backups and a successful drill are evidence; the target
recovery point and recovery time still require product and operational decisions.

## Decisions still required before hosting

- Provider, region and data-location requirements, including whether Hetzner fits.
- Expected load, database capacity and the availability commitment.
- Recovery point and recovery time objectives, backup schedule and retention.
- Production credential management, monitoring, alerting and incident ownership.
- TLS hostname/certificates, public authentication and deployment controls.
- Production mail/domain configuration, signup/activation policy, abuse thresholds and migration rehearsal for the chosen deployment.
- Live ChatGPT onboarding, free allowance, subscription units and prices.

This guide and its optional containers prepare those decisions. They do not
satisfy the hosting-and-recovery graph condition by themselves.

See [operations and recovery tooling](cloud-operations.md),
[open product decisions](cloud-product-decisions.md), and the
[private installation runbook](private-hosted.md) for the remaining preparation.

## Account verification and recovery

Account settings support email verification and password changes. Password reset
links use expiring single-use hashed action records. A password reset or change
revokes all browser sessions and issued OAuth grants for that account, including
concurrent issuance, while preserving graphs, ownership and verified-email state.
Another account signed into the reset browser keeps its own session.

Email delivery is optional, configured through verified-TLS SMTP and protected
operator secret files. Without it, the browser explicitly reports that recovery
and verification delivery are unavailable; authenticated password changes still
work. Requests do not return action tokens. Recovery request responses remain
generic before account lookup or delivery. Existing registration still distinguishes
a new address from an existing one; this is not a claim that every account
endpoint prevents enumeration.

The bounded in-memory delivery queue is not durable: restarting during delivery
can discard pending jobs, so the person may request another link. Action tokens
are never placed in a plaintext database outbox or ordinary logs. Link credentials
use fragments, are removed from the displayed URL, and require an explicit
same-origin confirmation. SMTP setup, account-action handling after restore and
operator limitations are described in [Cloud operations](cloud-operations.md).

Use the [monitoring tool](cloud-monitoring.md) for verified HTTPS health and
authenticated encrypted-backup freshness checks. Its alert events are local
operator output; notification routing and production thresholds remain explicit
operator configuration.

Enable `matters cloud --mail-events` to emit fixed-schema aggregate delivery
outcomes to the operator log without recipient, link, token or provider-error
fields. Counters are private to each worker and reset on process restart; they
are operational diagnostics, not billing usage or a public metrics endpoint.
