# Controlled pilot behind an existing Caddy proxy

This prepares a new installation on an operator-selected Docker server. The
existing proxy keeps ports 80/443 and its current sites. No hosting provider,
recovery promise, allowance or subscription price is chosen by this tool.
It does not connect a ChatGPT account or modify an existing proxy.

Use the reviewed application image with Cloud and MCP extras. The image must
also include `--disable-signup`: this initial pilot disables public account
registration. Operators can create explicitly approved smoke-test accounts
through controlled administrative access; general enrollment is a later,
explicit step. Sign-in and OAuth consent remain available to existing accounts.
This flag does not replace the pilot's access and storage policy. Default mail
configuration is absent, so email delivery/recovery is unavailable until an
operator configures a reviewed TLS SMTP service.

## Prepare private artifacts

Use Python 3.10+, Docker Compose v2 and the same reviewed Matters release as the
image. Install the operator dependencies in a private virtual environment:

```sh
python3 -m venv /opt/matters-operator-venv
/opt/matters-operator-venv/bin/pip install -e '.[cloud]' -r ops/cloud/requirements.txt
```

First record the actual Caddy configuration source/imports, Compose project,
network attachments, selected hostname/DNS ownership, TLS issuance/renewal
procedure, available disk and measured co-tenant load. An image-baked Caddyfile
needs a persistent, reviewed replacement/override; copying a file into the
running container or reloading an unpersisted configuration is insufficient.
Keep the old site's configuration, mounts and networks intact.

Choose **two unused private IPv4 /28 networks** that overlap neither existing
Docker networks nor host/VPN routes. The ingress network is dedicated to this
Matters app and the existing edge; the backend is dedicated to app/PostgreSQL.
The examples below are placeholders to replace with those inspected values:

```sh
/opt/matters-operator-venv/bin/python ops/cloud/pilot.py prepare \
  --directory /opt/matters-private/NEW-pilot \
  --project matters-pilot \
  --origin https://YOUR-MATTERS-HOST \
  --image YOUR-REVIEWED-IMAGE:TAG \
  --ingress-network matters-pilot-edge \
  --ingress-subnet 10.244.30.0/28 \
  --backend-subnet 10.244.31.0/28 \
  --proxy-ip 10.244.30.3
```

The destination must be new and outside the checkout. It becomes mode 0700;
every generated file is mode 0600. Four distinct passwords, `backup.key`,
`pilot.env`, `operator.json`, a Compose copy, `Caddyfile.site` and a separate `Caddyfile.logging.merge` stay there.
Only non-secret inventory is printed. Do not print/interpolate the environment
file, passwords or a complete Docker inspect/config result into logs or chat.
An image digest is preferable to a mutable tag for a recorded release.

Preparation is local file generation only. It neither verifies DNS/network
availability nor creates networks, starts containers, attaches Caddy, obtains a
certificate or publishes a service. The operator owns those explicit steps.
The app takes ingress address `.4`; `.1` is reserved for the gateway. Select a
different usable fixed proxy address. The backend uses app `.3` and database
`.4`. Do not put unrelated applications on either network.

The default app and database bounds are each 512 MiB, 0.75 CPU and 128 processes.
They are provisional pilot bounds, not a capacity guarantee or customer quota.
Adjust with `--app-memory-mib`, `--database-memory-mib`, `--app-cpus`,
`--database-cpus`, `--app-pids-limit`, `--database-pids-limit` after measuring
headroom for the existing services. Account password hashing, image build/cache,
PostgreSQL growth and retained backups also consume server resources.

## Provision the fresh database, then start the app

Create the dedicated ingress network with the exact subnet and the generated
`operations_owner` label. Record that network and the proxy's fixed address in
the existing edge's **authoritative Compose override/configuration**, so a later
edge recreation preserves them. These are operator actions, not performed by
`pilot.py`. Check for existing project containers, database volumes and network
names before starting; do not adopt an unrelated resource with a matching name.

Use the selected project name and private generated files consistently, in a
clean operator shell with no exported `MATTERS_*` or conflicting `COMPOSE_*`
settings. Shell values override `--env-file` during Compose interpolation. The
provision helper clears those ambient selectors itself:

```sh
docker compose --project-name matters-pilot \
  --env-file /opt/matters-private/NEW-pilot/pilot.env \
  -f /opt/matters-private/NEW-pilot/pilot.compose.yml config --quiet

docker compose --project-name matters-pilot \
  --env-file /opt/matters-private/NEW-pilot/pilot.env \
  -f /opt/matters-private/NEW-pilot/pilot.compose.yml up --wait -d database

/opt/matters-operator-venv/bin/python ops/cloud/pilot.py provision \
  --directory /opt/matters-private/NEW-pilot

docker compose --project-name matters-pilot \
  --env-file /opt/matters-private/NEW-pilot/pilot.env \
  -f /opt/matters-private/NEW-pilot/pilot.compose.yml up --wait -d app
```

Provisioning resolves only the stored project's database service and checks its
exact operations-owner label. It reuses the reviewed fresh-only
`database_roles.py` policy: new database, new app/migration/backup roles, no
existing passwords or grants changed. A pre-existing `roles.json` refuses a
rerun. An interrupted run requires ownership inspection, not deletion of the
inventory to force a retry. The database schema is prepared before the runtime
app starts; its health check uses the selected public Host header.

The administrator password is supplied only to PostgreSQL through a protected
file. The app receives its runtime-role database URL, never the administrator,
migration, backup or encryption-key secrets. The app runs non-root with a
read-only root filesystem, bounded temporary storage and dropped capabilities.
Neither app nor database publishes a host port. PostgreSQL is only on the
internal backend network; app and the controlled edge share ingress.

## Integrate and validate the edge route

`Caddyfile.site` is a **proposed site block**, not a complete replacement for the
existing Caddyfile. It routes the exact HTTPS hostname to the generated unique
app alias, sets Host to the configured authority (including a non-default port),
preserves Authorization, and overwrites X-Forwarded-For with
the edge's observed peer. The app trusts exactly the chosen proxy `/32`. Preserve
the original site's imports and any existing trusted upstream configuration;
review the real ingress chain if another proxy/CDN is in front of Caddy.

Integrate this site and the dedicated network through the actual persistent
configuration source. For an image-baked source, a reviewed Compose override
can bind a complete combined configuration that retains the original site;
validate it using the actual Caddy binary/version before changing the edge.
Do not assume a container port 2019 is a safe/public administration endpoint.
Keep the previous configuration and a concrete edge-only rollback available.

The site has its own named access logger with discarded output. Access-log
suppression alone does **not** protect Caddy runtime errors: upstream failures
can attach a complete request to the error event. `Caddyfile.logging.merge` is a
separate proposal to merge into the existing global logging block. Exclude only
this named site's error namespace from the default logger
so request-bearing runtime errors for this site are suppressed. Preserve all
co-tenant logger settings and validate the final adapted configuration with the
actual Caddy version. Do not load this artifact as a second global block or
replace the existing root configuration with it. Verify an induced Matters
upstream failure omits URLs, query values, headers and credentials while the
original site's logging remains as configured. If other configured loggers
also consume this namespace, exclude it there too; do not add another unfiltered
copy. Use aggregate health/error checks for this pilot rather than request-bearing
diagnostics. Review other proxy/debug
logging as well; do not enable diagnostics that bypass this filtering. Do not disable TLS certificate
verification for probes. No new ports 80/443 listener or second proxy is needed.

Before announcing the endpoint, verify DNS and trusted TLS, `/healthz`, browser
sign-in, an authorized graph edit and cross-account refusal, OAuth metadata with
the exact HTTPS issuer and `/mcp` resource, unauthenticated MCP refusal and
forwarded-address spoof refusal. Verify disabled sign-up and the original
site's behavior, then recreate the edge/app in a controlled rehearsal to prove
route/network/data persistence. A server-side SDK rehearsal does not establish
that a live ChatGPT connection has been completed.

## Backup, recovery, monitoring and rollback

Use the existing [role-aware operations guide](cloud-operations.md) for encrypted
backup as the backup role and restore to a **new** database. The container must
carry this installation's operations-owner label. Keep the encryption key
separate from encrypted archives; `backup.key` beside preparation secrets is
not an off-host backup strategy. Select and actually test an independently
protected off-host copy, retention, key access and restore location before
claiming recovery readiness. Record RPO/RTO only when approved and measured.

After restore, revoke sessions, issued OAuth grants and account action tokens,
retain recorded verification/graph data, and reapply reviewed runtime/backup
grants using the exact restore ownership marker. Use
[operator monitoring](cloud-monitoring.md) with an explicitly chosen backup-age
threshold and alert recipient. Collect resource/usage aggregates without graph
contents or credentials; commercial allowance and billing enforcement remain
separate product decisions.

For rollback, stop only the exact Matters app/project, remove or revert only its
new site/network entries through the persistent edge configuration, validate
and restore the edge's previous configuration, and check the original site.
Keep the Matters database volume and private inventory for investigation or an
authorized retry. Normal Compose `down` retains its volume. Do not use global
prune, force-drop databases, `down --volumes`, or disconnect unrelated networks.
Deleting any pilot volume/key needs an explicit data-retention decision.
