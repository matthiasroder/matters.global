# Hosted Matters MCP

The hosted adapter exposes the same 16 tools, input schemas and version-1 result
envelopes as [local MCP](local-mcp.md). It uses the same engine rules through the
database graph repository. It adds authenticated Streamable HTTP at `/mcp`.

This is implemented software for an operator-run installation. It does not mean
a public Matters service has been deployed or that a live ChatGPT connection
has been verified. Those require an actual endpoint, certificates, accounts and
the relevant workspace permissions.

## Start an installation

Install the optional hosted and MCP dependencies from the checkout:

```sh
python -m pip install -e '.[cloud,mcp]'
```

For a local demonstration with a new development database:

```sh
matters cloud --mcp --development \
  --database-url sqlite:///matters-pilot.sqlite \
  --host 127.0.0.1 --port 8766
```

Open `http://127.0.0.1:8766/`, create an account, then create a personal or
project graph. The MCP endpoint is `http://127.0.0.1:8766/mcp`. Keep the exact
configured origin: `localhost` and `127.0.0.1` are different OAuth origins and
resources. Development mode permits insecure cookies only on loopback.

For an operator-managed HTTPS installation, set `MATTERS_CLOUD_DATABASE_URL`
through the deployment's secret configuration, then run:

```sh
matters cloud --mcp \
  --public-origin https://matters.example \
  --host 0.0.0.0 --port 3000
```

`matters.example` is a placeholder. A trusted reverse proxy must provide HTTPS
at the configured origin and preserve the intended Host. The application port
is an internal HTTP listener; the `--public-origin` option does not create TLS
or provision certificates. Without `--mcp`, ordinary Cloud mode does not enable
the OAuth or MCP endpoints. Existing local commands need neither option nor a
Matters account.

See [Cloud foundation](cloud-foundation.md) for database versions, deployment
examples, backups and the public-launch work that remains.

## Connect a client

An MCP client discovers the protected resource from the `WWW-Authenticate`
challenge returned by `/mcp`. The installation provides protected-resource and
authorization-server metadata, public-client registration, authorization-code
flow with S256 PKCE, explicit graph consent, token refresh and revocation.

The account holder signs in to Matters and chooses the exact graphs and access
requested by the client. Consent is tied to that signed-in browser and its
anti-forgery token. The client cannot choose an acting account by passing a user
ID to a tool.

The supported scopes are:

| Scope | Meaning |
|---|---|
| `graphs:read` | Inspect and export selected graphs that the account can currently read. |
| `graphs:write` | Edit selected graphs that the account can currently edit; includes read access. |
| `offline_access` | Request a rotating refresh token for the same grant. |

An issued token is bound to the configured `/mcp` resource. A token for another
installation or resource is refused. Account session cookies are not accepted
as MCP bearer tokens.

For the intended Matters Cloud flow, Matters operates the public HTTPS
endpoint. A standard ChatGPT web user creates a Matters Cloud account, connects
the service through OAuth and selects their graphs. They do not run a local
server, publish a port or configure a tunnel. This checkout implements the
server software; operating that public service and verifying a real ChatGPT
connection remain separate work.

Codex clients can connect to an operator's HTTPS `/mcp` URL with explicit scopes:

```sh
codex mcp add matters-cloud --url https://matters.example/mcp
codex mcp login matters-cloud --scopes graphs:read,graphs:write,offline_access
```

Private company installations may need operator-managed connectivity. The
[local guide](local-mcp.md) explains tunnel and network options as an operator
reference, including that tool results sent to ChatGPT leave the installation.
These are exceptions for managed private access, not the standard Matters Cloud
customer setup.

## Reference a graph from a project

Use the same credential-free [project reference format](local-mcp.md#project-graph-references)
as local Matters. The [hosted example](../examples/mcp/hosted-project-graph-reference.json)
uses the connection name `matters-cloud` and an illustrative graph UUID. Replace
that UUID with the intended graph ID returned by `list_graphs`; it is not an
existing example graph or an authorization grant.

Put the reference in the material supplied to a ChatGPT project or in an
explicitly selected local directory reference. ChatGPT does not automatically
send its project identity. The client must read the reference, select its
configured connection and pass `graph_id` on every graph call. Keep passwords,
session cookies and OAuth tokens out of this file. The server still checks the
authenticated account, selected-graph grant and current graph permissions.

## Tool behavior

First call `list_graphs` and choose a returned graph ID. Every graph operation
requires `graph_id`; every mutation also requires `expected_revision` from the
reviewed snapshot. Condition indexes are 0-based and truth values must be JSON
booleans. Caller-supplied paths, actor IDs and unsupported fields are refused.

The selected-graph grant restricts the graph catalogue and every tool call.
Current database membership is checked as well: granting a connection access
does not let a viewer edit, preserve removed membership or add unselected
graphs. Token/grant validity is rechecked for each tool call, including invalid
or unknown tool requests. A grant revoked after HTTP authentication is still
refused at the tool boundary.

Graph failures retain the local structured error format: `isError: true` with
`structuredContent.error.code` and `.message`. Examples include `conflict` for
stale revisions, `forbidden` for an unselected graph or read-only grant, and
`not_found` for missing or no-longer-readable repository graphs. Clients must
reread and reassess a conflict rather than silently retrying the same edit.

Transport authentication failures use HTTP 401 and an OAuth discovery
challenge. An unavailable authorization store returns a safe HTTP 503. Graph
contents, credentials, raw exceptions and bearer tokens are not included in
those authentication messages.

Hosted tool descriptors advertise their OAuth scopes. A read-only connection
that attempts an edit receives an insufficient-scope challenge in tool result
metadata, so compatible clients can request broader consent. Removed graph
membership is a separate permission failure and does not trigger a scope-upgrade
loop. This metadata follows [OpenAI's tool authentication contract](https://developers.openai.com/plugins/build/auth#triggering-authentication-ui);
an actual ChatGPT connection still needs live verification.

`get_matter(include_sources=true)` reads stored citations. `export_graph`
returns the portable graph package with citations and without accounts,
permissions, tokens or caches. Imports are explicit account-based graph
operations that create new graph identities; no MCP tool accepts a server
destination path. There are no terminal or general filesystem tools.

## Operational boundaries

The transport uses stateless JSON responses. It does not retain MCP sessions or
session-owned credentials between requests. Each process bounds MCP request
bodies to 256 KiB and active HTTP requests to 64 by default; excess requests
receive a retryable 503. These safeguards are separate from future subscription
allowances and distributed rate limits.

The SDK enforces exact configured Host and Origin values. Authentication runs
before MCP JSON parsing and request-body validation. Normal account/browser
protections still apply to consent and other account routes. Request-target
access logs are disabled by the Cloud CLI so credentials accidentally placed in
URLs do not enter those logs.

When embedding the adapter in another ASGI app, construct
`create_hosted_mcp(oauth, repository)`, include the OAuth router first, and mount
the returned `.app` last at `/`. Explicitly enter the component's `.lifespan()`
from the parent lifespan. Do not enter both parent-managed and child-managed
lifespans for the same component. A fresh component is required after shutdown.

Automated checks use persistent OAuth records, real SDK HTTP calls, graph and
grant revocation, concurrent edits, sources/export and a real loopback HTTPS
socket with an explicitly trusted test certificate. This verifies the transport
without asserting that an external account or public deployment has been
connected.
