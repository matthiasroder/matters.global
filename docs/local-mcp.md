# Local Matters MCP

The local MCP server lets a compatible client read and edit explicitly selected
Matters graphs through the existing Python engine. Graphs remain JSON files on
the machine running the server. No Matters account or subscription is required.
The CLI and browser continue to use those files and the same mutation rules.

This implementation supplies a **stdio MCP server**. It does not provide a
hosted account service, a public HTTP endpoint, billing, or company hosting.
Model access belongs to the calling client; ordinary graph tools do not call a
model provider or require model credentials.

## Install and start

From a checkout, install the optional MCP dependency in your environment:

```sh
python -m pip install -e '.[mcp]'
```

For an installed release, use `python -m pip install 'matters-global[mcp]'`.
Core Matters remains usable without this extra. The current local package uses
POSIX file locks and terminal facilities; this release does not add native
Windows support.

Create a disposable copy of the repository example before trying writes:

```sh
mkdir -p /tmp/matters-mcp-demo
cp examples/matters.example.json /tmp/matters-mcp-demo/project.json
matters mcp --state /tmp/matters-mcp-demo/project.json --graph-id project
```

An MCP client normally launches that command and owns its stdin/stdout. A
terminal running it directly waits for MCP messages. Stdout is reserved for
the protocol; diagnostics go to stderr. The graph must already exist. To create
a new empty graph first, use `matters init --state /absolute/path/to/new.json`.

Use `--read-only` to prohibit all graph mutations:

```sh
matters mcp --state /absolute/path/to/project.json --graph-id project --read-only
```

For several graphs, give each one an explicit connection-local identifier:

```sh
matters mcp \
  --graph personal=/absolute/path/to/personal.json \
  --graph project=/absolute/path/to/project.json
```

`--graph` cannot be combined with `--state` or `--graph-id`. No tool accepts a
filesystem path or switches a shared active graph. With a single graph and no
`--graph-id`, the ID is the SHA-256 hash of its canonical file path, matching
the existing browser identifier. Moving the file changes that default ID.
An explicit alias survives a move if the operator updates its configured path.
Aliases are scoped to the configured connection; they are not global IDs or
credentials. An alias starts with a letter or digit, followed by letters,
digits, underscores, dots or hyphens, and is at most 128 characters long.

Without `--state` or `--graph`, normal CLI path selection applies:
`MATTERS_STATE`, then `~/.local/share/matters/matters.json`. The selected file
must still exist. Repository-local graph files are not automatically selected.

## Connect Codex or another local client

Replace both absolute paths below with the executable and graph on the machine
where Codex will launch the server:

```sh
codex mcp add matters-local -- \
  /absolute/path/to/.venv/bin/matters mcp \
  --state /absolute/path/to/project.json --graph-id project
codex mcp list
```

The equivalent Codex configuration is:

```toml
[mcp_servers.matters-local]
command = "/absolute/path/to/.venv/bin/matters"
args = ["mcp", "--state", "/absolute/path/to/project.json", "--graph-id", "project"]
```

See [the configuration example](../examples/mcp/codex-config.toml). Other
clients use the same executable and argument list in their stdio MCP settings.
See [OpenAI's MCP configuration documentation](https://developers.openai.com/codex/mcp/)
for supported Codex configuration options.

Start by asking the client to list its configured Matters graphs and show the
universe of `project`. Inspect a matter and its conditions before authorizing a
write. Changes made through MCP are visible in subsequent CLI reads and browser
refreshes. Opening the browser remains a separate command:

```sh
matters web --state /absolute/path/to/project.json
```

## Version 1 tool contract

Successful graph results carry `schema_version: 1`, `graph_id`, and the graph's
`revision`. `schema_version` here versions the tool result envelope; persisted
graph JSON continues to use its own schema version. `list_graphs` is a catalogue
operation and returns the configured graphs and their revisions.

Every graph-specific tool requires `graph_id`. Every mutation additionally
requires a nonempty `expected_revision` obtained from the graph snapshot the
caller reviewed. A graph identifier alone never grants access.

| Tool | Purpose and additional inputs |
| --- | --- |
| `list_graphs` | Discover only graphs configured for this server. |
| `list_matters` | List matters with `offset` and `limit` pagination. |
| `get_matter` | Inspect `matter_id`; optionally include its source citations. |
| `universe` | Find currently actionable matters. |
| `frontier` | Find the actionable immediate dependents of `matter_id`. |
| `horizon` | Find the farthest unresolved descendants of `matter_id`. |
| `unlock` | Return actionable matters, conditions, suggested actions and blocked IDs. |
| `create_matter` | Add `matter_id` with a nonempty `conditions` array of labels and boolean truth values; an omitted condition truth defaults to false. |
| `set_condition` | Set a condition's `truth` using its `matter_id` and `index`. |
| `add_condition` | Add a condition with a `label`; `truth` defaults to false. |
| `edit_condition` | Change a condition's `label` using `matter_id` and `index`. |
| `delete_condition` | Remove a condition using `matter_id` and `index`; removing the last condition requires `confirmed`. |
| `delete_matter` | Remove `matter_id`; explicit confirmation is required and `cascade` controls removal of incident dependencies. |
| `add_dependency` | Add a `source` prerequisite to a `target` dependent. |
| `remove_dependency` | Remove the edge from `source` to `target`. |
| `export_graph` | Return a portable package in the result's `bundle` field. |

Use the discovered input schemas as the authoritative argument reference.
Read tools advertise `readOnlyHint`; mutation annotations identify write and
destructive behavior. An annotation or a caller-supplied `confirmed` value is
not a substitute for actual user authorization.

`list_matters` defaults to `offset: 0` and `limit: 100`, with a maximum page size
of 500. Its result includes `matters`, `total`, `offset` and `limit`. Refresh the
listing if pages have different revisions. `get_matter` includes the matter's
`id`, numbered conditions, prerequisites and dependents; `include_sources` adds
citations. The graph revision covers graph facts, not independently edited
citation sidecars.

The MCP result includes the same JSON payload in `structuredContent` and text
content. For example, an actionable-matter read has this shape:

```json
{
  "schema_version": 1,
  "graph_id": "project",
  "revision": "revision-returned-by-the-server",
  "matters": ["ship_first_version"]
}
```

### Conditions, dependencies and revisions

Condition `index` values are **0-based** in MCP and JSON. CLI condition numbers
are **1-based**. For example, MCP index `0` addresses the condition shown as
number `1` by `matters show`. Indices are integers, not strings or booleans.
Truth values must be JSON `true` or `false`, not `"true"`, `"false"`, `0`, or `1`.
`set_condition` specifies the desired state; there is no toggle operation.

A dependency points from prerequisite to dependent. Removing a matter's last
condition can make it resolved and unlock its dependents, so review that change
before confirming it. The engine rejects edits that would create a cycle and
retains its existing repair behavior for removing edges from a cyclic graph.

Read the graph, review the intended edit, and supply that read's revision with
the mutation. The revision is checked while holding the shared graph-write
lock. If another CLI, browser or MCP client changes the graph, a stale write
fails without applying the edit. Refresh, inspect the new conditions and
prerequisites, and reassess the action before retrying. Do not merely replace
the revision token and repeat a positional edit: another edit may have changed
which condition occupies that index.

This is conflict detection, not automatic merging or live collaboration. It
coordinates cooperating local writers; direct edits by unrelated programs do
not participate in the Matters lock protocol.

### Errors

Tool failures return MCP `isError: true` with structured error data, for example:

```json
{
  "schema_version": 1,
  "error": {
    "code": "conflict",
    "message": "The graph changed since it was displayed. Refresh the graph and review your edit before trying again."
  }
}
```

| Code | Meaning |
| --- | --- |
| `invalid` | Rejected input or an attempted invalid graph change. |
| `forbidden` | An unconfigured graph ID or a write through a read-only connection. |
| `not_found` | A configured file, matter, condition or tool is unavailable. |
| `conflict` | Stale or missing revision, duplicate creation, or required destructive confirmation. |
| `locked` | Another cooperating process holds the graph-write lock. |
| `state_cycle` | The existing graph contains a dependency cycle. |
| `unavailable` | The operation could not access a required local facility. |
| `internal` | An unexpected implementation failure; private details are omitted. |

The SDK also rejects arguments that do not match a tool's input schema.
Do not treat an error result as a successful graph update. Initialization
problems, such as a missing MCP extra or missing configured graph, are reported
on stderr and the server does not begin serving tools.

## Access boundary

The server process runs as the operating-system user who started it. That user
selects the graph allowlist and read-only setting before launch. Filesystem
permissions still apply. Tool callers can access only configured graph IDs;
they cannot select new paths, run shell commands, or use the browser terminal.
The local adapter does not create a separate user account or permission system.
Anyone entrusted with a connected client can exercise that connection's allowed
graph operations. Configure separate connections or a read-only server when
different access is needed.

## Project graph references

A directory or ChatGPT project can contain an explicitly supplied reference:

```json
{
  "schema_version": 1,
  "connection": "matters-local",
  "graph_id": "project"
}
```

See [the reference example](../examples/mcp/project-graph-reference.json). The
`connection` names an already configured MCP connection. The reference contains
no credentials and grants no access. It is a convention for agents and people,
not an automatically loaded Matters configuration file.

For a directory, the operator may choose `.matters/graph.json` and explicitly
tell the client to use it. For a ChatGPT project, supply the reference in project
material or instructions. Do not assume ChatGPT passes its project identity to
MCP. If no graph has been explicitly selected, list the permitted graphs and ask
the user to choose. Include the chosen `graph_id` on every call; do not maintain
a server-wide active graph.

## Export, import and caches

Export a versioned package to a file or stdout:

```sh
matters export --state /absolute/path/to/project.json --output project.bundle.json
matters export --state /absolute/path/to/project.json
```

`--output` refuses to replace an existing file, including the source graph.

Import it as a new graph:

```sh
matters import project.bundle.json --state /absolute/path/to/imported.json
```

The package uses `format: "matters-graph"` and `schema_version: 1`; its `graph`
uses graph schema version 2 and its `sources` holds the canonical schema version
1 citation sidecar, empty when no citations exist. This is a full-graph export,
not a redacted public view. Imports validate graph facts and dependencies through the
shared engine. The destination must be new; imports refuse to replace an
existing graph or its associated artifacts. Native graph JSON can also be
imported. Pass `-` as the import source to read standard input. Export/import
does not modify the source graph.

The package excludes installation account permissions, client connection
settings, credential files and embedding cache blobs. Graph labels and canonical
citation content are exported in full, so review the package before sharing it.
Assign destination access explicitly. Cached embeddings can be regenerated
deliberately:

```sh
matters import project.bundle.json --state /absolute/path/to/imported.json --rebuild-embeddings
```

Rebuilding uses the configured local embedding backend and may download its
model on first use. Ordinary import does not make model calls or download a
model. If rebuilding fails, treat the import as failed and inspect the reported
result before retrying. Keep the source until the imported graph and citations
have been verified. This release provides explicit transfer, not background
synchronization between installations.

## Connect ChatGPT web to a local or private server

The standard ChatGPT web product will use a Matters Cloud account and our
hosted HTTPS MCP connection. Local CLI, browser and local MCP clients remain
account-free. The first Cloud foundation does not yet implement hosted
MCP/OAuth; see [Cloud foundation](cloud-foundation.md) for its current scope.

The connectivity details below are a technical reference for operator testing
or managed private installations. They are not standard customer onboarding,
and normal ChatGPT web users will not be asked to configure a local tunnel.

ChatGPT web cannot reach a laptop's `localhost` simply because a browser tab can
open it. A running network path from OpenAI to the MCP server is required, and
the local machine must remain awake and connected.

**Secure MCP Tunnel** can forward the stdio server without opening a public
listener. Run OpenAI's `tunnel-client` on the machine that can launch Matters;
configure the scoped `matters mcp` command. The operator needs a Platform tunnel
identity, runtime credentials, appropriate tunnel permissions, and permission
to add the connection in the target ChatGPT workspace. Associate the tunnel with
that workspace, configure the private connection, then verify discovery and a
representative read. Start with `--read-only`. Follow the current
[Secure MCP Tunnel setup guide](https://developers.openai.com/api/docs/guides/secure-mcp-tunnels)
for credential handling and exact commands.

An alternative is an authenticated, publicly reachable HTTPS MCP service using
Streamable HTTP. That requires a separate HTTP deployment or bridge; this stdio
command does not supply one. Keep per-user authorization and graph access checks
at that boundary. OpenAI's [ChatGPT connection guide](https://developers.openai.com/plugins/deploy/connect-chatgpt)
describes both routes and their workspace requirements. Public plugin submission
has additional requirements; a private testing tunnel alone is not a public
distribution endpoint.

**Do not forward `matters web` as the MCP service.** That server's launch token
also grants its interactive shell. Forward the graph-only MCP process, retain
its graph allowlist, and control who may use the connection.

Private graph storage does not keep all model input on the private machine:
tool results sent to ChatGPT or another hosted model leave the installation.
Choose the connection and permitted graph scope accordingly.

Repository tests verify local protocol behavior using example graphs. An actual
ChatGPT web connection requires an operator-owned account, workspace permissions
and tunnel configuration. A local SDK test or these instructions do not establish
that live connection; it must be demonstrated separately before claiming it works
for a specific installation.
