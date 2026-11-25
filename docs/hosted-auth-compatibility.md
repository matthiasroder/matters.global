# Hosted authorization compatibility review

Reviewed on 25 November 2026 against the installed MCP Python SDK **1.30.0**
and the primary sources below. Repository tests demonstrate implementation
behavior; they do not demonstrate that a particular ChatGPT account has connected.

## Supported route

The hosted service uses OAuth authorization code with S256 PKCE, opaque
database-backed access tokens, and an explicit account-holder consent screen.
The user chooses permitted graph IDs. A grant restricts access further than
account membership: both the grant and current graph permissions must allow an
operation. Local stdio MCP continues to use its configured local graph registry.

| Capability | Implementation boundary |
| --- | --- |
| Protected-resource and authorization-server discovery | Published for the configured issuer and canonical `/mcp` resource. |
| Client registration | Dynamic Client Registration for public clients using `none` token authentication. |
| Authorization | S256 PKCE, exact registered callback and resource binding, session-bound browser consent. |
| Tokens | Persisted hashes, expiry, refresh rotation, replay detection and grant revocation. |
| MCP access | Bearer authentication, the shared 16-tool contract, selected graph grants and current database roles. |
| Client ID Metadata Documents | Not implemented or advertised. The server does not fetch client-supplied metadata URLs. |
| OpenID Connect / verified corporate email | Not implemented or advertised. |
| Actual ChatGPT web connection | Requires separately operated public HTTPS hosting and an authorized live account test. |

## Standards and product limits

The [MCP authorization specification](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization)
prefers Client ID Metadata Documents and retains Dynamic Client Registration
for compatibility. This implementation supports the DCR route; it does not claim
complete support for every feature of the latest specification. Discovery,
resource/audience checks and PKCE remain essential regardless of registration
method. The installed SDK's supported protocol versions are a separate concern
from authorization capabilities.

The [OpenAI authentication guide](https://developers.openai.com/plugins/build/auth)
still describes DCR with public-client token authentication and S256 PKCE.
It also describes additional corporate-domain protections that depend on
OpenID Connect and verified email. Those corporate identity protections are not supplied by the
current account model. Optional SMTP email verification now records control of
a mailbox, but it does not create an OpenID Connect provider or verified
corporate-domain claims. Using a tested development authorization server is not
evidence of production identity-provider readiness.

The [ChatGPT connection guide](https://help.openai.com/en/articles/12584461-connecting-custom-mcp-servers-in-chatgpt)
requires a reachable endpoint and permits account/workspace policy restrictions.
The standard Matters product route is an operated Matters Cloud HTTPS endpoint
and a Matters account. Repository tests do not select a hosting provider,
publish an endpoint, change ChatGPT workspace permissions, or establish that a
customer can complete that live connection.

## Independent evidence

`tests/test_cloud_hosted_review.py` exercises selected-graph restrictions,
membership revocation, PKCE/resource/client/callback binding, refresh replay and
scope narrowing, account/session/CSRF substitution, conflicting simultaneous
writes, and graph transfer without inherited ownership or sharing. It runs on
SQLite and, with `MATTERS_TEST_DATABASE_URL`, isolated PostgreSQL schemas.

The SDK test starts with no client registration or bearer token. Its own OAuth
client discovers metadata, registers, generates PKCE and state, completes the
HTTP consent/token flow, refreshes its token and reconnects using stored client
state. Integrated HTTP checks also cover visual/export access after membership
changes. This is protocol interoperability evidence, not a live ChatGPT test.

Three independently reproduced defects were corrected during review: an older
editor invitation could restore editing after an owner changed the member to a
viewer, and a valid noncanonical configured issuer could fail the browser's
origin comparison. Both cases have regressions on SQLite and PostgreSQL. A
delayed invitation-acceptance response could also override a graph the user
subsequently selected; its browser-script regression now preserves the newer
selection. Additional browser-script checks keep pending sharing details and
new invitation links from crossing an account change.

The original local CLI, visual, stdio MCP and portability regression suites
remain relevant. An operational release also needs the hosting, account
recovery, abuse control, migrations and recovery checks described in the Cloud
operations documentation.

A real HTTPS Chromium consent run also caught a form-navigation issue that
simulated requests missed: `no-referrer` made the browser send `Origin: null`.
OAuth form pages now use an origin-only `strict-origin` policy and the outer
middleware preserves it. Exact Origin, browser session and CSRF checks remain
required; null origins are rejected. A native Chromium regression and the
combined HTTPS browser/SDK proof cover this behavior.
