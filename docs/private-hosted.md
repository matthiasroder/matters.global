# Private hosted Matters: managed-installation preparation

Status: **draft operating model and requirements template**. No company has
agreed these requirements merely because this document exists. No private
installation, contract, recovery commitment or support entitlement is claimed.

Private hosted Matters reuses the shared engine, browser software, graph format
and MCP tools. Matters provisions and operates a separately isolated company
installation. Companies do not need a public Matters Cloud account to use it.
Any ChatGPT web endpoint is configured and operated by Matters for that
installation; customers are not asked to expose a personal computer or set up a
local tunnel. Tool results sent to a hosted model still leave the installation
and require an agreed data-egress policy.

## Company requirements record

Use one record per prospective company. Fill it with the company's authorized
representative and identify the agreed evidence. Keep names and confidential
requirements in the operator's approved private location, not a public repository.

| Requirement | Information to agree |
| --- | --- |
| Authority and ownership | Company representative, installation/data owner and who may approve changes |
| Identity | Company identity provider, account lifecycle, admin roles and access revocation |
| Network | Permitted ingress/egress, TLS endpoint, remote administration and any restricted connectivity |
| Model use | Approved clients/models and which graph content may leave the installation |
| Data location | Region, storage location, backups and any contractual constraints |
| Isolation | Dedicated database/storage/keys, network boundary and permitted shared infrastructure |
| Recovery | Acceptable lost work, recovery time, retention and who validates drills |
| Support | Hours, incident route, escalation, maintenance windows and operator access approval |
| Change control | Release approval, migration rehearsal, rollback and integration ownership |
| Export and exit | Authorized recipients, export format, handover, deletion and key-retention evidence |

Record status as draft, proposed or agreed, with decision owner, date and
evidence. A managed-service requirement must not become a completed graph
condition until the appropriate company actually agrees it.

## Isolation and provisioning

Provision a distinct installation identity, database, persistent storage,
network, secrets and backup destination. Use separate TLS/authentication
configuration and OAuth audience/issuer boundaries so an account or token from
one installation cannot access another. Choose separate keys and narrowly
scoped operator credentials. Container names or different graph IDs alone are
not an isolation boundary.

Use a reviewed shared software release, not a customer engine fork. Express
company-specific integration and policy through configuration or explicit
adapters. A deployment controller may manage installations without pooling
their graph contents into a central public service.

Provisioning should record the approved release/image digest, configuration
version, schema version, network boundary, secret references and ownership.
The reviewed [database role tool](cloud-operations.md#database-role-maintenance-and-upgrades)
provides separate application, migration and read-only backup roles with an
installation ownership inventory. Support access remains separately approved
and time-limited. Keep bootstrap credentials out of the application; trust only
explicit controlled proxies and overwrite client-forwarded headers. Verify both
expected access and cross-installation refusal before handing over the endpoint.
Package assets and required model artifacts locally when external downloads are
prohibited. The localhost staging setup is a rehearsal pattern, not proof that
a company's actual isolation requirements are satisfied.

## Upgrades and rollback

For each release, inspect schema and configuration changes, take an authenticated
encrypted backup, restore it to isolated test capacity, and rehearse the upgrade
with representative graphs. Verify account permissions, MCP schemas, graph
validation, integrations and export after the migration.

Record the maintenance approval and observed results. Roll back application
code only when it supports the current database schema. If recovery requires
restoring data, record the lost-work window and obtain the authorization needed
for that action. Never reset or downgrade an existing schema automatically.

## Operator support access

Support access is installation-specific, time-limited and limited to the
required task. Use the company's agreed approval route. Record who accessed the
installation, why, when and what administrative operation occurred. Avoid
putting graph contents, email addresses, OAuth codes, session cookies or secret
values into routine logs. Revoke temporary credentials after the work.

An operator's ability to run the infrastructure does not imply unrestricted
permission to inspect company graphs. Provide incident evidence through the
agreed private channel. The company must know how access can be revoked and
which emergency access process, if any, has been agreed.

## Recovery and offboarding

Run encrypted backup and restore drills against the actual agreed boundary and
data size. Retain keys separately from backups, validate new sign-in and graph
permissions after recovery, and revoke old sessions, issued OAuth grants and
password-reset/email-verification credentials by default. Preserve recorded
verification state and graph data, then reapply reviewed runtime/backup grants
to the exact new restore database before admitting traffic. Record drill outcomes; do not convert a successful small fixture test
into a service-level recovery promise.

For offboarding, verify the requesting authority, export graph facts and source
citations, validate the import at the intended destination, and hand over through
the approved channel. Agree when the source becomes read-only and when it may
be retired. Revoke endpoints, client credentials, support access and automation
after handover as agreed. Delete source data, backups and encryption keys only
under the recorded retention/exit decision and retain the authorized completion
evidence. Do not erase another installation's resources or shared controller
state as part of one company's exit.
