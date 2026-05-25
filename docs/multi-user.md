# Multi-User Model

This document defines the first multi-user shape for `matters.global`.

## Accounts

A user account is an actor that can own, edit, or view matters. Accounts are identified by a stable string id such as an email address, organization slug, or local username.

The engine remains account-neutral. Ownership and visibility live in metadata around the primitive state so the core graph can stay portable.

## Ownership

Each matter can have:

- `owner`: the account responsible for moving the matter forward.
- `editors`: accounts allowed to change the matter, its conditions, or dependency edges touching it.
- `viewers`: accounts allowed to see a non-public matter.

If no owner is present, the matter is treated as locally owned by the state maintainer.

## Visibility

Matter visibility has three levels:

- `private`: visible only to the owner and explicit viewers.
- `shared`: visible to named viewers or collaborators.
- `public`: safe to publish into a world-readable state file.

Conditions inherit their matter's visibility. Dependencies are exported only when both endpoint matters are visible in the target export.

## Public Repository Format

A public repository can contain a normal matters state file with only public matters:

```json
{
  "schema_version": 2,
  "matters": ["publish_matters_global_system"],
  "conditions": {
    "publish_matters_global_system": [
      { "label": "First public demo or walkthrough is published", "truth": false }
    ]
  },
  "dependencies": []
}
```

Private matter ids, private conditions, and dependency edges touching private matters are omitted. Public exports should be generated from a richer private state plus visibility metadata instead of being edited by hand.

## Current Workflow

The implemented sharing workflow exports a sanitized public state from:

- the normal primitive matters state
- a visibility map where `matter_id -> public | shared | private`

Only matters marked `public` are included in the export. This lets publicly relevant matters be committed or published without exposing private matters.

The write path accepts an edited public state and merges it back into the private state, but only for matters already marked `public`. It rejects incoming public files that contain private matter ids. Dependency edits are merged only when both endpoint matters are public.

Full conflict handling, invitations, and account-aware editing UI are still future work.
