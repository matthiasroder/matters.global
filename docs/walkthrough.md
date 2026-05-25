# matters.global Walkthrough

This walkthrough shows the minimum loop: model a matter, find what is actionable, and prepare a public-safe export.

## 1. Create A State

```json
{
  "schema_version": 2,
  "matters": ["define_offer", "send_proposal"],
  "conditions": {
    "define_offer": [
      { "label": "Scope and price are written down", "truth": true }
    ],
    "send_proposal": [
      { "label": "Proposal has been sent to the client", "truth": false }
    ]
  },
  "dependencies": [["define_offer", "send_proposal"]]
}
```

Save it as `demo.matters.json`.

## 2. Ask What Can Move Now

```sh
matters universe --state demo.matters.json
```

Expected output:

```text
send_proposal
```

`send_proposal` is actionable because `define_offer` is already resolved.

## 3. Generate An Unlock Report

```sh
matters unlock --state demo.matters.json
```

The report lists actionable matters, false conditions, and candidate next actions. It separates work an agent can start from work that needs human confirmation.

## 4. Extract Candidate Matters

```sh
printf 'Goal: Invite a pilot user\n' | matters extract - --source-type notes --state demo.matters.json
```

Extraction prints a proposal only. It does not update state until a human confirms the candidate matters and dependencies.

## 5. Export A Public View

Create `visibility.json`:

```json
{
  "define_offer": "private",
  "send_proposal": "public"
}
```

Then run:

```sh
matters export-public --state demo.matters.json --visibility visibility.json
```

The exported state contains only public matters and public-to-public dependency edges.

## 6. Merge Public Edits

If collaborators edit the public state, review it and merge it back:

```sh
matters merge-public --state demo.matters.json --public-state public.matters.json --visibility visibility.json
```

The merge rejects incoming matter ids that are not marked public.
