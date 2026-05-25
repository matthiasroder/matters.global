# matters.global

`matters.global` contains the reusable matters engine and the first-party agent skill that uses it.

## Layout

```text
src/matters/         # reusable Python package
skills/matters/      # agent skill instructions
examples/            # non-personal example state
scripts/             # repository maintenance scripts
tests/               # engine and storage tests
```

The skill is intentionally thin. Shared behavior belongs in `src/matters`, not in a skill directory.

## Local Development

```sh
python -m pip install -e .
python -m pytest
```

The package installs a `matters` CLI:

```sh
matters universe --state examples/matters.example.json
```

## State Files

Runtime state should live outside installed skill directories. Use one of:

- an explicit `--state` path
- the `MATTERS_STATE` environment variable
- project-local `.matters/matters.json`
- `~/.local/share/matters/matters.json`

## Publishing the Skill

The canonical skill source lives in this repo at `skills/matters`. To publish it into the local `SKILLS` repository:

```sh
scripts/sync_skill_to_skills_repo.sh
```
