#!/usr/bin/env python3
"""Hold a local triage coordinator lock until stdin closes or Ctrl-C.

Run in a retained terminal while the calling agent coordinates the work. This
POSIX lock is separate from Matters' short graph-write lock: ordinary graph
reads and writes remain available. It coordinates cooperating local triage
agents, not remote machines or arbitrary external tools.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, help="Existing graph file to pin.")
    parser.add_argument("--run-id", required=True, help="Identifier for this triage run.")
    args = parser.parse_args(argv)
    if not args.run_id.strip():
        parser.error("--run-id must not be empty")

    try:
        state = Path(args.state).expanduser().resolve(strict=True)
        if not state.is_file():
            parser.error(f"state is not a file: {state}")
        lock = state.with_name(f".{state.name}.triage.lock")
        fd = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
    except OSError as error:
        parser.error(f"cannot open coordinator lock for {args.state}: {error}")

    # Never replace or unlink the sidecar: all holders must lock the same inode.
    # The kernel releases ownership on normal exit, EOF, or process death.
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error(f"another local triage coordinator holds {lock}")
        except OSError as error:
            parser.error(f"cannot acquire coordinator lock: {error}")

        print(json.dumps({
            "status": "acquired", "state_path": str(state),
            "lock_path": str(lock), "run_id": args.run_id, "pid": os.getpid(),
        }), flush=True)
        try:
            for _line in sys.stdin:
                pass
        except KeyboardInterrupt:
            pass
        print(json.dumps({"status": "released", "run_id": args.run_id}), flush=True)
    finally:
        os.close(fd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
