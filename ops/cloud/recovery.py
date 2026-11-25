#!/usr/bin/env python3
"""Bounded, labelled-container PostgreSQL recovery drills; no public deployment.

Install ops/cloud/requirements.txt in an operator environment. Database secrets
stay in an operator-mounted PGPASSFILE inside the database container. Encryption
uses an operator-owned 32-byte key file, never a command-line secret.
"""

import argparse
import hashlib
import json
import os
import re
import selectors
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


REPOSITORY = Path(__file__).resolve().parents[2]
OWNER_LABEL = "global.matters.operations-owner"
MAGIC = b"MATTERS-PG-AESGCM\x01"
NAME = re.compile(r"[a-z][a-z0-9_]{0,62}\Z")
CONTAINER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
DEFAULT_MAX_BYTES = 64 * 1024 * 1024


class OpsError(ValueError):
    pass


def outside_repository(path):
    path = Path(path).expanduser().absolute()
    resolved = path.parent.resolve() / path.name
    if resolved == REPOSITORY or REPOSITORY in resolved.parents:
        raise OpsError("Keep credentials, backups and operator artifacts outside the repository.")
    return resolved


def load_key(path):
    path = outside_repository(path)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077 or info.st_uid not in {os.getuid(), 0}:
                raise OpsError("The encryption key must be a private regular file with mode 0600 or stricter.")
            key = stream.read(33)
    except OSError:
        raise OpsError("The encryption key cannot be opened safely.") from None
    if len(key) != 32:
        raise OpsError("The encryption key must contain exactly 32 random bytes.")
    return key


def seal(dump, key, metadata):
    header = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    nonce = os.urandom(12)
    plaintext = len(header).to_bytes(4, "big") + header + dump
    return MAGIC + nonce + AESGCM(key).encrypt(nonce, plaintext, MAGIC)


def unseal(package, key):
    if not package.startswith(MAGIC) or len(package) < len(MAGIC) + 12 + 16 + 4:
        raise OpsError("Unsupported or truncated encrypted backup.")
    start = len(MAGIC)
    try:
        plaintext = AESGCM(key).decrypt(package[start:start + 12], package[start + 12:], MAGIC)
    except InvalidTag:
        raise OpsError("Backup authentication failed; no destination was changed.") from None
    size = int.from_bytes(plaintext[:4], "big")
    if size > 65536 or size > len(plaintext) - 4:
        raise OpsError("The authenticated backup manifest is invalid.")
    try:
        metadata = json.loads(plaintext[4:4 + size])
    except (ValueError, UnicodeError):
        raise OpsError("The authenticated backup manifest is invalid.") from None
    dump = plaintext[4 + size:]
    if (
        not isinstance(metadata, dict)
        or metadata.get("format") != "matters-postgresql-backup"
        or type(metadata.get("schema_version")) is not int
        or metadata.get("schema_version") != 1
        or type(metadata.get("database_schema")) is not int
        or metadata.get("database_schema") not in (1, 2, 3)
        or not isinstance(metadata.get("source_database"), str)
        or not NAME.fullmatch(metadata["source_database"])
        or not isinstance(metadata.get("source_operations_owner"), str)
        or not CONTAINER.fullmatch(metadata["source_operations_owner"])
        or metadata.get("dump_bytes") != len(dump)
        or metadata.get("dump_sha256") != hashlib.sha256(dump).hexdigest()
        or not dump.startswith(b"PGDMP")
    ):
        raise OpsError("The backup manifest or PostgreSQL archive is unsupported.")
    return metadata, dump


def write_exclusive(path, data):
    path = outside_repository(path)
    if os.path.lexists(path):
        raise OpsError("The output already exists; nothing was replaced.")
    staged = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".matters-backup-", delete=False) as stream:
            staged = Path(stream.name)
            os.fchmod(stream.fileno(), 0o600)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(staged, path)
    except OSError:
        raise OpsError("Cannot publish a new backup without replacing an existing file.") from None
    finally:
        if staged is not None:
            staged.unlink(missing_ok=True)


class PostgreSQL:
    def __init__(self, container, owner, username, *, passfile=None, timeout=120, max_bytes=DEFAULT_MAX_BYTES):
        if not CONTAINER.fullmatch(container) or not CONTAINER.fullmatch(owner):
            raise OpsError("An explicit container and operations-owner label are required.")
        if not username or timeout <= 0 or max_bytes <= 0:
            raise OpsError("Positive bounds and a database username are required.")
        if passfile is not None and not passfile.startswith("/"):
            raise OpsError("PGPASSFILE must be an absolute path inside the container.")
        self.container, self.owner, self.username = container, owner, username
        self.passfile, self.timeout, self.max_bytes = passfile, timeout, max_bytes
        self.docker = ["docker", "--host=unix:///var/run/docker.sock"]
        self.environment = dict(os.environ)
        for name in ("DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_TLS", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH"):
            self.environment.pop(name, None)
        try:
            result = subprocess.run(
                [*self.docker, "inspect", "--format", "{{json .Config.Labels}}", container],
                capture_output=True, timeout=timeout, env=self.environment, check=True,
            )
            labels = json.loads(result.stdout)
        except (OSError, subprocess.SubprocessError, ValueError):
            raise OpsError("The selected Docker container cannot be inspected.") from None
        if not isinstance(labels, dict) or labels.get(OWNER_LABEL) != owner:
            raise OpsError("The selected container does not carry the expected operations-owner label.")

    @staticmethod
    def database_name(value):
        if not NAME.fullmatch(value):
            raise OpsError("Database names must start with a lowercase letter and use lowercase letters, digits or underscores.")
        return value

    def command(self, executable, database, *arguments):
        self.database_name(database)
        command = [*self.docker, "exec", "-i"]
        if self.passfile:
            command += ["--env", "PGPASSFILE=" + self.passfile]
        return [*command, self.container, executable, "--no-password", "--username", self.username, "--dbname", database, *arguments]

    def capture(self, command, *, maximum=None):
        maximum = self.max_bytes if maximum is None else maximum
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=self.environment, bufsize=0)
        output = bytearray()
        deadline = time.monotonic() + self.timeout
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not selector.select(remaining):
                        raise OpsError("The PostgreSQL operation exceeded its time bound.")
                    chunk = os.read(process.stdout.fileno(), 65536)
                    if not chunk:
                        break
                    if len(output) + len(chunk) > maximum:
                        raise OpsError("The PostgreSQL operation exceeded its byte bound.")
                    output.extend(chunk)
            if process.wait(timeout=max(0.1, deadline - time.monotonic())):
                raise OpsError("The PostgreSQL operation failed; inspect approved operator logs.")
            return bytes(output)
        except subprocess.TimeoutExpired:
            raise OpsError("The PostgreSQL operation exceeded its time bound.") from None
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdout.close()

    def sql(self, database, statement):
        return self.capture(self.command("psql", database, "--no-psqlrc", "--tuples-only", "--no-align", "--set", "ON_ERROR_STOP=1", "--command", statement), maximum=65536).decode().strip()

    def schema(self, database):
        version = self.sql(database, "SELECT version FROM schema_versions WHERE id=1")
        if version not in {"1", "2", "3"}:
            raise OpsError("The database schema is unsupported; review recovery before proceeding.")
        return int(version)

    def sample(self, database):
        return json.loads(self.sql(database, "SELECT json_build_object('database_bytes',pg_database_size(current_database()),'accounts',(SELECT count(*) FROM users),'graphs',(SELECT count(*) FROM graphs),'graph_events',(SELECT count(*) FROM graph_events))"))


def backup(pg, database, key, output):
    output = outside_repository(output)
    if os.path.lexists(output):
        raise OpsError("The output already exists; nothing was replaced.")
    pg.database_name(database)
    version = pg.schema(database)
    dump = pg.capture(pg.command("pg_dump", database, "--format=custom", "--no-owner", "--no-acl"))
    if not dump.startswith(b"PGDMP"):
        raise OpsError("PostgreSQL did not return a custom-format archive.")
    metadata = {
        "format": "matters-postgresql-backup", "schema_version": 1,
        "database_schema": version, "source_database": database,
        "source_operations_owner": pg.owner,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dump_sha256": hashlib.sha256(dump).hexdigest(), "dump_bytes": len(dump),
    }
    write_exclusive(output, seal(dump, key, metadata))
    return {"schema_version": 1, "encrypted_backup": str(outside_repository(output)), "archive_bytes": len(dump), "database_schema": version}


def restore(pg, database, confirmation, package, key):
    metadata, dump = unseal(package, key)  # Authenticate before touching PostgreSQL.
    if metadata["source_operations_owner"] != pg.owner:
        raise OpsError("The backup belongs to a different installation owner; no destination was changed.")
    pg.database_name(database)
    if not database.startswith("restore_") or confirmation != database or database == metadata.get("source_database"):
        raise OpsError("Restore requires a new restore_* database and an exact confirmation; source databases cannot be targets.")
    if pg.sql("postgres", f"SELECT count(*) FROM pg_database WHERE datname='{database}'") != "0":
        raise OpsError("The restore destination already exists; nothing was replaced.")
    marker = "matters-restore:" + uuid.uuid4().hex
    created = False
    try:
        pg.sql("postgres", f'CREATE DATABASE "{database}"')
        created = True
        pg.sql("postgres", f'COMMENT ON DATABASE "{database}" IS \'{marker}\'')
        try:
            result = subprocess.run(
                pg.command("pg_restore", database, "--exit-on-error", "--single-transaction", "--no-owner", "--no-acl"),
                input=dump, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                env=pg.environment, timeout=pg.timeout,
            )
        except subprocess.SubprocessError:
            raise OpsError("The PostgreSQL restore did not finish within its bounds.") from None
        if result.returncode or pg.schema(database) != metadata["database_schema"]:
            raise OpsError("The PostgreSQL restore or schema validation failed.")
        # Old browser sessions and issued OAuth grants must not revive after a restore.
        invalidations = ["DELETE FROM sessions;"]
        if metadata["database_schema"] >= 2:
            invalidations.append("DELETE FROM oauth_records WHERE kind <> 'client';")
        if metadata["database_schema"] >= 3:
            invalidations += ["DELETE FROM account_actions;", "DELETE FROM rate_limit_buckets;"]
        pg.sql(database, "BEGIN;\n" + "\n".join(invalidations) + "\nCOMMIT;")
        result = {"schema_version": 1, "restored_database": database, "restore_ownership_marker": marker, "database_schema": metadata["database_schema"], "sessions_and_issued_grants_revoked": True, "account_actions_revoked": metadata["database_schema"] >= 3, "measurements": pg.sample(database)}
        return result
    except BaseException:
        if created:
            # Never force-drop or delete a destination whose ownership marker changed.
            try:
                observed = pg.sql("postgres", f"SELECT shobj_description(oid,'pg_database') FROM pg_database WHERE datname='{database}'")
                if observed == marker:
                    pg.sql("postgres", f'DROP DATABASE "{database}"')
            except OpsError:
                pass
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("command", choices=("backup", "restore", "sample"))
    result.add_argument("--container", required=True)
    result.add_argument("--owner", required=True, help="Expected global.matters.operations-owner label.")
    result.add_argument("--username", required=True)
    result.add_argument("--database", required=True)
    result.add_argument("--passfile", help="Operator-mounted absolute PGPASSFILE inside the database container.")
    result.add_argument("--key-file")
    result.add_argument("--output")
    result.add_argument("--source")
    result.add_argument("--confirm-database")
    result.add_argument("--timeout", type=int, default=120)
    result.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.max_bytes <= 0 or args.timeout <= 0:
            raise OpsError("Positive byte and time bounds are required.")
        key = load_key(args.key_file) if args.command != "sample" and args.key_file else None
        if args.command != "sample" and key is None:
            raise OpsError("An operator-owned encryption key file is required.")
        package = None
        if args.command == "restore":
            if not args.source:
                raise OpsError("An encrypted source backup is required.")
            source = outside_repository(args.source)
            with os.fdopen(os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise OpsError("The encrypted source must be a regular file.")
                package = stream.read(args.max_bytes + 131073)
            if len(package) > args.max_bytes + 131072:
                raise OpsError("The encrypted source exceeds its byte bound.")
            unseal(package, key)
        if args.command == "backup" and not args.output:
            raise OpsError("A new output path outside the repository is required.")
        pg = PostgreSQL(args.container, args.owner, args.username, passfile=args.passfile, timeout=args.timeout, max_bytes=args.max_bytes)
        if args.command == "backup":
            result = backup(pg, args.database, key, args.output)
        elif args.command == "restore":
            result = restore(pg, args.database, args.confirm_database, package, key)
        else:
            result = {"schema_version": 1, "measurements": pg.sample(args.database)}
        print(json.dumps(result, sort_keys=True))
        return 0
    except (OpsError, OSError):
        error = sys.exc_info()[1]
        message = str(error) if isinstance(error, OpsError) else "An operator file or command was unavailable."
        print(json.dumps({"schema_version": 1, "error": message}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
