#!/usr/bin/env python3
"""Provision fresh, owner-labelled PostgreSQL databases with separate roles.

Run from the same reviewed Matters release as the application. Password files
are private operator files outside the repository. Existing roles/databases are
never adopted or overwritten. This tool does not provision a hosting provider.
"""

import argparse
import json
import os
import re
import stat
import subprocess
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recovery import OpsError, PostgreSQL, outside_repository, write_exclusive


PREFIX = re.compile(r"[a-z][a-z0-9_]{0,47}\Z")
VERSION_TABLES = {
    1: {"schema_versions", "users", "sessions", "graphs", "graph_members", "graph_events"},
    2: {"graph_sources", "graph_invitations", "oauth_records"},
    3: {"account_security", "account_actions", "rate_limit_buckets"},
}
PASSWORD_LOG_OPTIONS = "-c log_statement=none -c log_min_duration_statement=-1 -c log_min_error_statement=panic -c log_parameter_max_length=0 -c log_parameter_max_length_on_error=0"
LARGE_OBJECT_WRITES = (
    "lo_create(oid)", "lo_creat(integer)", "lo_from_bytea(oid,bytea)",
    "lo_put(oid,bigint,bytea)", "lo_unlink(oid)", "lowrite(integer,bytea)",
    "lo_truncate(integer,integer)", "lo_truncate64(integer,bigint)",
    "lo_import(text)", "lo_import(text,oid)", "lo_export(oid,text)",
)


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,62}", value):
        raise OpsError("Use lowercase PostgreSQL identifiers with letters, digits and underscores.")
    return '"' + value + '"'


def literal(value):
    return "'" + value.replace("'", "''") + "'"


def private_password(path):
    path = outside_repository(path)
    try:
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077 or info.st_uid not in {os.getuid(), 0}:
                raise OpsError("Password files must be private regular files with mode 0600 or stricter.")
            raw = stream.read(4097)
        password = raw.decode("utf-8").removesuffix("\n").removesuffix("\r")
    except (OSError, UnicodeError):
        raise OpsError("An operator password file cannot be opened safely.") from None
    return checked_password(password)


def checked_password(password):
    if not isinstance(password, str) or not 20 <= len(password) <= 1024 or any(character in password for character in "\n\r\x00"):
        raise OpsError("Use a password file containing one strong 20..1024-character value.")
    return password


def script(pg, database, statements):
    command = [*pg.docker, "exec", "-i", "--env", "PGOPTIONS=" + PASSWORD_LOG_OPTIONS]
    if pg.passfile:
        command += ["--env", "PGPASSFILE=" + pg.passfile]
    command += [pg.container, "psql", "--no-password", "--username", pg.username, "--dbname", database, "--no-psqlrc", "--quiet", "--set", "ON_ERROR_STOP=1"]
    try:
        result = subprocess.run(command, input=statements.encode("utf-8"), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=pg.timeout, env=pg.environment)
    except (OSError, subprocess.SubprocessError):
        raise OpsError("The private PostgreSQL provisioning operation failed.") from None
    if result.returncode:
        raise OpsError("The private PostgreSQL provisioning operation failed; inspect approved operator diagnostics.")


def role_plan(database, prefix, owner):
    identifier(database)
    if not PREFIX.fullmatch(prefix) or prefix.startswith("pg_"):
        raise OpsError("Use a new non-reserved role prefix of at most 48 lowercase identifier characters.")
    return {
        "schema_version": 1,
        "database": database,
        "role_prefix": prefix,
        "operations_owner": owner,
        "marker": "matters-role-provision:" + owner + ":" + uuid.uuid4().hex,
        "roles": {"app": prefix + "_app", "migration": prefix + "_migration", "backup": prefix + "_backup"},
    }


def managed_tables(version):
    if type(version) is not int or version not in VERSION_TABLES:
        raise OpsError("The database schema needs a reviewed role-policy update.")
    return set().union(*(VERSION_TABLES[number] for number in range(1, version + 1)))


def schema_ddl():
    try:
        from matters.cloud.db import SCHEMA_VERSION, metadata
        from sqlalchemy.dialects.postgresql import dialect
        from sqlalchemy.schema import CreateIndex, CreateTable
    except ImportError:
        raise OpsError("Install this reviewed Matters release with its Cloud extra before schema provisioning.") from None
    if set(metadata.tables) != managed_tables(SCHEMA_VERSION):
        raise OpsError("Application tables differ from the reviewed role policy.")
    statements = [str(CreateTable(table).compile(dialect=dialect())) + ";" for table in metadata.sorted_tables]
    for table in metadata.sorted_tables:
        statements += [str(CreateIndex(index).compile(dialect=dialect())) + ";" for index in sorted(table.indexes, key=lambda index: index.name)]
    statements.append(f"INSERT INTO schema_versions(id,version) VALUES(1,{SCHEMA_VERSION});")
    return SCHEMA_VERSION, "\n".join(statements)


def verify_ownership(pg, plan, *, require_database=True):
    if not isinstance(plan, dict) or plan.get("schema_version") != 1 or plan.get("operations_owner") != pg.owner:
        raise OpsError("The role record belongs to a different operations owner.")
    database = plan.get("database")
    identifier(database)
    roles = plan.get("roles", {})
    if not isinstance(roles, dict) or set(roles) != {"app", "migration", "backup"} or not all(isinstance(value, str) for value in roles.values()) or len(set(roles.values())) != 3:
        raise OpsError("The role record is invalid.")
    marker = plan.get("marker")
    if not isinstance(marker, str) or not re.fullmatch("matters-role-provision:" + re.escape(pg.owner) + ":[0-9a-f]{32}", marker):
        raise OpsError("The role record has no valid ownership marker.")
    suffixes = {"app": "_app", "migration": "_migration", "backup": "_backup"}
    prefixes = {roles[kind].removesuffix(suffix) for kind, suffix in suffixes.items() if roles[kind].endswith(suffix)}
    if len(prefixes) != 1 or len([kind for kind, suffix in suffixes.items() if roles[kind].endswith(suffix)]) != 3:
        raise OpsError("Role names do not match their reviewed application/migration/backup purposes.")
    if require_database:
        observed = pg.sql("postgres", "SELECT shobj_description(oid,'pg_database') FROM pg_database WHERE datname=" + literal(database))
        if observed != marker:
            raise OpsError("The database ownership marker does not match this provision record.")
    for name in roles.values():
        identifier(name)
        observed = pg.sql("postgres", "SELECT shobj_description(oid,'pg_authid') FROM pg_roles WHERE rolname=" + literal(name))
        if observed != marker:
            raise OpsError("A role ownership marker does not match this provision record.")
        flags = pg.sql("postgres", "SELECT (rolsuper OR rolcreatedb OR rolcreaterole OR rolreplication OR rolinherit OR rolbypassrls)::int FROM pg_roles WHERE rolname=" + literal(name))
        if flags != "0":
            raise OpsError("An owned role has unexpected administrative privileges.")
        memberships = pg.sql("postgres", "SELECT count(*) FROM pg_auth_members m JOIN pg_roles r ON r.oid=m.member WHERE r.rolname=" + literal(name))
        if memberships != "0":
            raise OpsError("An owned role has unexpected role memberships.")
    if require_database:
        owner = pg.sql("postgres", "SELECT pg_get_userbyid(datdba) FROM pg_database WHERE datname=" + literal(database))
        if owner != roles["migration"]:
            raise OpsError("The database is not owned by its migration role.")


def apply_grants(pg, plan):
    verify_ownership(pg, plan)
    database, roles = plan["database"], plan["roles"]
    version = pg.schema(database)
    tables = managed_tables(version)
    actual = set(pg.sql(database, "SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename").splitlines())
    if actual != tables:
        raise OpsError("The database has unknown or missing application tables; grants were refused.")
    if pg.sql(database, "SELECT count(*) FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace WHERE n.nspname='public'") != "0":
        raise OpsError("Public schema functions require an explicit privilege-policy review.")
    if pg.sql(database, "SELECT count(*) FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname='public' AND c.relkind IN ('v','m','f')") != "0":
        raise OpsError("Public views or foreign tables require an explicit privilege-policy review.")
    for table in tables:
        owner = pg.sql(database, "SELECT pg_get_userbyid(c.relowner) FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname='public' AND c.relname=" + literal(table))
        if owner != roles["migration"]:
            raise OpsError("Application tables must be owned by the migration role.")
    app, migration, backup = (identifier(roles[name]) for name in ("app", "migration", "backup"))
    application = ",".join("public." + identifier(name) for name in sorted(tables - {"schema_versions"}))
    statements = [
        "BEGIN;",
        f"REVOKE ALL ON DATABASE {identifier(database)} FROM PUBLIC;",
        f"REVOKE ALL ON DATABASE {identifier(database)} FROM {app},{backup};",
        f"REVOKE ALL ON SCHEMA public FROM PUBLIC; ALTER SCHEMA public OWNER TO {migration};",
        f"REVOKE ALL ON SCHEMA public FROM {app},{backup};",
        f"GRANT CONNECT ON DATABASE {identifier(database)} TO {app},{backup};",
        f"GRANT USAGE ON SCHEMA public TO {app},{backup};",
        f"REVOKE ALL ON ALL TABLES IN SCHEMA public FROM {app},{backup};",
        f"REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM {app},{backup};",
        "REVOKE ALL ON ALL TABLES IN SCHEMA public FROM PUBLIC;",
        "REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM PUBLIC;",
        f"GRANT SELECT,INSERT,UPDATE,DELETE ON TABLE {application} TO {app};",
        f"GRANT SELECT ON TABLE public.schema_versions TO {app};",
        f"GRANT SELECT ON ALL TABLES IN SCHEMA public TO {backup};",
        f"GRANT SELECT ON ALL SEQUENCES IN SCHEMA public TO {backup};",
        # No broad default grant to app: a future configuration table cannot
        # silently become writable. Re-run the reviewed grants after migration.
        f"ALTER DEFAULT PRIVILEGES FOR ROLE {migration} IN SCHEMA public REVOKE ALL ON TABLES FROM PUBLIC;",
        f"ALTER DEFAULT PRIVILEGES FOR ROLE {migration} IN SCHEMA public REVOKE ALL ON SEQUENCES FROM PUBLIC;",
        f"ALTER DEFAULT PRIVILEGES FOR ROLE {migration} IN SCHEMA public GRANT SELECT ON TABLES TO {backup};",
    ]
    # PostgreSQL otherwise lets any login create large objects through PUBLIC
    # catalog-function privileges even without schema CREATE/TEMP. The engine
    # stores graph data as JSON and needs none of these write functions.
    for function in LARGE_OBJECT_WRITES:
        statements += [f"REVOKE EXECUTE ON FUNCTION pg_catalog.{function} FROM PUBLIC,{app},{backup};", f"GRANT EXECUTE ON FUNCTION pg_catalog.{function} TO {migration};"]
    statements.append("COMMIT;")
    script(pg, database, "\n".join(statements))
    for name in ("app", "backup"):
        role = roles[name]
        flags = pg.sql("postgres", "SELECT (rolsuper OR rolcreatedb OR rolcreaterole OR rolreplication OR rolinherit OR rolbypassrls)::int FROM pg_roles WHERE rolname=" + literal(role))
        if flags != "0":
            raise OpsError("A runtime role has unexpected administrative privileges.")
    return {"schema_version": 1, "database": database, "database_schema": version, "roles": roles, "grants_applied": True}


def cleanup_failed(pg, plan):
    database, marker = plan["database"], plan["marker"]
    try:
        if pg.sql("postgres", "SELECT shobj_description(oid,'pg_database') FROM pg_database WHERE datname=" + literal(database)) == marker:
            pg.sql("postgres", "DROP DATABASE " + identifier(database))  # No FORCE.
    except OpsError:
        pass
    for name in plan["roles"].values():
        try:
            if pg.sql("postgres", "SELECT shobj_description(oid,'pg_authid') FROM pg_roles WHERE rolname=" + literal(name)) == marker:
                pg.sql("postgres", "DROP ROLE " + identifier(name))
        except OpsError:
            pass


def grant_restored(pg, plan, database, marker, confirmation, output_record):
    """Bind only a freshly restored, exactly marked DB to existing owned roles."""
    verify_ownership(pg, plan, require_database=False)
    identifier(database)
    if not database.startswith("restore_") or database != confirmation or not isinstance(marker, str) or not re.fullmatch(r"matters-restore:[0-9a-f]{32}", marker):
        raise OpsError("Confirm the fresh restore_* database and its exact restore ownership marker.")
    observed = pg.sql("postgres", "SELECT shobj_description(oid,'pg_database') FROM pg_database WHERE datname=" + literal(database))
    if observed != marker:
        raise OpsError("The fresh restore ownership marker does not match; no grants were changed.")
    tables = managed_tables(pg.schema(database))
    actual = set(pg.sql(database, "SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename").splitlines())
    if actual != tables:
        raise OpsError("The restored database has unknown or missing tables.")
    sequences = pg.sql(database, "SELECT sequencename FROM pg_sequences WHERE schemaname='public' ORDER BY sequencename").splitlines()
    for name in tables | set(sequences):
        identifier(name)
        owner = pg.sql(database, "SELECT pg_get_userbyid(c.relowner) FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname='public' AND c.relname=" + literal(name))
        if owner not in {pg.username, plan["roles"]["migration"]}:
            raise OpsError("A restored object has unexpected ownership.")
    restored_plan = {**plan, "database": database, "restored_from_database": plan["database"], "restore_marker": marker}
    write_exclusive(output_record, json.dumps(restored_plan, indent=2).encode() + b"\n")
    migration = identifier(plan["roles"]["migration"])
    statements = ["BEGIN;", f"ALTER DATABASE {identifier(database)} OWNER TO {migration};", f"ALTER SCHEMA public OWNER TO {migration};"]
    statements += [f"ALTER TABLE public.{identifier(name)} OWNER TO {migration};" for name in sorted(tables)]
    statements += [f"ALTER SEQUENCE public.{identifier(name)} OWNER TO {migration};" for name in sequences]
    statements += [f"COMMENT ON DATABASE {identifier(database)} IS {literal(plan['marker'])};", "COMMIT;"]
    # Alter only named objects in this new DB. REASSIGN OWNED could affect
    # shared database/tablespace ownership elsewhere and is intentionally absent.
    script(pg, database, "\n".join(statements))
    return {**apply_grants(pg, restored_plan), "ownership_record": str(outside_repository(output_record))}


def provision(pg, database, prefix, passwords, record):
    plan = role_plan(database, prefix, pg.owner)
    version, ddl = schema_ddl()  # Validate the reviewed policy before any DDL.
    if not isinstance(passwords, dict) or set(passwords) != set(plan["roles"]):
        raise OpsError("Supply three distinct operator passwords.")
    passwords = {kind: checked_password(value) for kind, value in passwords.items()}
    if len(set(passwords.values())) != 3:
        raise OpsError("Supply three distinct operator passwords.")
    if pg.sql("postgres", "SELECT rolsuper::int FROM pg_roles WHERE rolname=current_user") != "1":
        raise OpsError("Fresh role provisioning needs the controlled cluster administrator.")
    if pg.sql("postgres", "SELECT count(*) FROM pg_database WHERE datname=" + literal(database)) != "0":
        raise OpsError("The database already exists; provisioning never replaces it.")
    for name in plan["roles"].values():
        if pg.sql("postgres", "SELECT count(*) FROM pg_roles WHERE rolname=" + literal(name)) != "0":
            raise OpsError("A role already exists; provisioning never changes its password or grants.")
    write_exclusive(record, json.dumps(plan, indent=2).encode() + b"\n")
    try:
        statements = ["BEGIN;"]
        for name in plan["roles"].values():
            statements += [f"CREATE ROLE {identifier(name)} LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS NOINHERIT;", f"COMMENT ON ROLE {identifier(name)} IS {literal(plan['marker'])};"]
        statements.append("COMMIT;")
        script(pg, "postgres", "\n".join(statements))
        for kind, name in plan["roles"].items():
            # psql encrypts a prompted password client-side. It never becomes
            # SQL text or argv; standard statement/parameter logs are disabled
            # only for this privileged provisioning session.
            script(pg, "postgres", f"\\password {name}\n{passwords[kind]}\n{passwords[kind]}\n")
        pg.sql("postgres", f"CREATE DATABASE {identifier(database)} OWNER {identifier(plan['roles']['migration'])}")
        pg.sql("postgres", f"COMMENT ON DATABASE {identifier(database)} IS {literal(plan['marker'])}")
        script(pg, database, f"BEGIN; SET LOCAL ROLE {identifier(plan['roles']['migration'])};\n{ddl}\nCOMMIT;")
        result = apply_grants(pg, plan)
        return {**result, "ownership_record": str(outside_repository(record)), "initialized_schema": version}
    except BaseException:
        cleanup_failed(pg, plan)
        raise


def read_record(path):
    path = outside_repository(path)
    try:
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077 or info.st_uid not in {os.getuid(), 0}:
                raise OpsError("The ownership record must be a private regular file.")
            raw = stream.read(65537)
            if len(raw) > 65536:
                raise OpsError("The ownership record is too large.")
            return json.loads(raw)
    except (OSError, ValueError):
        raise OpsError("The role ownership record cannot be read safely.") from None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("provision", "grant", "grant-restored"))
    parser.add_argument("--container", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--admin-user", required=True)
    parser.add_argument("--passfile")
    parser.add_argument("--record", required=True)
    parser.add_argument("--database")
    parser.add_argument("--role-prefix")
    parser.add_argument("--app-password-file")
    parser.add_argument("--migration-password-file")
    parser.add_argument("--backup-password-file")
    parser.add_argument("--restore-marker")
    parser.add_argument("--confirm-database")
    parser.add_argument("--output-record")
    args = parser.parse_args(argv)
    try:
        pg = PostgreSQL(args.container, args.owner, args.admin_user, passfile=args.passfile)
        if args.command == "provision":
            if not args.database or not args.role_prefix:
                raise OpsError("A new database and role prefix are required.")
            paths = {"app": args.app_password_file, "migration": args.migration_password_file, "backup": args.backup_password_file}
            if any(path is None for path in paths.values()):
                raise OpsError("Provide all three private password files.")
            result = provision(pg, args.database, args.role_prefix, {kind: private_password(path) for kind, path in paths.items()}, args.record)
        elif args.command == "grant":
            result = apply_grants(pg, read_record(args.record))
        else:
            if not args.database or not args.output_record:
                raise OpsError("The restored database and a new output ownership record are required.")
            result = grant_restored(pg, read_record(args.record), args.database, args.restore_marker, args.confirm_database, args.output_record)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (OpsError, OSError) as error:
        print(json.dumps({"schema_version": 1, "error": str(error) if isinstance(error, OpsError) else "An operator file was unavailable."}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
