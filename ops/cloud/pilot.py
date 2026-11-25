#!/usr/bin/env python3
"""Prepare a fresh Matters pilot behind an explicitly configured existing proxy.

Preparation writes private local artifacts only. It does not edit, reload or
attach an existing proxy, publish ports, create networks or deploy a provider.
"""

import argparse
import ipaddress
import json
import math
import os
import re
import secrets
import stat
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recovery import OpsError, PostgreSQL, outside_repository
from database_roles import private_password, provision, read_record

NAME = re.compile(r"[a-z][a-z0-9_-]{0,47}\Z")
NETWORK_NAME = re.compile(r"[a-z][a-z0-9_.-]{0,62}\Z")
IMAGE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./:@-]{0,255}\Z")


def exact_origin(value):
    try:
        if not isinstance(value, str) or any(character.isspace() or ord(character) < 32 for character in value):
            raise ValueError()
        parts = urlsplit(value)
        host = parts.hostname
        if (parts.scheme != "https" or not host or parts.username is not None
                or parts.password is not None or parts.path not in {"", "/"}
                or parts.query or parts.fragment or not 1 <= (parts.port or 443) <= 65535):
            raise ValueError()
        try:
            address = ipaddress.ip_address(host)
            host = "[" + str(address) + "]" if address.version == 6 else str(address)
        except ValueError:
            if len(host) > 253 or any(not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label) for label in host.split(".")):
                raise ValueError()
        authority = host + (":" + str(parts.port) if parts.port and parts.port != 443 else "")
        return "https://" + authority
    except (ValueError, TypeError, AttributeError):
        raise OpsError("Supply an exact HTTPS origin with no credentials, path or query.") from None


def private_subnet(value):
    try:
        network = ipaddress.ip_network(value)
        private = [ipaddress.ip_network(item) for item in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")]
        if network.version != 4 or network.prefixlen != 28 or not any(network.subnet_of(item) for item in private):
            raise ValueError()
        return network
    except (ValueError, TypeError):
        raise OpsError("Choose distinct unused private IPv4 /28 networks.") from None


def positive(value, *, integer=False):
    if type(value) is bool or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 or (integer and type(value) is not int):
        raise OpsError("Resource bounds must be positive finite numbers; memory and process bounds are integers.")
    return str(value)


def prepare(directory, project, origin, image, ingress_network, ingress_subnet, backend_subnet, proxy_ip, *, app_memory_mib=512, database_memory_mib=512, app_cpus=0.75, database_cpus=0.75, app_pids_limit=128, database_pids_limit=128):
    directory = outside_repository(directory)
    if not re.fullmatch(r"/[A-Za-z0-9_/.-]+", str(directory)):
        raise OpsError("Use an absolute operator directory without whitespace or configuration characters.")
    if not NAME.fullmatch(project) or not NETWORK_NAME.fullmatch(ingress_network) or ingress_network == project + "-backend":
        raise OpsError("Use explicit lowercase project and dedicated ingress-network names.")
    if not IMAGE.fullmatch(image) or "://" in image or ("@" in image and not re.fullmatch(r"[^@]+@sha256:[a-f0-9]{64}", image)):
        raise OpsError("Supply a reviewed image tag or digest without configuration characters.")
    origin = exact_origin(origin)
    ingress, backend = private_subnet(ingress_subnet), private_subnet(backend_subnet)
    if ingress.overlaps(backend):
        raise OpsError("The ingress and backend networks must not overlap.")
    try:
        proxy = ipaddress.ip_address(proxy_ip)
        if proxy not in ingress or int(proxy) - int(ingress.network_address) not in {2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}:
            raise ValueError()
    except (ValueError, TypeError):
        raise OpsError("Choose the controlled proxy's fixed usable ingress address, excluding gateway and app address.") from None
    resources = {
        "MATTERS_PILOT_APP_MEMORY": positive(app_memory_mib, integer=True) + "m",
        "MATTERS_PILOT_DATABASE_MEMORY": positive(database_memory_mib, integer=True) + "m",
        "MATTERS_PILOT_APP_CPUS": positive(app_cpus),
        "MATTERS_PILOT_DATABASE_CPUS": positive(database_cpus),
        "MATTERS_PILOT_APP_PIDS": positive(app_pids_limit, integer=True),
        "MATTERS_PILOT_DATABASE_PIDS": positive(database_pids_limit, integer=True),
    }
    owner = "matters-pilot-" + secrets.token_hex(8)
    passwords = {kind: secrets.token_hex(24) for kind in ("admin", "app", "migration", "backup")}
    upstream = project + "-app"
    log_name = "matters_" + project.replace("-", "_") + "_" + owner[-16:]
    settings = {
        "MATTERS_OPERATIONS_OWNER": owner,
        "MATTERS_PILOT_DIRECTORY": str(directory),
        "MATTERS_CLOUD_IMAGE": image,
        "MATTERS_CLOUD_PUBLIC_ORIGIN": origin,
        "MATTERS_CLOUD_DATABASE_URL": "postgresql+psycopg://matters_app:" + passwords["app"] + "@database:5432/matters",
        "MATTERS_PILOT_INGRESS_NETWORK": ingress_network,
        "MATTERS_PILOT_INGRESS_SUBNET": str(ingress),
        "MATTERS_PILOT_BACKEND_NETWORK": project + "-backend",
        "MATTERS_PILOT_BACKEND_SUBNET": str(backend),
        "MATTERS_PILOT_PROXY_IP": str(proxy),
        "MATTERS_PILOT_APP_INGRESS_IP": str(ingress.network_address + 4),
        "MATTERS_PILOT_APP_BACKEND_IP": str(backend.network_address + 3),
        "MATTERS_PILOT_DATABASE_IP": str(backend.network_address + 4),
        "MATTERS_PILOT_UPSTREAM": upstream,
        **resources,
    }
    metadata = {
        "schema_version": 1, "format": "matters-pilot-preparation",
        "directory": str(directory), "project": project, "origin": origin,
        "image": image, "operations_owner": owner,
        "ingress_network": ingress_network, "ingress_subnet": str(ingress),
        "backend_network": settings["MATTERS_PILOT_BACKEND_NETWORK"], "backend_subnet": str(backend),
        "proxy_ip": str(proxy), "trusted_proxy": str(proxy) + "/32",
        "upstream": upstream + ":3000", "database": "matters", "role_prefix": "matters",
        "caddy_log_name": log_name,
        "resource_bounds_are_provisional": True,
    }
    snippet = (origin + " {\n"
        "    # Integrate through the existing proxy's persistent configuration source.\n"
        "    # Discard site access logs; keep OAuth query values and credentials out of diagnostics.\n"
        "    log " + log_name + " {\n        output discard\n    }\n"
        "    reverse_proxy " + upstream + ":3000 {\n"
        "        header_up Host " + urlsplit(origin).netloc + "\n"
        "        header_up Authorization {http.request.header.Authorization}\n"
        "        header_up X-Forwarded-Proto https\n"
        "        header_up X-Forwarded-For {http.request.remote.host}\n"
        "    }\n}\n")
    logging_merge = (
        "# REVIEW/MERGE ONLY: extend the existing global block; do not replace co-tenant logging.\n"
        "# Add this namespace to the existing default logger's excludes. Preserve all other settings.\n"
        "{\n"
        "    log default {\n        exclude http.log.error." + log_name + "\n    }\n"
        "}\n"
    )
    os.mkdir(directory, 0o700)  # No adoption, overwrite or symlink following.
    os.chmod(directory, 0o700)
    try:
        def private(name, content):
            with (directory / name).open("xb") as stream:
                os.fchmod(stream.fileno(), 0o600)
                stream.write(content)
        for kind, password in passwords.items():
            private(kind + ".password", (password + "\n").encode())
        private("backup.key", secrets.token_bytes(32))
        private("pilot.env", "".join(key + "=" + value + "\n" for key, value in settings.items()).encode())
        private("operator.json", (json.dumps(metadata, indent=2) + "\n").encode())
        private("Caddyfile.site", snippet.encode())
        private("Caddyfile.logging.merge", logging_merge.encode())
        private("pilot.compose.yml", Path(__file__).with_name("pilot.compose.yml").read_bytes())
    except BaseException:
        shutil.rmtree(directory)  # Only the directory just created exclusively here.
        raise
    return metadata


def provision_database(directory):
    directory = outside_repository(directory)
    info = directory.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_mode & 0o077 or info.st_uid not in {os.getuid(), 0}:
        raise OpsError("The pilot directory must remain private and operator-owned.")
    metadata = read_record(directory / "operator.json")
    if (not isinstance(metadata, dict) or metadata.get("format") != "matters-pilot-preparation" or metadata.get("schema_version") != 1
            or metadata.get("directory") != str(directory) or not isinstance(metadata.get("project"), str)
            or not NAME.fullmatch(metadata["project"]) or metadata.get("database") != "matters"
            or metadata.get("role_prefix") != "matters" or not isinstance(metadata.get("operations_owner"), str)
            or not re.fullmatch(r"matters-pilot-[0-9a-f]{16}", metadata["operations_owner"])):
        raise OpsError("Use the unchanged private inventory for this pilot directory.")
    if os.path.lexists(directory / "roles.json"):
        raise OpsError("The role inventory already exists; provisioning never reruns or overwrites it.")
    environment = dict(os.environ)
    for key in list(environment):
        if key.startswith("MATTERS_") or key in {"DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_TLS", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH", "COMPOSE_FILE", "COMPOSE_PROJECT_NAME", "COMPOSE_PROFILES"}:
            environment.pop(key, None)
    command = ["docker", "--host=unix:///var/run/docker.sock", "compose", "--project-name", metadata["project"], "--env-file", str(directory / "pilot.env"), "-f", str(directory / "pilot.compose.yml"), "ps", "-q", "database"]
    try:
        result = subprocess.run(command, capture_output=True, timeout=20, env=environment, check=True)
        container = result.stdout.decode().strip()
    except (OSError, UnicodeError, subprocess.SubprocessError):
        raise OpsError("The selected pilot database could not be resolved safely.") from None
    pg = PostgreSQL(container, metadata["operations_owner"], "matters_ops")
    return provision(pg, metadata["database"], metadata["role_prefix"], {kind: private_password(directory / (kind + ".password")) for kind in ("app", "migration", "backup")}, directory / "roles.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("prepare")
    for name in ("directory", "project", "origin", "image", "ingress-network", "ingress-subnet", "backend-subnet", "proxy-ip"):
        setup.add_argument("--" + name, required=True)
    for kind in ("app", "database"):
        setup.add_argument("--" + kind + "-memory-mib", type=int, default=512)
        setup.add_argument("--" + kind + "-cpus", type=float, default=0.75)
        setup.add_argument("--" + kind + "-pids-limit", type=int, default=128)
    initialize = commands.add_parser("provision")
    initialize.add_argument("--directory", required=True)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    try:
        result = prepare(**args) if command == "prepare" else provision_database(**args)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (OpsError, OSError, ValueError):
        print(json.dumps({"schema_version": 1, "error": "Pilot preparation or fresh role provisioning failed; existing proxy configuration was not changed."}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
