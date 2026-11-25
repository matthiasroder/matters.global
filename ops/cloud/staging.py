#!/usr/bin/env python3
"""Prepare private, disposable localhost staging secrets; never deploy publicly."""

import argparse
import ipaddress
import json
import os
import re
import secrets
import shutil
import ssl
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit


REPOSITORY = Path(__file__).resolve().parents[2]


def prepare(directory, image, port=19443, subnet="10.243.17.0/28"):
    directory = Path(directory).expanduser().absolute()
    directory = directory.parent.resolve() / directory.name
    if directory == REPOSITORY or REPOSITORY in directory.parents:
        raise ValueError("Staging credentials must be outside the repository.")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./:@-]{0,255}", image) or not re.fullmatch(r"/[A-Za-z0-9_/.-]+", str(directory)) or not 1024 <= port <= 65535:
        raise ValueError("Supply an application image and an unprivileged HTTPS port.")
    if not shutil.which("openssl"):
        raise ValueError("OpenSSL is required to create a disposable localhost certificate.")
    network = ipaddress.ip_network(subnet)
    private_networks = [ipaddress.ip_network(value) for value in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")]
    if network.version != 4 or network.prefixlen != 28 or not any(network.subnet_of(private) for private in private_networks):
        raise ValueError("Select an unused private IPv4 /28 subnet for this staging project.")
    os.mkdir(directory, 0o700)  # Refuses existing paths and symlinks.
    password = secrets.token_hex(24)
    role_passwords = {kind: secrets.token_hex(24) for kind in ("app", "migration", "backup")}
    owner = "matters-staging-" + secrets.token_hex(6)

    def private(name, data, mode=0o600):
        with (directory / name).open("xb") as stream:
            os.fchmod(stream.fileno(), mode)
            stream.write(data)

    try:
        private("backup.key", secrets.token_bytes(32))
        for kind, value in role_passwords.items():
            private(kind + ".password", (value + "\n").encode())
        subprocess.run([
            "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "2",
            "-keyout", str(directory / "server.key"), "-out", str(directory / "server.crt"),
            "-subj", "/CN=localhost", "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1",
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Readable by the non-root proxy's read-only mount. The host directory
        # remains mode 0700; no other service receives the TLS private key.
        os.chmod(directory / "server.key", 0o444)
        os.chmod(directory / "server.crt", 0o444)
        origin = f"https://localhost:{port}"
        settings = {
            "MATTERS_OPERATIONS_OWNER": owner,
            "MATTERS_CLOUD_IMAGE": image,
            "MATTERS_PG_PASSWORD": password,
            "MATTERS_CLOUD_DATABASE_URL": f"postgresql+psycopg://matters_app:{role_passwords['app']}@database:5432/matters",
            "MATTERS_STAGING_SECRETS": str(directory),
            "MATTERS_STAGING_HTTPS_PORT": str(port),
            "MATTERS_CLOUD_PUBLIC_ORIGIN": origin,
            "MATTERS_STAGING_SUBNET": str(network),
            "MATTERS_STAGING_PROXY_IP": str(network.network_address + 3),
            "MATTERS_STAGING_APP_IP": str(network.network_address + 4),
            "MATTERS_STAGING_DATABASE_IP": str(network.network_address + 5),
        }
        private("staging.env", "".join(f"{key}={value}\n" for key, value in settings.items()).encode())
        public = {"schema_version": 1, "origin": origin, "environment_file": str(directory / "staging.env"), "operations_owner": owner, "certificate": str(directory / "server.crt"), "subnet": str(network), "trusted_proxy": settings["MATTERS_STAGING_PROXY_IP"] + "/32", "database": "matters", "role_prefix": "matters"}
        private("operator.json", (json.dumps(public, indent=2) + "\n").encode())
        return public
    except BaseException:
        # The directory was exclusively created by this invocation.
        shutil.rmtree(directory)
        raise


def probe(origin, ca_file, samples=3):
    parsed = urlsplit(origin)
    if (
        parsed.scheme != "https" or parsed.hostname not in {"localhost", "127.0.0.1"}
        or parsed.username or parsed.password or parsed.path not in {"", "/"}
        or parsed.query or parsed.fragment or not 1 <= samples <= 20
    ):
        raise ValueError("Use an exact localhost HTTPS origin and 1..20 health samples.")
    context = ssl.create_default_context(cafile=str(ca_file))
    # Direct loopback transport; no external URL or credential-bearing request.
    class NoRedirects(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *_args, **_kwargs):
            raise ValueError("A staging health probe must not redirect.")

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), urllib.request.HTTPSHandler(context=context), NoRedirects())
    timings = []
    for _ in range(samples):
        start = time.monotonic()
        with opener.open(origin.rstrip("/") + "/healthz", timeout=5) as response:
            if response.status != 200 or json.loads(response.read(1024)).get("status") != "ok":
                raise ValueError("The staging health check failed.")
        timings.append((time.monotonic() - start) * 1000)
    return {"schema_version": 1, "verified_https": True, "samples": samples, "health_median_ms": round(statistics.median(timings), 3), "health_max_ms": round(max(timings), 3)}


def provision_database(directory, project):
    """Initialize a fresh labelled database before starting the runtime app."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from database_roles import PostgreSQL, private_password, provision, read_record
    directory = Path(directory).expanduser().resolve()
    metadata = read_record(directory / "operator.json")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,62}", project):
        raise ValueError("Use the exact staging Compose project name.")
    environment = dict(os.environ)
    for key in ("DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_TLS", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH"):
        environment.pop(key, None)
    command = ["docker", "--host=unix:///var/run/docker.sock", "compose", "-p", project, "--env-file", str(directory / "staging.env"), "-f", str(Path(__file__).with_name("staging.compose.yml")), "ps", "-q", "database"]
    result = subprocess.run(command, capture_output=True, timeout=20, env=environment, check=True)
    container = result.stdout.decode().strip()
    pg = PostgreSQL(container, metadata["operations_owner"], "matters_ops")
    return provision(pg, metadata["database"], metadata["role_prefix"], {kind: private_password(directory / (kind + ".password")) for kind in ("app", "migration", "backup")}, directory / "roles.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("prepare")
    setup.add_argument("--directory", required=True)
    setup.add_argument("--image", required=True)
    setup.add_argument("--port", type=int, default=19443)
    setup.add_argument("--subnet", default="10.243.17.0/28", help="Unused private /28 for the isolated project network.")
    provision = commands.add_parser("provision")
    provision.add_argument("--directory", required=True)
    provision.add_argument("--project", required=True)
    health = commands.add_parser("probe")
    health.add_argument("--origin", required=True)
    health.add_argument("--ca-file", required=True)
    health.add_argument("--samples", type=int, default=3)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(args.directory, args.image, args.port, args.subnet)
        elif args.command == "provision":
            result = provision_database(args.directory, args.project)
        else:
            result = probe(args.origin, args.ca_file, args.samples)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (OSError, ValueError, subprocess.SubprocessError):
        print(json.dumps({"schema_version": 1, "error": "Staging preparation or verified HTTPS probe failed; no public deployment was attempted."}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
