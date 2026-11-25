#!/usr/bin/env python3
"""Prepare private, disposable localhost staging secrets; never deploy publicly."""

import argparse
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


def prepare(directory, image, port=19443):
    directory = Path(directory).expanduser().absolute()
    directory = directory.parent.resolve() / directory.name
    if directory == REPOSITORY or REPOSITORY in directory.parents:
        raise ValueError("Staging credentials must be outside the repository.")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./:@-]{0,255}", image) or not re.fullmatch(r"/[A-Za-z0-9_/.-]+", str(directory)) or not 1024 <= port <= 65535:
        raise ValueError("Supply an application image and an unprivileged HTTPS port.")
    if not shutil.which("openssl"):
        raise ValueError("OpenSSL is required to create a disposable localhost certificate.")
    os.mkdir(directory, 0o700)  # Refuses existing paths and symlinks.
    password = secrets.token_hex(24)
    owner = "matters-staging-" + secrets.token_hex(6)

    def private(name, data, mode=0o600):
        with (directory / name).open("xb") as stream:
            os.fchmod(stream.fileno(), mode)
            stream.write(data)

    try:
        private("backup.key", secrets.token_bytes(32))
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
            "MATTERS_CLOUD_DATABASE_URL": f"postgresql+psycopg://matters_ops:{password}@database:5432/matters",
            "MATTERS_STAGING_SECRETS": str(directory),
            "MATTERS_STAGING_HTTPS_PORT": str(port),
            "MATTERS_CLOUD_PUBLIC_ORIGIN": origin,
        }
        private("staging.env", "".join(f"{key}={value}\n" for key, value in settings.items()).encode())
        return {"schema_version": 1, "origin": origin, "environment_file": str(directory / "staging.env"), "operations_owner": owner, "certificate": str(directory / "server.crt")}
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("prepare")
    setup.add_argument("--directory", required=True)
    setup.add_argument("--image", required=True)
    setup.add_argument("--port", type=int, default=19443)
    health = commands.add_parser("probe")
    health.add_argument("--origin", required=True)
    health.add_argument("--ca-file", required=True)
    health.add_argument("--samples", type=int, default=3)
    args = parser.parse_args(argv)
    try:
        result = prepare(args.directory, args.image, args.port) if args.command == "prepare" else probe(args.origin, args.ca_file, args.samples)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (OSError, ValueError, subprocess.SubprocessError):
        print(json.dumps({"schema_version": 1, "error": "Staging preparation or verified HTTPS probe failed; no public deployment was attempted."}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
