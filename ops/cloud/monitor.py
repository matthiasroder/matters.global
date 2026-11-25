#!/usr/bin/env python3
"""Bounded HTTPS and authenticated backup monitoring for operators.

This tool emits safe JSON and private append-only state. It sends no messages,
deploys no service and chooses no production recovery or availability target.
"""

import argparse
import fcntl
import hashlib
import http.client
import importlib.util
import json
import math
import os
import re
import selectors
import ssl
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit


REPOSITORY = Path(__file__).resolve().parents[2]
MAX_BACKUP_BYTES = 64 * 1024 * 1024
MAX_OUTPUT_BYTES = 8192
MAX_RECORDS = 10000
STATES = {"healthy", "failure", "stale"}
RECORD_NAME = re.compile(r"sample-([0-9]{6})\.json\Z")


class MonitorError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def outside_repository(value):
    path = Path(value).expanduser().absolute()
    if ".." in path.parts or path == REPOSITORY or REPOSITORY in path.parents:
        raise MonitorError("unsafe_path")
    return path


def open_directory(path):
    """Walk with directory descriptors; reject symlinks in every component."""
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in Path(path).parts[1:]:
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except OSError:
        os.close(descriptor)
        raise MonitorError("unsafe_path") from None


def open_artifact(value, *, private=True, maximum=MAX_BACKUP_BYTES):
    path = outside_repository(value)
    parent = open_directory(path.parent)
    try:
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
    except OSError:
        raise MonitorError("artifact_unreadable") from None
    finally:
        os.close(parent)
    info = os.fstat(descriptor)
    if (
        not stat.S_ISREG(info.st_mode) or info.st_uid not in ({os.getuid()} if private else {0, os.getuid()})
        or info.st_nlink != 1 or info.st_size > maximum
        or (private and info.st_mode & 0o077)
    ):
        os.close(descriptor)
        raise MonitorError("unsafe_artifact")
    return descriptor


def read_bounded(descriptor, maximum):
    result = bytearray()
    while True:
        chunk = os.read(descriptor, min(65536, maximum + 1 - len(result)))
        if not chunk:
            return bytes(result)
        result.extend(chunk)
        if len(result) > maximum:
            raise MonitorError("byte_limit")


def _module(name):
    specification = importlib.util.spec_from_file_location("monitor_" + name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def validated_origin(origin, allow_remote=False):
    try:
        parsed = urlsplit(origin)
        if (
            not isinstance(origin, str) or len(origin) > 2048
            or any(ord(character) <= 32 or ord(character) == 127 for character in origin)
            or parsed.scheme != "https" or not parsed.hostname
            or (not allow_remote and parsed.hostname not in {"localhost", "127.0.0.1"})
            or parsed.username is not None or parsed.password is not None
            or parsed.path not in {"", "/"} or parsed.query or parsed.fragment
            or (parsed.port is not None and not 1 <= parsed.port <= 65535)
        ):
            raise ValueError
        return parsed
    except (TypeError, ValueError):
        raise MonitorError("invalid_configuration") from None


def worker_health(origin, certificate_descriptor, allow_remote=False):
    # Reuse the repository's verified-CA, no-proxy, no-redirect localhost probe.
    if not allow_remote:
        result = _module("staging").probe(origin, f"/proc/self/fd/{certificate_descriptor}", samples=1)
        return {"state": "healthy", "code": "https_ok", "latency_ms": result["health_max_ms"]}
    parsed = validated_origin(origin, allow_remote=True)
    context = ssl.create_default_context(cafile=f"/proc/self/fd/{certificate_descriptor}")
    # Explicit operator-selected remote/private targets use direct transport,
    # never ambient proxies. The parent process bounds DNS, TLS and body reads.
    connection = http.client.HTTPSConnection(parsed.hostname, parsed.port or 443, timeout=5, context=context)
    started = time.monotonic()
    try:
        connection.request("GET", "/healthz", headers={"Accept": "application/json"})
        response = connection.getresponse()
        if response.status != 200:
            raise MonitorError("health_failed")
        declared = response.getheader("Content-Length")
        if declared is not None and (not declared.isdecimal() or int(declared) > 1024):
            raise MonitorError("health_body_limit")
        body = response.read(1025)
        if len(body) > 1024:
            raise MonitorError("health_body_limit")
        document = json.loads(body)
        if not isinstance(document, dict) or document.get("status") != "ok":
            raise MonitorError("health_failed")
        return {"state": "healthy", "code": "https_ok", "latency_ms": round((time.monotonic() - started) * 1000, 3)}
    finally:
        connection.close()


def worker_backup(backup_descriptor, key_descriptor, maximum, now, max_age, clock_skew):
    recovery = _module("recovery")
    key = read_bounded(key_descriptor, 32)
    if len(key) != 32:
        raise MonitorError("backup_key_invalid")
    package = read_bounded(backup_descriptor, maximum)
    # Authenticates AES-GCM and checks the archive hash/size using the recovery
    # codec. No dump is restored and no plaintext or manifest fields are logged.
    metadata, _dump = recovery.unseal(package, key)
    try:
        created = datetime.fromisoformat(metadata["created_at_utc"])
        if created.tzinfo is None or created.utcoffset() is None:
            raise ValueError
        age = now - created.timestamp()
        if not math.isfinite(age) or age < -clock_skew:
            raise ValueError
    except (KeyError, TypeError, ValueError, OverflowError):
        raise MonitorError("backup_time_invalid") from None
    stale = age > max_age
    return {"state": "stale" if stale else "healthy", "code": "backup_stale" if stale else "backup_ok", "age_seconds": round(max(0, age), 3), "encrypted_bytes": len(package), "authenticated": True}


def bounded_worker(arguments, descriptors, timeout):
    """Bound the whole check, including TLS/body reads and archive decryption."""
    process = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "_worker", *arguments],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        pass_fds=tuple(descriptors), bufsize=0,
    )
    deadline = time.monotonic() + timeout
    output = bytearray()
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise MonitorError("check_timeout")
                chunk = os.read(process.stdout.fileno(), 4096)
                if not chunk:
                    break
                output.extend(chunk)
                if len(output) > MAX_OUTPUT_BYTES:
                    raise MonitorError("check_output_limit")
        if process.wait(timeout=max(0.01, deadline - time.monotonic())) != 0:
            raise MonitorError("check_failed")
        value = json.loads(output)
        if not isinstance(value, dict) or value.get("state") not in STATES:
            raise MonitorError("check_invalid")
        return value
    except MonitorError:
        raise
    except (OSError, ValueError, subprocess.SubprocessError):
        raise MonitorError("check_failed") from None
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stdout.close()


def checked_health(origin, ca_file, timeout, *, allow_remote=False):
    descriptor = None
    try:
        descriptor = open_artifact(ca_file, private=False, maximum=1024 * 1024)
        return bounded_worker(["health", origin, str(descriptor), str(int(allow_remote))], [descriptor], timeout)
    except MonitorError as error:
        return {"state": "failure", "code": error.code}
    finally:
        if descriptor is not None:
            os.close(descriptor)


def checked_backup(backup_file, key_file, *, timeout, maximum, now, max_age, clock_skew):
    descriptors = []
    try:
        descriptors.append(open_artifact(backup_file, maximum=maximum))
        descriptors.append(open_artifact(key_file, maximum=32))
        return bounded_worker(["backup", *(str(descriptor) for descriptor in descriptors), str(maximum), str(now), str(max_age), str(clock_skew)], descriptors, timeout)
    except MonitorError as error:
        return {"state": "failure", "code": error.code}
    finally:
        for descriptor in descriptors:
            os.close(descriptor)


class StateStore:
    """Private immutable records; never replace an existing operator file."""

    def __init__(self, path, binding):
        self.path = outside_repository(path)
        self.binding = binding
        self.descriptor = None
        self.sequence = 0
        self.previous = None

    def _read(self, name):
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=self.descriptor)
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077 or info.st_nlink != 1:
                raise MonitorError("unsafe_state")
            return json.loads(read_bounded(descriptor, MAX_OUTPUT_BYTES))
        finally:
            os.close(descriptor)

    def _write(self, name, data):
        encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        if len(encoded) > MAX_OUTPUT_BYTES:
            raise MonitorError("state_limit")
        descriptor = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=self.descriptor)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.fsync(self.descriptor)
        except BaseException:
            # A partial immutable record remains evidence of interruption.
            # It is never silently replaced on the next invocation.
            raise

    def __enter__(self):
        parent = open_directory(self.path.parent)
        created = False
        try:
            try:
                os.mkdir(self.path.name, 0o700, dir_fd=parent)
                created = True
            except FileExistsError:
                pass
            self.descriptor = os.open(self.path.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        except OSError:
            raise MonitorError("unsafe_state") from None
        finally:
            os.close(parent)
        try:
            info = os.fstat(self.descriptor)
            if info.st_uid != os.getuid() or info.st_mode & 0o077:
                raise MonitorError("unsafe_state")
            try:
                fcntl.flock(self.descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise MonitorError("state_busy") from None
            header = {"schema_version": 1, "format": "matters-operator-monitor", "binding": self.binding, "owner_uid": os.getuid()}
            if created:
                self._write("monitor.json", header)
            elif self._read("monitor.json") != header:
                raise MonitorError("state_binding_mismatch")
            records = []
            with os.scandir(self.descriptor) as entries:
                for entry in entries:
                    if len(records) > MAX_RECORDS:
                        raise MonitorError("state_limit")
                    if entry.name == "monitor.json":
                        continue
                    match = RECORD_NAME.fullmatch(entry.name)
                    if not match or not entry.is_file(follow_symlinks=False):
                        raise MonitorError("unsafe_state")
                    records.append(int(match.group(1)))
            records.sort()
            if records != list(range(1, len(records) + 1)):
                raise MonitorError("invalid_state")
            self.sequence = len(records)
            if records:
                last = self._read(f"sample-{self.sequence:06d}.json")
                if not isinstance(last, dict) or last.get("schema_version") != 1 or last.get("sequence") != self.sequence or last.get("state") not in STATES:
                    raise MonitorError("invalid_state")
                self.previous = last["state"]
            return self
        except MonitorError:
            self.__exit__(None, None, None)
            raise
        except (OSError, ValueError):
            self.__exit__(None, None, None)
            raise MonitorError("unsafe_state") from None

    def append(self, sample):
        if self.sequence >= MAX_RECORDS:
            raise MonitorError("state_limit")
        self.sequence += 1
        record = {**sample, "schema_version": 1, "sequence": self.sequence}
        self._write(f"sample-{self.sequence:06d}.json", record)
        self.previous = record["state"]
        return record

    def __exit__(self, *_):
        if self.descriptor is not None:
            os.close(self.descriptor)
            self.descriptor = None


def validate_configuration(origin, max_age, timeout, maximum, iterations, interval, clock_skew, *, allow_remote=False):
    try:
        if type(allow_remote) is not bool:
            raise ValueError
        validated_origin(origin, allow_remote)
        numbers = [max_age, timeout, maximum, iterations, interval, clock_skew]
        if any(isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) for number in numbers):
            raise ValueError
        if (
            not 0 < max_age <= 366 * 86400 or not 0 < timeout <= 30
            or type(maximum) is not int or not 1 <= maximum <= MAX_BACKUP_BYTES
            or type(iterations) is not int or not 1 <= iterations <= 100
            or not 0 <= interval <= 60 or (iterations > 1 and interval == 0)
            or not 0 <= clock_skew <= 3600
            or iterations * timeout * 2 + max(0, iterations - 1) * interval > 3600
        ):
            raise ValueError
    except (TypeError, ValueError):
        raise MonitorError("invalid_configuration") from None


def sample(origin, ca_file, backup_file, key_file, *, max_age, timeout, maximum, clock_skew, previous=None, now=None, allow_remote=False):
    now = time.time() if now is None else now
    started = time.monotonic()
    health = checked_health(origin, ca_file, timeout, allow_remote=allow_remote)
    backup = checked_backup(backup_file, key_file, timeout=timeout, maximum=maximum, now=now, max_age=max_age, clock_skew=clock_skew)
    state = "failure" if "failure" in {health["state"], backup["state"]} else backup["state"]
    event = "recovered" if state == "healthy" and previous in {"failure", "stale"} else "unchanged" if state == previous else state
    return {"schema_version": 1, "checked_at_utc": datetime.fromtimestamp(now, timezone.utc).isoformat(), "state": state, "event": event, "health": health, "backup": backup, "elapsed_ms": round((time.monotonic() - started) * 1000, 3)}


class SafeParser(argparse.ArgumentParser):
    def error(self, _message):
        print(json.dumps({"schema_version": 1, "error": "invalid_configuration"}), file=sys.stderr)
        raise SystemExit(2)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "_worker":
        try:
            if argv[1] == "health" and len(argv) == 5 and argv[4] in {"0", "1"}:
                result = worker_health(argv[2], int(argv[3]), argv[4] == "1")
            elif argv[1] == "backup" and len(argv) == 8:
                result = worker_backup(int(argv[2]), int(argv[3]), int(argv[4]), float(argv[5]), float(argv[6]), float(argv[7]))
            else:
                raise MonitorError("check_invalid")
            print(json.dumps(result, sort_keys=True))
            return 0
        except Exception:
            print(json.dumps({"state": "failure", "code": "check_failed"}))
            return 0
    parser = SafeParser(description=__doc__)
    parser.add_argument("--origin", required=True, help="Exact HTTPS origin; local by default, credentials and query strings are refused.")
    parser.add_argument("--allow-remote-origin", action="store_true", help="Permit an explicit private/public HTTPS target; still verify TLS and refuse redirects/proxies.")
    parser.add_argument("--ca-file", required=True)
    parser.add_argument("--backup-file", required=True)
    parser.add_argument("--key-file", required=True)
    parser.add_argument("--state-directory", required=True, help="Private operator directory outside the repository; created exclusively on first use.")
    parser.add_argument("--max-backup-age-seconds", type=float, required=True, help="Explicit operator test/deployment threshold, not a product recovery commitment.")
    parser.add_argument("--timeout-seconds", type=float, default=5, help="Whole-check deadline, capped at 30 seconds.")
    parser.add_argument("--max-backup-bytes", type=int, default=MAX_BACKUP_BYTES)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--interval-seconds", type=float, default=0)
    parser.add_argument("--clock-skew-seconds", type=float, default=60)
    args = parser.parse_args(argv)
    try:
        validate_configuration(args.origin, args.max_backup_age_seconds, args.timeout_seconds, args.max_backup_bytes, args.iterations, args.interval_seconds, args.clock_skew_seconds, allow_remote=args.allow_remote_origin)
        origin = args.origin.rstrip("/")
        paths = [outside_repository(path) for path in (args.ca_file, args.backup_file, args.key_file)]
        binding = hashlib.sha256(json.dumps([origin, args.allow_remote_origin, *(str(path) for path in paths)], separators=(",", ":")).encode()).hexdigest()
        with StateStore(args.state_directory, binding) as state:
            if state.sequence + args.iterations > MAX_RECORDS:
                raise MonitorError("state_limit")
            for iteration in range(args.iterations):
                result = sample(origin, *paths, max_age=args.max_backup_age_seconds, timeout=args.timeout_seconds, maximum=args.max_backup_bytes, clock_skew=args.clock_skew_seconds, previous=state.previous, allow_remote=args.allow_remote_origin)
                print(json.dumps(state.append(result), sort_keys=True), flush=True)
                if iteration + 1 < args.iterations:
                    time.sleep(args.interval_seconds)
        return 0 if result["state"] == "healthy" else 1
    except (OSError, ValueError):
        print(json.dumps({"schema_version": 1, "error": "monitor_configuration_or_state_failed"}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
