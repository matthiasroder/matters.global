"""Atomic security budgets shared across connections, processes and restarts."""

import json
import multiprocessing
import os
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import pytest

pytest.importorskip("sqlalchemy")

from sqlalchemy import create_engine, func, select
from sqlalchemy.engine import make_url

from matters.cloud.db import Database, rate_limit_buckets
from matters.cloud.rate_limits import CAPACITY_KEY, DatabaseRateLimiter, RateLimitUnavailable


BACKENDS = ["sqlite"] + (["postgresql"] if os.environ.get("MATTERS_TEST_DATABASE_URL") else [])


@pytest.fixture(params=BACKENDS)
def database(request, tmp_path):
    admin = None
    schema = None
    if request.param == "sqlite":
        url = f"sqlite:///{tmp_path / 'shared-limits.sqlite'}"
    else:
        pytest.importorskip("psycopg")
        configured = make_url(os.environ["MATTERS_TEST_DATABASE_URL"])
        assert configured.get_backend_name() == "postgresql"
        admin = create_engine(configured)
        schema = "rate_limits_" + uuid.uuid4().hex
        with admin.begin() as connection:
            connection.exec_driver_sql(f'CREATE SCHEMA "{schema}"')
        url = configured.update_query_dict({"options": f"-csearch_path={schema}"})
    db = Database(url).initialize()
    try:
        yield db, url
    finally:
        db.close()
        if admin is not None:
            with admin.begin() as connection:
                connection.exec_driver_sql(f'DROP SCHEMA "{schema}" CASCADE')
            admin.dispose()


def rows(db):
    with db.transaction() as connection:
        return [dict(row) for row in connection.execute(select(rate_limit_buckets)).mappings()]


def process_consume(parameters):
    url, attempts, requests = parameters
    db = Database(url)
    try:
        limiter = DatabaseRateLimiter(db, attempts=attempts, now=lambda: 1000)
        return sum(limiter.allow("203.0.113.24") for _ in range(requests))
    finally:
        db.close()


def test_separate_database_instances_share_exact_concurrent_budget(database):
    db, url = database
    other = Database(url)
    try:
        limiters = [DatabaseRateLimiter(instance, attempts=7, now=lambda: 1000) for instance in (db, other)]
        with ThreadPoolExecutor(max_workers=16) as pool:
            results = list(pool.map(lambda index: limiters[index % 2].allow("203.0.113.24"), range(40)))
        assert sum(results) == 7
        assert [row["hits"] for row in rows(db) if row["key"] != CAPACITY_KEY] == [7]
        assert DatabaseRateLimiter(other, attempts=7, now=lambda: 1000).allow("203.0.113.24") is False
    finally:
        other.close()


def test_independent_processes_cannot_reset_or_duplicate_budget(database):
    db, url = database
    parameters = [(url, 5, 4)] * 4
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn")) as pool:
        accepted = list(pool.map(process_consume, parameters))
    assert sum(accepted) == 5
    restarted = Database(url)
    try:
        assert DatabaseRateLimiter(restarted, attempts=5, now=lambda: 1000).allow("203.0.113.24") is False
    finally:
        restarted.close()


def test_expiry_opens_a_new_shared_window_without_extending_denied_attempts(database):
    db, url = database
    now = [1000]
    first = DatabaseRateLimiter(db, attempts=2, window_seconds=60, now=lambda: now[0])
    assert first.allow("peer") is True
    assert first.allow("peer") is True
    now[0] = 1059
    assert first.allow("peer") is False
    bucket = next(row for row in rows(db) if row["key"] != CAPACITY_KEY)
    assert bucket["expires_at"] == 1060
    now[0] = 1060
    restarted = Database(url)
    try:
        second = DatabaseRateLimiter(restarted, attempts=2, window_seconds=60, now=lambda: now[0])
        assert second.allow("peer") is True
        assert first.allow("peer") is True
        assert second.allow("peer") is False
    finally:
        restarted.close()


def test_categories_and_configured_policies_have_independent_shared_budgets(database):
    db, _url = database
    limiter = DatabaseRateLimiter(db, attempts=2, limits={"recovery": (1, 3600)}, now=lambda: 1000)
    assert limiter.allow("peer", category="auth") is True
    assert limiter.allow("peer", category="auth") is True
    assert limiter.allow("peer", category="auth") is False
    assert limiter.allow("peer", category="recovery") is True
    assert limiter.allow("peer", category="recovery") is False
    assert limiter.policy("recovery") == (1, 3600)
    assert limiter.policy("oauth.token") == (30, 60)


def test_subjects_and_categories_are_not_persisted_in_plaintext(database):
    db, _url = database
    limiter = DatabaseRateLimiter(db, now=lambda: 1000)
    subject = "203.0.113.24/alice@example.com"
    category = "recovery:alice@example.com"
    assert limiter.allow(subject, category=category) is True
    stored = rows(db)
    dumped = json.dumps(stored)
    assert "203.0.113.24" not in dumped
    assert "alice@example.com" not in dumped
    assert "recovery:" not in dumped
    assert all(len(row["key"]) == 64 and set(row["key"]) <= set("0123456789abcdef") for row in stored)
    assert set(stored[0]) == {"key", "window_started", "hits", "expires_at"}


def test_capacity_is_shared_and_rotating_subjects_cannot_evict_live_counters(database):
    db, url = database
    first = DatabaseRateLimiter(db, attempts=2, capacity=3, now=lambda: 1000)
    assert first.allow("known") is True
    second_db = Database(url)
    try:
        second = DatabaseRateLimiter(second_db, attempts=2, capacity=3, now=lambda: 1000)
        accepted = [second.allow(f"rotating-{index}") for index in range(20)]
        assert sum(accepted) == 2
        assert first.allow("known") is True
        assert first.allow("known") is False
        assert len(rows(db)) == 4  # three subjects plus the global capacity lock
        assert any(row["key"] == first.bucket_key("known") and row["hits"] == 2 for row in rows(db))
    finally:
        second_db.close()


def test_cleanup_is_bounded_and_expiry_releases_capacity(database):
    db, _url = database
    with db.transaction(write=True) as connection:
        for index in range(5):
            connection.execute(rate_limit_buckets.insert().values(
                key=DatabaseRateLimiter.bucket_key(f"expired-{index}"), window_started=0, hits=1, expires_at=10,
            ))
    limiter = DatabaseRateLimiter(db, attempts=2, capacity=5, cleanup_batch_size=2, now=lambda: 1000)
    assert limiter.allow("new-peer") is True
    assert sum(row["expires_at"] == 10 for row in rows(db)) == 3
    assert limiter.allow("second-new-peer") is True
    assert sum(row["expires_at"] == 10 for row in rows(db)) == 1
    assert len(rows(db)) <= 6


def test_expired_bucket_outside_cleanup_batch_is_reset_safely(database):
    db, _url = database
    subjects = sorted(("expired-a", "expired-b", "expired-c"), key=DatabaseRateLimiter.bucket_key)
    with db.transaction(write=True) as connection:
        for subject in subjects:
            connection.execute(rate_limit_buckets.insert().values(
                key=DatabaseRateLimiter.bucket_key(subject), window_started=0, hits=99, expires_at=10,
            ))
    limiter = DatabaseRateLimiter(db, attempts=2, capacity=3, cleanup_batch_size=1, now=lambda: 1000)
    assert limiter.allow(subjects[-1]) is True
    current = next(row for row in rows(db) if row["key"] == limiter.bucket_key(subjects[-1]))
    assert current["hits"] == 1
    assert current["window_started"] == 1000
    assert current["expires_at"] == 1060


def test_database_time_is_authoritative_when_no_test_clock_is_injected(database, monkeypatch):
    db, _url = database
    monkeypatch.setattr(time, "time", lambda: 1)
    limiter = DatabaseRateLimiter(db, attempts=1)
    assert limiter.allow("peer") is True
    assert limiter.allow("peer") is False
    bucket = next(row for row in rows(db) if row["key"] != CAPACITY_KEY)
    assert bucket["window_started"] > 1_000_000


def test_database_failure_refuses_safely_without_local_fallback(database, monkeypatch, caplog):
    db, _url = database
    limiter = DatabaseRateLimiter(db)
    secret = "postgresql://someone:PRIVATE-PASSWORD@host/database"

    def fail_transaction(*_args, **_kwargs):
        from matters.cloud.db import CloudDatabaseError
        raise CloudDatabaseError(secret)

    monkeypatch.setattr(db, "transaction", fail_transaction)
    with pytest.raises(RateLimitUnavailable) as caught:
        limiter.allow("203.0.113.24/alice@example.com")
    assert caught.value.code == "unavailable"
    assert secret not in str(caught.value)
    assert "alice@example.com" not in str(caught.value)
    assert secret not in caplog.text


def test_untrusted_proxy_peer_cannot_split_budget_with_forwarded_headers(database):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    pytest.importorskip("argon2")
    pytest.importorskip("email_validator")
    from fastapi.testclient import TestClient
    from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware
    from matters.cloud.app import create_app

    _db, url = database
    app = create_app(url, secure_cookies=False)
    app.state.auth_limiter.attempts = 1
    wrapped = ProxyHeadersMiddleware(app, trusted_hosts=["127.0.0.1"])
    with TestClient(wrapped, base_url="http://localhost", client=("203.0.113.24", 4000)) as client:
        first = client.post("/api/auth/login", json={}, headers={"X-Forwarded-For": "198.51.100.1"})
        second = client.post("/api/auth/login", json={}, headers={"X-Forwarded-For": "198.51.100.2", "X-Real-IP": "198.51.100.3"})
        assert first.status_code == 422
        assert second.status_code == 429


def test_explicit_trusted_proxy_uses_nearest_untrusted_peer_not_spoofed_prefix(database):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    pytest.importorskip("argon2")
    pytest.importorskip("email_validator")
    from fastapi.testclient import TestClient
    from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware
    from matters.cloud.app import create_app

    _db, url = database
    app = create_app(url, secure_cookies=False)
    app.state.auth_limiter.attempts = 1
    wrapped = ProxyHeadersMiddleware(app, trusted_hosts=["127.0.0.1"])
    with TestClient(wrapped, base_url="http://localhost", client=("127.0.0.1", 4000)) as client:
        first = client.post("/api/auth/login", json={}, headers={"X-Forwarded-For": "198.51.100.1, 203.0.113.24"})
        spoofed_prefix = client.post("/api/auth/login", json={}, headers={"X-Forwarded-For": "198.51.100.2, 203.0.113.24"})
        different_peer = client.post("/api/auth/login", json={}, headers={"X-Forwarded-For": "198.51.100.3, 203.0.113.25"})
        assert first.status_code == 422
        assert spoofed_prefix.status_code == 429
        assert different_peer.status_code == 422


@pytest.mark.parametrize("subject,category", [("", "auth"), (None, "auth"), ("x" * 2049, "auth"), ("peer", ""), ("peer", "x" * 81), ("\ud800", "auth")])
def test_invalid_subjects_are_refused_without_storage(database, subject, category):
    db, _url = database
    assert DatabaseRateLimiter(db).allow(subject, category=category) is False
    assert rows(db) == []


@pytest.mark.parametrize("options", [
    {"attempts": 0}, {"attempts": True}, {"window_seconds": 0},
    {"capacity": 0}, {"cleanup_batch_size": 0}, {"cleanup_batch_size": 501},
    {"limits": {"auth": (0, 60)}}, {"limits": {"": (1, 60)}},
])
def test_invalid_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        DatabaseRateLimiter(None, **options)
