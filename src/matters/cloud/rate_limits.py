"""Shared, atomic security throttles backed by the installation database.

Only a SHA-256 digest of the category and subject is persisted. Callers must
normalize subjects (for example an email address) before using them and must
derive network peers from the trusted request boundary, not forwarded headers.
These counters are abuse controls, independent of subscription allowances.
"""

import hashlib
import json
from collections.abc import Mapping

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import SQLAlchemyError

from .db import CloudDatabaseError, rate_limit_buckets


CAPACITY_KEY = hashlib.sha256(b"matters/security-rate-limits/v1/capacity").hexdigest()
MAX_TIMESTAMP = (1 << 63) - 1
DEFAULT_LIMITS = {
    "oauth.register": (30, 60),
    "oauth.authorize": (30, 60),
    "oauth.consent": (60, 60),
    "oauth.token": (30, 60),
    "oauth.revoke": (30, 60),
    "auth-recovery-peer": (10, 60),
    "auth-recovery-email": (5, 3600),
    "auth-token-peer": (30, 60),
    "auth-token-value": (10, 60),
    "auth-verification-peer": (10, 60),
    "auth-verification-actor": (5, 3600),
}


class RateLimitUnavailable(RuntimeError):
    """Fail closed without exposing database details or the subject."""

    code = "unavailable"
    message = "The security limit service is temporarily unavailable."

    def __init__(self):
        super().__init__(self.message)


def _policy(value):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("Rate-limit policies require an attempts/window pair.")
    attempts, window = value
    if type(attempts) is not int or not 1 <= attempts <= 1_000_000:
        raise ValueError("Rate-limit attempts must be a positive bounded integer.")
    if type(window) is not int or not 1 <= window <= 86400:
        raise ValueError("Rate-limit windows must be between one second and one day.")
    return attempts, window


class DatabaseRateLimiter:
    """Share one finite window across workers and application restarts.

    PostgreSQL serializes the short admission transaction using a single hashed
    sentinel row. SQLite's BEGIN IMMEDIATE provides the same exclusion across
    connections/processes. This deliberately favors a small, auditable critical
    section over per-process caches. The table holds at most ``capacity`` subject
    buckets plus one sentinel; active buckets are never evicted to admit a new
    subject, which would let rotating identities reset an existing quota.

    ``now`` exists for deterministic verification. In normal use, database time
    is authoritative, so application-worker clock differences cannot reset a
    shared window. All workers must use the same configured security policies.
    """

    def __init__(
        self, db, *, attempts=20, window_seconds=60, capacity=10000,
        cleanup_batch_size=128, limits=None, now=None,
    ):
        _policy((attempts, window_seconds))
        if type(capacity) is not int or not 1 <= capacity <= 1_000_000:
            raise ValueError("Rate-limit capacity must be a positive bounded integer.")
        if type(cleanup_batch_size) is not int or not 1 <= cleanup_batch_size <= 500:
            raise ValueError("Rate-limit cleanup batches must contain one to 500 rows.")
        if now is not None and not callable(now):
            raise ValueError("Rate-limit clock must be callable.")
        if limits is not None and not isinstance(limits, Mapping):
            raise ValueError("Rate-limit policies must be a category mapping.")
        self.db = db
        self.attempts = attempts
        self.window_seconds = window_seconds
        self.capacity = capacity
        self.cleanup_batch_size = cleanup_batch_size
        self.limits = dict(DEFAULT_LIMITS)
        for category, value in (limits or {}).items():
            if not self._valid_category(category):
                raise ValueError("Rate-limit categories must be nonempty bounded strings.")
            self.limits[category] = _policy(value)
        self._now = now

    @staticmethod
    def _valid_category(category):
        return isinstance(category, str) and bool(category.strip()) and len(category) <= 80

    def policy(self, category="auth"):
        """Return the configured security policy, useful for retry headers."""

        return _policy(self.limits.get(category, (self.attempts, self.window_seconds)))

    @staticmethod
    def bucket_key(subject, category="auth"):
        value = json.dumps(
            ["matters/security-rate-limits/v1/bucket", category, subject],
            ensure_ascii=False, separators=(",", ":"),
        )
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _clock(self, connection):
        if self._now is not None:
            value = self._now()
        elif connection.dialect.name == "postgresql":
            value = connection.execute(select(func.extract("epoch", func.clock_timestamp()))).scalar_one()
        else:
            value = connection.execute(select(func.strftime("%s", "now"))).scalar_one()
        if type(value) is bool:
            raise ValueError("Invalid security clock.")
        now = int(value)
        if not 0 <= now <= MAX_TIMESTAMP - 86400:
            raise ValueError("Invalid security clock.")
        return now

    def allow(self, subject, *, category="auth"):
        """Consume one attempt, or refuse without evicting a live bucket.

        This is synchronous database work; async HTTP boundaries should offload
        it to their thread pool. A database failure raises a safe unavailable
        error instead of returning success or falling back to an in-memory
        counter. Invalid subjects refuse without storing any record.
        """

        if not isinstance(subject, str) or not subject or len(subject) > 2048 or not self._valid_category(category):
            return False
        try:
            key = self.bucket_key(subject, category)
        except (UnicodeError, ValueError, TypeError):
            return False
        try:
            attempts, window = self.policy(category)
            with self.db.transaction(write=True) as connection:
                insert = postgres_insert if connection.dialect.name == "postgresql" else sqlite_insert
                connection.execute(insert(rate_limit_buckets).values(
                    key=CAPACITY_KEY, window_started=0, hits=0, expires_at=MAX_TIMESTAMP,
                ).on_conflict_do_nothing(index_elements=[rate_limit_buckets.c.key]))
                # The sentinel is a durable lock, not a process-local cache or
                # counter. COUNT below observes the actual bounded table.
                connection.execute(select(rate_limit_buckets.c.key).where(
                    rate_limit_buckets.c.key == CAPACITY_KEY
                ).with_for_update()).scalar_one()
                now = self._clock(connection)
                expired = list(connection.execute(select(rate_limit_buckets.c.key).where(
                    rate_limit_buckets.c.key != CAPACITY_KEY,
                    rate_limit_buckets.c.expires_at <= now,
                ).order_by(rate_limit_buckets.c.expires_at, rate_limit_buckets.c.key).limit(self.cleanup_batch_size)).scalars())
                if expired:
                    connection.execute(delete(rate_limit_buckets).where(rate_limit_buckets.c.key.in_(expired)))
                bucket = connection.execute(select(rate_limit_buckets).where(
                    rate_limit_buckets.c.key == key
                ).with_for_update()).mappings().one_or_none()
                if bucket is None:
                    count = connection.execute(select(func.count()).select_from(rate_limit_buckets).where(
                        rate_limit_buckets.c.key != CAPACITY_KEY
                    )).scalar_one()
                    if count >= self.capacity:
                        return False
                    connection.execute(rate_limit_buckets.insert().values(
                        key=key, window_started=now, hits=1, expires_at=now + window,
                    ))
                    return True
                if bucket["expires_at"] <= now:
                    connection.execute(rate_limit_buckets.update().where(rate_limit_buckets.c.key == key).values(
                        window_started=now, hits=1, expires_at=now + window,
                    ))
                    return True
                if bucket["hits"] >= attempts:
                    return False
                connection.execute(rate_limit_buckets.update().where(rate_limit_buckets.c.key == key).values(
                    hits=bucket["hits"] + 1,
                ))
                return True
        except (CloudDatabaseError, SQLAlchemyError, ValueError, TypeError, OverflowError):
            raise RateLimitUnavailable() from None
