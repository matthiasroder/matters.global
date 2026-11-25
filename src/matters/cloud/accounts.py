"""Cloud accounts and revocable sessions, independent of graph semantics.

Only password hashes and session-token digests are persisted. A successful
login returns the raw session token once to the HTTP adapter, which can put
it in an HttpOnly cookie; public account records never contain credentials.
"""

import hashlib
import hmac
import re
import secrets
import threading
import time
import uuid
from contextlib import contextmanager
from functools import lru_cache

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from email_validator import EmailNotValidError, validate_email
from sqlalchemy import delete, insert, select, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from .db import CloudDatabaseError, sessions, users


MIN_PASSWORD_LENGTH = 12
MAX_PASSWORD_LENGTH = 1024
DEFAULT_SESSION_TTL_SECONDS = 86400
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_-]{43}")
_LOGIN_FAILURE = "Email or password is incorrect."
_SESSION_FAILURE = "A valid session is required."
_UNAVAILABLE = "The account service is unavailable."
PASSWORD_WORK_SLOTS = 2
_PASSWORD_WORK = threading.BoundedSemaphore(PASSWORD_WORK_SLOTS)


class AccountError(ValueError):
    """A safe account error for the caller, without backend exception text."""

    def __init__(self, message, code="invalid"):
        super().__init__(message)
        self.message = message
        self.code = code


@contextmanager
def _password_work():
    """Bound Argon work across all account services in this process.

    Refuse excess work instead of holding an unbounded queue of credential
    requests. This is a memory/work bound, independent of account policy and
    the shared database request limits.
    """
    if not _PASSWORD_WORK.acquire(blocking=False):
        raise AccountError("The password service is busy. Try again shortly.", "unavailable")
    try:
        yield
    finally:
        _PASSWORD_WORK.release()


class _BoundedPasswordHasher:
    def __init__(self):
        self._hasher = PasswordHasher()

    def hash(self, *args, **kwargs):
        with _password_work():
            return self._hasher.hash(*args, **kwargs)

    def verify(self, *args, **kwargs):
        with _password_work():
            return self._hasher.verify(*args, **kwargs)

    def check_needs_rehash(self, *args, **kwargs):
        return self._hasher.check_needs_rehash(*args, **kwargs)


@lru_cache(maxsize=1)
def _dummy_password_hash():
    # Unknown-account logins still perform a real password verification.
    # This hash is not a credential for an account and is never persisted.
    return _BoundedPasswordHasher().hash(secrets.token_urlsafe(32))


def _normalized_email(value):
    if not isinstance(value, str):
        raise AccountError("Enter a valid email address.")
    candidate = value.strip()
    if not candidate or len(candidate) > 320:
        raise AccountError("Enter a valid email address.")
    try:
        normalized = validate_email(candidate, check_deliverability=False).normalized.casefold()
    except EmailNotValidError:
        raise AccountError("Enter a valid email address.") from None
    if len(normalized) > 320:
        raise AccountError("Enter a valid email address.")
    return normalized


def _validated_password(value):
    if not isinstance(value, str) or not MIN_PASSWORD_LENGTH <= len(value) <= MAX_PASSWORD_LENGTH:
        raise AccountError(
            f"Password must contain between {MIN_PASSWORD_LENGTH} and "
            f"{MAX_PASSWORD_LENGTH} characters."
        )
    return value


def _token_digest(token):
    if not isinstance(token, str) or not _TOKEN_PATTERN.fullmatch(token):
        raise AccountError(_SESSION_FAILURE, "unauthorized")
    return hashlib.sha256(token.encode("ascii")).hexdigest()


def _public_user(record):
    return {
        "id": record["id"],
        "email": record["email"],
        "created_at": int(record["created_at"]),
    }


class CloudAccounts:
    """Manage account identity and opaque database-backed login sessions.

    The caller initializes and owns the database lifecycle. Registration does
    not create graphs or grant graph membership. Session expiry is measured in
    epoch seconds; an injected clock lets callers verify expiry without sleeps.
    """

    def __init__(self, db, *, session_ttl_seconds=DEFAULT_SESSION_TTL_SECONDS, now=None):
        if type(session_ttl_seconds) is not int or session_ttl_seconds <= 0:
            raise AccountError("Session lifetime must be a positive integer.")
        if now is not None and not callable(now):
            raise AccountError("The account clock must be callable.")
        self.db = db
        self.session_ttl_seconds = session_ttl_seconds
        self._now = now if now is not None else time.time
        self._hasher = _BoundedPasswordHasher()
        self._dummy_hash = _dummy_password_hash()

    def register(self, email, password):
        """Create one account and return only its public identity fields."""
        normalized = _normalized_email(email)
        password = _validated_password(password)
        # Password hashing does not hold a database write lock.
        password_hash = self._hasher.hash(password)
        user = {
            "id": str(uuid.uuid4()),
            "email": normalized,
            "password_hash": password_hash,
            "created_at": int(self._now()),
        }
        try:
            with self.db.transaction(write=True) as connection:
                connection.execute(insert(users).values(**user))
        except IntegrityError:
            raise AccountError("An account with these details already exists.", "conflict") from None
        except (CloudDatabaseError, SQLAlchemyError):
            raise AccountError(_UNAVAILABLE, "unavailable") from None
        return _public_user(user)

    def login(self, email, password):
        """Verify a credential and return a freshly committed session token."""
        try:
            normalized = _normalized_email(email)
            password = _validated_password(password)
        except AccountError:
            raise AccountError(_LOGIN_FAILURE, "unauthorized") from None
        try:
            with self.db.transaction() as connection:
                observed = connection.execute(
                    select(users).where(users.c.email == normalized)
                ).mappings().one_or_none()
        except (CloudDatabaseError, SQLAlchemyError):
            raise AccountError(_UNAVAILABLE, "unavailable") from None

        encoded = observed["password_hash"] if observed is not None else self._dummy_hash
        try:
            verified = self._hasher.verify(encoded, password)
        except (VerificationError, InvalidHashError, TypeError):
            verified = False
        if not verified or observed is None:
            raise AccountError(_LOGIN_FAILURE, "unauthorized")

        replacement_hash = (
            self._hasher.hash(password) if self._hasher.check_needs_rehash(encoded) else None
        )
        token = secrets.token_urlsafe(32)
        now = int(self._now())
        expires_at = now + self.session_ttl_seconds
        try:
            with self.db.transaction(write=True) as connection:
                current = connection.execute(
                    select(users).where(users.c.id == observed["id"]).with_for_update()
                ).mappings().one_or_none()
                # Do not issue a session after credentials were changed or
                # removed while the comparatively expensive hash was checked.
                if current is None or not hmac.compare_digest(current["password_hash"], encoded):
                    raise AccountError(_LOGIN_FAILURE, "unauthorized")
                if replacement_hash is not None:
                    connection.execute(
                        update(users).where(users.c.id == current["id"]).values(
                            password_hash=replacement_hash
                        )
                    )
                connection.execute(delete(sessions).where(
                    sessions.c.user_id == current["id"], sessions.c.expires_at <= now
                ))
                connection.execute(insert(sessions).values(
                    token_hash=_token_digest(token), user_id=current["id"],
                    created_at=now, expires_at=expires_at,
                ))
                public_user = _public_user(current)
        except (CloudDatabaseError, SQLAlchemyError):
            raise AccountError(_UNAVAILABLE, "unavailable") from None
        return {"token": token, "user": public_user, "expires_at": expires_at}

    def authenticate(self, token):
        """Resolve one unexpired session to its current account identity."""
        digest = _token_digest(token)
        now = int(self._now())
        try:
            with self.db.transaction() as connection:
                current = connection.execute(
                    select(users.c.id, users.c.email, users.c.created_at)
                    .join(sessions, sessions.c.user_id == users.c.id)
                    .where(sessions.c.token_hash == digest, sessions.c.expires_at > now)
                ).mappings().one_or_none()
        except (CloudDatabaseError, SQLAlchemyError):
            raise AccountError(_UNAVAILABLE, "unavailable") from None
        if current is None:
            raise AccountError(_SESSION_FAILURE, "unauthorized")
        return _public_user(current)

    def logout(self, token):
        """Revoke a session; repeated or malformed logout requests are harmless."""
        try:
            digest = _token_digest(token)
        except AccountError:
            return
        try:
            with self.db.transaction(write=True) as connection:
                connection.execute(delete(sessions).where(sessions.c.token_hash == digest))
        except (CloudDatabaseError, SQLAlchemyError):
            raise AccountError(_UNAVAILABLE, "unavailable") from None
