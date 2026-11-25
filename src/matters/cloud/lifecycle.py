"""Verification and credential recovery without graph or billing policy."""

import hashlib
import hmac
import queue
import secrets
import threading
import time
from contextlib import contextmanager

from argon2.exceptions import InvalidHashError, VerificationError
from sqlalchemy import and_, delete, or_, select
from sqlalchemy.exc import SQLAlchemyError

from .accounts import AccountError, _normalized_email, _token_digest, _validated_password
from .db import CloudDatabaseError, account_actions, account_security, oauth_records, sessions, users
from .mail_delivery import DELIVERY_OUTCOMES, MAX_DELIVERY_COUNT, DisabledMailDelivery
from .oauth import _url


RESET_TTL = 1800
VERIFY_TTL = 86400
RESET_RESPONSE = "If recovery is available for this address, instructions may arrive shortly. Check your inbox or contact the installation operator."
VERIFY_RESPONSE = "If verification is available, instructions may arrive shortly. Check your inbox or contact the installation operator."


class AccountLifecycle:
    """Persist action digests; deliver only through a transient bounded queue.

    Public reset work, including account lookup and SMTP, happens after the
    uniform response. Pending delivery is not durable: an interrupted process
    can lose mail jobs, and the person can request a new link. Neither durable
    action records nor a log/outbox contains a plaintext action credential.
    """

    def __init__(self, db, accounts, *, public_origin=None, delivery=None, allow_loopback_http=False, now=None, event_sink=None):
        self.db, self.accounts = db, accounts
        self.delivery = delivery if delivery is not None else DisabledMailDelivery()
        self.public_origin = _url(public_origin, loopback_http=allow_loopback_http, origin=True) if public_origin else None
        if self.delivery.configured and self.public_origin is None:
            raise ValueError("Email delivery requires an explicit trusted public origin.")
        self._now = now or time.time
        self._jobs = queue.Queue(maxsize=128)
        self._thread = None
        self._thread_lock = threading.Lock()
        self._stop = threading.Event()
        if event_sink is not None and not callable(event_sink):
            raise ValueError("event_sink must be callable")
        self._event_sink = event_sink
        self._metrics_lock = threading.Lock()
        self._metrics = {name: 0 for name in (*DELIVERY_OUTCOMES, "event_sink_failed")}

    def _event(self, outcome):
        with self._metrics_lock:
            self._metrics[outcome] = min(MAX_DELIVERY_COUNT, self._metrics[outcome] + 1)
            count = self._metrics[outcome]
        if self._event_sink is not None:
            try:
                self._event_sink({"event": "matters_mail_delivery", "outcome": outcome, "count": count})
            except Exception:
                # Sink failures cannot change a recovery response or disclose
                # the exception, which might itself contain a secret.
                with self._metrics_lock:
                    self._metrics["event_sink_failed"] = min(MAX_DELIVERY_COUNT, self._metrics["event_sink_failed"] + 1)

    def delivery_metrics(self):
        """Private aggregate health, reset for each worker/service instance.

        Operators collect this snapshot or the safe event sink out of band;
        neither belongs in public capabilities or per-account responses.
        """
        with self._metrics_lock:
            snapshot = dict(self._metrics)
        return {**snapshot, "queue_depth": self._jobs.qsize(), "queue_capacity": self._jobs.maxsize}

    @contextmanager
    def _transaction(self, *, write=False):
        try:
            with self.db.transaction(write=write) as connection:
                yield connection
        except (CloudDatabaseError, SQLAlchemyError):
            raise AccountError("The account service is unavailable.", "unavailable") from None

    def capabilities(self):
        enabled = bool(self.delivery.configured and self.public_origin)
        return {"email_delivery": enabled, "email_verification": enabled, "password_reset": enabled, "password_change": True}

    def status(self, actor_id):
        with self._transaction() as connection:
            if connection.execute(select(users.c.id).where(users.c.id == actor_id)).first() is None:
                raise AccountError("Sign in to continue.", "unauthorized")
            verified = connection.execute(select(account_security.c.email_verified_at).where(account_security.c.user_id == actor_id)).scalar_one_or_none()
        return {**self.capabilities(), "email_verified": verified is not None, "email_verified_at": verified}

    def _enqueue(self, kind, value):
        self._event("requested")
        if not self.capabilities()["email_delivery"]:
            self._event("skipped")
            return
        outcome = "skipped"
        with self._thread_lock:
            if not self._stop.is_set():
                if self._thread is None:
                    self._thread = threading.Thread(target=self._worker, name="matters-mail-delivery", daemon=True)
                    self._thread.start()
                try:
                    self._jobs.put_nowait((kind, value))
                    outcome = "queued"
                except queue.Full:
                    # No account/provider detail changes the public response.
                    outcome = "dropped"
        self._event(outcome)

    def request_password_reset(self, email):
        self._enqueue("reset_password", email)
        return {"message": RESET_RESPONSE}

    def request_verification(self, actor_id):
        self._enqueue("verify_email", actor_id)
        return {"message": VERIFY_RESPONSE}

    def _worker(self):
        while not self._stop.is_set() or self._jobs.unfinished_tasks:
            try:
                job = self._jobs.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if not self._stop.is_set():
                    self._event(self._deliver(*job))
                else:
                    self._event("dropped")
            except Exception:
                self._event("failed")
            finally:
                self._jobs.task_done()

    def _deliver(self, kind, value):
        if kind == "reset_password":
            try:
                value = _normalized_email(value)
            except AccountError:
                return "skipped"
        token = secrets.token_urlsafe(32)
        identity = _token_digest(token)
        with self._transaction(write=True) as connection:
            predicate = users.c.email == value if kind == "reset_password" else users.c.id == value
            user = connection.execute(select(users).where(predicate).with_for_update()).mappings().one_or_none()
            if user is None:
                return "skipped"
            if kind == "verify_email" and connection.execute(select(account_security.c.email_verified_at).where(account_security.c.user_id == user["id"])).scalar_one_or_none() is not None:
                return "skipped"
            now = int(self._now())
            connection.execute(account_actions.insert().values(
                id=identity, user_id=user["id"], kind=kind,
                data={"email": user["email"], "credential_version": hashlib.sha256(user["password_hash"].encode()).hexdigest()},
                created_at=now, expires_at=now + (RESET_TTL if kind == "reset_password" else VERIFY_TTL), used_at=None,
            ))
            email = user["email"]
        route = "reset" if kind == "reset_password" else "verify"
        link = self.public_origin + "/account/" + route + "#token=" + token
        subject = "Reset your Matters password" if kind == "reset_password" else "Verify your Matters email"
        body = ("A Matters account request was made for this address.\n\n" + link + "\n\n"
                "Open the link and explicitly confirm the request. If you did not request this, ignore this message.\n")
        try:
            self.delivery.send(email, subject, body)
            return "sent"
        except Exception:
            with self._transaction(write=True) as connection:
                connection.execute(account_actions.update().where(account_actions.c.id == identity, account_actions.c.used_at.is_(None)).values(used_at=int(self._now())))
            return "failed"

    def drain_delivery(self, timeout=5):
        deadline = time.monotonic() + timeout
        while self._jobs.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.01)
        return self._jobs.unfinished_tasks == 0

    def close(self, timeout=15):
        with self._thread_lock:
            notify = not self._stop.is_set()
            self._stop.set()
        if notify:
            self._event("closed")
        close_delivery = getattr(self.delivery, "close", None)
        if close_delivery is not None:
            close_delivery()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _observe_action(self, token, kind):
        try:
            identity = _token_digest(token)
        except AccountError:
            raise AccountError("This link is invalid or expired.") from None
        with self._transaction() as connection:
            row = connection.execute(select(account_actions).where(account_actions.c.id == identity, account_actions.c.kind == kind)).mappings().one_or_none()
            if row is None or row["used_at"] is not None or row["expires_at"] <= int(self._now()):
                raise AccountError("This link is invalid or expired.")
            return identity, row["user_id"]

    def _locked_action(self, connection, identity, actor_id, kind):
        # Every credential writer and login/consent issuance locks user first.
        user = connection.execute(select(users).where(users.c.id == actor_id).with_for_update()).mappings().one_or_none()
        action = connection.execute(select(account_actions).where(account_actions.c.id == identity, account_actions.c.kind == kind).with_for_update()).mappings().one_or_none()
        if user is None or action is None or action["user_id"] != actor_id or action["used_at"] is not None or action["expires_at"] <= int(self._now()) or action["data"].get("email") != user["email"] or not hmac.compare_digest(action["data"].get("credential_version", ""), hashlib.sha256(user["password_hash"].encode()).hexdigest()):
            raise AccountError("This link is invalid or expired.")
        return user

    def verify_email(self, token):
        identity, actor_id = self._observe_action(token, "verify_email")
        with self._transaction(write=True) as connection:
            self._locked_action(connection, identity, actor_id, "verify_email")
            now = int(self._now())
            exists = connection.execute(select(account_security.c.user_id).where(account_security.c.user_id == actor_id)).first()
            if exists:
                connection.execute(account_security.update().where(account_security.c.user_id == actor_id).values(email_verified_at=now))
            else:
                connection.execute(account_security.insert().values(user_id=actor_id, email_verified_at=now))
            connection.execute(account_actions.update().where(account_actions.c.user_id == actor_id, account_actions.c.kind == "verify_email", account_actions.c.used_at.is_(None)).values(used_at=now))
        return {"email_verified": True}

    @staticmethod
    def _invalidate(connection, actor_id, now):
        connection.execute(delete(sessions).where(sessions.c.user_id == actor_id))
        connection.execute(delete(oauth_records).where(or_(
            oauth_records.c.user_id == actor_id,
            and_(oauth_records.c.kind == "request", oauth_records.c.data["actor_id"].as_string() == actor_id),
        )))
        connection.execute(account_actions.update().where(account_actions.c.user_id == actor_id, account_actions.c.used_at.is_(None)).values(used_at=now))

    def reset_password(self, token, new_password):
        identity, actor_id = self._observe_action(token, "reset_password")
        encoded = self.accounts._hasher.hash(_validated_password(new_password))
        with self._transaction(write=True) as connection:
            self._locked_action(connection, identity, actor_id, "reset_password")
            connection.execute(users.update().where(users.c.id == actor_id).values(password_hash=encoded))
            self._invalidate(connection, actor_id, int(self._now()))
        return {"password_changed": True, "sign_in_required": True}

    def change_password(self, actor_id, session_token, current_password, new_password):
        digest = _token_digest(session_token)
        current_password = _validated_password(current_password)
        new_password = _validated_password(new_password)
        with self._transaction() as connection:
            observed = connection.execute(select(users).where(users.c.id == actor_id)).mappings().one_or_none()
        encoded = observed["password_hash"] if observed else self.accounts._dummy_hash
        try:
            verified = self.accounts._hasher.verify(encoded, current_password)
        except (VerificationError, InvalidHashError, TypeError):
            verified = False
        if not verified or observed is None:
            raise AccountError("The current password is incorrect.")
        replacement = self.accounts._hasher.hash(new_password)
        with self._transaction(write=True) as connection:
            current = connection.execute(select(users).where(users.c.id == actor_id).with_for_update()).mappings().one_or_none()
            session = connection.execute(select(sessions.c.user_id).where(sessions.c.token_hash == digest, sessions.c.user_id == actor_id, sessions.c.expires_at > int(self._now()))).first()
            if current is None or session is None or not hmac.compare_digest(current["password_hash"], encoded):
                raise AccountError("Sign in again before changing your password.", "unauthorized")
            connection.execute(users.update().where(users.c.id == actor_id).values(password_hash=replacement))
            self._invalidate(connection, actor_id, int(self._now()))
        return {"password_changed": True, "sign_in_required": True}
