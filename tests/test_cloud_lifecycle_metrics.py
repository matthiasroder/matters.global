"""Operator-only, bounded mail health must never disclose recovery secrets."""

import io
import json
import logging
import threading
import time

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("argon2")
pytest.importorskip("email_validator")

from matters.cloud.accounts import CloudAccounts
from matters.cloud.db import Database
from matters.cloud.lifecycle import AccountLifecycle
from matters.cloud.mail_delivery import CaptureMailDelivery, DELIVERY_OUTCOMES, JSONMailEventSink, MAX_DELIVERY_COUNT


EMAIL = "sensitive-recipient@example.com"
PASSWORD = "Private temporary test password!"
ORIGIN = "https://private-origin.example"


@pytest.fixture
def account(tmp_path):
    db = Database("sqlite:///" + str(tmp_path / "metrics.db")).initialize()
    accounts = CloudAccounts(db)
    user = accounts.register(EMAIL, PASSWORD)
    yield db, accounts, user
    db.close()


def safe_events(events):
    assert events
    for event in events:
        assert set(event) == {"event", "outcome", "count"}
        assert event["event"] == "matters_mail_delivery"
        assert event["outcome"] in DELIVERY_OUTCOMES
        assert type(event["count"]) is int and 0 <= event["count"] <= MAX_DELIVERY_COUNT


def test_success_skip_and_fault_counters_have_no_credentials(account, capsys, caplog):
    db, accounts, user = account
    events, capture = [], CaptureMailDelivery()
    lifecycle = AccountLifecycle(db, accounts, public_origin=ORIGIN, delivery=capture, event_sink=events.append)
    try:
        expected = lifecycle.request_password_reset(EMAIL)
        assert lifecycle.request_password_reset("unknown@example.com") == expected
        assert lifecycle.request_password_reset("invalid") == expected
        assert lifecycle.drain_delivery()
        snapshot = lifecycle.delivery_metrics()
        assert snapshot["requested"] == snapshot["queued"] == 3
        assert snapshot["sent"] == 1 and snapshot["skipped"] == 2 and snapshot["failed"] == 0
        assert snapshot["queue_depth"] == 0 and snapshot["queue_capacity"] == 128
        assert "sent" not in lifecycle.capabilities() and "failed" not in lifecycle.status(user["id"])
        safe_events(events)
        serialized = json.dumps(events) + json.dumps(snapshot)
        for secret in (EMAIL, PASSWORD, user["id"], ORIGIN, capture.messages[0].body, capture.messages[0].subject):
            assert secret not in serialized
    finally:
        lifecycle.close()
    class FailedDelivery:
        configured = True
        def send(self, recipient, subject, body):
            raise RuntimeError(recipient + PASSWORD + subject + body)
    failed_events = []
    failed = AccountLifecycle(db, accounts, public_origin=ORIGIN, delivery=FailedDelivery(), event_sink=failed_events.append)
    try:
        assert failed.request_password_reset(EMAIL) == expected
        assert failed.drain_delivery()
        assert failed.delivery_metrics()["failed"] == 1
        assert any(event["outcome"] == "failed" for event in failed_events)
        safe_events(failed_events)
        assert EMAIL not in json.dumps(failed_events) and PASSWORD not in json.dumps(failed_events)
        assert not capsys.readouterr().err and not caplog.records
    finally:
        failed.close()


def test_queue_saturation_and_shutdown_are_bounded_and_visible(account):
    db, accounts, _ = account
    entered, release = threading.Event(), threading.Event()
    class BlockingDelivery:
        configured = True
        def send(self, recipient, subject, body):
            entered.set()
            assert release.wait(timeout=5)
    events = []
    lifecycle = AccountLifecycle(db, accounts, public_origin=ORIGIN, delivery=BlockingDelivery(), event_sink=events.append)
    try:
        expected = lifecycle.request_password_reset(EMAIL)
        assert entered.wait(timeout=3)
        for _ in range(129):
            assert lifecycle.request_password_reset(EMAIL) == expected
        snapshot = lifecycle.delivery_metrics()
        assert snapshot["requested"] == 130 and snapshot["queued"] == 129
        assert snapshot["dropped"] == 1 and snapshot["queue_depth"] == snapshot["queue_capacity"] == 128
        lifecycle.close(timeout=0)
        assert lifecycle.request_password_reset(EMAIL) == expected
        release.set()
        assert lifecycle.drain_delivery()
        lifecycle.close()
        snapshot = lifecycle.delivery_metrics()
        assert snapshot["closed"] == 1 and snapshot["dropped"] == 129
        assert snapshot["sent"] == 1 and snapshot["skipped"] == 1 and snapshot["queue_depth"] == 0
        safe_events(events)
        assert EMAIL not in json.dumps(events) and ORIGIN not in json.dumps(events)
    finally:
        release.set()
        lifecycle.close()


def test_observer_exception_cannot_break_requests_or_worker(account, capsys, caplog):
    db, accounts, _ = account
    capture = CaptureMailDelivery()
    def observer(event):
        event["recipient"] = EMAIL
        raise RuntimeError(EMAIL + PASSWORD + ORIGIN)
    lifecycle = AccountLifecycle(db, accounts, public_origin=ORIGIN, delivery=capture, event_sink=observer)
    try:
        assert lifecycle.request_password_reset(EMAIL)["message"]
        assert lifecycle.drain_delivery()
        snapshot = lifecycle.delivery_metrics()
        assert snapshot["requested"] == snapshot["queued"] == snapshot["sent"] == 1
        assert snapshot["event_sink_failed"] == 3
        assert len(capture.messages) == 1
        assert not capsys.readouterr().err and not caplog.records
    finally:
        lifecycle.close()


def test_disabled_default_is_quiet_and_counters_restart_per_worker(account, capsys, caplog):
    db, accounts, _ = account
    lifecycle = AccountLifecycle(db, accounts)
    lifecycle.request_password_reset(EMAIL)
    assert lifecycle.delivery_metrics()["requested"] == lifecycle.delivery_metrics()["skipped"] == 1
    lifecycle.close()
    replacement = AccountLifecycle(db, accounts)
    assert all(value == 0 for key, value in replacement.delivery_metrics().items() if key not in {"queue_capacity"})
    replacement.close()
    assert not capsys.readouterr().err and not caplog.records


def test_counters_saturate_and_snapshots_are_detached(account):
    db, accounts, _ = account
    lifecycle = AccountLifecycle(db, accounts)
    with lifecycle._metrics_lock:
        lifecycle._metrics["requested"] = MAX_DELIVERY_COUNT
    lifecycle.request_password_reset(EMAIL)
    assert lifecycle.delivery_metrics()["requested"] == MAX_DELIVERY_COUNT
    snapshot = lifecycle.delivery_metrics()
    snapshot["requested"] = -1
    assert lifecycle.delivery_metrics()["requested"] == MAX_DELIVERY_COUNT
    assert set(lifecycle.delivery_metrics()) == set(DELIVERY_OUTCOMES) | {"event_sink_failed", "queue_depth", "queue_capacity"}
    lifecycle.close()


def test_json_sink_whitelists_values_and_does_not_touch_global_logging():
    root = logging.getLogger()
    third_party = logging.getLogger("sqlalchemy.engine")
    before = (root.level, tuple(root.handlers), third_party.level, tuple(third_party.handlers))
    output = io.StringIO()
    sink = JSONMailEventSink(output)
    sink({"event": PASSWORD, "outcome": "failed", "count": 1, "email": EMAIL, "body": PASSWORD, "url": ORIGIN, "exception": PASSWORD})
    assert json.loads(output.getvalue()) == {"event": "matters_mail_delivery", "outcome": "failed", "count": 1}
    with pytest.raises(ValueError, match="Invalid mail delivery event"):
        sink({"outcome": PASSWORD, "count": 1})
    assert PASSWORD not in output.getvalue() and EMAIL not in output.getvalue() and ORIGIN not in output.getvalue()
    assert (root.level, tuple(root.handlers), third_party.level, tuple(third_party.handlers)) == before
