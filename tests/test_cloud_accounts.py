"""Account and session behavior against actual persistent SQL storage."""

import hashlib
import os
import secrets
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("argon2")
pytest.importorskip("email_validator")

from argon2 import PasswordHasher
from sqlalchemy import func, select, update

from matters.cloud.accounts import AccountError, CloudAccounts
from matters.cloud.db import CloudDatabaseError, Database, sessions, users


PASSWORD = "correct horse battery staple"


@pytest.fixture
def db(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'accounts.db'}")
    database.initialize()
    yield database
    database.close()


@pytest.fixture
def clock():
    return [1_800_000_000]


@pytest.fixture
def accounts(db, clock):
    return CloudAccounts(db, session_ttl_seconds=60, now=lambda: clock[0])


def register(accounts, email="alice@example.com"):
    return accounts.register(email, PASSWORD)


def test_register_normalizes_email_and_persists_only_argon2_hash(accounts, db, clock):
    public = accounts.register("  Alice@EXAMPLE.COM  ", PASSWORD)
    assert set(public) == {"id", "email", "created_at"}
    assert str(uuid.UUID(public["id"])) == public["id"]
    assert public["email"] == "alice@example.com"
    assert public["created_at"] == clock[0]
    with db.transaction() as connection:
        stored = connection.execute(select(users)).mappings().one()
    assert stored["password_hash"].startswith("$argon2id$")
    assert PASSWORD not in stored.values()
    assert PasswordHasher().verify(stored["password_hash"], PASSWORD)
    assert "password_hash" not in public


def test_login_returns_independent_sessions_with_hashed_tokens(accounts, db, clock):
    public = register(accounts)
    first = accounts.login("ALICE@example.COM", PASSWORD)
    second = accounts.login("alice@example.com", PASSWORD)
    assert first["user"] == second["user"] == public
    assert len(first["token"]) == 43
    assert first["token"] != second["token"]
    assert first["expires_at"] == clock[0] + 60
    assert accounts.authenticate(first["token"]) == public
    assert accounts.authenticate(second["token"]) == public
    with db.transaction() as connection:
        stored = connection.execute(select(sessions)).mappings().all()
    assert {row["token_hash"] for row in stored} == {
        hashlib.sha256(first["token"].encode()).hexdigest(),
        hashlib.sha256(second["token"].encode()).hexdigest(),
    }
    assert not any(row["token_hash"] in {first["token"], second["token"]} for row in stored)
    assert all(row["user_id"] == public["id"] for row in stored)


@pytest.mark.parametrize("email,password", [
    ("alice@example.com", "incorrect password"),
    ("unknown@example.com", PASSWORD),
    ("invalid email", PASSWORD),
    (None, PASSWORD),
    ("alice@example.com", "short"),
    ("alice@example.com", None),
])
def test_failed_login_is_generic_and_creates_no_session(accounts, db, email, password):
    register(accounts)
    with pytest.raises(AccountError) as error:
        accounts.login(email, password)
    assert error.value.code == "unauthorized"
    assert error.value.message == "Email or password is incorrect."
    assert error.value.__cause__ is None
    assert "alice" not in str(error.value)
    with db.transaction() as connection:
        assert connection.scalar(select(func.count()).select_from(sessions)) == 0


def test_unknown_account_still_checks_a_real_password_hash(accounts, monkeypatch):
    hasher = accounts._hasher
    checked = []

    class ObservedHasher:
        def verify(self, encoded, password):
            checked.append(encoded)
            return hasher.verify(encoded, password)

    monkeypatch.setattr(accounts, "_hasher", ObservedHasher())
    with pytest.raises(AccountError, match="Email or password is incorrect"):
        accounts.login("nobody@example.com", PASSWORD)
    assert len(checked) == 1
    assert checked[0].startswith("$argon2id$")


def test_session_expiry_is_enforced_at_the_exact_boundary(accounts, clock):
    public = register(accounts)
    login = accounts.login(public["email"], PASSWORD)
    clock[0] = login["expires_at"] - 1
    assert accounts.authenticate(login["token"]) == public
    clock[0] += 1
    with pytest.raises(AccountError) as error:
        accounts.authenticate(login["token"])
    assert error.value.code == "unauthorized"


def test_logout_revokes_only_the_selected_session_and_is_idempotent(accounts, db):
    public = register(accounts)
    first = accounts.login(public["email"], PASSWORD)
    second = accounts.login(public["email"], PASSWORD)
    assert accounts.logout(first["token"]) is None
    assert accounts.logout(first["token"]) is None
    with pytest.raises(AccountError) as error:
        accounts.authenticate(first["token"])
    assert error.value.code == "unauthorized"
    assert accounts.authenticate(second["token"]) == public
    with db.transaction() as connection:
        assert connection.scalar(select(func.count()).select_from(sessions)) == 1


@pytest.mark.parametrize("token", [None, "", True, 123, "x" * 5000, "a" * 42, "é" * 43])
def test_malformed_sessions_never_authenticate_and_logout_is_harmless(accounts, token):
    with pytest.raises(AccountError) as error:
        accounts.authenticate(token)
    assert error.value.code == "unauthorized"
    if token not in (None, ""):
        assert str(token) not in str(error.value)
    assert accounts.logout(token) is None


def test_unknown_well_formed_session_does_not_authenticate(accounts):
    with pytest.raises(AccountError) as error:
        accounts.authenticate(secrets.token_urlsafe(32))
    assert error.value.code == "unauthorized"


def test_accounts_and_sessions_survive_database_reconstruction(tmp_path):
    url = f"sqlite:///{tmp_path / 'restart.db'}"
    original_db = Database(url)
    original_db.initialize()
    original = CloudAccounts(original_db, now=lambda: 1_800_000_000)
    public = register(original)
    login = original.login(public["email"], PASSWORD)
    original_db.close()

    reopened_db = Database(url)
    reopened_db.initialize()
    try:
        reopened = CloudAccounts(reopened_db, now=lambda: 1_800_000_001)
        assert reopened.authenticate(login["token"]) == public
        assert reopened.login(public["email"], PASSWORD)["user"] == public
        reopened.logout(login["token"])
        with pytest.raises(AccountError):
            reopened.authenticate(login["token"])
    finally:
        reopened_db.close()


def test_duplicate_email_does_not_replace_password_or_identity(accounts, db):
    public = register(accounts)
    with pytest.raises(AccountError) as error:
        accounts.register("ALICE@EXAMPLE.COM", "a different long password")
    assert error.value.code == "conflict"
    assert error.value.__cause__ is None
    assert accounts.login(public["email"], PASSWORD)["user"] == public
    with db.transaction() as connection:
        assert connection.scalar(select(func.count()).select_from(users)) == 1


@pytest.mark.parametrize("email,password", [
    ("not an email", PASSWORD),
    ("person@localhost", PASSWORD),
    ("alice@example.com", "short"),
    ("alice@example.com", "x" * 1025),
    ("alice@example.com", True),
])
def test_registration_validation_leaves_no_account(accounts, db, email, password):
    with pytest.raises(AccountError) as error:
        accounts.register(email, password)
    assert error.value.code == "invalid"
    with db.transaction() as connection:
        assert connection.scalar(select(func.count()).select_from(users)) == 0


def test_passwords_are_not_trimmed_or_normalized(accounts):
    password = "  keep these spaces  "
    public = accounts.register("alice@example.com", password)
    assert accounts.login(public["email"], password)["user"] == public
    with pytest.raises(AccountError):
        accounts.login(public["email"], password.strip())


def test_login_cleans_expired_sessions_only_for_its_account(accounts, db, clock):
    alice = register(accounts)
    bob = register(accounts, "bob@example.com")
    old_alice = accounts.login(alice["email"], PASSWORD)
    old_bob = accounts.login(bob["email"], PASSWORD)
    clock[0] = old_alice["expires_at"]
    current = accounts.login(alice["email"], PASSWORD)
    with db.transaction() as connection:
        records = connection.execute(select(sessions)).mappings().all()
    assert len(records) == 2
    assert hashlib.sha256(old_bob["token"].encode()).hexdigest() in {r["token_hash"] for r in records}
    assert hashlib.sha256(current["token"].encode()).hexdigest() in {r["token_hash"] for r in records}
    with pytest.raises(AccountError):
        accounts.authenticate(old_bob["token"])


def test_login_rehashes_existing_password_with_current_argon2_parameters(accounts, db):
    public = register(accounts)
    previous_hash = PasswordHasher(time_cost=1, memory_cost=8192, parallelism=1).hash(PASSWORD)
    with db.transaction(write=True) as connection:
        connection.execute(update(users).where(users.c.id == public["id"]).values(password_hash=previous_hash))
    assert accounts.login(public["email"], PASSWORD)["user"] == public
    with db.transaction() as connection:
        current_hash = connection.scalar(select(users.c.password_hash).where(users.c.id == public["id"]))
    assert current_hash != previous_hash
    assert not PasswordHasher().check_needs_rehash(current_hash)


def test_a_credential_change_during_verification_cannot_issue_a_session(accounts, db, monkeypatch):
    public = register(accounts)
    original_hasher = accounts._hasher
    replacement_hash = PasswordHasher().hash("replacement account password")

    class ChangingHasher:
        def verify(self, encoded, password):
            verified = original_hasher.verify(encoded, password)
            with db.transaction(write=True) as connection:
                connection.execute(update(users).where(users.c.id == public["id"]).values(password_hash=replacement_hash))
            return verified

        def check_needs_rehash(self, encoded):
            return original_hasher.check_needs_rehash(encoded)

    monkeypatch.setattr(accounts, "_hasher", ChangingHasher())
    with pytest.raises(AccountError) as error:
        accounts.login(public["email"], PASSWORD)
    assert error.value.code == "unauthorized"
    with db.transaction() as connection:
        assert connection.scalar(select(func.count()).select_from(sessions)) == 0


def test_concurrent_registration_has_one_winner(accounts, db):
    def attempt(_):
        try:
            return accounts.register("alice@example.com", PASSWORD)["id"]
        except AccountError as error:
            assert error.code == "conflict"
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, range(2)))
    assert sum(result is not None for result in results) == 1
    with db.transaction() as connection:
        assert connection.scalar(select(func.count()).select_from(users)) == 1


def test_corrupted_password_hash_returns_the_same_generic_login_error(accounts, db):
    public = register(accounts)
    with db.transaction(write=True) as connection:
        connection.execute(update(users).where(users.c.id == public["id"]).values(password_hash="broken hash"))
    with pytest.raises(AccountError) as error:
        accounts.login(public["email"], PASSWORD)
    assert error.value.code == "unauthorized"
    assert error.value.message == "Email or password is incorrect."


@pytest.mark.parametrize("operation", ["register", "login", "authenticate", "logout"])
def test_database_failure_uses_a_safe_service_error(accounts, db, monkeypatch, operation):
    def unavailable(*args, **kwargs):
        raise CloudDatabaseError("private database details")

    monkeypatch.setattr(db, "transaction", unavailable)
    arguments = ("alice@example.com", PASSWORD) if operation in {"register", "login"} else (secrets.token_urlsafe(32),)
    with pytest.raises(AccountError) as error:
        getattr(accounts, operation)(*arguments)
    assert error.value.code == "unavailable"
    assert error.value.message == "The account service is unavailable."
    assert "private" not in str(error.value)
    assert error.value.__cause__ is None


@pytest.mark.parametrize("ttl", [0, -1, True, "60"])
def test_session_lifetime_rejects_invalid_configuration(db, ttl):
    with pytest.raises(AccountError) as error:
        CloudAccounts(db, session_ttl_seconds=ttl)
    assert error.value.code == "invalid"


@pytest.mark.skipif(not os.environ.get("MATTERS_TEST_DATABASE_URL"), reason="PostgreSQL test URL not configured")
def test_postgresql_account_and_session_persistence():
    url = os.environ["MATTERS_TEST_DATABASE_URL"]
    if not url.startswith("postgresql"):
        pytest.skip("A PostgreSQL URL is required")
    db = Database(url)
    db.initialize()
    email = f"auth-{uuid.uuid4().hex}@example.com"
    accounts = CloudAccounts(db, now=lambda: 1_800_000_000)
    user = accounts.register(email, PASSWORD)
    login = accounts.login(email, PASSWORD)
    db.close()

    reopened = Database(url)
    reopened.initialize()
    try:
        restored = CloudAccounts(reopened, now=lambda: 1_800_000_001)
        assert restored.authenticate(login["token"]) == user
        restored.logout(login["token"])
        with pytest.raises(AccountError):
            restored.authenticate(login["token"])
    finally:
        reopened.close()
