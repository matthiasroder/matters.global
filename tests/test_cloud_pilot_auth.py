"""Closed-pilot signup and bounded expensive authentication work."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import threading
import time
from urllib.parse import parse_qs, urlsplit

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("argon2")
pytest.importorskip("email_validator")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient
from sqlalchemy import select

from matters.cli import main
from matters.cloud.accounts import AccountError, CloudAccounts
from matters.cloud.app import create_app
from matters.cloud.db import Database, users
from matters.cloud.lifecycle import AccountLifecycle
from matters.cloud.mail_delivery import CaptureMailDelivery


PASSWORD = "Temporary owned pilot test password!"


def test_signup_default_stays_open_and_closed_pilot_preserves_existing_accounts(tmp_path, monkeypatch):
    url = "sqlite:///" + str(tmp_path / "pilot.db")
    opened = create_app(url, secure_cookies=False)
    with TestClient(opened, base_url="http://localhost") as client:
        credentials = {"email": "existing@example.com", "password": PASSWORD}
        assert client.get("/api/auth/capabilities").json()["signup_enabled"] is True
        assert client.post("/api/auth/register", json=credentials).status_code == 201
        assert 'data-signup-enabled="true"' in client.get("/").text
    closed = create_app(url, secure_cookies=False, disable_signup=True)
    monkeypatch.setattr(closed.state.accounts, "register", lambda *_: pytest.fail("Closed signup performed password/account work"))
    with TestClient(closed, base_url="http://localhost") as client:
        capabilities = client.get("/api/auth/capabilities").json()
        assert capabilities["signup_enabled"] is False and capabilities["email_delivery"] is False
        first = client.post("/api/auth/register", json=credentials)
        second = client.post("/api/auth/register", json={"email": "new@example.com", "password": PASSWORD})
        assert first.status_code == second.status_code == 403 and first.json() == second.json()
        assert "existing@example.com" not in first.text and PASSWORD not in first.text
        page = client.get("/").text
        assert 'data-signup-enabled="false"' in page
        assert 'id="signup-controls" class="auth-switch" hidden' in page
        assert 'id="signup-message" class="field-help" hidden' not in page
        assert client.post("/api/auth/login", json=credentials).status_code == 200
        assert client.post("/api/graphs", json={"title": "Existing account project", "kind": "project"}).status_code == 201
        assert client.get("/api/graphs").json()["graphs"][0]["title"] == "Existing account project"
    with pytest.raises(ValueError, match="disable_signup must be a boolean"):
        create_app(url, disable_signup="true")


def test_disable_signup_cli_flag_is_explicit_and_does_not_change_default(monkeypatch):
    import matters.cloud.app
    import uvicorn
    calls = []
    for name in list(os.environ):
        if name.startswith("MATTERS_SMTP_"):
            monkeypatch.delenv(name)
    monkeypatch.setattr(matters.cloud.app, "create_app", lambda url, **options: calls.append(options) or object())
    monkeypatch.setattr(uvicorn, "run", lambda *_args, **_kwargs: None)
    assert main(["cloud", "--database-url", "sqlite:///unused.db", "--development"]) == 0
    assert "disable_signup" not in calls[-1]
    assert main(["cloud", "--database-url", "sqlite:///unused.db", "--development", "--disable-signup"]) == 0
    assert calls[-1]["disable_signup"] is True


def test_global_hash_budget_bounds_real_concurrency_and_all_password_paths(tmp_path):
    db = Database("sqlite:///" + str(tmp_path / "password-work.db")).initialize()
    first, second = CloudAccounts(db), CloudAccounts(db)
    person = first.register("existing@example.com", PASSWORD)
    session = first.login(person["email"], PASSWORD)["token"]
    delivery = CaptureMailDelivery()
    lifecycle = AccountLifecycle(db, first, public_origin="https://example.test", delivery=delivery)
    lifecycle.request_password_reset(person["email"])
    assert lifecycle.drain_delivery()
    link = next(line for line in delivery.messages[0].body.splitlines() if line.startswith("https://"))
    reset = parse_qs(urlsplit(link).fragment)["token"][0]
    gate, started = threading.Event(), threading.Event()
    counts, lock = {"active": 0, "maximum": 0}, threading.Lock()
    actual = first._hasher._hasher
    class PausedHasher:
        def hash(self, *args, **kwargs):
            with lock:
                counts["active"] += 1
                counts["maximum"] = max(counts["maximum"], counts["active"])
                if counts["active"] == 2:
                    started.set()
            try:
                assert gate.wait(timeout=5)
                return actual.hash(*args, **kwargs)
            finally:
                with lock:
                    counts["active"] -= 1
        def verify(self, *args, **kwargs):
            return actual.verify(*args, **kwargs)
        def check_needs_rehash(self, *args, **kwargs):
            return actual.check_needs_rehash(*args, **kwargs)
    first._hasher._hasher = PausedHasher()
    second._hasher._hasher = PausedHasher()
    pool = ThreadPoolExecutor(max_workers=2)
    pending = []
    try:
        pending = [pool.submit(service.register, email, PASSWORD) for service, email in ((first, "one@example.com"), (second, "two@example.com"))]
        assert started.wait(timeout=3)
        before = time.monotonic()
        refused = [
            lambda: first.register("overflow@example.com", PASSWORD),
            lambda: second.login(person["email"], PASSWORD),
            lambda: lifecycle.reset_password(reset, "A replacement temporary password!"),
            lambda: lifecycle.change_password(person["id"], session, PASSWORD, "A replacement temporary password!"),
        ]
        for operation in refused:
            with pytest.raises(AccountError) as error:
                operation()
            assert error.value.code == "unavailable" and PASSWORD not in str(error.value)
        overloaded = create_app("sqlite:///" + str(tmp_path / "password-work.db"), secure_cookies=False)
        with TestClient(overloaded, base_url="http://localhost") as client:
            response = client.post("/api/auth/login", json={"email": person["email"], "password": PASSWORD})
            assert response.status_code == 503 and response.json()["error"]["code"] == "unavailable"
            assert PASSWORD not in response.text and person["email"] not in response.text
        assert time.monotonic() - before < 1
        assert first.authenticate(session)["id"] == person["id"]
        assert counts["maximum"] == 2
    finally:
        gate.set()
        pool.shutdown(wait=True)
        lifecycle.close()
    assert all(future.result()["email"] in {"one@example.com", "two@example.com"} for future in pending)
    assert first.login(person["email"], PASSWORD)["user"]["id"] == person["id"]
    with db.transaction() as connection:
        assert connection.execute(select(users.c.id).where(users.c.email == "overflow@example.com")).first() is None
    db.close()


def test_closed_pilot_native_browser_shows_existing_account_signin(tmp_path):
    import asyncio
    import shutil
    import socket
    import subprocess
    uvicorn = pytest.importorskip("uvicorn")
    node = shutil.which("node")
    executable = os.environ.get("MATTERS_BROWSER_EXECUTABLE", "/usr/bin/chromium")
    if node is None or not Path(executable).is_file():
        pytest.skip("Native pilot browser check requires Node and Chromium")
    if subprocess.run([node, "-e", "require('playwright')"], capture_output=True, timeout=10).returncode:
        pytest.skip("Native pilot browser check requires Playwright")
    owned = socket.socket(); owned.bind(("127.0.0.1", 0)); owned.listen(128)
    origin = "http://127.0.0.1:" + str(owned.getsockname()[1])
    app = create_app("sqlite:///" + str(tmp_path / "browser.db"), secure_cookies=False, public_origin=origin, disable_signup=True, mail_delivery=CaptureMailDelivery())
    # An operator can provision an account through trusted service code; the
    # closed HTTP registration endpoint never performs this operation.
    app.state.accounts.register("existing@example.com", PASSWORD)
    server = uvicorn.Server(uvicorn.Config(app, access_log=False, log_level="error"))
    thread = threading.Thread(target=lambda: asyncio.run(server.serve(sockets=[owned])), daemon=True)
    thread.start()
    script = r"""
const fs=require('node:fs');const{chromium}=require('playwright');const c=JSON.parse(fs.readFileSync(0,'utf8'));
(async()=>{const b=await chromium.launch({executablePath:c.executable,headless:true,args:['--no-sandbox']});
try{const context=await b.newContext();const page=await context.newPage();page.setDefaultTimeout(10000);const errors=[];page.on('pageerror',e=>errors.push(e.message));
await page.goto(c.origin);await page.getByRole('heading',{name:'Welcome back',exact:true}).waitFor();
if(await page.locator('#signup-controls').isVisible()||!await page.locator('#signup-message').isVisible())throw Error('Closed signup was advertised');
await page.locator('#auth-switch').evaluate(button=>button.click());
if(await page.locator('#auth-title').innerText()!=='Welcome back')throw Error('Hidden signup control changed mode');
const caps=await(await context.request.get(c.origin+'/api/auth/capabilities')).json();
if(caps.signup_enabled!==false||caps.email_delivery!==true)throw Error('Signup and mail policies were conflated');
const refused=await context.request.post(c.origin+'/api/auth/register',{headers:{Origin:c.origin},data:{email:'new@example.com',password:c.password}});if(refused.status()!==403)throw Error('Public signup succeeded');
await page.locator('#auth-form').getByLabel('Email',{exact:true}).fill('existing@example.com');await page.locator('#auth-form').getByLabel('Password',{exact:true}).fill(c.password);await page.getByRole('button',{name:'Sign in',exact:true}).click();await page.locator('#account-email').filter({hasText:'existing@example.com'}).waitFor();
if(errors.length)throw Error('Browser script failed');console.log(JSON.stringify({closed_signup:true,existing_signin:true,independent_mail_capability:true,errors}));
}finally{await b.close();}})().catch(()=>{console.error('Owned pilot browser check failed');process.exitCode=1;});
"""
    try:
        deadline = time.monotonic() + 10
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert server.started
        checked = subprocess.run([node, "-e", script], input=json.dumps({"origin": origin, "password": PASSWORD, "executable": executable}), text=True, capture_output=True, timeout=30)
        assert checked.returncode == 0, checked.stderr
        assert json.loads(checked.stdout) == {"closed_signup": True, "existing_signin": True, "independent_mail_capability": True, "errors": []}
    finally:
        server.should_exit = True; thread.join(timeout=5)
        if thread.is_alive():
            server.force_exit = True; thread.join(timeout=5)
        owned.close()
        assert not thread.is_alive()
