"""CSRF-protected account lifecycle APIs and credential-free landing HTML."""

import hmac
import secrets
from html import escape
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from .accounts import AccountError, _normalized_email
from .oauth import OAuthError, TOKEN_PATTERN
from .oauth_routes import _body


CSRF_COOKIE = "matters_account_csrf"


def create_lifecycle_router(lifecycle, authenticated_user, *, public_origin=None, secure_cookies=True, cookie_name="matters_session", limiter=None):
    router = APIRouter()

    def response(payload, request, status=200, *, nonce=False):
        if nonce:
            token = request.cookies.get(CSRF_COOKIE, "")
            if not TOKEN_PATTERN.fullmatch(token):
                token = secrets.token_urlsafe(32)
            payload = {**payload, "csrf_token": token}
        result = JSONResponse({"schema_version": 1, **payload}, status_code=status, headers={"Cache-Control": "no-store"})
        if nonce:
            result.set_cookie(CSRF_COOKIE, token, max_age=86400, secure=secure_cookies, httponly=True, samesite="strict", path="/")
        return result

    def same_origin(request):
        expected = public_origin or str(request.base_url).rstrip("/")
        candidate = request.headers.get("origin")
        try:
            a, b = urlsplit(expected), urlsplit(candidate or "")
            valid = (b.scheme in {"http", "https"} and b.hostname and not b.username and not b.password and b.path in {"", "/"} and not b.query and not b.fragment
                     and (a.scheme, a.hostname, a.port or (443 if a.scheme == "https" else 80)) == (b.scheme, b.hostname, b.port or (443 if b.scheme == "https" else 80)))
        except ValueError:
            valid = False
        if not valid or request.headers.get("sec-fetch-site", "same-origin") not in {"same-origin", "none"}:
            raise AccountError("Submit this request from the Matters account page.", "forbidden")

    def check_csrf(request, submitted):
        expected = request.cookies.get(CSRF_COOKIE, "")
        if not TOKEN_PATTERN.fullmatch(expected) or not isinstance(submitted, str) or not hmac.compare_digest(expected, submitted):
            raise AccountError("Reload the account page and try again.", "forbidden")

    async def body(request, fields, *, csrf=False):
        same_origin(request)
        try:
            document = await _body(request, json_body=True)
        except OAuthError as error:
            raise AccountError(error.message) from None
        expected = set(fields) | ({"csrf_token"} if csrf else set())
        if set(document) != expected or any(not isinstance(document.get(key), str) for key in expected):
            raise AccountError("Request fields do not match the account schema.")
        if csrf:
            check_csrf(request, document.pop("csrf_token"))
        return document

    async def limit(request, category, subject=None):
        if limiter is None:
            return
        peer = request.client.host if request.client else "unknown"
        try:
            allowed = await run_in_threadpool(limiter.allow, peer, category=category + "-peer")
            if allowed and subject is not None:
                suffix = "email" if category == "auth-recovery" else "actor" if category in {"auth-verification", "auth-password-change"} else "value"
                allowed = await run_in_threadpool(limiter.allow, subject, category=category + "-" + suffix) and allowed
        except Exception:
            raise AccountError("The account request service is unavailable.", "unavailable") from None
        if not allowed:
            raise AccountError("Too many account requests. Try again shortly.", "rate_limited")

    @router.get("/api/auth/capabilities")
    def capabilities(request: Request):
        return response(lifecycle.capabilities(), request, nonce=True)

    @router.get("/api/auth/security")
    def security(request: Request, user=Depends(authenticated_user)):
        return response(lifecycle.status(user["id"]), request, nonce=True)

    @router.post("/api/auth/verification/request", status_code=202)
    async def verification_request(request: Request, user=Depends(authenticated_user)):
        await body(request, (), csrf=True)
        await limit(request, "auth-verification", user["id"])
        return response(lifecycle.request_verification(user["id"]), request, 202)

    @router.post("/api/auth/verification/confirm")
    async def verification_confirm(request: Request):
        data = await body(request, ("token",), csrf=True)
        await limit(request, "auth-token", data["token"])
        result = await run_in_threadpool(lifecycle.verify_email, data["token"])
        return response(result, request)

    @router.post("/api/auth/password/reset/request", status_code=202)
    async def reset_request(request: Request):
        data = await body(request, ("email",))
        if len(data["email"]) > 320:
            raise AccountError("Request fields do not match the account schema.")
        try:
            subject = _normalized_email(data["email"])
        except AccountError:
            subject = data["email"].strip().casefold()
        await limit(request, "auth-recovery", subject)
        return response(lifecycle.request_password_reset(data["email"]), request, 202)

    @router.post("/api/auth/password/reset/confirm")
    async def reset_confirm(request: Request):
        data = await body(request, ("token", "new_password"), csrf=True)
        await limit(request, "auth-token", data["token"])
        result = response(await run_in_threadpool(lifecycle.reset_password, data["token"], data["new_password"]), request)
        return result

    @router.post("/api/auth/password/change")
    async def password_change(request: Request, user=Depends(authenticated_user)):
        data = await body(request, ("current_password", "new_password"), csrf=True)
        await limit(request, "auth-password-change", user["id"])
        result = await run_in_threadpool(lifecycle.change_password, user["id"], request.cookies.get(cookie_name), data["current_password"], data["new_password"])
        result = response(result, request)
        return result

    def landing(action):
        title = "Verify your email" if action == "verify" else "Reset your password"
        content = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
        content += '<title>' + escape(title) + ' · Matters</title><link rel="stylesheet" href="/assets/styles.css"><script src="/assets/lifecycle.js" defer></script></head>'
        content += '<body data-action="' + action + '"><header class="topbar"><a class="brand" href="/">matters.</a></header><main id="lifecycle-view" class="auth-layout"><section class="intro"><h1>Keep your<br>account yours.</h1></section>'
        content += '<section class="auth-card"><h2 id="lifecycle-title">' + escape(title) + '</h2><p id="lifecycle-message" class="message" role="status" aria-live="polite"></p><form id="lifecycle-form" class="stack" method="post">'
        if action == "reset":
            content += '<label>New password<input id="reset-password" name="new_password" type="password" autocomplete="new-password" minlength="12" maxlength="1024" required></label><label>Confirm new password<input id="reset-password-confirm" name="confirm_password" type="password" autocomplete="new-password" minlength="12" maxlength="1024" required></label>'
        content += '<button id="lifecycle-submit" class="primary-button" type="submit" disabled>Confirm</button></form><p><a href="/">Return to sign in</a></p></section></main></body></html>'
        return HTMLResponse(content, headers={"Cache-Control": "no-store", "Referrer-Policy": "strict-origin", "Content-Security-Policy": "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'"})

    @router.get("/account/verify")
    def verify_landing(request: Request):
        if request.url.query:
            raise AccountError("Open the complete link from your email.")
        return landing("verify")

    @router.get("/account/reset")
    def reset_landing(request: Request):
        if request.url.query:
            raise AccountError("Open the complete link from your email.")
        return landing("reset")

    return router
