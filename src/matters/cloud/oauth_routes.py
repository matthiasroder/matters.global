"""OAuth discovery, PKCE exchange and explicit account/graph consent pages."""

import hmac
from html import escape
import json
import secrets
from urllib.parse import parse_qsl, urlencode

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from starlette.concurrency import run_in_threadpool

from .accounts import AccountError
from .oauth import OAuthError, TOKEN_PATTERN
from .rate_limits import DatabaseRateLimiter, RateLimitUnavailable


MAX_OAUTH_BODY = 32 * 1024
CSRF_COOKIE = "matters_oauth_csrf"


def _response(payload, status=200):
    return JSONResponse(payload, status_code=status, headers={"Cache-Control": "no-store", "Pragma": "no-cache"})


def _error(error):
    response = _response({"error": error.code, "error_description": error.message}, error.status)
    if error.status == 429:
        response.headers["Retry-After"] = str(getattr(error, "retry_after", 60))
    return response


def _page(title, content):
    return HTMLResponse(
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>' + escape(title) + ' · Matters</title><link rel="stylesheet" href="/assets/styles.css"></head>'
        '<body><header class="topbar"><a class="brand" href="/">matters.</a></header>'
        '<main class="auth-layout"><section class="intro"><h1>Connect what<br>matters.</h1>'
        '<p class="intro-copy">Choose which graphs this client may use. Your other graphs stay private.</p></section>'
        '<section class="auth-card"><h2>' + escape(title) + '</h2>' + content + '</section></main></body></html>',
        # Chromium also suppresses a native form's Origin under no-referrer.
        # Send only the origin as Referrer, retaining a real Origin for the
        # strict same-origin + CSRF checks without leaking pending request URLs.
        headers={"Cache-Control": "no-store", "Referrer-Policy": "strict-origin", "Content-Security-Policy": "default-src 'none'; style-src 'self'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'"},
    )


async def _body(request, *, json_body=False, repeated=()):
    raw = bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw) > MAX_OAUTH_BODY:
            raise OAuthError("The request is too large.", status=413)
    media_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    try:
        if json_body:
            if media_type != "application/json":
                raise OAuthError("Use application/json.", status=415)
            def unique_keys(pairs):
                result = {}
                for key, value in pairs:
                    if key in result:
                        raise ValueError("duplicate field")
                    result[key] = value
                return result
            result = json.loads(raw, object_pairs_hook=unique_keys)
            if not isinstance(result, dict):
                raise OAuthError()
            return result
        if media_type != "application/x-www-form-urlencoded":
            raise OAuthError("Use form-encoded parameters.", status=415)
        pairs = parse_qsl(raw.decode("utf-8"), keep_blank_values=True, strict_parsing=True, max_num_fields=120)
        result = {key: [] for key in repeated}
        for key, value in pairs:
            if key in repeated:
                result[key].append(value)
            elif key in result:
                raise OAuthError("Duplicate parameters are not permitted.")
            else:
                result[key] = value
        return result
    except OAuthError:
        raise
    except (ValueError, UnicodeError):
        raise OAuthError("The request could not be parsed.") from None


def _query(request, allowed):
    if len(request.scope.get("query_string", b"")) > 8192:
        raise OAuthError("The request is too large.", status=413)
    pairs = list(request.query_params.multi_items())
    result = {}
    for key, value in pairs:
        if key in result or key not in allowed:
            raise OAuthError("Query fields do not match the authorization schema.")
        result[key] = value
    return result


def build_oauth_router(oauth, accounts, graphs, *, secure_cookies=True, cookie_name="matters_session", limiter=None, signup_enabled=True):
    """Create routes for a configured, trusted issuer/resource origin.

    Consent uses the existing CloudAccounts login and session cookie. The
    browser submits its own session-bound pending request and a double-submit
    CSRF nonce; the server rechecks both before creating a grant.
    """
    router = APIRouter()
    limiter = limiter if limiter is not None else DatabaseRateLimiter(oauth.db)

    def limit(request, category):
        peer = request.client.host if request.client else "unknown"
        category = "auth" if category == "login" else "oauth." + category
        try:
            allowed = limiter.allow(peer, category=category)
        except RateLimitUnavailable:
            raise OAuthError("The authorization service is temporarily unavailable.", "temporarily_unavailable", 503) from None
        if not allowed:
            error = OAuthError("Too many authorization attempts. Try again shortly.", "temporarily_unavailable", 429)
            error.retry_after = limiter.policy(category)[1]
            raise error

    def browser_origin(request):
        if request.headers.get("origin") != oauth.issuer_url or request.headers.get("sec-fetch-site", "same-origin") not in {"same-origin", "none"}:
            raise OAuthError("Submit this form from the Matters authorization page.", "access_denied", 403)

    def csrf(request, submitted):
        expected = request.cookies.get(CSRF_COOKIE, "")
        if not TOKEN_PATTERN.fullmatch(expected) or not isinstance(submitted, str) or not hmac.compare_digest(expected, submitted):
            raise OAuthError("Reload the authorization page and try again.", "access_denied", 403)

    def set_csrf(response, value):
        response.set_cookie(CSRF_COOKIE, value, max_age=600, httponly=True, secure=secure_cookies, samesite="lax", path="/oauth")
        return response

    @router.get("/.well-known/oauth-authorization-server")
    def metadata():
        return _response(oauth.metadata())

    @router.get("/.well-known/oauth-protected-resource")
    @router.get("/.well-known/oauth-protected-resource/mcp")
    def resource_metadata():
        return _response(oauth.protected_resource_metadata())

    @router.post("/oauth/register")
    async def register(request: Request):
        try:
            await run_in_threadpool(limit, request, "register")
            document = await _body(request, json_body=True)
            return _response(await run_in_threadpool(oauth.register_client, document), 201)
        except OAuthError as error:
            return _error(error)

    @router.get("/oauth/authorize")
    def authorize(request: Request):
        try:
            limit(request, "authorize")
            parameters = _query(request, {"client_id", "redirect_uri", "response_type", "state", "scope", "resource", "code_challenge", "code_challenge_method"})
            pending = oauth.create_authorization_request(parameters)
            return RedirectResponse("/oauth/consent?" + urlencode({"request": pending}), status_code=303, headers={"Cache-Control": "no-store"})
        except OAuthError as error:
            # Invalid redirects are never followed, including with an error.
            return _error(error)

    @router.get("/oauth/consent")
    def consent_page(request: Request):
        try:
            limit(request, "consent")
            parameters = _query(request, {"request"})
            pending = parameters.get("request")
            data = oauth.inspect_request(pending)
            nonce = request.cookies.get(CSRF_COOKIE, "")
            if not TOKEN_PATTERN.fullmatch(nonce):
                nonce = secrets.token_urlsafe(32)
            session = request.cookies.get(cookie_name)
            try:
                user = accounts.authenticate(session)
            except AccountError as error:
                if error.code != "unauthorized":
                    raise
                content = '<p class="muted">Sign in to your Matters account to review access for ' + escape(data["client_name"]) + '.</p>'
                content += '<form class="stack" method="post" action="/oauth/consent/login">'
                content += '<input type="hidden" name="request" value="' + escape(pending, quote=True) + '">'
                content += '<input type="hidden" name="csrf" value="' + nonce + '">'
                content += '<label>Email<input name="email" type="email" maxlength="254" autocomplete="username" required></label>'
                content += '<label>Password<input name="password" type="password" maxlength="1024" autocomplete="current-password" required></label>'
                content += '<button class="primary-button" type="submit">Sign in</button></form>'
                if signup_enabled:
                    content += '<p class="muted">Need an account? <a href="/" target="_blank" rel="noopener">Create one in Matters</a>, then return to this page.</p>'
                else:
                    content += '<p class="muted">New account registration is disabled. Use the account provided by this installation’s operator.</p>'
                return set_csrf(_page("Sign in to connect", content), nonce)
            data = oauth.prepare_consent(pending, user["id"], session, nonce)
            available = graphs.list_graphs(user["id"])["graphs"]
            editing = "graphs:write" in data["scopes"]
            available = [g for g in available if not editing or g["role"] in ("owner", "editor")]
            content = '<p class="muted">Signed in as ' + escape(user["email"]) + '.</p><p><strong>' + escape(data["client_name"]) + '</strong> requests ' + ("read and edit" if editing else "read-only") + ' access to the graphs you select.</p>'
            content += '<p class="muted">The application supplied this name. Continue only if you recognize its return address: <strong>' + escape(data["redirect_uri"]) + '</strong>.</p>'
            if "offline_access" in data["scopes"]:
                content += '<p class="muted">This connection may renew access until it expires or is revoked.</p>'
            content += '<form class="stack" method="post" action="/oauth/consent">'
            content += '<input type="hidden" name="request" value="' + escape(pending, quote=True) + '"><input type="hidden" name="csrf" value="' + nonce + '">'
            for graph in available:
                content += '<label class="condition-label"><input type="checkbox" name="graph_id" value="' + escape(graph["graph_id"], quote=True) + '"><span>' + escape(graph["title"]) + ' (' + escape(graph["kind"]) + ')</span></label>'
            if available:
                content += '<button class="primary-button" type="submit" name="decision" value="approve">Allow selected graphs</button>'
            else:
                content += '<p>Create a graph in Matters before approving this connection.</p>'
            content += '<button class="secondary-button" type="submit" name="decision" value="deny">Decline</button></form>'
            content += '<p class="muted">A project reference chooses a graph; it never grants access to another graph.</p>'
            return set_csrf(_page("Review graph access", content), nonce)
        except OAuthError as error:
            return _error(error)
        except AccountError:
            return _error(OAuthError("Sign in to continue.", "access_denied", 401))

    @router.post("/oauth/consent/login")
    async def consent_login(request: Request):
        try:
            await run_in_threadpool(limit, request, "login")
            browser_origin(request)
            body = await _body(request)
            if set(body) != {"request", "csrf", "email", "password"}:
                raise OAuthError()
            csrf(request, body["csrf"])
            await run_in_threadpool(oauth.inspect_request, body["request"])
            session = await run_in_threadpool(accounts.login, body["email"], body["password"])
            response = RedirectResponse("/oauth/consent?" + urlencode({"request": body["request"]}), status_code=303, headers={"Cache-Control": "no-store"})
            response.set_cookie(cookie_name, session["token"], max_age=accounts.session_ttl_seconds, httponly=True, secure=secure_cookies, samesite="strict", path="/")
            return response
        except OAuthError as error:
            return _error(error)
        except AccountError as error:
            if error.code == "unavailable":
                return _error(OAuthError("The account service is temporarily unavailable. Try again shortly.", "temporarily_unavailable", 503))
            return _error(OAuthError("Email or password is incorrect.", "access_denied", 401))

    @router.post("/oauth/consent")
    async def submit_consent(request: Request):
        try:
            await run_in_threadpool(limit, request, "consent")
            browser_origin(request)
            body = await _body(request, repeated=("graph_id",))
            if set(body) != {"request", "csrf", "graph_id", "decision"} or body["decision"] not in {"approve", "deny"}:
                raise OAuthError()
            csrf(request, body["csrf"])
            session = request.cookies.get(cookie_name)
            user = await run_in_threadpool(accounts.authenticate, session)
            location = await run_in_threadpool(oauth.consent, body["request"], user["id"], session, body["csrf"], body["graph_id"], repository=graphs, approve=body["decision"] == "approve")
            return RedirectResponse(location, status_code=303, headers={"Cache-Control": "no-store"})
        except OAuthError as error:
            return _error(error)
        except AccountError:
            return _error(OAuthError("Sign in to continue.", "access_denied", 401))

    @router.post("/oauth/token")
    async def token(request: Request):
        try:
            await run_in_threadpool(limit, request, "token")
            if request.headers.get("authorization"):
                raise OAuthError("This endpoint accepts public PKCE clients.", "invalid_client", 401)
            body = await _body(request)
            grant = body.pop("grant_type", None)
            if grant == "authorization_code":
                if set(body) != {"client_id", "code", "redirect_uri", "code_verifier", "resource"}:
                    raise OAuthError()
                result = await run_in_threadpool(oauth.exchange_code, **body)
            elif grant == "refresh_token":
                if not {"client_id", "refresh_token", "resource"} <= set(body) or set(body) - {"client_id", "refresh_token", "resource", "scope"}:
                    raise OAuthError()
                result = await run_in_threadpool(oauth.refresh, **body)
            else:
                raise OAuthError("The grant type is not supported.", "unsupported_grant_type")
            return _response(result)
        except OAuthError as error:
            return _error(error)

    @router.post("/oauth/revoke")
    async def revoke(request: Request):
        try:
            await run_in_threadpool(limit, request, "revoke")
            if request.headers.get("authorization"):
                raise OAuthError("This endpoint accepts public PKCE clients.", "invalid_client", 401)
            body = await _body(request)
            if not {"client_id", "token"} <= set(body) or set(body) - {"client_id", "token", "token_type_hint"}:
                raise OAuthError()
            await run_in_threadpool(oauth.revoke, client_id=body["client_id"], token=body["token"])
            return _response({})
        except OAuthError as error:
            return _error(error)

    @router.get("/oauth/connections")
    def connections(request: Request):
        try:
            _query(request, set())
            user = accounts.authenticate(request.cookies.get(cookie_name))
            nonce = request.cookies.get(CSRF_COOKIE, "")
            if not TOKEN_PATTERN.fullmatch(nonce):
                nonce = secrets.token_urlsafe(32)
            grants = oauth.list_grants(user["id"])
            content = '<p class="muted">Signed in as ' + escape(user["email"]) + '.</p>'
            for grant in grants:
                content += '<form class="stack" method="post" action="/oauth/connections/revoke"><p><strong>' + escape(grant["client_name"]) + '</strong> · ' + str(len(grant["graph_ids"])) + ' selected graphs</p>'
                content += '<input type="hidden" name="grant_id" value="' + grant["grant_id"] + '"><input type="hidden" name="csrf" value="' + nonce + '">'
                content += '<button class="secondary-button" type="submit">Disconnect</button></form>'
            if not grants:
                content += '<p>No active client connections.</p>'
            content += '<p><a href="/">Return to Matters</a></p>'
            return set_csrf(_page("Connected clients", content), nonce)
        except AccountError:
            return _error(OAuthError("Sign in to Matters before managing connections.", "access_denied", 401))
        except OAuthError as error:
            return _error(error)

    @router.post("/oauth/connections/revoke")
    async def disconnect(request: Request):
        try:
            browser_origin(request)
            body = await _body(request)
            if set(body) != {"grant_id", "csrf"}:
                raise OAuthError()
            csrf(request, body["csrf"])
            user = await run_in_threadpool(accounts.authenticate, request.cookies.get(cookie_name))
            await run_in_threadpool(oauth.revoke_grant, user["id"], body["grant_id"])
            return RedirectResponse("/oauth/connections", status_code=303, headers={"Cache-Control": "no-store"})
        except OAuthError as error:
            return _error(error)
        except AccountError:
            return _error(OAuthError("Sign in to continue.", "access_denied", 401))

    return router
