"""Browser account controls use explicit, session-scoped lifecycle requests."""

import json
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


ASSETS = Path(__file__).parents[1] / "src" / "matters" / "cloud" / "assets"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")

MAIN = r"""
const fs = require('node:fs'), vm = require('node:vm');
class Element {
  constructor() { this.children=[]; this.elements=[]; this.events={}; this.value=''; this.textContent=''; this.hidden=false; }
  addEventListener(name, fn) { this.events[name]=fn; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children=[...children]; }
  setAttribute() {}
  reset() { for (const item of this.elements) item.value=''; }
}
const nodes = new Map();
const document = {getElementById(id) {if (!nodes.has(id)) nodes.set(id,new Element());return nodes.get(id);},createElement(){return new Element();}};
const get = id => document.getElementById(id);
const scenario=process.argv[2];
let session=scenario.startsWith('recovery') ? null : 'alice';
let holdSecurity=false, holdPassword=false, releaseSecurity=null, releasePassword=null;
const requests=[];
const people={alice:{id:'alice',email:'alice@example.com'},bob:{id:'bob',email:'bob@example.com'}};
const csrf='C'.repeat(43);
const capabilities={schema_version:1,email_delivery:!scenario.includes('disabled'),email_verification:!scenario.includes('disabled'),password_reset:!scenario.includes('disabled'),password_change:true,csrf_token:csrf};
const response=(data,status=200)=>({ok:status>=200&&status<300,status,json:async()=>data});
const security=()=>({...capabilities,email_verified:scenario==='already-verified',email_verified_at:null});
const fetch=async(path,options={})=>{
  requests.push({path,method:options.method||'GET',body:options.body?JSON.parse(options.body):null});
  if(path==='/api/auth/me') return session?response({user:people[session]}):response({},401);
  if(path==='/api/auth/login'){session=JSON.parse(options.body).email.startsWith('bob')?'bob':'alice';return response({user:people[session]});}
  if(path==='/api/auth/logout'){session=null;return response({logged_out:true});}
  if(path==='/api/graphs')return response({graphs:[]});
  if(path==='/api/auth/capabilities')return response(capabilities);
  if(path==='/api/auth/security'){
    if(holdSecurity){holdSecurity=false;return new Promise(resolve=>{releaseSecurity=resolve;});}
    return response(security());
  }
  if(path==='/api/auth/password/reset/request')return response({message:'If recovery is available for this address, instructions may arrive shortly.'},202);
  if(path==='/api/auth/verification/request')return response({message:'If verification is available, instructions may arrive shortly.'},202);
  if(path==='/api/auth/password/change'){
    if(holdPassword){holdPassword=false;return new Promise(resolve=>{releasePassword=resolve;});}
    if(scenario==='wrong-password')return response({error:{message:'The current password is incorrect.'}},400);
    if(scenario==='expired-password')return response({error:{message:'Sign in to continue.'}},401);
    session=null;return response({password_changed:true,sign_in_required:true});
  }
  return response({},404);
};
const context=vm.createContext({document,fetch,console,FormData:class{}});
vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),context);
const run=source=>vm.runInContext(source,context);
const flush=async()=>{for(let i=0;i<12;i++)await new Promise(resolve=>setImmediate(resolve));};
async function openSecurity(){get('account-security-panel').open=true;get('account-security-panel').events.toggle();await flush();}
async function switchBob(){
  await get('sign-out').events.click();
  get('email').value='bob@example.com';get('password').value='Example-login-password-42!';
  await get('auth-form').events.submit({preventDefault(){}});
}
(async()=>{
  await flush();
  if(scenario.startsWith('recovery')){
    get('recovery-panel').open=true;get('recovery-panel').events.toggle();await flush();
    get('recovery-email').value='alice@example.com';
    await get('recovery-request-form').events.submit({preventDefault(){}});
  }else{
    await openSecurity();
    if(scenario==='verification-request')await get('request-verification').events.click();
    if(['password-change','wrong-password','expired-password','stale-password'].includes(scenario)){
      get('current-password').value='Example-current-password-42!';
      get('new-password').value='Example-new-password-42!';
      get('confirm-new-password').value='Example-new-password-42!';
      if(scenario==='stale-password')holdPassword=true;
      const pending=get('password-change-form').events.submit({preventDefault(){}});
      if(scenario==='stale-password'){
        await flush();await switchBob();
        releasePassword(response({password_changed:true,sign_in_required:true}));
      }
      await pending;
    }
    if(scenario==='stale-security'){
      get('current-password').value='Alice-private-draft';
      get('new-password').value='Alice-private-new-password';
      get('confirm-new-password').value='Alice-private-new-password';
      holdSecurity=true;get('account-security-panel').events.toggle();await flush();
      await switchBob();releaseSecurity(response({...security(),email_verified:true}));
    }
  }
  await flush();
  console.log(JSON.stringify({
    user:run('state.user?.id ?? null'),requests,
    recoveryHidden:get('recovery-request-form').hidden,recoveryCapability:get('recovery-capability').textContent,recoveryMessage:get('recovery-message').textContent,
    verifyHidden:get('request-verification').hidden,passwordHidden:get('password-change-form').hidden,
    verifiedStatus:get('security-verified-status').textContent,securityMessage:get('security-message').textContent,delivery:get('security-delivery').textContent,
    authMessage:get('auth-message').textContent,passwords:['current-password','new-password','confirm-new-password'].map(id=>get(id).value)
  }));
})().catch(error=>{console.error(error);process.exitCode=1;});
"""

LANDING = r"""
const fs=require('node:fs'),vm=require('node:vm');
class Element{constructor(){this.value='';this.textContent='';this.events={};this.disabled=false;this.hidden=false;}addEventListener(name,fn){this.events[name]=fn;}}
const action=process.argv[2],scenario=process.argv[3];
const token='A'.repeat(43),csrf='C'.repeat(43),requests=[],nodes=new Map();
for(const id of ['lifecycle-form','lifecycle-submit','lifecycle-message','reset-password','reset-password-confirm'])nodes.set(id,new Element());
const document={body:{dataset:{action}},getElementById(id){return nodes.get(id)||null;}};
const window={location:{hash:scenario==='missing'?'':'#token='+token,pathname:'/account/'+action,search:''},events:{},addEventListener(name,fn){this.events[name]=fn;},history:{replaceState(_a,_b,url){window.location.hash='';window.location.search='';window.replaced=url;}}};
const fetch=async(path,options={})=>{
  if(window.location.hash)throw new Error('Fragment remained visible when fetching');
  requests.push({path,method:options.method||'GET',body:options.body?JSON.parse(options.body):null});
  if(path==='/api/auth/capabilities')return {ok:scenario!=='unavailable',json:async()=>({csrf_token:csrf})};
  if(scenario==='expired')return{ok:false,json:async()=>({error:{message:'This link is no longer available.'}})};
  return{ok:true,json:async()=>action==='verify'?{email_verified:true}:{password_changed:true,sign_in_required:true}};
};
const context=vm.createContext({window,document,fetch,URLSearchParams,console});
vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),context);
const flush=async()=>{for(let i=0;i<8;i++)await new Promise(resolve=>setImmediate(resolve));};
(async()=>{
  await flush();const postsBeforeClick=requests.filter(request=>request.method==='POST').length;
  nodes.get('reset-password').value='Example-reset-password-42!';
  nodes.get('reset-password-confirm').value=scenario==='mismatch'?'Different-reset-password-42!':'Example-reset-password-42!';
  if(scenario==='pagehide')window.events.pagehide();
  if(['success','expired','mismatch','pagehide'].includes(scenario))await nodes.get('lifecycle-form').events.submit({preventDefault(){}});
  await flush();
  console.log(JSON.stringify({requests,postsBeforeClick,url:window.replaced,hash:window.location.hash,disabled:nodes.get('lifecycle-submit').disabled,hidden:nodes.get('lifecycle-form').hidden,message:nodes.get('lifecycle-message').textContent,password:nodes.get('reset-password').value,confirmation:nodes.get('reset-password-confirm').value}));
})().catch(error=>{console.error(error);process.exitCode=1;});
"""


def run_browser(script, asset, *arguments):
    result = subprocess.run(
        ["node", "-e", script, str(ASSETS / asset), *arguments],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


def posts(result, path):
    return [request for request in result["requests"] if request["path"] == path and request["method"] == "POST"]


def test_unconfigured_recovery_is_explicit_and_does_not_claim_delivery():
    result = run_browser(MAIN, "app.js", "recovery-disabled")
    assert result["recoveryHidden"] is True
    assert "not configured" in result["recoveryCapability"]
    assert not posts(result, "/api/auth/password/reset/request")
    assert "sent" not in result["recoveryMessage"].lower()


def test_recovery_request_uses_generic_non_enumerating_response():
    result = run_browser(MAIN, "app.js", "recovery-configured")
    assert result["recoveryHidden"] is False
    assert posts(result, "/api/auth/password/reset/request")[0]["body"] == {"email": "alice@example.com"}
    assert "If recovery is available" in result["recoveryMessage"]
    assert "may arrive" in result["recoveryMessage"]


@pytest.mark.parametrize("scenario,verified", [("already-verified", True), ("security-disabled", False)])
def test_security_status_and_unconfigured_delivery_are_honest(scenario, verified):
    result = run_browser(MAIN, "app.js", scenario)
    assert result["user"] == "alice"
    assert result["verifyHidden"] is True
    assert result["passwordHidden"] is False
    if verified:
        assert result["verifiedStatus"] == "Email verified."
    else:
        assert "not configured" in result["delivery"]


def test_verification_request_carries_csrf_without_exposing_nonce():
    result = run_browser(MAIN, "app.js", "verification-request")
    assert posts(result, "/api/auth/verification/request")[0]["body"] == {"csrf_token": "C" * 43}
    assert "may arrive" in result["securityMessage"]
    assert "C" * 43 not in result["securityMessage"]


def test_successful_password_change_clears_private_drafts_and_returns_to_sign_in():
    result = run_browser(MAIN, "app.js", "password-change")
    submitted = posts(result, "/api/auth/password/change")[0]["body"]
    assert set(submitted) == {"current_password", "new_password", "csrf_token"}
    assert submitted["csrf_token"] == "C" * 43
    assert result["passwords"] == ["", "", ""]
    assert result["user"] is None
    assert result["authMessage"] == "Password changed. Sign in with your new password."


def test_wrong_current_password_does_not_clear_an_authenticated_workspace():
    result = run_browser(MAIN, "app.js", "wrong-password")
    assert result["user"] == "alice"
    assert result["passwords"] == ["", "", ""]
    assert result["securityMessage"] == "The current password is incorrect."


def test_expired_password_change_session_clears_private_workspace():
    result = run_browser(MAIN, "app.js", "expired-password")
    assert result["user"] is None
    assert result["passwords"] == ["", "", ""]


@pytest.mark.parametrize("scenario", ["stale-password", "stale-security"])
def test_old_account_security_responses_do_not_overwrite_a_new_session(scenario):
    result = run_browser(MAIN, "app.js", scenario)
    assert result["user"] == "bob"
    assert result["passwords"] == ["", "", ""]
    assert result["verifiedStatus"] == "Open this panel to check email verification."
    assert result["securityMessage"] == ""
    assert "Password changed" not in result["authMessage"]


@pytest.mark.parametrize("action", ["verify", "reset"])
def test_landing_scrubs_fragment_and_waits_for_explicit_post_with_csrf(action):
    result = run_browser(LANDING, "lifecycle.js", action, "success")
    assert result["url"] == "/account/" + action
    assert result["hash"] == ""
    assert result["postsBeforeClick"] == 0
    sent = [request for request in result["requests"] if request["method"] == "POST"]
    assert len(sent) == 1
    assert sent[0]["body"]["token"] == "A" * 43
    assert sent[0]["body"]["csrf_token"] == "C" * 43
    assert ("new_password" in sent[0]["body"]) is (action == "reset")
    assert result["disabled"] is True
    assert result["hidden"] is True
    assert result["password"] == result["confirmation"] == ""
    assert "A" * 43 not in result["message"]


@pytest.mark.parametrize("scenario", ["missing", "unavailable", "mismatch", "pagehide"])
def test_incomplete_or_unprepared_reset_forms_cannot_submit(scenario):
    result = run_browser(LANDING, "lifecycle.js", "reset", scenario)
    assert not [request for request in result["requests"] if request["method"] == "POST"]
    if scenario == "mismatch":
        assert "do not match" in result["message"]
    else:
        assert result["disabled"] is True


def test_expired_link_does_not_claim_password_reset():
    result = run_browser(LANDING, "lifecycle.js", "reset", "expired")
    assert result["message"] == "This link is no longer available."
    assert result["password"] == result["confirmation"] == ""
    assert result["hidden"] is False


@pytest.mark.parametrize("action", ["reset", "change"])
def test_credential_action_response_cannot_clear_an_unrelated_new_browser_session(action):
    pytest.importorskip("fastapi")
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from fastapi.testclient import TestClient

    from matters.cloud.lifecycle_routes import create_lifecycle_router

    entered = threading.Event()
    release = threading.Event()

    class Lifecycle:
        def capabilities(self):
            return {"email_delivery": True, "email_verification": True, "password_reset": True, "password_change": True}

        def reset_password(self, token, _password):
            assert token == "A" * 43
            entered.set()
            assert release.wait(timeout=10)
            return {"password_changed": True, "sign_in_required": True}

        def change_password(self, actor, session, _current_password, _new_password):
            assert actor == "alice"
            assert session == "alice-session"
            entered.set()
            assert release.wait(timeout=10)
            return {"password_changed": True, "sign_in_required": True}

    app = FastAPI()

    def authenticated_user(request: Request):
        return {"id": request.cookies["matters_session"].removesuffix("-session")}

    @app.get("/test/login/{name}")
    def login(name: str):
        response = JSONResponse({"signed_in": name})
        response.set_cookie("matters_session", name + "-session", path="/", httponly=True, samesite="strict")
        return response

    app.include_router(create_lifecycle_router(
        Lifecycle(), authenticated_user, public_origin="http://localhost", secure_cookies=False,
    ))
    with TestClient(app, base_url="http://localhost") as client, ThreadPoolExecutor(max_workers=1) as executor:
        client.get("/test/login/" + ("bob" if action == "reset" else "alice"))
        csrf = client.get("/api/auth/capabilities").json()["csrf_token"]
        payload = {"new_password": "Example-new-password-42!", "csrf_token": csrf}
        if action == "reset":
            payload["token"] = "A" * 43
        else:
            payload["current_password"] = "Example-current-password-42!"
        path = "/api/auth/password/reset/confirm" if action == "reset" else "/api/auth/password/change"
        future = executor.submit(client.post, path, headers={"Origin": "http://localhost"}, json=payload)
        try:
            assert entered.wait(timeout=10)
            if action == "change":
                # The original request authenticated Alice, but Bob signs in
                # before its delayed response reaches the browser cookie jar.
                client.get("/test/login/bob")
            assert client.cookies.get("matters_session") == "bob-session"
        finally:
            release.set()
        response = future.result(timeout=10)
        assert response.status_code == 200, response.text
        assert not any(value.startswith("matters_session=") for value in response.headers.get_list("set-cookie"))
        assert client.cookies.get("matters_session") == "bob-session"
