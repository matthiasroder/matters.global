"""Native-browser checks of the compact hosted graph workspace.

The server and accounts belong to each test. Graph content comes from the
repository example; no deployed installation or personal account is used.
"""

import asyncio
import copy
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import threading
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("argon2")
pytest.importorskip("email_validator")

from matters.cloud.app import create_app


PASSWORD = "Matters-ui-example-password-42!"
EXAMPLE = Path(__file__).parents[1] / "examples" / "matters.example.json"


@pytest.fixture
def native_workspace(tmp_path):
    uvicorn = pytest.importorskip("uvicorn")
    node = shutil.which("node")
    executable = os.environ.get("MATTERS_BROWSER_EXECUTABLE", "/usr/bin/chromium")
    if node is None or not Path(executable).is_file():
        pytest.skip("Native workspace checks require Node and Chromium")
    if subprocess.run([node, "-e", "require('playwright')"], capture_output=True, timeout=10).returncode:
        pytest.skip("Native workspace checks require Playwright")
    owned = socket.socket()
    owned.bind(("127.0.0.1", 0))
    owned.listen(128)
    origin = "http://127.0.0.1:" + str(owned.getsockname()[1])
    app = create_app(
        "sqlite:///" + str(tmp_path / "workspace.db"), secure_cookies=False,
        public_origin=origin, disable_signup=True,
    )
    alice = app.state.accounts.register("alice@example.com", PASSWORD)
    bob = app.state.accounts.register("bob@example.com", PASSWORD)
    app.state.accounts.register("empty@example.com", PASSWORD)
    example = json.loads(EXAMPLE.read_text())
    # One real example matter makes a native canvas hit independent of viewport
    # size and the renderer's projection. The full example remains available.
    single = copy.deepcopy(example)
    single["matters"] = ["ship_first_version"]
    single["conditions"] = {"ship_first_version": single["conditions"]["ship_first_version"]}
    single["dependencies"] = []
    graph = app.state.graphs.create_graph(alice["id"], "Shared example", state=single)
    other = app.state.graphs.create_graph(alice["id"], "Full example", state=example)
    app.state.graphs.set_member(alice["id"], graph["graph_id"], bob["id"], "viewer")
    private = app.state.graphs.create_graph(bob["id"], "Bob personal example", "personal")
    server = uvicorn.Server(uvicorn.Config(app, access_log=False, log_level="error"))
    thread = threading.Thread(target=lambda: asyncio.run(server.serve(sockets=[owned])), daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert server.started
        yield {
            "node": node, "executable": executable, "origin": origin,
            "password": PASSWORD, "graph_id": graph["graph_id"],
            "other_graph_id": other["graph_id"], "private_graph_id": private["graph_id"],
            "screenshot_directory": str(tmp_path),
        }
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        if thread.is_alive():
            server.force_exit = True
            thread.join(timeout=5)
        owned.close()
        assert not thread.is_alive()


def run_native(config, script, **options):
    checked = subprocess.run(
        [config["node"], "-e", script], input=json.dumps({**config, **options}),
        text=True, capture_output=True, timeout=75,
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr
    return json.loads(checked.stdout)


NATIVE = r"""
const fs = require('node:fs'), assert = require('node:assert/strict');
const {chromium} = require('playwright');
const c = JSON.parse(fs.readFileSync(0, 'utf8'));
(async () => {
  const browser = await chromium.launch({executablePath:c.executable,headless:true,args:['--no-sandbox']});
  try {
    const context = await browser.newContext({viewport:{width:c.width,height:844},reducedMotion:'reduce'});
    const page = await context.newPage();
    page.setDefaultTimeout(10000);
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    const hidden = async id => {
      await page.locator('#'+id).waitFor({state:'hidden'});
      assert.equal(await page.locator('#'+id).isVisible(),false,id+' remained visible');
    };
    const openDetails = async id => {
      const details = page.locator('#'+id);
      if (!await details.evaluate(node=>node.open)) await details.locator(':scope > summary').click();
    };
    const openAncestors = async locator => {
      for (const details of await locator.locator('xpath=ancestor::details').all()) {
        if (!await details.evaluate(node=>node.open)) await details.locator(':scope > summary').click();
      }
    };
    const selectGraph = async title => {
      await openDetails('graph-menu');
      await page.locator('#graph-list .graph-link').filter({hasText:title}).click();
      await page.waitForFunction(title=>document.getElementById('graph-title').textContent===title,title);
      await page.locator('#visual-canvas canvas').waitFor();
    };
    const nativeNode = async () => {
      const canvas = page.locator('#visual-canvas canvas');
      const rect = await canvas.boundingBox();
      assert.ok(rect && rect.width>100 && rect.height>200,'Canvas was not usable');
      // This fixture has one actual example node, projected at the canvas centre.
      await canvas.click({position:{x:rect.width/2,y:rect.height/2}});
      await page.locator('#matter-inspector').waitFor({state:'visible'});
      assert.equal(await page.locator('#matter-inspector').evaluate(node=>document.activeElement===node),true,'Inspector did not receive keyboard focus');
      assert.equal(await page.locator('#matters-list .matter-card').count(),1);
      assert.equal(await page.locator('#matters-list .matter-card').getAttribute('data-matter-id'),'ship_first_version');
    };
    await page.goto(c.origin);
    await page.locator('#auth-form').getByLabel('Email',{exact:true}).fill(c.viewer?'bob@example.com':'alice@example.com');
    await page.locator('#auth-form').getByLabel('Password',{exact:true}).fill(c.password);
    await page.locator('#auth-submit').click();
    await page.locator('#workspace-view').waitFor({state:'visible'});
    await page.locator('#visual-canvas canvas').waitFor({state:'visible'});
    await selectGraph('Shared example');
    for (const id of ['matter-inspector','new-matter-form','new-graph-form','invitation-form','password-change-form']) await hidden(id);
    assert.equal(await page.locator('.sidebar, .stat-grid, footer, .intro').count(),0,'Permanent chrome remained');
    const dimensions = await page.locator('#visual-canvas').evaluate(node=>{
      const rect=node.getBoundingClientRect();
      return {width:rect.width,height:rect.height,bottom:rect.bottom,screenWidth:innerWidth,screenHeight:innerHeight,
        toolbarHeight:document.querySelector('.visual-toolbar').getBoundingClientRect().height,
        headerHeight:document.querySelector('.topbar').getBoundingClientRect().height,
        scrollWidth:document.scrollingElement.scrollWidth,scrollHeight:document.scrollingElement.scrollHeight};
    });
    await page.screenshot({path:c.screenshot_directory+'/'+(c.viewer?'viewer-':'owner-')+c.width+'.png'});
    assert.ok(dimensions.width>=dimensions.screenWidth-24,'Canvas lost width to permanent controls');
    assert.ok(dimensions.height>=dimensions.screenHeight-dimensions.toolbarHeight-dimensions.headerHeight-2,'Canvas did not fill the remaining workspace: '+JSON.stringify(dimensions));
    assert.ok(dimensions.headerHeight+dimensions.toolbarHeight<=180,'Permanent controls consumed too much canvas space');
    assert.ok(dimensions.bottom<=dimensions.screenHeight+2,'Canvas extended below the viewport');
    assert.ok(dimensions.scrollWidth<=dimensions.screenWidth+1,'Workspace overflowed horizontally');
    assert.ok(dimensions.scrollHeight<=dimensions.screenHeight+2,'Default workspace required scrolling');
    await nativeNode();
    const condition = page.locator('#matter-inspector').getByRole('checkbox',{name:'First version is usable by one real user'});
    assert.equal(await condition.isChecked(),false);
    if (c.viewer) {
      assert.equal(await condition.isDisabled(),true,'Viewer received a condition editor');
      for (const id of ['new-matter-panel','sharing-panel','dependency-editor']) {
        assert.equal(await page.locator('#'+id).evaluate(node=>node.hidden),true,'Viewer received '+id);
      }
      await page.locator('#close-inspector').click();
      await hidden('matter-inspector');
      await selectGraph('Bob personal example');
      assert.equal(await page.locator('#new-matter-panel').isVisible(),true,'Viewer could not manage own graph');
      await selectGraph('Shared example');
      await hidden('new-matter-panel');
      await page.locator('#toggle-list').focus();
      await page.keyboard.press('Enter');
      await page.locator('#matter-inspector').waitFor({state:'visible'});
      assert.equal(await page.locator('#matter-inspector input[type=checkbox]').isDisabled(),true);
    } else {
      const saved = page.waitForResponse(response=>response.url().endsWith('/operations')&&response.request().method()==='POST');
      await condition.focus();
      await page.keyboard.press('Space');
      assert.equal((await saved).status(),200);
      await page.waitForFunction(()=>{
        const checkbox=document.querySelector('#matter-inspector input[type=checkbox]');
        return checkbox?.checked===true&&!checkbox.disabled;
      });
      assert.equal(await condition.evaluate(node=>document.activeElement===node),true,'Saving a condition lost keyboard focus');
      await page.keyboard.press('Escape');
      await hidden('matter-inspector');
      assert.equal(await page.locator('#visual-canvas canvas').evaluate(node=>document.activeElement===node),true,'Inspector did not restore canvas focus');
      await page.reload();
      await page.locator('#workspace-view').waitFor({state:'visible'});
      await page.locator('#visual-canvas canvas').waitFor({state:'visible'});
      await selectGraph('Shared example');
      await hidden('matter-inspector');
      await nativeNode();
      assert.equal(await condition.isChecked(),true,'Condition did not survive reload');
      await page.locator('#close-inspector').click();
      await page.locator('#toggle-list').focus();
      await page.keyboard.press('Enter');
      await page.locator('#matter-inspector').waitFor({state:'visible'});
      await page.keyboard.press('Escape');
      await hidden('matter-inspector');
      assert.equal(await page.locator('#toggle-list').evaluate(node=>document.activeElement===node),true,'List dismissal lost keyboard focus');
      await openDetails('graph-menu');
      await page.keyboard.press('Escape');
      assert.equal(await page.locator('#graph-menu').evaluate(node=>node.open),false);
      assert.equal(await page.locator('#graph-menu > summary').evaluate(node=>document.activeElement===node),true,'Menu dismissal lost keyboard focus');
      await openDetails('graph-menu');
      await page.locator('#visual-canvas canvas').click({position:{x:2,y:20}});
      assert.equal(await page.locator('#graph-menu').evaluate(node=>node.open),false,'Outside click left menu open');
      await openAncestors(page.locator('#sharing-panel'));
      await openDetails('sharing-panel');
      await page.waitForFunction(()=>document.getElementById('sharing-owner').textContent.includes('alice@example.com'));
      assert.match(await page.locator('#sharing-members').innerText(),/bob@example.com/,'Owner sharing controls were unreachable');
      await openAncestors(page.locator('#dependency-editor'));
      await openDetails('dependency-editor');
      assert.equal(await page.locator('#dependency-form select[name=source] option').count(),1,'Dependency controls lost the current graph');
      await openDetails('new-matter-panel');
      await page.locator('#new-matter-form input[name=name]').fill('Define project goal');
      await page.locator('#new-matter-form input[name=condition]').fill('Goal is written in one sentence');
      const added = page.waitForResponse(response=>response.url().endsWith('/operations')&&response.request().method()==='POST');
      await page.locator('#new-matter-form button[type=submit]').click();
      assert.equal((await added).status(),200);
      await hidden('new-matter-form');
      const persisted = await (await context.request.get(c.origin+'/api/graphs/'+c.graph_id)).json();
      assert.deepEqual(persisted.state.matters.sort(),['define_project_goal','ship_first_version']);
      assert.equal(persisted.state.conditions.ship_first_version[0].truth,true);
      await openAncestors(page.locator('#new-graph-form'));
      await page.locator('#new-graph-form input[name=title]').fill('Created example');
      await page.locator('#new-graph-form select[name=kind]').selectOption('personal');
      await page.locator('#new-graph-form button[type=submit]').click();
      await page.waitForFunction(()=>document.getElementById('graph-title').textContent==='Created example');
      await hidden('matter-inspector');
      await hidden('new-graph-form');
      const catalog = await (await context.request.get(c.origin+'/api/graphs')).json();
      assert.ok(catalog.graphs.some(graph=>graph.title==='Created example'&&graph.kind==='personal'&&graph.role==='owner'));
      await page.locator('#add-first-matter').click();
      await page.locator('#new-matter-form input[name=name]').fill('Define project goal');
      await page.locator('#new-matter-form input[name=condition]').fill('Goal is written in one sentence');
      await page.locator('#new-matter-form button[type=submit]').click();
      await page.locator('#matter-inspector .matter-card').waitFor();
      const current = catalog.graphs.find(graph=>graph.title==='Created example');
      const firstMatter = await (await context.request.get(c.origin+'/api/graphs/'+current.graph_id)).json();
      assert.deepEqual(firstMatter.state.matters,['define_project_goal'],'Empty-state add did not save the first matter');
      await selectGraph('Full example');
      await page.locator('#toggle-list').click();
      assert.equal(await page.locator('#matters-list .matter-card').count(),2);
      await page.locator('#matters-list .matter-select').filter({hasText:'define project goal'}).click();
      assert.equal(await page.locator('#matters-list .matter-card').count(),1,'Selected inspector displayed unrelated matters');
      assert.equal(await page.locator('#matter-inspector').getByRole('checkbox',{name:'Goal is written in one sentence'}).isChecked(),true);
    }
    await page.keyboard.press('Escape');
    await hidden('matter-inspector');
    await openAncestors(page.locator('#account-security-panel'));
    await openDetails('account-security-panel');
    await page.locator('#password-change-form').waitFor({state:'visible'});
    assert.match(await page.locator('#security-delivery').innerText(),/not configured/);
    await page.locator('#sign-out').click();
    await page.locator('#auth-view').waitFor({state:'visible'});
    for(const id of ['workspace-view','account-controls','graph-menu','matter-inspector']) await hidden(id);
    assert.equal(await page.locator('#account-email').innerText(),'');
    assert.deepEqual(errors,[],'Browser script error');
    console.log(JSON.stringify({viewer:c.viewer,width:c.width,canvas:true,native_inspector:true,permissions:true,
      condition_persistence:!c.viewer,graph_creation:!c.viewer,keyboard_dismissal:true,account_controls:true,errors}));
    await context.close();
  } finally { await browser.close(); }
})().catch(error=>{console.error(error.stack);process.exitCode=1;});
"""


@pytest.mark.parametrize("width", [1440, 390])
@pytest.mark.parametrize("viewer", [False, True], ids=["owner", "viewer"])
def test_compact_workspace_native_editing_permissions_and_navigation(native_workspace, width, viewer):
    result = run_native(native_workspace, NATIVE, width=width, viewer=viewer)
    assert result == {
        "viewer": viewer, "width": width, "canvas": True,
        "native_inspector": True, "permissions": True,
        "condition_persistence": not viewer, "graph_creation": not viewer,
        "keyboard_dismissal": True, "account_controls": True, "errors": [],
    }


def test_mobile_empty_workspace_exposes_first_graph_and_matter_actions(native_workspace):
    result = run_native(native_workspace, r"""
const fs=require('node:fs'),assert=require('node:assert/strict'),{chromium}=require('playwright');
const c=JSON.parse(fs.readFileSync(0,'utf8'));
(async()=>{
 const browser=await chromium.launch({executablePath:c.executable,headless:true,args:['--no-sandbox']});
 try {
  const context=await browser.newContext({viewport:{width:390,height:844},reducedMotion:'reduce'});
  const page=await context.newPage();page.setDefaultTimeout(10000);
  const errors=[];page.on('pageerror',error=>errors.push(error.message));
  await page.goto(c.origin);
  await page.locator('#email').fill('empty@example.com');await page.locator('#password').fill(c.password);
  await page.locator('#auth-submit').click();
  await page.locator('#create-first-graph').waitFor({state:'visible'});
  await page.screenshot({path:c.screenshot_directory+'/empty-mobile.png'});
  await page.locator('#create-first-graph').click();
  await page.locator('#new-graph-form input[name=title]').fill('First example');
  await page.locator('#new-graph-form button[type=submit]').click();
  await page.locator('#add-first-matter').waitFor({state:'visible'});
  await page.locator('#new-graph-form').waitFor({state:'hidden'});
  await page.locator('#add-first-matter').click();
  await page.locator('#new-matter-form input[name=name]').fill('Define project goal');
  await page.locator('#new-matter-form input[name=condition]').fill('Goal is written in one sentence');
  await page.locator('#new-matter-form button[type=submit]').click();
  await page.locator('#matter-inspector .matter-card').waitFor({state:'visible'});
  assert.equal(await page.locator('#matter-inspector').getByRole('checkbox',{name:'Goal is written in one sentence'}).isEnabled(),true);
  const catalog=await(await context.request.get(c.origin+'/api/graphs')).json();
  assert.equal(catalog.graphs.length,1);assert.equal(catalog.graphs[0].title,'First example');
  const graph=await(await context.request.get(c.origin+'/api/graphs/'+catalog.graphs[0].graph_id)).json();
  assert.deepEqual(graph.state.matters,['define_project_goal']);
  await page.keyboard.press('Escape');await page.locator('#matter-inspector').waitFor({state:'hidden'});
  assert.equal(await page.locator('#new-matter-form').isVisible(),false);
  assert.ok(await page.evaluate(()=>document.scrollingElement.scrollWidth<=innerWidth+1));
  await page.screenshot({path:c.screenshot_directory+'/first-matter-mobile.png'});
  assert.deepEqual(errors,[]);
  console.log(JSON.stringify({first_graph:true,first_matter:true,mobile:true,errors}));
 }finally{await browser.close();}
})().catch(error=>{console.error(error.stack);process.exitCode=1;});
""")
    assert result == {"first_graph": True, "first_matter": True, "mobile": True, "errors": []}
