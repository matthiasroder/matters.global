"""Native condition editing and delayed-response isolation on owned fixtures."""

import pytest

from test_cloud_minimal_ui import native_workspace, run_native


COMMON = r"""
const fs=require('node:fs'),assert=require('node:assert/strict'),{chromium}=require('playwright');
const c=JSON.parse(fs.readFileSync(0,'utf8'));
const tick=page=>page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
(async()=>{
 const browser=await chromium.launch({executablePath:c.executable,headless:true,args:['--no-sandbox']});
 try {
  const context=await browser.newContext({viewport:{width:c.width||1440,height:844},reducedMotion:'reduce'});
  const page=await context.newPage();page.setDefaultTimeout(10000);
  const errors=[];page.on('pageerror',error=>errors.push(error.message));
  const open=async id=>{const panel=page.locator('#'+id);if(!await panel.evaluate(node=>node.open))await panel.locator(':scope > summary').click();};
  const login=async email=>{
   await page.locator('#email').fill(email);await page.locator('#password').fill(c.password);await page.locator('#auth-submit').click();
   await page.locator('#workspace-view').waitFor({state:'visible'});await page.locator('#visual-canvas canvas').waitFor({state:'visible'});
  };
  const graph=async title=>{
   await open('graph-menu');await page.locator('#graph-list .graph-link').filter({hasText:title}).click();
   await page.waitForFunction(title=>document.getElementById('graph-title').textContent===title,title);await tick(page);
  };
  const matter=async id=>{
   await page.locator('#toggle-list').click();
   await page.locator('#matters-list .matter-card[data-matter-id="'+id+'"] .matter-select').click();
   await page.locator('#matter-inspector').waitFor({state:'visible'});
  };
  const submit=async form=>page.locator('#'+form+' button[type=submit]').click();
  const conditions=page.locator('#matter-condition-fields input[name=condition]');
  const newName=page.locator('#new-matter-form input[name=name]');
  const addition=page.locator('#add-condition-form input[name=condition]');
  const operation=()=>page.waitForResponse(response=>response.url().endsWith('/operations')&&response.request().method()==='POST');
  const snapshot=async id=>(await context.request.get(c.origin+'/api/graphs/'+id)).json();
  await page.goto(c.origin);await login('alice@example.com');await graph('Shared example');
  /* SCENARIO */
  assert.deepEqual(errors,[],'Browser script error');
  console.log(JSON.stringify({scenario:c.scenario,width:c.width,passed:true,errors}));
 }finally{await browser.close();}
})().catch(error=>{console.error(error.stack);process.exitCode=1;});
"""


EDITING = r"""
  await open('new-matter-panel');
  assert.equal(await conditions.count(),1);
  assert.equal(await page.getByRole('button',{name:'Remove condition 1',exact:true}).isDisabled(),true);
  await newName.fill('Release checks');await conditions.first().fill('   ');
  let requests=0;page.on('request',request=>{if(request.url().endsWith('/operations')&&request.method()==='POST')requests++;});
  await submit('new-matter-form');await tick(page);
  assert.equal(requests,0,'Whitespace-only conditions were submitted');
  assert.equal(await newName.inputValue(),'Release checks');
  await conditions.first().fill('Goal is written in one sentence');
  await page.locator('#add-matter-condition').click();
  await page.getByRole('textbox',{name:'Condition 2',exact:true}).fill('Remove this draft');
  await page.locator('#add-matter-condition').click();
  await page.getByRole('textbox',{name:'Condition 3',exact:true}).fill('First version is usable by one real user');
  await page.getByRole('button',{name:'Remove condition 2',exact:true}).click();
  assert.equal(await conditions.count(),2);
  assert.equal(await page.getByRole('textbox',{name:'Condition 2',exact:true}).inputValue(),'First version is usable by one real user');
  await page.locator('#add-matter-condition').click();
  await conditions.nth(2).fill('   ');
  const expected=['Goal is written in one sentence','First version is usable by one real user'];
  await submit('new-matter-form');await tick(page);
  assert.equal(requests,0,'A visible blank condition was silently omitted');
  assert.deepEqual(await conditions.evaluateAll(nodes=>nodes.map(node=>node.value)),[...expected,'   ']);
  assert.equal(await conditions.nth(2).evaluate(node=>document.activeElement===node),true,'Blank condition was not focused');
  await page.getByRole('button',{name:'Remove condition 3',exact:true}).click();
  let refused=false,createPayload;
  await page.route('**/api/graphs/*/operations',async route=>{
   const body=route.request().postDataJSON();
   if(body.operation==='create_matter'&&!refused){refused=true;createPayload=body.payload;await route.fulfill({status:409,contentType:'application/json',body:JSON.stringify({error:{message:'Example conflict',code:'conflict'}})});}
   else await route.continue();
  });
  let saved=operation();await submit('new-matter-form');assert.equal((await saved).status(),409);
  await page.locator('#new-matter-form button[type=submit]').waitFor({state:'visible'});
  await page.waitForFunction(()=>!document.querySelector('#new-matter-form button[type=submit]').disabled);
  assert.deepEqual(createPayload.conditions,expected.map(label=>({label,truth:false})),'Created condition order changed');
  assert.deepEqual(await conditions.evaluateAll(nodes=>nodes.map(node=>node.value)),expected,'Conflict cleared condition drafts');
  assert.equal(await newName.inputValue(),'Release checks');
  saved=operation();await submit('new-matter-form');assert.equal((await saved).status(),200);
  await page.locator('#matter-inspector .matter-card[data-matter-id=release_checks]').waitFor();
  assert.equal(await conditions.count(),1,'Success did not reset creation rows');
  assert.equal(await conditions.first().inputValue(),'');
  let persisted=await snapshot(c.graph_id);
  assert.deepEqual(persisted.state.conditions.release_checks,expected.map(label=>({label,truth:false})));
  const first=page.locator('#matter-inspector').getByRole('checkbox',{name:expected[0],exact:true});
  saved=operation();await first.check();assert.equal((await saved).status(),200);
  await page.waitForFunction(()=>{const first=document.querySelector('#matter-inspector input[type=checkbox]');return first?.checked&&!first.disabled;});
  await open('add-condition-panel');
  const beforeBlankAppend=requests;
  await addition.fill('   ');await submit('add-condition-form');await tick(page);
  assert.equal(requests,beforeBlankAppend,'Whitespace-only append reached the server');
  await addition.fill('Another repository example check');
  const appendFailures=[409,503];
  await page.unroute('**/api/graphs/*/operations');
  await page.route('**/api/graphs/*/operations',async route=>{
   if(route.request().postDataJSON().operation==='add_condition'&&appendFailures.length){await route.fulfill({status:appendFailures.shift(),contentType:'application/json',body:JSON.stringify({error:{message:'Example save failure',code:'unavailable'}})});}
   else await route.continue();
  });
  for(const status of [409,503]){
   saved=operation();await submit('add-condition-form');assert.equal((await saved).status(),status);
   await page.waitForFunction(()=>!document.querySelector('#add-condition-form button[type=submit]').disabled);
   assert.equal(await addition.inputValue(),'Another repository example check','Failed append cleared its draft');
   assert.equal(await first.isChecked(),true);
  }
  saved=operation();await submit('add-condition-form');assert.equal((await saved).status(),200);
  await page.waitForFunction(()=>document.querySelectorAll('#matter-inspector input[type=checkbox]').length===3);
  persisted=await snapshot(c.graph_id);
  assert.deepEqual(persisted.state.conditions.release_checks,[{label:expected[0],truth:true},{label:expected[1],truth:false},{label:'Another repository example check',truth:false}],'Adding a condition replaced existing truth or order');
  assert.equal(await addition.inputValue(),'');
  await open('add-condition-panel');await addition.fill('Private selected matter draft');
  await page.locator('#toggle-list').click();
  assert.equal(await page.locator('#add-condition-panel').isVisible(),false,'List exposed selected-matter addition');
  await page.locator('#matters-list .matter-card[data-matter-id=ship_first_version] .matter-select').click();
  await open('add-condition-panel');assert.equal(await addition.inputValue(),'','Selected-matter draft crossed node selection');
  await addition.fill('Private graph draft');await graph('Full example');await matter('define_project_goal');
  await open('add-condition-panel');assert.equal(await addition.inputValue(),'','Selected-matter draft crossed graph navigation');
  await page.reload();await page.locator('#visual-canvas canvas').waitFor();await graph('Shared example');await matter('release_checks');
  assert.equal(await page.locator('#matter-inspector input[type=checkbox]').count(),3);
  assert.equal(await page.locator('#matter-inspector').getByRole('checkbox',{name:expected[0],exact:true}).isChecked(),true);
  await page.screenshot({path:c.screenshot_directory+'/conditions-'+c.width+'.png'});
  await open('account-menu');await page.locator('#sign-out').click();await page.locator('#auth-view').waitFor();
  await login('bob@example.com');await graph('Shared example');await matter('release_checks');
  assert.equal(await page.locator('#add-condition-panel').evaluate(node=>node.hidden),true,'Viewer received Add condition');
  assert.equal(await addition.inputValue(),'','Selected-matter draft crossed an account change');
  for(const checkbox of await page.locator('#matter-inspector input[type=checkbox]').all())assert.equal(await checkbox.isDisabled(),true);
"""


@pytest.mark.parametrize("width", [1440, 390])
def test_multiple_conditions_create_append_persist_and_preserve_drafts(native_workspace, width):
    result = run_native(native_workspace, COMMON.replace("/* SCENARIO */", EDITING), width=width, scenario="editing")
    assert result == {"scenario": "editing", "width": width, "passed": True, "errors": []}


PENDING = r"""
  const creating=c.scenario.startsWith('create');
  if(!creating){await graph('Full example');await matter('define_project_goal');}
  const gates=[],ready=[];
  const waiting=[0,1].map(index=>new Promise(resolve=>{ready[index]=resolve;}));
  let intercepted=0;
  await page.route('**/api/graphs/*/operations',async route=>{
   const index=intercepted++;
   if(index>1){await route.continue();return;}
   const gate=new Promise(resolve=>{gates[index]=resolve;});
   if(index===0&&c.scenario!=='append-graph'){ready[index]();await gate;}
   const response=await route.fetch();
   if(index===0&&c.scenario==='append-graph'){ready[index]();await gate;}
   if(index===1){ready[index]();await gate;}
   await route.fulfill({response});
  });
  if(creating){
   await open('new-matter-panel');await newName.fill('Pending original');
   await conditions.first().fill('Original example condition');await page.locator('#add-matter-condition').click();await conditions.nth(1).fill('Second original condition');
   await submit('new-matter-form');
  }else{
   await open('add-condition-panel');await addition.fill('Original selected matter condition');await submit('add-condition-form');
  }
  await waiting[0];
  if(c.scenario==='create-account'){
   await open('account-menu');await page.locator('#sign-out').click();await page.locator('#auth-view').waitFor();
   await login('bob@example.com');await graph('Bob personal example');
  }else if(creating){await graph('Full example');if(c.scenario==='create-return')await graph('Shared example');}
  else if(c.scenario==='append-graph'){await graph('Shared example');await matter('ship_first_version');}
  else if(c.scenario==='append-reopen'){await page.locator('#close-inspector').click();await matter('define_project_goal');}
  else {await matter('ship_first_version');}
  if(creating){
   await open('new-matter-panel');assert.equal(await conditions.count(),1,'Earlier condition rows crossed scope');
   assert.equal(await conditions.first().inputValue(),'','Earlier condition text crossed scope');
   await newName.fill('Pending current');await conditions.first().fill('Current example condition');
   await page.locator('#add-matter-condition').click();await conditions.nth(1).fill('Second current condition');
   await submit('new-matter-form');
  }else{
   await open('add-condition-panel');assert.equal(await addition.inputValue(),'','Earlier selected matter draft crossed scope');
   await addition.fill('Current selected matter condition');await submit('add-condition-form');
  }
  await waiting[1];
  const form=creating?'new-matter-form':'add-condition-form';
  const oldCompletion=operation();gates[0]();await oldCompletion;await tick(page);
  assert.equal(await page.locator('#'+form+' button[type=submit]').isDisabled(),true,'Earlier request enabled the current pending form');
  if(creating){
   assert.equal(await newName.inputValue(),'Pending current','Earlier create cleared the current name');
   assert.deepEqual(await conditions.evaluateAll(nodes=>nodes.map(node=>node.value)),['Current example condition','Second current condition'],'Earlier create cleared current condition rows');
   assert.equal(await page.locator('#add-matter-condition').isDisabled(),true);
  }else{
   assert.equal(await addition.inputValue(),'Current selected matter condition','Earlier append cleared the current selected-matter draft');
   assert.equal(await page.locator('#matters-list .matter-card').getAttribute('data-matter-id'),c.scenario==='append-reopen'?'define_project_goal':'ship_first_version','Earlier append replaced the current selection');
  }
  const currentCompletion=operation();gates[1]();assert.equal((await currentCompletion).status(),200);
  if(creating){
   await page.locator('#matter-inspector .matter-card[data-matter-id=pending_current]').waitFor();
   assert.equal(await conditions.count(),1);assert.equal(await conditions.first().inputValue(),'');
   const id=c.scenario==='create-account'?c.private_graph_id:c.scenario==='create-return'?c.graph_id:c.other_graph_id;
   const current=await snapshot(id);
   assert.deepEqual(current.state.conditions.pending_current,[{label:'Current example condition',truth:false},{label:'Second current condition',truth:false}]);
  }else{
   await page.waitForFunction(()=>document.querySelectorAll('#matter-inspector input[type=checkbox]').length===2);
   await page.waitForFunction(()=>document.querySelector('#add-condition-form input[name=condition]').value===''&&!document.querySelector('#add-condition-form button[type=submit]').disabled);
   assert.equal(await addition.inputValue(),'');
   const current=await snapshot(c.scenario==='append-graph'?c.graph_id:c.other_graph_id);
   if(c.scenario==='append-reopen'){
    assert.deepEqual(current.state.conditions.define_project_goal,[{label:'Goal is written in one sentence',truth:true},{label:'Current selected matter condition',truth:false}]);
    assert.deepEqual(current.state.conditions.ship_first_version,[{label:'First version is usable by one real user',truth:false}]);
   }else{
    if(c.scenario!=='append-graph')assert.deepEqual(current.state.conditions.define_project_goal,[{label:'Goal is written in one sentence',truth:true}],'An obsolete append mutated a different selected matter');
    assert.deepEqual(current.state.conditions.ship_first_version,[{label:'First version is usable by one real user',truth:false},{label:'Current selected matter condition',truth:false}]);
   }
  }
"""


@pytest.mark.parametrize("scenario", ["create-graph", "create-account", "create-return", "append-matter", "append-graph", "append-reopen"])
def test_old_condition_requests_cannot_clear_or_enable_a_new_pending_draft(native_workspace, scenario):
    result = run_native(native_workspace, COMMON.replace("/* SCENARIO */", PENDING), width=390, scenario=scenario)
    assert result == {"scenario": scenario, "width": 390, "passed": True, "errors": []}
