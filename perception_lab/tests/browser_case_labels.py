from playwright.sync_api import sync_playwright
from pathlib import Path
import json,argparse
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out;out.mkdir(parents=True,exist_ok=True)
with sync_playwright() as pw:
 b=pw.chromium.launch(executable_path='/usr/bin/google-chrome',headless=False,args=['--no-sandbox','--ignore-gpu-blocklist'])
 page=b.new_page(viewport={'width':1900,'height':1250});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
 base='http://127.0.0.1:9092';cases=page.request.get(base+'/cases.json').json();target=next(c for c in cases if c['kind']=='false_positive')['case_id']
 original=page.request.get(base+'/api/labels').json();old=original['items'].get(target,{'label':'original','note':''})
 try:
  page.goto(base+'/#'+target);page.wait_for_selector('#caseLabel')
  page.select_option('#caseLabel','gt_missing');page.fill('#labelNote','QA 临时保存测试，不代表实际 GT 漏标结论')
  page.click('#saveLabel');page.wait_for_function("document.getElementById('labelMessage').textContent.startsWith('已保存')")
  saved=page.request.get(base+'/api/labels').json()['items'][target]
  assert saved['original']['case_id']==target and saved['original']['kind']=='false_positive'
  stale=page.request.post(base+'/api/labels',data=json.dumps(dict(dataset_id=original['dataset_id'],case_id=target,revision=saved['revision']-1,label='original')),headers={'Content-Type':'application/json'})
  assert stale.status==409
  page.select_option('#labelFilter','gt_missing');assert page.locator('.case').count()>=1
  page.reload();page.wait_for_selector('#caseLabel');assert page.input_value('#caseLabel')=='gt_missing'
  assert 'QA 临时' in page.input_value('#labelNote')
  page.wait_for_function("document.getElementById('status').textContent.startsWith('已就绪')",timeout=180000)
  page.wait_for_function("document.getElementById('selection').textContent.includes('已定位帧')",timeout=60000)
  page.locator('#labelEditor').scroll_into_view_if_needed();page.screenshot(path=str(out/'editing.png'))
  with page.expect_download() as info:page.click('#exportLabels')
  d=info.value;d.save_as(str(out/'qa_export.json'));export=json.loads((out/'qa_export.json').read_text());assert export['items'][target]['label']=='gt_missing'
  page.select_option('#caseLabel',old['label']);page.fill('#labelNote',old['note']);page.click('#saveLabel');page.wait_for_function("document.getElementById('labelMessage').textContent.startsWith('已保存')")
  page.click('#reloadLabel');assert page.input_value('#caseLabel')==old['label']
  page.select_option('#labelFilter','original');page.select_option('#kind','false_negative');page.locator('.case').first.click()
  assert page.locator('#caseLabel option[value=gt_missing]').is_disabled()
  events=page.request.get(base+'/events.json').json()['events']
  event=next(e for e in events if e['kind']=='false_positive' and len(e['case_ids'])>1)
  page.select_option('#labelFilter','');page.select_option('#kind','false_positive');page.select_option('#view','events')
  page.locator('[data-case-id="'+event['event_id']+'"]').click()
  page.locator('#detail summary').click();page.locator('.member').nth(1).click()
  member=event['case_ids'][1]
  assert member in page.locator('#labelEditor h2').inner_text()
  page.wait_for_function('(id)=>document.getElementById("selection").dataset.caseId===id',arg=member,timeout=60000)
  page.set_viewport_size({'width':700,'height':950});page.locator('#labelEditor').scroll_into_view_if_needed();page.screenshot(path=str(out/'narrow.png'))
  assert errors==[],errors
  (out/'verification.json').write_text(json.dumps(dict(errors=errors,target=target,checks=['save and reload','GT-missing filter','original FP retained','stale revision 409','download JSON','restore original','FN cannot be labelled correct detection','member editor follows selected frame','desktop and narrow'],qa_labels_restored=True),indent=2))
  print('Manual label browser checks passed',target,flush=True)
 finally:
  current=page.request.get(base+'/api/labels').json()['items'].get(target)
  if current and (current['label']!=old['label'] or current['note']!=old['note']):
   r=page.request.post(base+'/api/labels',data=json.dumps(dict(dataset_id=original['dataset_id'],case_id=target,revision=current['revision'],label=old['label'],note=old['note'])),headers={'Content-Type':'application/json'});assert r.ok
  b.close()
