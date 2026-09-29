from playwright.sync_api import sync_playwright
from pathlib import Path
import json,argparse
parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);a=parser.parse_args();out=a.out;out.mkdir(parents=True,exist_ok=True)
root=Path(__file__).resolve().parents[1]
cases=json.loads((root/'reports/mini_evaluation/cases.json').read_text());scores=json.loads((root/'reports/confidence_filter/scores.json').read_text())['scores'];data=json.loads((root/'reports/failure_events/events.json').read_text());results=[]
with sync_playwright() as p:
 b=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=False,args=['--no-sandbox','--ignore-gpu-blocklist'])
 page=b.new_page(viewport={'width':1900,'height':1250});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
 page.goto('http://127.0.0.1:9092/');page.wait_for_selector('.case')
 for value,lo,hi in [('0:10',0,10),('10:20',10,20),('20:40',20,40),('40:Infinity',40,float('inf'))]:
  selected={c['case_id'] for c in cases if c['kind']!='gap_recovery' and lo<=c['distance_m']<hi}
  events={e['event_id'] for e in data['events'] if selected.intersection(e['case_ids'])}
  groups={g['group_id'] for g in data['groups'] if events.intersection(g['event_ids'])}
  page.select_option('#distance',value)
  for mode,expected in [('events',events),('cases',selected),('groups',groups)]:
   page.select_option('#view',mode);assert page.locator('.case').count()==len(expected),(value,mode)
  results.append(dict(distance=value,cases=len(selected),events=len(events),groups=len(groups)))
 page.select_option('#view','events');page.select_option('#distance','0:10');page.select_option('#confidence','0.6:0.8')
 selected={c['case_id'] for c in cases if c['kind']!='gap_recovery' and c['distance_m']<10 and scores[c['case_id']] is not None and .6<=scores[c['case_id']]<.8}
 expected=[e for e in data['events'] if selected.intersection(e['case_ids'])]
 assert page.locator('.case').count()==len(expected)
 page.wait_for_function("document.getElementById('status').textContent.startsWith('已就绪')",timeout=180000)
 page.locator('.case').first.click()
 page.wait_for_function("document.getElementById('selection').textContent.includes('已定位帧')",timeout=60000)
 ident=page.locator('#selection').get_attribute('data-case-id');assert ident in selected,ident
 page.wait_for_timeout(2000);page.screenshot(path=str(out/'distance_filter.png'))
 page.select_option('#confidence','na');page.select_option('#view','cases')
 assert page.locator('.case').count()==sum(c['kind']=='false_negative' and c['distance_m']<10 for c in cases)
 page.select_option('#kind','false_positive');assert page.locator('#empty').count()==1
 page.wait_for_function("document.getElementById('selection').dataset.caseId===''")
 page.select_option('#kind','');page.select_option('#confidence','');page.select_option('#distance','');assert page.locator('.case').count()==1910
 page.set_viewport_size({'width':700,'height':950});page.screenshot(path=str(out/'narrow.png'))
 assert errors==[],errors
 (out/'verification.json').write_text(json.dumps(dict(errors=errors,counts=results,combined_selected_case=ident,checks=['all distance bins in events/cases/groups','same-record confidence intersection','selection seek/highlight/focus','FN and distance','empty clears highlight','reset restores 1910 cases','desktop and narrow']),indent=2))
 print('Distance browser checks passed',results,flush=True);b.close()
