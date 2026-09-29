from playwright.sync_api import sync_playwright
from pathlib import Path
import json,argparse
parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
out=args.out;out.mkdir(parents=True,exist_ok=True)
with sync_playwright() as p:
 b=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=False,args=['--no-sandbox','--ignore-gpu-blocklist'])
 page=b.new_page(viewport={'width':1900,'height':1250});errors=[]
 page.on('pageerror',lambda e:errors.append(str(e)))
 page.goto('http://127.0.0.1:9092/')
 page.wait_for_selector('.case')
 assert 'CARperception' in page.title()
 assert page.locator('#reviewForm').count()==0
 page.select_option('#confidence','na');assert page.locator('.case').count()==97
 page.select_option('#view','cases');assert page.locator('.case').count()==114
 page.select_option('#confidence','0.8:1');page.select_option('#kind','false_positive')
 assert page.locator('.case').count()==10
 page.select_option('#module','detection');assert page.locator('.case').count()==2
 page.select_option('#kind','false_negative');assert page.locator('#empty').count()==1
 assert page.locator('#detail').inner_text()==''
 page.select_option('#module','tracking');page.select_option('#kind','id_switch')
 assert page.locator('.case').count()==4
 page.select_option('#view','events')
 page.wait_for_function("document.getElementById('status').textContent.startsWith('已就绪')",timeout=240000)
 page.locator('.case').first.click()
 page.wait_for_function("document.getElementById('selection').textContent.includes('已定位帧')")
 expected=page.evaluate("""async()=>{const d=await fetch('/events.json').then(r=>r.json()),s=await fetch('/confidence.json').then(r=>r.json()),c=await fetch('/cases.json').then(r=>r.json());const e=d.events.find(e=>e.event_id===location.hash.slice(1));const a=e.case_ids.filter(id=>s.scores[id]>=.8);const id=a.includes(e.representative_case_id)?e.representative_case_id:a[0];return c.find(c=>c.case_id===id).frame}""")
 actual=page.evaluate("()=>{const v=window.caseViewer;return v.get_current_time(v.get_active_recording_id(),'frame')}")
 assert actual==expected,(actual,expected)
 page.screenshot(path=str(out/'filtered.png'))
 page.select_option('#view','groups');assert page.locator('.case').count()>0
 page.locator('.case').first.click();page.click('#openGroup');assert page.input_value('#confidence')=='0.8:1'
 assert page.locator('.case').count()>0
 page.set_viewport_size({'width':700,'height':950});page.screenshot(path=str(out/'narrow.png'))
 assert errors==[],errors
 (out/'verification.json').write_text(json.dumps(dict(errors=errors,expected_frame=expected,actual_frame=actual,checks=['FN 97 events / 114 cases','high-score FP 10 / detection 2','high-score ID 4','empty results clear detail','event seeks qualifying frame','group drilldown retains filter','review inputs absent','desktop and narrow screenshots']),indent=2))
 print('Confidence browser checks passed',expected,flush=True)
 b.close()
