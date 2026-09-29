"""Local browser QA; Browser plugin not available, using system Playwright."""
from playwright.sync_api import sync_playwright
import json,time,argparse
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--url',default='http://127.0.0.1:9092/');args=parser.parse_args()
out=args.out;out.mkdir(parents=True,exist_ok=True)
with sync_playwright() as p:
 b=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=False,args=['--no-sandbox','--ignore-gpu-blocklist'])
 page=b.new_page(viewport={'width':1900,'height':1250});errors=[]
 page.on('pageerror',lambda e:errors.append(str(e)))
 page.goto(args.url)
 page.wait_for_selector('.case')
 assert page.locator('#reviewForm').count()==0
 page.select_option('#view','groups');page.locator('.case').first.click()
 assert page.locator('#openGroup').count()==1
 page.click('#openGroup');assert page.input_value('#view')=='events'
 assert page.locator('#groupScope').is_visible()
 page.click('#clearGroup')
 for tick in range(40):
  state=page.locator('#status').inner_text();print(tick,state,flush=True)
  if state.startswith('已就绪'):break
  if state.startswith('加载失败'):raise AssertionError(state)
  page.wait_for_timeout(10000)
 assert page.locator('#status').inner_text().startswith('已就绪')
 page.select_option('#module','tracking');page.select_option('#kind','id_switch');page.select_option('#className','pedestrian')
 page.locator('.case').first.click();page.wait_for_timeout(1200)
 assert page.evaluate("()=>window.caseViewer.get_current_time(window.caseViewer.get_active_recording_id(),'frame')")==2
 assert not page.evaluate("()=>window.caseViewer.get_playing(window.caseViewer.get_active_recording_id())")
 page.locator('#detail details').click()
 page.locator('.member').first.click();page.wait_for_timeout(600)
 event_id=page.evaluate("()=>location.hash.slice(1)")
 expected_end=page.evaluate("""async id=>{const d=await fetch('/events.json').then(r=>r.json()),e=d.events.find(x=>x.event_id===id);
 return d.frames.filter(f=>f.scene_token===e.scene_token&&f.elapsed_seconds>=e.start_seconds-2&&f.elapsed_seconds<=e.end_seconds+2).at(-1).frame}""",event_id)
 page.click('#playClip')
 page.wait_for_function("document.getElementById('selection').textContent.startsWith('片段播放完成')",timeout=25000)
 actual=page.evaluate("()=>{let v=window.caseViewer,id=v.get_active_recording_id();return {frame:v.get_current_time(id,'frame'),playing:v.get_playing(id)}}")
 print('clip',actual,flush=True);assert not actual['playing'];assert actual['frame']==expected_end
 page.screenshot(path=str(out/'events.png'))
 page.select_option('#view','cases');page.select_option('#module','detection');page.select_option('#kind','false_negative');page.select_option('#className','')
 page.locator('.case').first.click();page.wait_for_timeout(600)
 assert page.evaluate("()=>window.caseViewer.get_current_time(window.caseViewer.get_active_recording_id(),'frame')")==0
 page.click('#parentEvent');assert page.input_value('#view')=='events'
 page.fill('#query','no_matching_event');assert page.locator('#empty').count()==1;page.fill('#query','')
 page.set_viewport_size({'width':700,'height':950});page.screenshot(path=str(out/'narrow.png'))
 (out/'verification.json').write_text(json.dumps(dict(errors=errors,clip=actual,checks=['review editing removed','group drill-down','GT ID switch seek frame2','expand original cases','bounded clip completes paused','raw FN seek frame0','parent event','empty search','narrow viewport']),ensure_ascii=False,indent=2))
 assert errors==[],errors
 b.close()
