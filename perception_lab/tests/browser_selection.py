from playwright.sync_api import sync_playwright
from pathlib import Path
import json,argparse
parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
out=args.out;out.mkdir(parents=True,exist_ok=True)
with sync_playwright() as p:
 b=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=False,args=['--no-sandbox','--ignore-gpu-blocklist'])
 page=b.new_page(viewport={'width':1900,'height':1250});errors=[];requests=[]
 page.on('pageerror',lambda e:errors.append(str(e)));page.on('request',lambda r:requests.append(r.url))
 page.goto('http://127.0.0.1:9092/')
 page.wait_for_function("document.getElementById('status').textContent.startsWith('已就绪')",timeout=180000)
 page.wait_for_function("document.getElementById('selection').textContent.includes('已定位帧')",timeout=60000)
 assert page.locator('#selection').get_attribute('data-case-id')
 page.select_option('#module','tracking');page.select_option('#kind','id_switch');page.select_option('#className','pedestrian')
 page.locator('.case').first.click()
 page.wait_for_function("document.getElementById('selection').dataset.caseId==='case_01219' && document.getElementById('selection').textContent.includes('已定位帧 2')",timeout=60000)
 page.wait_for_timeout(2000);page.screenshot(path=str(out/'selected_frame2.png'))
 page.locator('#detail summary').click();page.locator('.member').nth(1).click()
 page.wait_for_function("document.getElementById('selection').dataset.caseId==='case_01230' && document.getElementById('selection').textContent.includes('已定位帧 3')",timeout=60000)
 page.wait_for_timeout(2000);page.screenshot(path=str(out/'selected_frame3.png'))
 page.check('#showNormal');page.wait_for_function("document.getElementById('displayStatus').dataset.mode==='normal'",timeout=60000)
 assert page.locator('#selection').get_attribute('data-case-id')=='case_01230'
 assert page.evaluate("()=>window.caseViewer.get_current_time(window.caseViewer.get_active_recording_id(),'frame')")==3
 page.uncheck('#showNormal');page.wait_for_function("document.getElementById('displayStatus').dataset.mode==='errors'",timeout=60000)
 page.click('#playClip');page.wait_for_function("document.getElementById('selection').textContent.startsWith('片段播放完成')",timeout=20000)
 page.wait_for_timeout(1200);page.screenshot(path=str(out/'after_clip.png'))
 page.fill('#query','no_such_case');page.wait_for_function("document.getElementById('selection').dataset.caseId===''")
 assert page.locator('#detail').inner_text()==''
 page.fill('#query','');page.select_option('#view','groups')
 page.wait_for_function("document.getElementById('selection').dataset.caseId===''")
 # Rapid last-wins changes, then FN branch with exact target.
 page.select_option('#view','cases');page.select_option('#className','');page.select_option('#module','detection');page.select_option('#kind','false_negative')
 page.locator('.case').nth(1).click();page.locator('.case').first.click()
 page.wait_for_function("document.getElementById('selection').dataset.caseId==='case_00000' && document.getElementById('selection').textContent.includes('case_00000 · 已定位')",timeout=60000)
 page.wait_for_timeout(2000);page.screenshot(path=str(out/'fn_selected.png'))
 assert not page.evaluate("()=>window.caseViewer.get_playing(window.caseViewer.get_active_recording_id())")
 assert page.request.get('http://127.0.0.1:9092/selection/tracking/errors/case_00000.rrd').status==400
 page.set_viewport_size({'width':700,'height':950});page.screenshot(path=str(out/'narrow.png'))
 from PIL import Image
 cyan_counts={}
 for name in ['selected_frame2','selected_frame3','after_clip']:
  im=Image.open(out/(name+'.png')).convert('RGB')
  cyan_counts[name]=[sum(1 for r,g,b in im.crop(box).getdata() if 150<=r<=170 and g>=245 and b>=245)
                     for box in [(370,160,1135,510),(370,535,1135,803),(1135,160,1900,1000)]]
 assert all(n>0 for n in cyan_counts['selected_frame2'])
 assert all(n>0 for n in cyan_counts['selected_frame3'])
 assert cyan_counts['after_clip']==[0,0,0]
 assert errors==[],errors
 (out/'verification.json').write_text(json.dumps(dict(errors=errors,cyan_counts=cyan_counts,checks=['automatic initial selection','ID case frame2 highlighted','member changes to frame3','normal toggles preserve selection and time','clip ends paused','empty and group clear selection','rapid last choice wins','FN selection','invalid branch rejected'],scene_requests=sum('/mini_scene.rrd' in u for u in requests),selection_requests=[u for u in requests if '/selection/' in u]),indent=2))
 assert sum('/mini_scene.rrd' in u for u in requests)==1
 print('Selection browser QA passed',flush=True);b.close()
