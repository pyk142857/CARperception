from playwright.sync_api import sync_playwright
from pathlib import Path
import json,argparse
parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);a=parser.parse_args()
out=a.out;out.mkdir(parents=True,exist_ok=True)
with sync_playwright() as p:
 b=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=False,args=['--no-sandbox','--ignore-gpu-blocklist'])
 page=b.new_page(viewport={'width':1900,'height':1250});errors=[];requests=[]
 page.on('pageerror',lambda e:errors.append(str(e)));page.on('request',lambda r:requests.append(r.url))
 page.goto('http://127.0.0.1:9092/')
 for tick in range(40):
  page.wait_for_timeout(5000);print(tick,page.locator('#status').inner_text(),flush=True)
  if not page.locator('#showNormal').is_disabled():break
  assert not page.locator('#status').inner_text().startswith('加载失败')
 assert page.evaluate("()=>getComputedStyle(document.body).margin")=='0px'
 assert not page.locator('#showNormal').is_checked()
 assert not any('/normal_targets.rrd' in u for u in requests)
 assert page.evaluate("()=>window.caseViewer.get_time_range(window.caseViewer.get_active_recording_id(),'normal_ready')") is None
 page.screenshot(path=str(out/'detection_errors.png'))
 page.select_option('#module','tracking');page.select_option('#kind','id_switch');page.select_option('#className','pedestrian')
 page.locator('.case').first.click()
 page.wait_for_function("document.getElementById('displayStatus').dataset.module==='tracking'")
 page.wait_for_timeout(1200)
 def state():
  return page.evaluate("()=>{const v=window.caseViewer,id=v.get_active_recording_id();return {id,frame:v.get_current_time(id,'frame'),playing:v.get_playing(id)}}")
 before=state();assert before['frame']==2,before
 page.screenshot(path=str(out/'tracking_errors.png'))
 page.check('#showNormal')
 page.wait_for_function("document.getElementById('displayStatus').dataset.mode==='normal'",timeout=60000)
 page.wait_for_timeout(4500);after=state();assert before==after,(before,after)
 assert page.evaluate("()=>window.caseViewer.get_time_range(window.caseViewer.get_active_recording_id(),'normal_ready')")['max']==1
 page.screenshot(path=str(out/'tracking_normal.png'))
 from PIL import Image,ImageStat
 crop=Image.open(out/'tracking_normal.png').crop((500,580,1000,760))
 assert sum(ImageStat.Stat(crop).mean)>15,'3D viewport remains blank'
 page.uncheck('#showNormal');page.wait_for_function("document.getElementById('displayStatus').dataset.mode==='errors'")
 page.wait_for_timeout(4500);assert state()==before
 page.screenshot(path=str(out/'tracking_hidden.png'))
 page.check('#showNormal');page.wait_for_function("document.getElementById('displayStatus').dataset.mode==='normal'")
 assert sum('/normal_targets.rrd' in u for u in requests)==1
 page.uncheck('#showNormal');page.wait_for_function("document.getElementById('displayStatus').dataset.mode==='errors'")
 page.click('#playClip');page.wait_for_function("document.getElementById('selection').textContent.startsWith('片段播放完成')",timeout=20000)
 assert not state()['playing']
 page.set_viewport_size({'width':700,'height':950});page.screenshot(path=str(out/'narrow.png'))
 (out/'verification.json').write_text(json.dumps(dict(page_errors=errors,before=before,after=after,normal_requests=sum('/normal_targets.rrd' in u for u in requests),checks=['no normal download by default','branch change','normal_ready marker','same recording/frame/paused on toggle','uncheck hides normal','recheck no redownload','clip still ends paused']),indent=2))
 assert errors==[],errors
 print('Display browser checks passed',before,after,flush=True);b.close()
