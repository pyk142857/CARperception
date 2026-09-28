"""Run against the local case browser with system Playwright and headed Chrome."""
from playwright.sync_api import sync_playwright
import json,argparse
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument("--url",default="http://127.0.0.1:9092/")
parser.add_argument("--out",type=Path,required=True)
args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
with sync_playwright() as p:
 b=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=False,args=['--no-sandbox','--ignore-gpu-blocklist'])
 page=b.new_page(viewport={'width':1900,'height':1150});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
 page.goto(args.url+'#case_01217')
 assert page.locator('#module option').count()==3
 for tick in range(36):
  page.wait_for_timeout(10000)
  state=page.locator('#status').inner_text()
  print('loading',tick,state,flush=True)
  if state.startswith(('已就绪','加载失败')):break
 page.screenshot(path=str(args.out/'loading.png'))
 print('status',page.locator('#status').inner_text(),flush=True)
 page.screenshot(path=str(args.out/'initial.png'))
 assert page.evaluate("()=>window.caseViewer.get_current_time(window.caseViewer.get_active_recording_id(),'frame')")==2
 if '加载失败' in page.locator('#status').inner_text():print(errors);b.close();raise SystemExit(1)
 page.select_option('#module','tracking');page.select_option('#kind','id_switch');page.select_option('#className','pedestrian')
 page.locator('.case').first.click();page.wait_for_timeout(1500)
 result=page.evaluate("()=>{const v=window.caseViewer,id=v.get_active_recording_id();return {frame:v.get_current_time(id,'frame'),timeline:v.get_active_timeline(id),playing:v.get_playing(id),selection:document.getElementById('selection').textContent,count:document.getElementById('count').textContent}}")
 assert result['frame']==2 and result['timeline']=='frame' and result['playing']==False,result
 page.screenshot(path=str(args.out/'selected.png'));page.click('#next');page.wait_for_timeout(600);assert page.locator('.case[aria-pressed=true]').count()==1
 page.click('#prev');page.wait_for_timeout(600);assert page.locator('.case[aria-pressed=true]').get_attribute('data-case-id')=='case_01217'
 page.fill('#query','no_case_exists');assert page.locator('#empty').count()==1
 page.fill('#query','');page.select_option('#module','detection');page.select_option('#kind','false_negative');page.select_option('#className','');page.locator('.case').first.click();page.wait_for_timeout(800)
 assert page.evaluate("()=>window.caseViewer.get_current_time(window.caseViewer.get_active_recording_id(),'frame')")==0
 page.set_viewport_size({'width':700,'height':900});page.screenshot(path=str(args.out/'narrow.png'))
 print(json.dumps({'selected_case':result,'errors':errors,'checks':['filtered pedestrian ID switch -> frame2 paused','next case','empty filter','detection FN -> frame0','narrow viewport']},ensure_ascii=False),flush=True)
 open(args.out/'verification.json','w').write(json.dumps({'selected_case':result,'errors':errors,'checks':['deep_link','filter','seek','next','previous','empty','seek_back','narrow']},ensure_ascii=False,indent=2))
 b.close()
