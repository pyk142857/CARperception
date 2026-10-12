"""Actual Rerun WebGPU rendering/timeline QA via SDK-bundled WebHandle API, physical RTX 4090 WebGL."""
from playwright.sync_api import sync_playwright
from pathlib import Path
import json,time,re
out=Path(__file__).resolve().parents[1]/'reports/deployment_4090_20261008/visualize';errors=[];console=[];checks=[]
(out/'rerun_browser_qa.json').unlink(missing_ok=True)
with sync_playwright() as p:
 b=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=False,args=['--no-sandbox','--ignore-gpu-blocklist'])
 page=b.new_page(viewport={'width':1800,'height':1100});page.on('pageerror',lambda e:errors.append(str(e)));page.on('console',lambda m:console.append(dict(type=m.type,text=m.text)) if m.type in ['error','warning'] else None)
 url='http://127.0.0.1:9095/?url=http%3A%2F%2F127.0.0.1%3A9094%2Fdeployment_4090.rrd&renderer=webgl&persist=0'
 def expose_handle(route):
  response=route.fetch();body=response.text().replace('let handle = new wasm_bindgen.WebHandle(options);', 'let handle = window._qa_handle = new wasm_bindgen.WebHandle(options);');route.fulfill(response=response,body=body)
 page.route(re.compile(r'^http://127\.0\.0\.1:9095/\?'),expose_handle)
 page.goto(url,wait_until='domcontentloaded',timeout=60000);page.wait_for_timeout(5000)
 print('BOOTSTRAP',page.evaluate('({hook:!!window._qa_handle, body:document.body.innerText.slice(0,500)})'),flush=True)
 page.wait_for_function('window._qa_handle && window._qa_handle.get_active_recording_id()',timeout=120000)
 print('RANGE',page.evaluate('window._qa_handle.get_timeline_time_range(window._qa_handle.get_active_recording_id(),"frame")'),flush=True)
 page.wait_for_function('(()=>{let h=window._qa_handle,r=h.get_timeline_time_range(h.get_active_recording_id(),"frame");return r && r.min===0 && r.max===38})()',timeout=120000)
 page.wait_for_timeout(3000)
 for frame in [0,19,38]:
  page.evaluate('(f)=>{let h=window._qa_handle,id=h.get_active_recording_id();h.set_playing(id,false);h.set_active_timeline(id,"frame");h.set_time_for_timeline(id,"frame",f)}',frame)
  page.wait_for_function('(f)=>window._qa_handle.get_time_for_timeline(window._qa_handle.get_active_recording_id(),"frame")===f',arg=frame)
  page.wait_for_timeout(3000)
  state=page.evaluate('()=>{let h=window._qa_handle,id=h.get_active_recording_id();return {recording:id,timeline:h.get_active_timeline(id),frame:h.get_time_for_timeline(id,"frame"),playing:h.get_playing(id),range:h.get_timeline_time_range(id,"frame"),panic:h.has_panicked()}}');checks.append(state)
  page.screenshot(path=str(out/f'rerun_browser_{frame:02d}.png'))
 (out/'rerun_browser_qa.json').write_text(json.dumps(dict(status='passed' if not errors and all(c['frame']==f and c['timeline']=='frame' and not c['playing'] and not c['panic'] and c['range']=={'min':0,'max':38} for c,f in zip(checks,[0,19,38])) else 'failed',export_rrd_sha256=json.loads((out/'summary.json').read_text())['rrd_sha256'],url=url,title=page.title(),renderer=page.evaluate('()=>{let g=document.createElement("canvas").getContext("webgl2"),e=g.getExtension("WEBGL_debug_renderer_info");return {backend:"WebGL2",renderer:g.getParameter(e.UNMASKED_RENDERER_WEBGL),max_color_attachments:g.getParameter(g.MAX_COLOR_ATTACHMENTS)}}'),checks=checks,page_errors=errors,console=console),indent=2))
 b.close()
assert not errors
assert json.loads((out/'rerun_browser_qa.json').read_text())['status']=='passed'
print(checks)
