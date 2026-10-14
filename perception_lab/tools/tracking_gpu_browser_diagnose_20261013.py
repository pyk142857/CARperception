"""Bounded SwiftShader-only viewer initialization diagnosis, separate from QA."""
import json,re,hashlib
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/tracking_gpu_20261013/visualization'
export=json.loads((OUT/'status.json').read_text())
result={'purpose':'Reproduce WebGL initialization limit failure; software only','console':[],'network':[]}
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=False,args=['--no-sandbox','--headless=new','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader'])
    page=browser.new_page(viewport={'width':1800,'height':1100});page.set_default_timeout(15000)
    page.on('console',lambda m:result['console'].append({'type':m.type,'text':m.text}))
    page.on('request',lambda r:result['network'].append({'event':'request','url':r.url}))
    page.on('response',lambda r:result['network'].append({'event':'response','url':r.url,'status':r.status,'content_length':r.headers.get('content-length')}))
    page.on('requestfinished',lambda r:result['network'].append({'event':'finished','url':r.url}))
    def expose(route):
        response=route.fetch();route.fulfill(response=response,body=response.text().replace('let handle = new wasm_bindgen.WebHandle(options);','let handle = window._qa_handle = new wasm_bindgen.WebHandle(options);'))
    page.route(re.compile(r'^http://127\.0\.0\.1:9095/\?'),expose)
    page.goto(export['rerun_url'],wait_until='domcontentloaded');page.wait_for_timeout(10000)
    result['body']=page.locator('body').inner_text()
    result['state']=page.evaluate('''()=>{let h=window._qa_handle,id=h?.get_active_recording_id();return {handle:!!h,id:id??null,timeline:id?h.get_active_timeline(id):null,range:id?h.get_timeline_time_range(id,"frame"):null,panic:h?.has_panicked(),visibility:document.visibilityState,memory:performance.memory?{used:performance.memory.usedJSHeapSize,total:performance.memory.totalJSHeapSize,limit:performance.memory.jsHeapSizeLimit}:null}}''')
    result['webgl']=page.evaluate('''()=>{let g=document.createElement('canvas').getContext('webgl2'),e=g.getExtension('WEBGL_debug_renderer_info');return {renderer:g.getParameter(e.UNMASKED_RENDERER_WEBGL),max_color_attachments:g.getParameter(g.MAX_COLOR_ATTACHMENTS),max_draw_buffers:g.getParameter(g.MAX_DRAW_BUFFERS)}}''')
    assert 'SwiftShader' in result['webgl']['renderer']
    result['viewer_requested_rrd']=any('.rrd' in n['url'] and ':9094/' in n['url'] for n in result['network'])
    try:page.screenshot(path=str(OUT/'swiftshader_initialization_failure.png'),timeout=15000)
    except Exception as exc:result['screenshot_error']=str(exc)
    result['independent_browser_fetch']=page.evaluate('''async url=>{let r=await fetch(url),a=await r.arrayBuffer(),hash=await crypto.subtle.digest('SHA-256',a);return {status:r.status,bytes:a.byteLength,sha256:Array.from(new Uint8Array(hash)).map(x=>x.toString(16).padStart(2,'0')).join('')}}''',export['url']+'tracking_gpu.rrd')
    assert result['independent_browser_fetch']['sha256']==export['rrd_sha256']
    browser.close()
(OUT/'swiftshader_initialization_diagnosis.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='network'}))
