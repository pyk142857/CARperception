"""Validate measured 2025 recording and timeline using existing local services.

Browser plugin not available: uses installed Playwright/Chrome. Requires exclusive
GPU slot; run after inference/timing completes, never concurrently with benchmarks.
"""
from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime, timezone
import hashlib
import json
import re
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/perception_2025_20261013/visualization'
URL = 'http://127.0.0.1:9095/?url=http%3A%2F%2F127.0.0.1%3A9094%2Fperception_2025%2Fperception_2025.rrd&renderer=webgl&persist=0'

def classify_console(result):
    external_hosts={'fonts.gstatic.com','fonts.googleapis.com','app.rerun.io','tel.rerun.io'}
    external_failures=bool(result['failed_requests']) and all(urlparse(x['url']).hostname in external_hosts for x in result['failed_requests'])
    app_errors=[x for x in result['rerun_console_before_navigation'] if 'ERROR:' in x['text'] or (x['type']=='error' and not (external_failures and ('Failed to fetch' in x['text'] or 'Failed to load resource' in x['text'])))]
    result['console_review']={'external_optional_request_failures':external_failures,'unexplained_app_errors':app_errors,'note':'External font/examples/telemetry requests do not carry local recording data. SDK initialization and attention warnings are retained.'}
    result['status']='passed' if not app_errors else 'partial'

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = dict(status='running', browser='Playwright Chromium', fallback_reason='Browser plugin not available', url=URL, viewport={'width':1800,'height':1100}, page_errors=[], console=[], failed_requests=[], http_errors=[], checks=[], generated_at=datetime.now(timezone.utc).isoformat())
    destination = OUT / 'rerun_browser_qa.json'
    destination.write_text(json.dumps(result, indent=2))
    try:
        from playwright.sync_api import sync_playwright
        export = json.loads((OUT/'status.json').read_text())
        assert export['export']=='passed'
        result['export_rrd_sha256'] = hashlib.sha256(Path(export['rrd']).read_bytes()).hexdigest()
        assert result['export_rrd_sha256'] == export['rrd_sha256']
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path='/usr/bin/google-chrome', headless=False, args=['--no-sandbox','--ignore-gpu-blocklist'])
            page = browser.new_page(viewport=result['viewport'])
            page.on('requestfailed', lambda r:result['failed_requests'].append(dict(url=r.url,failure=r.failure)))
            page.on('response', lambda r:result['http_errors'].append(dict(url=r.url,status=r.status)) if r.status>=400 else None)
            page.on('pageerror', lambda e:result['page_errors'].append(str(e)))
            page.on('console', lambda m:result['console'].append(dict(type=m.type,text=m.text)) if m.type in ['warning','error'] else None)
            def expose(route):
                response = route.fetch()
                body = response.text().replace('let handle = new wasm_bindgen.WebHandle(options);','let handle = window._qa_handle = new wasm_bindgen.WebHandle(options);')
                route.fulfill(response=response,body=body)
            page.route(re.compile(r'^http://127\.0\.0\.1:9095/\?'),expose)
            page.goto(URL,wait_until='domcontentloaded',timeout=60000)
            page.wait_for_function('window._qa_handle && window._qa_handle.get_active_recording_id()',timeout=120000)
            page.wait_for_function('(()=>{let h=window._qa_handle,r=h.get_timeline_time_range(h.get_active_recording_id(),"frame");return r && r.min===0 && r.max===38})()',timeout=120000)
            result['title']=page.title()
            result['actual_url']=page.url
            result['canvas_count']=page.locator('canvas').count()
            assert page.url==URL and result['canvas_count']>0
            for frame in [0,19,38]:
                page.evaluate('(f)=>{let h=window._qa_handle,id=h.get_active_recording_id();h.set_playing(id,false);h.set_active_timeline(id,"frame");h.set_time_for_timeline(id,"frame",f)}',frame)
                page.wait_for_function('(f)=>window._qa_handle.get_time_for_timeline(window._qa_handle.get_active_recording_id(),"frame")===f',arg=frame)
                page.wait_for_timeout(3000)
                state=page.evaluate('()=>{let h=window._qa_handle,id=h.get_active_recording_id();return {recording:id,timeline:h.get_active_timeline(id),frame:h.get_time_for_timeline(id,"frame"),playing:h.get_playing(id),range:h.get_timeline_time_range(id,"frame"),panic:h.has_panicked()}}')
                state['sample_token']=export['counts'][frame]['sample_token']
                result['checks'].append(state)
                page.screenshot(path=str(OUT/f'rerun_browser_{frame:02d}.png'))
                assert state['frame']==frame and state['timeline']=='frame' and not state['playing'] and not state['panic'] and state['range']=={'min':0,'max':38}
            # Rerun renders tab controls inside a canvas; coordinates follow the inspected desktop layout.
            page.mouse.click(195,40)
            page.wait_for_timeout(1500)
            page.screenshot(path=str(OUT/'rerun_independent_gt.png'))
            page.mouse.click(74,40)
            page.wait_for_timeout(500)
            result['gt_tab_interaction']='clicked Independent GT tab and returned to predictions; screenshot requires visual review'
            result['renderer']=page.evaluate('()=>{let g=document.createElement("canvas").getContext("webgl2"),e=g.getExtension("WEBGL_debug_renderer_info");return {backend:"WebGL2",renderer:g.getParameter(e.UNMASKED_RENDERER_WEBGL)}}')
            result['rerun_console_before_navigation']=list(result['console'])
            # Independent lightweight overview verifies the visible slider updates metadata/image.
            page.goto('http://127.0.0.1:9094/perception_2025/',wait_until='networkidle')
            page.wait_for_function('document.querySelector("#info").textContent.length>0')
            result['overview_checks']=[]
            for frame in [0,19,38]:
                page.locator('#frame').fill(str(frame))
                page.wait_for_function('(f)=>JSON.parse(document.querySelector("#info").textContent).frame===f',arg=frame)
                page.wait_for_function('document.querySelector("#image").complete && document.querySelector("#image").naturalWidth>0')
                result['overview_checks'].append(json.loads(page.locator('#info').inner_text()))
            page.set_viewport_size({'width':390,'height':844})
            page.screenshot(path=str(OUT/'overview_mobile.png'))
            result['mobile_note']='Overview smoke only; dense Rerun desktop workspace not evaluated on mobile.'
            browser.close()
        assert not result['page_errors'], result['page_errors']
        classify_console(result)
        export['browser_validation']=result['status']
        (OUT/'status.json').write_text(json.dumps(export,indent=2)+'\n')
    except BaseException as exc:
        result.update(status='failed',error=str(exc),traceback=traceback.format_exc())
        if (OUT/'status.json').exists():
            export=json.loads((OUT/'status.json').read_text());export['browser_validation']='failed'
            (OUT/'status.json').write_text(json.dumps(export,indent=2)+'\n')
        raise
    finally:
        destination.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
    main()
