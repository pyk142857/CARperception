"""Validate measured tracking comparison recording and timeline using existing local services.

Browser plugin not available: uses installed Playwright/Chrome. Requires exclusive
GPU slot; run after inference/timing completes, never concurrently with benchmarks.
"""
from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime, timezone
import argparse
import hashlib
import json
import re
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/tracking_gpu_20261013/visualization'
URL = None

def classify_console(result):
    external_hosts={'fonts.gstatic.com','fonts.googleapis.com','app.rerun.io','tel.rerun.io'}
    external_failures=bool(result['failed_requests']) and all(urlparse(x['url']).hostname in external_hosts for x in result['failed_requests'])
    app_errors=[x for x in result['rerun_console_before_navigation'] if 'ERROR:' in x['text'] or (x['type']=='error' and not (external_failures and ('Failed to fetch' in x['text'] or 'Failed to load resource' in x['text'])))]
    result['console_review']={'external_optional_request_failures':external_failures,'unexplained_app_errors':app_errors,'note':'External font/examples/telemetry requests do not carry local recording data. SDK initialization and attention warnings are retained.'}
    result['status']='passed' if not app_errors else 'partial'

def main():
    global OUT, URL
    parser=argparse.ArgumentParser()
    parser.add_argument('--split', choices=['scene0061','mini_val'], default='scene0061')
    parser.add_argument('--headless',action='store_true',help='Avoid desktop compositor/occlusion stalls on unattended X11 sessions')
    parser.add_argument('--software-rendering',action='store_true',help='Explicit independent SwiftShader fallback; not GPU performance evidence')
    args=parser.parse_args()
    if args.split=='mini_val': OUT=OUT/'mini_val'
    export=json.loads((OUT/'status.json').read_text())
    URL=export['rerun_url']
    last=export['frames']-1
    selected=[0,export['frames']//2,last]
    scene_changes=[i for i,row in enumerate(export['counts']) if i and row.get('scene_token')!=export['counts'][i-1].get('scene_token')]
    selected=sorted(set(selected+[j for i in scene_changes for j in (i-1,i)]))
    OUT.mkdir(parents=True, exist_ok=True)
    result = dict(status='running', browser='Playwright Chromium', fallback_reason='Browser plugin not available',headless=args.headless or args.software_rendering,requested_renderer='SwiftShader' if args.software_rendering else 'Chrome default hardware', url=URL, viewport={'width':1800,'height':1100}, page_errors=[], console=[], failed_requests=[], http_errors=[], checks=[], generated_at=datetime.now(timezone.utc).isoformat())
    destination = OUT / 'rerun_browser_qa.json'
    if destination.exists():
        previous=json.loads(destination.read_text())
        if previous.get('status')=='failed':
            (OUT/('failed_attempt_'+previous.get('generated_at','unknown').replace(':','-')+'.json')).write_text(json.dumps(previous,indent=2)+'\n')
    destination.write_text(json.dumps(result, indent=2))
    try:
        from playwright.sync_api import sync_playwright
        export = json.loads((OUT/'status.json').read_text())
        assert export['export']=='passed'
        result['export_rrd_sha256'] = hashlib.sha256(Path(export['rrd']).read_bytes()).hexdigest()
        assert result['export_rrd_sha256'] == export['rrd_sha256']
        with sync_playwright() as p:
            chrome_args=['--no-sandbox','--ignore-gpu-blocklist']
            if args.software_rendering:
                chrome_args+=['--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader']
            if args.headless or args.software_rendering:
                # Installed Playwright emits removed --headless=old; request current Chrome mode explicitly.
                chrome_args+=['--headless=new']
            result['chrome_args']=chrome_args
            browser = p.chromium.launch(executable_path='/usr/bin/google-chrome', headless=False, args=chrome_args)
            page = browser.new_page(viewport=result['viewport'])
            page.set_default_timeout(90000)
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
            result['document_visibility']=page.evaluate('({visibility:document.visibilityState,focused:document.hasFocus()})')
            page.bring_to_front()
            result['renderer']=page.evaluate('()=>{let g=document.createElement("canvas").getContext("webgl2"),e=g.getExtension("WEBGL_debug_renderer_info");return {backend:"WebGL2",renderer:g.getParameter(e.UNMASKED_RENDERER_WEBGL)}}')
            if args.software_rendering:
                assert 'SwiftShader' in result['renderer']['renderer'], result['renderer']
            destination.write_text(json.dumps(result,indent=2)+'\n')
            page.wait_for_function('window._qa_handle && window._qa_handle.get_active_recording_id()',timeout=120000,polling=100)
            page.wait_for_function('(last)=>{let h=window._qa_handle,r=h.get_timeline_time_range(h.get_active_recording_id(),"frame");return r && r.min===0 && r.max===last}',arg=last,timeout=120000,polling=100)
            result['title']=page.title()
            result['actual_url']=page.url
            result['canvas_count']=page.locator('canvas').count()
            assert page.url==URL and result['canvas_count']>0
            for frame in selected:
                page.evaluate('(f)=>{let h=window._qa_handle,id=h.get_active_recording_id();h.set_playing(id,false);h.set_active_timeline(id,"frame");h.set_time_for_timeline(id,"frame",f)}',frame)
                page.wait_for_function('(f)=>window._qa_handle.get_time_for_timeline(window._qa_handle.get_active_recording_id(),"frame")===f',arg=frame,polling=100)
                page.wait_for_timeout(3000)
                state=page.evaluate('()=>{let h=window._qa_handle,id=h.get_active_recording_id();return {recording:id,timeline:h.get_active_timeline(id),frame:h.get_time_for_timeline(id,"frame"),playing:h.get_playing(id),range:h.get_timeline_time_range(id,"frame"),panic:h.has_panicked()}}')
                state['expected_sample_token']=export['counts'][frame]['sample_token']
                state['token_evidence']='Token ordering independently verified by decoded RRD audit; browser API reports frame only'
                state['model_counts']=export['counts'][frame].get('model_boxes',{})
                result['checks'].append(state)
                destination.write_text(json.dumps(result,indent=2)+'\n')
                page.screenshot(path=str(OUT/f'rerun_browser_{frame:02d}.png'))
                assert state['frame']==frame and state['timeline']=='frame' and not state['playing'] and not state['panic'] and state['range']=={'min':0,'max':last}
            # Canvas tab positions require screenshot verification for this 1800x1100 layout.
            # Keep screenshots as proof; the WebHandle reports timeline state, not active tab.
            result['model_tab_checks']=[]
            for family,x in [('grae',152),('optipmb',226),('ground_truth',324)]:
                page.mouse.click(x,40)
                page.wait_for_timeout(1500)
                for frame in selected:
                    page.evaluate('(f)=>{let h=window._qa_handle,id=h.get_active_recording_id();h.set_playing(id,false);h.set_active_timeline(id,"frame");h.set_time_for_timeline(id,"frame",f)}',frame)
                    page.wait_for_function('(f)=>window._qa_handle.get_time_for_timeline(window._qa_handle.get_active_recording_id(),"frame")===f',arg=frame,polling=100)
                    page.wait_for_timeout(1200)
                    screenshot=f'rerun_{family}_{frame:02d}.png'
                    page.screenshot(path=str(OUT/screenshot))
                    assert not page.evaluate('window._qa_handle.has_panicked()')
                    result['model_tab_checks'].append(dict(family=family,frame=frame,screenshot=screenshot,tab_evidence='Inspected canvas tab coordinates; rendered screenshot must be reviewed',panic=False))
            result['model_tabs']='Model/family tabs clicked; screenshot evidence captured.'
            result['rerun_console_before_navigation']=list(result['console'])
            # Independent lightweight overview verifies the visible slider updates metadata/image.
            page.goto(export['url'],wait_until='networkidle')
            page.wait_for_function('document.querySelector("#info").textContent.length>0')
            result['overview_checks']=[]
            for frame in selected:
                page.locator('#frame').fill(str(frame))
                page.wait_for_function('(f)=>JSON.parse(document.querySelector("#info").textContent).frame===f',arg=frame,polling=100)
                page.wait_for_function('document.querySelector("#image").complete && document.querySelector("#image").naturalWidth>0')
                result['overview_checks'].append(json.loads(page.locator('#info').inner_text()))
            page.set_viewport_size({'width':390,'height':844})
            page.screenshot(path=str(OUT/'overview_mobile.png'))
            result['mobile_note']='Overview smoke only; dense Rerun desktop workspace not evaluated on mobile.'
            browser.close()
        assert not result['page_errors'], result['page_errors']
        classify_console(result)
        export['browser_validation']=result['status']
        export['status']='passed' if result['status']=='passed' and not export.get('missing_models') else 'partial'
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
