"""System Chrome page interaction QA. Run with /usr/bin/python3 (installed Playwright)."""
from playwright.sync_api import sync_playwright
from pathlib import Path
import json
out=Path(__file__).resolve().parents[1]/'reports/deployment_4090_20261008/visualize';errors=[];checks=[];console=[]
(out/'browser_qa.json').unlink(missing_ok=True)
with sync_playwright() as p:
 b=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=False,args=['--headless=new','--no-sandbox','--disable-gpu'])
 page=b.new_page(viewport={'width':1600,'height':1100});page.on('pageerror',lambda e:errors.append(str(e)));page.on('console',lambda m:console.append(dict(type=m.type,text=m.text)) if m.type in ['error','warning'] else None)
 page.goto('http://127.0.0.1:9094/');page.wait_for_selector('#model option',state='attached');models=page.locator('#model option').evaluate_all('(items)=>items.map(x=>x.value)')
 for model in models:
  page.select_option('#model',model)
  for frame,label in [(0,'首帧'),(19,'中帧'),(38,'末帧')]:
   page.get_by_role('button',name=label,exact=True).click();page.wait_for_function('(f)=>view.complete && view.naturalWidth===1920 && view.src.endsWith("frame_"+String(f).padStart(2,"0")+".jpg")',arg=frame)
   assert page.locator('#index').inner_text()==f'{frame} / 38'
   token=page.locator('#token').inner_text();checks.append(dict(model=model,frame=frame,token=token,image_loaded=True))
   page.screenshot(path=str(out/f'browser_{model}_{frame:02d}.png'))
 page.set_viewport_size({'width':900,'height':720});page.screenshot(path=str(out/'browser_compact.png'))
 title=page.title();b.close()
(out/'browser_qa.json').write_text(json.dumps(dict(status='passed' if not errors else 'failed',export_rrd_sha256=json.loads((out/'summary.json').read_text())['rrd_sha256'],browser='System Chrome / Playwright; Browser plugin not available',url='http://127.0.0.1:9094/',title=title,checks=checks,page_errors=errors,console=console),indent=2))
assert not errors
print(json.dumps(checks,indent=2))
