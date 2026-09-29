import {WebViewer} from '/vendor/index.js';
import {seekCase} from '/case_logic.mjs';
import {EventUI} from '/event_ui.js';
import {createDisplayController} from '/display_layers.mjs';
const $=id=>document.getElementById(id);
const viewer=new WebViewer();window.caseViewer=viewer;
let request=0;
const setStatus=text=>$('status').textContent=text;
async function choose(row){
 const serial=++request;
 try{const id=seekCase(viewer,row);$('selection').textContent=`定位中：${row.case_id} → 帧 ${row.frame}`;
 await new Promise(r=>setTimeout(r,250));if(serial!==request)return;
 const actual=viewer.get_current_time(id,'frame');
 if(viewer.get_active_timeline(id)!=='frame'||Math.abs(actual-row.frame)>.001||viewer.get_playing(id))throw new Error('时间轴未完成同步，请重试');
 $('selection').textContent=`${row.case_id} · 已定位帧 ${row.frame}（${row.elapsed_seconds.toFixed(3)} 秒）· 已暂停`;
 $('selection').dataset.frame=String(actual);
 }catch(error){$('selection').textContent='定位失败：'+error.message;}
}
async function init(){
 const [raw,recording]=await Promise.all([fetch('/cases.json').then(r=>r.arrayBuffer()),fetch('/recording.json').then(r=>r.json())]);
 const data=JSON.parse(new TextDecoder().decode(raw));
 // Ensure the selected cases correspond to the actual exported recording.
 const digest=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',raw)),b=>b.toString(16).padStart(2,'0')).join('');
 if(!recording.triage_only)throw new Error('请先导出仅异常录制');
 if(!recording.failure_overlay||digest!==recording.failure_report_sha256)throw new Error('案例与记录不一致，请重新导出 Rerun 记录');
 const events=await fetch('/events.json').then(r=>r.json());
 if(events.cases_sha256!==digest)throw new Error('事件与案例版本不一致，请重新归并');
 const confidence=await fetch('/confidence.json').then(r=>{if(!r.ok)throw new Error('请先运行 export_case_confidence.py');return r.json();});
 if(confidence.cases_sha256!==digest)throw new Error('置信度与案例版本不一致，请重新导出置信度');
 for(const row of data){
  const score=confidence.scores[row.case_id];
  if(score!==null&&(!Number.isFinite(score)||score<0||score>1))throw new Error('案例置信度缺失或无效：'+row.case_id);
  if((row.kind==='false_negative')!==(score===null))throw new Error('FN 置信度口径不一致');
  row.confidence=score;
 }
 const display=createDisplayController(viewer,async url=>{
  const r=await fetch(url);if(!r.ok)throw new Error('图层读取失败：'+url);return r.arrayBuffer();
 },(module,normal,loading=false)=>{
  $('displayStatus').textContent=loading?'加载正常目标…':(module==='detection'?'检测':'跟踪')+' · '+(normal?'异常＋正常目标':'仅异常目标')+' · 选中目标：青白粗框';
  $('displayStatus').dataset.mode=loading?'loading':(normal?'normal':'errors');
  $('displayStatus').dataset.module=module;
 });
 const ui=new EventUI(events,data,viewer,choose,async(module,caseId=null)=>{await display.apply(module,$('showNormal').checked,caseId);$('selection').dataset.caseId=display.getSelection()||'';});await ui.init();
 $('showNormal').onchange=async()=>{
  ui.player.stop();
  try{await display.apply(display.getBranch(),$('showNormal').checked,display.getSelection());}
  catch(e){$('displayStatus').textContent=e.message;}
 };
 setStatus('加载 Rerun 0.23.4…');await viewer.start(null,$('viewer'),{render_backend:'webgl',hide_welcome_screen:true,width:'100%',height:'100%'});
 setStatus('加载场景记录…');const buffer=await fetch('/mini_scene.rrd').then(r=>{if(!r.ok)throw new Error('记录读取失败');return r.arrayBuffer();});
 const channel=viewer.open_channel('CARperception failures');channel.send_rrd(new Uint8Array(buffer));channel.close();
 const started=Date.now();
 while(true){const id=viewer.get_active_recording_id();const range=id?viewer.get_time_range(id,'frame'):null;
 if(id){viewer.set_playing(id,false);viewer.set_active_timeline(id,'frame');viewer.set_current_time(id,'frame',0);}
 if(range&&range.max>=recording.frame_count-1){await new Promise(r=>setTimeout(r,800));viewer.set_playing(id,false);break;}
 if(Date.now()-started>360000)throw new Error('记录加载超时，请刷新重试');setStatus(`解析场景：${range?Math.min(recording.frame_count,Math.floor(range.max)+1):0} / ${recording.frame_count} 帧`);await new Promise(r=>setTimeout(r,300));}
 setStatus(`已就绪 · ${recording.frame_count} 帧 · ${events.events.length} 事件`);await ui.setReady();$('showNormal').disabled=false;
}
init().catch(error=>{setStatus('加载失败：'+error.message);$('selection').textContent='请检查服务与记录后刷新页面。';console.error(error);});
