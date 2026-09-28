import {WebViewer} from '/vendor/index.js';
import {kinds,filterCases,seekCase} from '/case_logic.mjs';
const $=id=>document.getElementById(id);
const viewer=new WebViewer();window.caseViewer=viewer;
let rows=[],filtered=[],selected=null,ready=false,request=0;
const setStatus=text=>$('status').textContent=text;
function option(value,text){const el=document.createElement('option');el.value=value;el.textContent=text;return el;}
function render(){
 const filters=Object.fromEntries(['module','kind','className','query'].map(k=>[k,$(k).value]));filtered=filterCases(rows,filters);
 $('count').textContent=`${filtered.length} 条匹配案例 · 点击跳转并暂停`;
 const fragment=document.createDocumentFragment();
 for(const row of filtered){const button=document.createElement('button');button.className='case';button.disabled=!ready;button.dataset.caseId=row.case_id;button.setAttribute('aria-pressed',selected?.case_id===row.case_id);
 const title=document.createElement('strong');const name=document.createElement('span');name.textContent=kinds[row.kind];name.className=row.kind;const frame=document.createElement('span');frame.textContent=`帧 ${row.frame}`;title.append(name,frame);
 const info=document.createElement('small');info.textContent=`${row.class_name} · ${row.module==='detection'?'检测':'跟踪'} · ${row.case_id}`;button.append(title,info);button.onclick=()=>choose(row);fragment.append(button);}
 if(!filtered.length){const p=document.createElement('p');p.id='empty';p.textContent='没有匹配案例，请调整筛选条件。';fragment.append(p);}
 $('list').replaceChildren(fragment);nav();
}
function nav(){const i=filtered.findIndex(r=>r.case_id===selected?.case_id);$('prev').disabled=!ready||i<=0;$('next').disabled=!ready||!filtered.length||i>=filtered.length-1;}
function details(row){const dl=document.createElement('dl');for(const [k,v] of Object.entries({'案例':row.case_id,'类型':kinds[row.kind],'分支':row.module,'类别':row.class_name,'帧 / 秒':`${row.frame} / ${row.elapsed_seconds.toFixed(3)}`,'目标 ID':row.previous_tracking_id?`${row.previous_tracking_id} → ${row.tracking_id}`:(row.tracking_id||'—'),'目标距离':`${row.distance_m.toFixed(2)} m`,'置信度':row.score==null?'—':row.score.toFixed(3),'sample':row.sample_token})){const dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=k;dd.textContent=v;dl.append(dt,dd);}$('detail').replaceChildren(dl);}
async function choose(row){
 if(!ready)return;const serial=++request;selected=row;details(row);render();
 try{const id=seekCase(viewer,row);$('selection').textContent=`定位中：${row.case_id} → 帧 ${row.frame}`;
 await new Promise(r=>setTimeout(r,250));if(serial!==request)return;
 const actual=viewer.get_current_time(id,'frame');const timeline=viewer.get_active_timeline(id);
 if(timeline!=='frame'||Math.abs(actual-row.frame)>.001||viewer.get_playing(id))throw new Error('时间轴未完成同步，请重试');
 $('selection').textContent=`${row.case_id} · ${kinds[row.kind]} · ${row.class_name} · 已定位帧 ${row.frame}（${row.elapsed_seconds.toFixed(3)} 秒）· 已暂停`;
 $('selection').dataset.frame=String(actual);location.hash=row.case_id;
 }catch(error){$('selection').textContent=`定位失败：${error.message}`;}
}
for(const id of ['module','kind','className','query'])$(id).addEventListener(id==='query'?'input':'change',render);
$('prev').onclick=()=>choose(filtered[filtered.findIndex(r=>r.case_id===selected?.case_id)-1]);
$('next').onclick=()=>choose(filtered[Math.max(0,filtered.findIndex(r=>r.case_id===selected?.case_id)+1)]);
async function init(){
 const [raw,recording]=await Promise.all([fetch('/cases.json').then(r=>r.arrayBuffer()),fetch('/recording.json').then(r=>r.json())]);
 const data=JSON.parse(new TextDecoder().decode(raw));
 // Ensure the selected cases correspond to the actual exported recording.
 const digest=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',raw)),b=>b.toString(16).padStart(2,'0')).join('');
 if(!recording.failure_overlay||digest!==recording.failure_report_sha256)throw new Error('案例与记录不一致，请重新导出 Rerun 记录');
 rows=data;for(const [k,v] of Object.entries(kinds))$('kind').append(option(k,v));for(const c of [...new Set(rows.map(r=>r.class_name))].sort())$('className').append(option(c,c));render();
 setStatus('加载 Rerun 0.23.4…');await viewer.start(null,$('viewer'),{render_backend:'webgl',hide_welcome_screen:true,width:'100%',height:'100%'});
 setStatus('加载场景记录…');const buffer=await fetch('/mini_scene.rrd').then(r=>{if(!r.ok)throw new Error('记录读取失败');return r.arrayBuffer();});
 const channel=viewer.open_channel('CARperception failures');channel.send_rrd(new Uint8Array(buffer));channel.close();
 const started=Date.now();
 while(true){const id=viewer.get_active_recording_id();const range=id?viewer.get_time_range(id,'frame'):null;
 if(id){viewer.set_playing(id,false);viewer.set_active_timeline(id,'frame');viewer.set_current_time(id,'frame',0);}
 if(range&&range.max>=recording.frame_count-1){await new Promise(r=>setTimeout(r,800));viewer.set_playing(id,false);break;}
 if(Date.now()-started>360000)throw new Error('记录加载超时，请刷新重试');setStatus(`解析场景：${range?Math.min(recording.frame_count,Math.floor(range.max)+1):0} / ${recording.frame_count} 帧`);await new Promise(r=>setTimeout(r,300));}
 ready=true;setStatus(`已就绪 · ${recording.frame_count} 帧 · ${rows.filter(r=>kinds[r.kind]).length} 条案例`);render();
 const linked=rows.find(r=>r.case_id===location.hash.slice(1)&&kinds[r.kind]);await choose(linked||filtered[0]);
}
init().catch(error=>{setStatus('加载失败：'+error.message);$('selection').textContent='请检查服务与记录后刷新页面。';console.error(error);});
