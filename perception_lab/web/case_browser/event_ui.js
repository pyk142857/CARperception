import {kinds} from '/case_logic.mjs';
import {clipFrames,createClipPlayer,matchingCases} from '/event_logic.mjs';
const $=id=>document.getElementById(id);
const labels={original:'未修正／原始判定',gt_missing:'检测正确／GT 漏标',prediction_correct:'检测正确／其他匹配问题',gt_error:'GT 标注错误',confirmed_failure:'确认模型失败',needs_review:'待进一步确认'};
const statuses={pending:'待复核',confirmed:'确认失败',not_issue:'非问题',annotation_error:'标注问题',duplicate:'重复问题',investigating:'调查中',resolved:'已解决'};
function el(tag,text,attrs={}){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;Object.assign(e,attrs);return e;}
function opt(value,text){return el('option',text,{value});}
async function json(url,options){const r=await fetch(url,options),data=await r.json();if(!r.ok)throw new Error(data.error||r.statusText);return data;}

export class EventUI {
 constructor(data,cases,viewer,seek,display=async()=>{}){
  this.display=display;
  this.data=data;this.cases=cases;this.viewer=viewer;this.seek=seek;this.ready=false;
  this.caseMap=new Map(cases.map(r=>[r.case_id,r]));this.events=new Map(data.events.map(e=>[e.event_id,e]));
  this.groups=new Map(data.groups.map(g=>[g.group_id,g]));this.reviews={items:{}};this.labels={items:{}};this.selected=null;
  this.player=createClipPlayer(viewer,last=>{$('selection').textContent='片段播放完成 · 已暂停在帧 '+last.frame;$('selection').dataset.frame=last.frame;});
  for(const [k,v] of Object.entries(kinds))$('kind').append(opt(k,v));
  for(const c of [...new Set(cases.map(r=>r.class_name))].sort())$('className').append(opt(c,c));
  for(const [k,v] of Object.entries(labels))$('labelFilter').append(opt(k,v));
  for(const [k,v] of Object.entries(statuses))$('reviewFilter').append(opt(k,v));
  for(const id of ['view','module','kind','className','query','reviewFilter','confidence','distance','labelFilter'])
   $(id).addEventListener(id==='query'?'input':'change',()=>{this.player.stop();this.render();if(this.filtered.length)this.choose(this.filtered[0]);else{this.choiceToken=(this.choiceToken||0)+1;this.selected=null;$('detail').replaceChildren();$('selection').textContent='没有匹配事件；画面保留上一帧';if(this.ready)this.display($('module').value||'detection',null).catch(e=>{$('selection').textContent=e.message;});}});
  $('clearGroup').onclick=()=>{this.groupId=null;$('groupScope').hidden=true;this.render();};
  $('prev').onclick=()=>this.move(-1);$('next').onclick=()=>this.move(1);
  $('exportReviews').onclick=async()=>{
   try{const d=await json('/api/reviews');const url=URL.createObjectURL(new Blob([JSON.stringify(d,null,2)],{type:'application/json'}));
    const a=el('a','',{href:url,download:'reviews_'+data.dataset_id.slice(0,12)+'.json'});a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
   }catch(e){$('selection').textContent=e.message;}
  };
  $('exportLabels').onclick=async()=>{try{const data=await json('/api/labels');const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const a=el('a','',{href:url,download:'case_labels_'+data.dataset_id.slice(0,12)+'.json'});a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}catch(e){$('selection').textContent=e.message;}};
  $('overview').textContent=`${data.events.length} 事件 · ${data.groups.length} 分组 · ${cases.filter(r=>kinds[r.kind]).length} 原始失败`;
 }
 async init(){
  this.reviews=await json('/api/reviews');if(this.reviews.dataset_id!==this.data.dataset_id)throw new Error('复核数据版本不一致');
  this.labels=await json('/api/labels');if(this.labels.dataset_id!==this.data.cases_sha256)throw new Error('人工标签与案例版本不一致');
  this.render();this.openHash(false);
 }
 id(row){return row.event_id||row.group_id||row.case_id;}
 status(id){return this.reviews.items[id]?.status||'pending';}
 label(id){return this.labels.items[id]?.label||'original';}
 matches(row){return matchingCases(row,$('confidence').value,this.caseMap,this.events,$('distance').value).filter(c=>!$('labelFilter').value||this.label(c.case_id)===$('labelFilter').value);}
 focus(row){const matches=this.matches(row);return matches.find(c=>c.case_id===row.representative_case_id)||matches[0];}
 render(){
  const mode=$('view').value;let list=mode==='events'?this.data.events:mode==='groups'?this.data.groups:this.cases.filter(r=>kinds[r.kind]);
  if(this.groupId&&mode==='events')list=list.filter(e=>e.group_id===this.groupId);
  list=list.filter(r=>this.matches(r).length&&(!$('module').value||r.module===$('module').value)&&(!$('kind').value||r.kind===$('kind').value)&&
   (!$('className').value||r.class_name===$('className').value)&&
   (mode==='cases'||!$('reviewFilter').value||this.status(this.id(r))===$('reviewFilter').value)&&
   (! $('query').value.trim()||JSON.stringify(r).toLowerCase().includes($('query').value.trim().toLowerCase())));
  this.filtered=list;$('reviewFilter').disabled=mode==='cases';
  $('count').textContent=`${list.length} 条匹配${mode==='groups'?'分组（按事件数排序）':mode==='events'?'事件':'原始案例'}`;
  const fragment=document.createDocumentFragment();
  for(const row of list){
   const id=this.id(row),button=el('button',undefined,{className:'case'});
   button.dataset.caseId=id;button.setAttribute('aria-pressed',this.selected&&this.id(this.selected)===id);
   const title=el('strong');title.append(el('span','原始 '+kinds[row.kind],{className:row.kind}),
     el('span',mode==='groups'?row.event_count+'事件（总计）':mode==='events'?`帧 ${row.start_frame}–${row.end_frame}`:'帧 '+row.frame));
   button.append(title,el('small',`${row.class_name} · ${row.module==='detection'?'检测':'跟踪'} · ${mode==='groups'?row.distance_bucket:mode==='events'?row.failure_frame_count+'个失败帧':id}`));
   const matched=this.matches(row);
   const judged=matched.filter(c=>this.label(c.case_id)!=='original');
   if(mode==='cases')button.append(el('small','人工：'+labels[this.label(row.case_id)]));
   else if(judged.length)button.append(el('small',`${judged.length} 条有人工修正（按逐帧记录）`));
   button.append(el('small',`${matched.length} 条符合筛选`));
   if(mode!=='cases')button.append(el('small',statuses[this.status(id)]));
   button.onclick=()=>this.choose(row);fragment.append(button);
  }
  if(!list.length)fragment.append(el('p','没有匹配结果，请调整筛选。',{id:'empty'}));
  $('list').replaceChildren(fragment);this.nav();
 }
 nav(){
  const i=this.filtered.findIndex(r=>this.selected&&this.id(r)===this.id(this.selected));
  $('prev').disabled=i<=0;$('next').disabled=!this.filtered.length||i>=this.filtered.length-1;
 }
 move(offset){const i=this.filtered.findIndex(r=>this.selected&&this.id(r)===this.id(this.selected));
  const row=this.filtered[Math.max(0,i+offset)];if(row)this.choose(row);}
 async choose(row,doSeek=true){
  const token=this.choiceToken=(this.choiceToken||0)+1;
  this.player.stop();this.selected=row;location.hash=this.id(row);this.render();this.details(row);
  const focused=row.event_id?this.focus(row):row.case_id?row:null;
  this.focusedCase=focused;
  if(this.ready){try{await this.display(row.module,focused?.case_id||null);}catch(e){$('selection').textContent='高亮加载失败：'+e.message;return;}}
  if(token!==this.choiceToken)return;
  if(!row.event_id&&row.group_id){$('selection').textContent='问题分组：相似现象，根因待复核';return;}
  const original=row.event_id?this.focus(row):row;
  if(this.ready&&doSeek)await this.seek(original);
  else $('selection').textContent='已选择 '+this.id(row)+' · 记录就绪后可定位';
 }
 details(row){
  const box=$('detail');box.replaceChildren();const isGroup=!row.event_id&&!!row.group_id;
  box.append(el('h2',isGroup?'问题分组':row.event_id?'目标事件':'原始案例'));
  const dl=el('dl');
  const scored=this.matches(row).map(c=>c.confidence).filter(Number.isFinite);
  const distances=this.matches(row).map(c=>c.distance_m).filter(Number.isFinite);
  const fields={'符合筛选的距离':distances.length?`${Math.min(...distances).toFixed(2)}–${Math.max(...distances).toFixed(2)} m`:'N/A','符合筛选的置信度':scored.length?`${Math.min(...scored).toFixed(3)}–${Math.max(...scored).toFixed(3)}`:'N/A（无匹配预测）','编号':this.id(row),'现象':kinds[row.kind],'类别':row.class_name};
  if(row.event_id)Object.assign(fields,{'帧范围':`${row.start_frame}–${row.end_frame}`,'观测跨度':row.span_seconds.toFixed(3)+'秒',
   '失败帧数':row.failure_frame_count,'关联方式':row.association_method,'关联可信度':row.association_quality,
   '转换次数':row.transition_count,'距离中位数':row.median_distance_m.toFixed(2)+'m'});
  else if(isGroup)Object.assign(fields,{'事件数':row.event_count,'原始失败':row.case_count,'距离分组':row.distance_bucket,'覆盖场景':row.scene_tokens.length});
  else Object.assign(fields,{'帧':row.frame,'sample':row.sample_token,'目标距离':row.distance_m.toFixed(2)+'m'});
  for(const [key,value] of Object.entries(fields))dl.append(el('dt',key),el('dd',String(value)));
  box.append(dl);
  if(isGroup){
   box.append(el('p','分组描述相似现象；没有曝光量，不显示失败率。'));
   const open=el('button','查看组内事件',{id:'openGroup'});open.onclick=()=>{
    this.groupId=row.group_id;$('view').value='events';$('groupScope').hidden=false;
    $('groupLabel').textContent=row.class_name+' / '+row.distance_bucket;
    $('reviewFilter').value='';this.render();if(this.filtered[0])this.choose(this.filtered[0]);
   };box.append(open);
  }else{
   const event=row.event_id?row:this.data.events.find(e=>e.case_ids.includes(row.case_id));
   const controls=el('div',undefined,{className:'nav'});
   const jump=el('button','定位符合筛选的帧',{id:'jump',disabled:!this.ready});jump.onclick=()=>this.choose(row);
   const play=el('button','播放前后 2 秒',{id:'playClip',disabled:!this.ready});
   play.onclick=()=>{this.player.play(clipFrames(event,this.data.frames));$('selection').textContent='播放事件片段（按采样时间）';};
   const stop=el('button','停止',{id:'stopClip'});stop.onclick=()=>{this.player.stop();$('selection').textContent='片段已停止';};
   controls.append(jump,play,stop);box.append(controls);
   if(row.event_id){
    const members=el('details');members.append(el('summary','展开 '+this.matches(row).length+' 条符合筛选的记录（共 '+row.case_ids.length+' 条）'));
    for(const r of this.matches(row)){const id=r.case_id,b=el('button',id+' · 帧 '+r.frame+' · '+r.distance_m.toFixed(2)+' m · score '+(r.confidence===null?'N/A':r.confidence.toFixed(3)),{className:'member'});
     b.onclick=async()=>{this.player.stop();this.focusedCase=r;this.labelEditor(r);const token=this.choiceToken=(this.choiceToken||0)+1;
      if(this.ready){try{await this.display(r.module,r.case_id);if(token===this.choiceToken)await this.seek(r);}catch(e){$('selection').textContent='高亮加载失败：'+e.message;}}
      else $('selection').textContent='记录尚未就绪';};members.append(b);}
    box.append(members);
    if(row.transitions.length)box.append(el('p',row.transitions.map(t=>`帧${t.frame}: ${t.previous_id}→${t.new_id}`).join('；')));
   }else if(event){const b=el('button','查看所属事件',{id:'parentEvent'});b.onclick=()=>{$('view').value='events';this.choose(event);};box.append(b);}
  }
  if(!isGroup)this.labelEditor(row.event_id?this.focus(row):row);
 }
 labelEditor(row){
  document.getElementById('labelEditor')?.remove();if(!row)return;
  const panel=el('section',undefined,{id:'labelEditor'});$('detail').append(panel);
  const record=this.labels.items[row.case_id]||{revision:0,label:'original',note:''};
  panel.append(el('h2','人工标签 · '+row.case_id+' · 帧 '+row.frame));
  panel.append(el('p','只修改此条记录的人工判定，不修改 GT 框或原始指标。'));
  const select=el('select',undefined,{id:'caseLabel'});
  for(const [key,value] of Object.entries(labels)){const option=opt(key,value);option.disabled=['gt_missing','prediction_correct'].includes(key)&&row.kind!=='false_positive';select.append(option);}select.value=record.label;
  const input=el('textarea',undefined,{id:'labelNote',value:record.note,maxLength:1000,placeholder:'可选：修正依据，例如该帧图像中可见真实目标'});
  const save=el('button','保存人工标签',{id:'saveLabel'}),reload=el('button','重新读取',{id:'reloadLabel'});
  const message=el('p','',{id:'labelMessage',role:'status'});panel.append(select,input,save,reload,message);
  reload.onclick=async()=>{try{this.labels=await json('/api/labels');this.render();this.labelEditor(row);}catch(e){message.textContent=e.message;}};
  save.onclick=async()=>{save.disabled=true;try{
   const item=await json('/api/labels',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({dataset_id:this.labels.dataset_id,case_id:row.case_id,revision:record.revision,label:select.value,note:input.value})});
   this.labels.items[row.case_id]=item;record.revision=item.revision;this.render();message.textContent='已保存 · '+labels[item.label]+' · 版本 '+item.revision+(!this.matches(row).length?'；当前记录不符合列表筛选':'');
  }catch(e){message.textContent='保存失败：'+e.message;}finally{save.disabled=false;}};
 }
 openHash(seek=true){
  const key=location.hash.slice(1);let row=this.events.get(key)||this.groups.get(key)||this.caseMap.get(key);
  if(row){$('view').value=row.event_id?'events':row.group_id?'groups':'cases';this.choose(row,seek);}
  else if(!this.selected&&this.data.events.length)this.choose(this.data.events[0],seek);
 }
 async setReady(){
  this.ready=true;
  if(this.selected)await this.display(this.selected.module,this.focusedCase?.case_id||null);
  for(const id of ['jump','playClip'])if($(id))$(id).disabled=false;
  if(this.selected&&(this.selected.event_id||this.selected.case_id)){
   const row=this.focusedCase||(this.selected.event_id?this.focus(this.selected):this.selected);
   this.seek(row);
  }
 }
}
