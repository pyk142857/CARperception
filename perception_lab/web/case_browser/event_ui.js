import {kinds} from '/case_logic.mjs';
import {clipFrames,createClipPlayer} from '/event_logic.mjs';
const $=id=>document.getElementById(id);
const statuses={pending:'待复核',confirmed:'确认失败',not_issue:'非问题',annotation_error:'标注问题',duplicate:'重复问题',investigating:'调查中',resolved:'已解决'};
function el(tag,text,attrs={}){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;Object.assign(e,attrs);return e;}
function opt(value,text){return el('option',text,{value});}
async function json(url,options){const r=await fetch(url,options),data=await r.json();if(!r.ok)throw new Error(data.error||r.statusText);return data;}

export class EventUI {
 constructor(data,cases,viewer,seek,display=async()=>{}){
  this.display=display;
  this.data=data;this.cases=cases;this.viewer=viewer;this.seek=seek;this.ready=false;
  this.caseMap=new Map(cases.map(r=>[r.case_id,r]));this.events=new Map(data.events.map(e=>[e.event_id,e]));
  this.groups=new Map(data.groups.map(g=>[g.group_id,g]));this.reviews={items:{}};this.selected=null;
  this.player=createClipPlayer(viewer,last=>{$('selection').textContent='片段播放完成 · 已暂停在帧 '+last.frame;$('selection').dataset.frame=last.frame;});
  for(const [k,v] of Object.entries(kinds))$('kind').append(opt(k,v));
  for(const c of [...new Set(cases.map(r=>r.class_name))].sort())$('className').append(opt(c,c));
  for(const [k,v] of Object.entries(statuses))$('reviewFilter').append(opt(k,v));
  for(const id of ['view','module','kind','className','query','reviewFilter'])
   $(id).addEventListener(id==='query'?'input':'change',()=>{this.player.stop();this.render();});
  $('clearGroup').onclick=()=>{this.groupId=null;$('groupScope').hidden=true;this.render();};
  $('prev').onclick=()=>this.move(-1);$('next').onclick=()=>this.move(1);
  $('exportReviews').onclick=async()=>{
   try{const d=await json('/api/reviews');const url=URL.createObjectURL(new Blob([JSON.stringify(d,null,2)],{type:'application/json'}));
    const a=el('a','',{href:url,download:'reviews_'+data.dataset_id.slice(0,12)+'.json'});a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
   }catch(e){$('selection').textContent=e.message;}
  };
  $('overview').textContent=`${data.events.length} 事件 · ${data.groups.length} 分组 · ${cases.filter(r=>kinds[r.kind]).length} 原始失败`;
 }
 async init(){
  this.reviews=await json('/api/reviews');if(this.reviews.dataset_id!==this.data.dataset_id)throw new Error('复核数据版本不一致');
  this.render();this.openHash(false);
 }
 id(row){return row.event_id||row.group_id||row.case_id;}
 status(id){return this.reviews.items[id]?.status||'pending';}
 render(){
  const mode=$('view').value;let list=mode==='events'?this.data.events:mode==='groups'?this.data.groups:this.cases.filter(r=>kinds[r.kind]);
  if(this.groupId&&mode==='events')list=list.filter(e=>e.group_id===this.groupId);
  list=list.filter(r=>(!$('module').value||r.module===$('module').value)&&(!$('kind').value||r.kind===$('kind').value)&&
   (!$('className').value||r.class_name===$('className').value)&&
   (mode==='cases'||!$('reviewFilter').value||this.status(this.id(r))===$('reviewFilter').value)&&
   (! $('query').value.trim()||JSON.stringify(r).toLowerCase().includes($('query').value.trim().toLowerCase())));
  this.filtered=list;$('reviewFilter').disabled=mode==='cases';
  $('count').textContent=`${list.length} 条匹配${mode==='groups'?'分组（按事件数排序）':mode==='events'?'事件':'原始案例'}`;
  const fragment=document.createDocumentFragment();
  for(const row of list){
   const id=this.id(row),button=el('button',undefined,{className:'case'});
   button.dataset.caseId=id;button.setAttribute('aria-pressed',this.selected&&this.id(this.selected)===id);
   const title=el('strong');title.append(el('span',kinds[row.kind],{className:row.kind}),
     el('span',mode==='groups'?row.event_count+'事件':mode==='events'?`帧 ${row.start_frame}–${row.end_frame}`:'帧 '+row.frame));
   button.append(title,el('small',`${row.class_name} · ${row.module==='detection'?'检测':'跟踪'} · ${mode==='groups'?row.distance_bucket:mode==='events'?row.failure_frame_count+'个失败帧':id}`));
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
  if(this.ready)await this.display(row.module);
  if(token!==this.choiceToken)return;
  if(!row.event_id&&row.group_id){$('selection').textContent='问题分组：相似现象，根因待复核';return;}
  const original=row.event_id?this.caseMap.get(row.representative_case_id):row;
  if(this.ready&&doSeek)await this.seek(original);
  else $('selection').textContent='已选择 '+this.id(row)+' · 记录就绪后可定位';
 }
 details(row){
  const box=$('detail');box.replaceChildren();const isGroup=!row.event_id&&!!row.group_id;
  box.append(el('h2',isGroup?'问题分组':row.event_id?'目标事件':'原始案例'));
  const dl=el('dl');
  const fields={'编号':this.id(row),'现象':kinds[row.kind],'类别':row.class_name};
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
   const jump=el('button','定位代表帧',{id:'jump',disabled:!this.ready});jump.onclick=()=>this.choose(row);
   const play=el('button','播放前后 2 秒',{id:'playClip',disabled:!this.ready});
   play.onclick=()=>{this.player.play(clipFrames(event,this.data.frames));$('selection').textContent='播放事件片段（按采样时间）';};
   const stop=el('button','停止',{id:'stopClip'});stop.onclick=()=>{this.player.stop();$('selection').textContent='片段已停止';};
   controls.append(jump,play,stop);box.append(controls);
   if(row.event_id){
    const members=el('details');members.append(el('summary','展开 '+row.case_ids.length+' 条原始记录'));
    for(const id of row.case_ids){const r=this.caseMap.get(id),b=el('button',id+' · 帧 '+r.frame,{className:'member'});
     b.onclick=()=>{this.player.stop();if(this.ready)this.seek(r);else $('selection').textContent='记录尚未就绪';};members.append(b);}
    box.append(members);
    if(row.transitions.length)box.append(el('p',row.transitions.map(t=>`帧${t.frame}: ${t.previous_id}→${t.new_id}`).join('；')));
   }else if(event){const b=el('button','查看所属事件',{id:'parentEvent'});b.onclick=()=>{$('view').value='events';this.choose(event);};box.append(b);}
  }
  if(row.event_id||isGroup)this.reviewForm(box,this.id(row));
 }
 reviewForm(box,id){
  const form=el('form',undefined,{id:'reviewForm'}),record=this.reviews.items[id]||{revision:0,status:'pending'};
  form.append(el('h2','复核结论（本地持久保存）'));
  const state=el('select',undefined,{name:'status',id:'reviewStatus'});
  for(const [k,v] of Object.entries(statuses))state.append(opt(k,v));state.value=record.status;
  const statusLabel=el('label','状态');statusLabel.append(state);form.append(statusLabel);
  for(const [key,label] of Object.entries({owner:'负责人',root_cause:'根因 / 调查记录',action:'改进措施',validation:'验证证据（已解决必填）',duplicate_of:'重复目标编号'})){
   const l=el('label',label),input=el(key==='owner'||key==='duplicate_of'?'input':'textarea',undefined,{name:key,value:record[key]||'',maxLength:4000});
   l.append(input);form.append(l);
  }
  const save=el('button','保存复核',{type:'submit',id:'saveReview'}),refresh=el('button','重新读取',{type:'button',id:'reloadReview'});
  const message=el('p','',{id:'reviewMessage',role:'status'});
  refresh.onclick=async()=>{try{this.reviews=await json('/api/reviews');this.details(this.selected);this.render();}catch(e){message.textContent=e.message;}};
  form.append(save,refresh,message);box.append(form);
  form.onsubmit=async e=>{
   e.preventDefault();save.disabled=true;
   const payload=Object.fromEntries(new FormData(form));Object.assign(payload,{dataset_id:this.data.dataset_id,target_id:id,revision:record.revision});
   try{const item=await json('/api/reviews',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    this.reviews.items[id]=item;record.revision=item.revision;message.textContent='已保存 · 版本 '+item.revision;this.render();
   }catch(error){message.textContent='保存失败：'+error.message;}finally{save.disabled=false;}
  };
 }
 openHash(seek=true){
  const key=location.hash.slice(1);let row=this.events.get(key)||this.groups.get(key)||this.caseMap.get(key);
  if(row){$('view').value=row.event_id?'events':row.group_id?'groups':'cases';this.choose(row,seek);}
  else if(!this.selected&&this.data.events.length)this.choose(this.data.events[0],seek);
 }
 async setReady(){
  this.ready=true;
  if(this.selected)await this.display(this.selected.module);
  for(const id of ['jump','playClip'])if($(id))$(id).disabled=false;
  if(this.selected&&(this.selected.event_id||this.selected.case_id)){
   const row=this.selected.event_id?this.caseMap.get(this.selected.representative_case_id):this.selected;
   this.seek(row);
  }
 }
}
