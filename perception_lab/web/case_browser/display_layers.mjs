/** Load normal data only on explicit opt-in; switch a small blueprint, not the scene. */
export function createDisplayController(viewer,fetchBytes,onStatus=()=>{},sleep=ms=>new Promise(r=>setTimeout(r,ms))){
 let normalPromise=null,serial=0,applied='',branch='detection',selection=null;
 const cache=new Map();
 async function bytes(url){if(!cache.has(url))cache.set(url,fetchBytes(url).catch(e=>{cache.delete(url);throw e;}));return cache.get(url);}
 function send(buffer,name){const channel=viewer.open_channel(name);channel.send_rrd(new Uint8Array(buffer));channel.close();}
 async function loadNormal(){
  if(!normalPromise)normalPromise=(async()=>{
   send(await bytes('/normal_targets.rrd'),'normal target overlay');
   const start=Date.now(),id=viewer.get_active_recording_id();
   while(!viewer.get_time_range(id,'normal_ready')){
    if(Date.now()-start>60000)throw new Error('正常目标图层加载超时');
    await sleep(100);
   }
  })().catch(e=>{normalPromise=null;throw e;});
  return normalPromise;
 }
 return {getBranch:()=>branch,getSelection:()=>selection,async apply(module,normal,caseId=null){
  if(!['detection','tracking'].includes(module))throw new Error('未知检测分支');
  branch=module;selection=caseId;const mode=normal?'normal':'errors';
  const key=module+'_'+mode+(caseId?'_'+caseId:''),token=++serial;
  if(key===applied){onStatus(module,normal);return;}
  if(normal){onStatus(module,normal,true);await loadNormal();}
  if(token!==serial)return;
  const buffer=await bytes(caseId?'/selection/'+module+'/'+mode+'/'+caseId+'.rrd':'/view_'+key+'.rrd');if(token!==serial)return;
  const id=viewer.get_active_recording_id(),frame=viewer.get_current_time(id,'frame');
  applied='';send(buffer,'display '+key);await sleep(150);if(token!==serial)return;
  viewer.set_playing(id,false);viewer.set_active_timeline(id,'frame');
  if(frame!==null&&frame!==undefined)viewer.set_current_time(id,'frame',frame);
  applied=key;onStatus(module,normal);
 }};
}
