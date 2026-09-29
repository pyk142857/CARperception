export function clipFrames(event, frames, context=2) {
 return frames.filter(f=>f.scene_token===event.scene_token &&
  f.elapsed_seconds>=event.start_seconds-context && f.elapsed_seconds<=event.end_seconds+context);
}
export function createClipPlayer(viewer,onDone,schedule=setTimeout,cancel=clearTimeout) {
 let timer=null,generation=0,recording=null;
 function stop(){generation++;if(timer!==null)cancel(timer);timer=null;
  if(recording)viewer.set_playing(recording,false);}
 return {stop,play(frames){
  stop();if(!frames.length)return;const token=generation,id=viewer.get_active_recording_id();
  if(!id)throw new Error('记录未加载');recording=id;
  viewer.set_active_timeline(id,'frame');viewer.set_playing(id,false);
  function step(index){
   if(token!==generation)return;
   viewer.set_playing(id,false);viewer.set_current_time(id,'frame',frames[index].frame);
   if(index===frames.length-1){timer=null;onDone(frames[index]);return;}
   const delay=Math.max(1,(frames[index+1].elapsed_seconds-frames[index].elapsed_seconds)*1000);
   timer=schedule(()=>step(index+1),delay);
  }step(0);
 }};
}
