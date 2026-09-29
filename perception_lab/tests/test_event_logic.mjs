import assert from 'assert';
import {clipFrames,createClipPlayer} from '../web/case_browser/event_logic.mjs';
const frames=[0,1,2,3,4].map(i=>({frame:i,elapsed_seconds:i*.5,scene_token:i<4?'a':'b'}));
const clip=clipFrames({scene_token:'a',start_seconds:.5,end_seconds:1},frames,2);
assert.deepEqual(clip.map(f=>f.frame),[0,1,2,3]);
const calls=[],queue=[];
const viewer={get_active_recording_id:()=> 'r',set_playing:(id,v)=>calls.push(['playing',v]),
 set_active_timeline:()=>{},set_current_time:(id,t,f)=>calls.push(['frame',f])};
let finished=false;
const player=createClipPlayer(viewer,()=>{finished=true},fn=>{queue.push(fn);return queue.length},()=>{});
player.play(clip);while(queue.length)queue.shift()();
assert.equal(finished,true);assert.deepEqual(calls.filter(x=>x[0]==='frame').map(x=>x[1]),[0,1,2,3]);
finished=false;calls.length=0;player.play(clip);player.stop();while(queue.length)queue.shift()();
assert.equal(finished,false);assert.equal(calls.filter(x=>x[0]==='frame').length,1);
console.log('Clip boundaries, completion and cancellation passed');
