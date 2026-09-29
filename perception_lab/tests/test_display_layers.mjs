import assert from 'assert';
import {createDisplayController} from '../web/case_browser/display_layers.mjs';
async function test(){
 const requests=[],sent=[],statuses=[];let frame=7;
 const v={get_active_recording_id:()=> 'r',get_current_time:()=>frame,get_time_range:()=>({min:1,max:1}),
  set_playing:()=>{},set_active_timeline:()=>{},set_current_time:(id,t,value)=>{frame=value},
  open_channel:name=>({send_rrd:()=>sent.push(name),close:()=>{}})};
 const control=createDisplayController(v,async url=>{requests.push(url);return new ArrayBuffer(1)},(...args)=>statuses.push(args),async()=>{});
 await control.apply('detection',false);
 assert.equal(requests.includes('/normal_targets.rrd'),false);
 await control.apply('detection',true);await control.apply('tracking',true);await control.apply('tracking',false);
 assert.equal(requests.filter(r=>r==='/normal_targets.rrd').length,1);
 assert.equal(frame,7);
 assert.deepEqual(statuses[statuses.length-1],['tracking',false]);
 let resolve;
 const race=createDisplayController(v,url=>url==='/normal_targets.rrd'?new Promise(r=>resolve=r):Promise.resolve(new ArrayBuffer(1)),()=>{},async()=>{});
 await race.apply('detection',false);
 const pending=race.apply('detection',true);await race.apply('tracking',false);
 resolve(new ArrayBuffer(1));await pending;
 assert.equal(race.getBranch(),'tracking');
 assert.equal(sent[sent.length-1],'normal target overlay');
 console.log('Default excludes normal request; lazy single load, branch, frame preservation and cancellation passed');
}
test().catch(e=>{console.error(e);process.exit(1)});
