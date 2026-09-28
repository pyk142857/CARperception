import assert from 'assert';
import {filterCases,seekCase} from '../web/case_browser/case_logic.mjs';
const rows=[{kind:'id_switch',module:'tracking',class_name:'car',case_id:'case_1',frame:2},{kind:'gap_recovery',module:'tracking',class_name:'car',case_id:'case_2',frame:3}];
assert.equal(filterCases(rows,{module:'tracking'}).length,1);
assert.equal(filterCases(rows,{className:'pedestrian'}).length,0);
assert.equal(filterCases(rows,{query:'case_1'})[0].frame,2);
let calls=[];let v={get_active_recording_id:()=> 'r',get_time_range:()=>({min:0,max:38}),set_playing:(...x)=>calls.push(['pause',...x]),set_active_timeline:(...x)=>calls.push(['timeline',...x]),set_current_time:(...x)=>calls.push(['seek',...x])};
seekCase(v,rows[0]);assert.deepEqual(calls,[['pause','r',false],['timeline','r','frame'],['seek','r','frame',2]]);
assert.throws(()=>seekCase(v,{frame:39}),/尚未加载/);console.log('Filter and seek tests passed');
