"""Per-frame CPU cProfile decomposition; inclusive categories are not additive."""
import cProfile,json,time
from tracking_gpu_geometry_20261013 import grae,REPORT,array_hash
from tracking_gpu_inputs_20261013 import paths,SPECS,sha

def main():
 rows=[];reads=[]
 expected={r['sample_token']:r['input_array_sha256'] for r in json.loads((REPORT/'cpu_profile.json').read_text())['rows']}
 for split in SPECS:
  dp,pp=paths(split);start=time.perf_counter();raw=json.loads(dp.read_text())['results'];packets=json.loads(pp.read_text());reads.append(dict(split=split,json_packets_ms=(time.perf_counter()-start)*1000))
  for index,p in enumerate(packets):
   profiler=cProfile.Profile();start=time.perf_counter();profiler.enable();frame=grae.frame_arrays(raw[p['sample_token']],p);profiler.disable();elapsed=(time.perf_counter()-start)*1000
   if array_hash(frame)!=expected[p['sample_token']]:raise ValueError('Profile input mismatch')
   entries=[]
   for item in profiler.getstats():
    code=item.code
    label=code if isinstance(code,str) else f'{code.co_filename}:{code.co_firstlineno}:{code.co_name}'
    entries.append(dict(function=label,calls=item.callcount,self_ms=item.inlinetime*1000,inclusive_ms=item.totaltime*1000))
   categories={
    'reader_arrays_allowlist_validation_result_index_score_filter_self':lambda s:'grae3dmot_2025.py' in s,
    'bbox_constructor_and_corners':lambda s:'/data_protos/bbox.py:' in s,
    'quaternion_and_yaw':lambda s:'pyquaternion/' in s,
    'coarse_grid':lambda s:'bbox_coarse_hash.py:' in s,
    'sort':lambda s:'argsort' in s,
    'candidate_and_greedy_nms_self':lambda s:'/preprocessing/nms.py:' in s,
    'intersection_and_height':lambda s:'/utils/geometry.py:' in s and ':iou3d' in s,
    'polygon_shapely':lambda s:'shapely/' in s,
   }
   values={key:dict(self_ms=sum(x['self_ms'] for x in entries if pred(x['function'])),inclusive_ms=sum(x['inclusive_ms'] for x in entries if pred(x['function'])),calls=sum(x['calls'] for x in entries if pred(x['function']))) for key,pred in categories.items()}
   rows.append(dict(split=split,index=index,sample_token=p['sample_token'],profiled_ms=elapsed,categories=values,functions=sorted(entries,key=lambda x:x['inclusive_ms'],reverse=True)[:35]))
 result=dict(status='passed',frames=len(rows),json_reads=reads,rows=rows,scope='Unmodified per-frame cProfile. Self-time category sums are disjoint for named functions; inclusive times nest and MUST NOT be summed. Reader line-level allowlist/array/index/score work grouped in reader self because vectorized list/dict expressions cannot be reliably split without changing code. Full function records and prior plain/instrumented pairs retained.',source_sha256=sha(__file__))
 (REPORT/'cpu_segments.json').write_text(json.dumps(result,indent=2)+'\n');print('profiled',len(rows))
if __name__=='__main__':main()
