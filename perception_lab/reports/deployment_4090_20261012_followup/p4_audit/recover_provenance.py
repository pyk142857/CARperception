"""Recover historical source in memory from five inspected local edit records.
No historical shell command or model code is executed; writes only this folder.
"""
import pathlib,json,hashlib,io,difflib,ast
out=pathlib.Path(__file__).resolve().parent
log=pathlib.Path('/home/minglei/.codex/sessions/2026/10/12/rollout-2026-10-12T08-46-52-01a12d93-a0ed-7202-94cb-2920e4222c84.jsonl')
records={i:json.loads(l) for i,l in enumerate(log.open(),1) if i in [89,142,163,215,270]}
cmds={i:r['payload']['item']['command'][-1] for i,r in records.items()}
src=cmds[89].split("<<'PY'\n",1)[1].split('\nPY\n',1)[0]+'\n'
# In-memory file shim limits reviewed string-only edits to this one source.
state={'source':src}
class Mem(io.StringIO):
 def __init__(self,mode):super().__init__(state['source'] if mode=='r' else '');self.mode=mode
 def write(self,s):state['source']=s;return len(s)
def memory_open(p,mode='r'):
 assert p=='perception_lab/tools/gaussianworld_4090.py'
 assert mode in ['r','w']
 return Mem(mode)
steps=[]
for i in [142,163,215,270]:
 cmd=cmds[i]
 if '\n' not in cmd:cmd=cmd.replace('\\n','\n')
 script=cmd.split("<<'PY'\n",1)[1].split('\nPY\n',1)[0]
 # Scripts are inspected edit expressions only; no builtins other than memory open / str.
 exec(compile(script,f'<historical-edit-{i}>','exec'),{'__builtins__':{'open':memory_open}})
 steps.append({'session_line':i,'timestamp':records[i]['timestamp'],'sha256':hashlib.sha256(state['source'].encode()).hexdigest(),'bytes':len(state['source'].encode())})
s=state['source'];current=(out.parents[2]/'tools/gaussianworld_4090.py').read_text()
expected='9b99b79bba08c2838f7c57c8ec51a3ea840a7a788559d4df85782542221124c3'
result={'session_file':log.name,'reviewed_edit_lines':[89,142,163,215,270],'steps':steps,'recovered_sha256':hashlib.sha256(s.encode()).hexdigest(),'expected_sha256':expected,'matches':hashlib.sha256(s.encode()).hexdigest()==expected,'diff_to_current':list(difflib.unified_diff(s.splitlines(True),current.splitlines(True),fromfile='provenance_source',tofile='ce24482_source'))}
result['AST_equal']=ast.dump(ast.parse(s))==ast.dump(ast.parse(current))
result['exact_difference_verified']=s.replace('\n    \n','\n\n')==current
(out/'provenance_recovery.json').write_text(json.dumps(result,indent=2)+'\n')
if result['matches']:(out/'gaussianworld_4090_provenance_snapshot.py').write_text(s)
print(json.dumps(result,indent=2))
