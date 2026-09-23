import argparse,json,platform,shutil,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
commands=[['nvidia-smi'],['df','-h'],['free','-h'],['lscpu'],['gcc','--version'],['/usr/local/cuda-11.8/bin/nvcc','--version'],['/usr/local/cuda-12.2/bin/nvcc','--version'],['/home/minglei/anaconda3/bin/conda','env','list']]
r={'python':sys.version,'os':platform.platform(),'disk_free_bytes':shutil.disk_usage(out).free,'probes':[]}
for command in commands:
 try:
  s=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=30);r['probes'].append({'command':command,'exit_code':s.returncode,'output':s.stdout})
 except (OSError,subprocess.TimeoutExpired) as e:r['probes'].append({'command':command,'error':str(e)})
(out/'hardware.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
