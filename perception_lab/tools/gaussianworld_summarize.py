#!/usr/bin/env python3
"""Summarize actual occupancy grids and render xyz-oriented semantic BEV."""
import argparse,csv,json,shutil,re,hashlib
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args();a.report.mkdir(parents=True,exist_ok=True)
    m=json.loads((a.run/'manifest.json').read_text());rows=list(csv.DictReader((a.run/'timing.csv').open()));summary={k:v for k,v in m.items() if k!='frames'}
    summary['inference_status']=m['status'];summary['evaluation_status']=m['evaluation']['status']
    if m.get('label_independence_audit'):
        summary['label_independence_audit']=m['label_independence_audit']
        (a.report/'audit.json').write_text(json.dumps(m['label_independence_audit'],indent=2))
    elif (a.report/'audit.json').exists():summary['label_independence_audit']=json.loads((a.report/'audit.json').read_text())
    summary['frames']=len(m['frames']);summary['sample_tokens']=[f['sample_token'] for f in m['frames']]
    for metric in ['model_ms','e2e_ms','preprocess_ms','h2d_ms','postprocess_d2h_ms']:
        v=np.array([float(r[metric]) for r in rows]);summary[metric]={'mean':float(v.mean()),'p50':float(np.median(v)),'p95':float(np.percentile(v,95)),'first':float(v[0]),'sum':float(v.sum())}
    for metric in ['allocated_peak_mb','reserved_peak_mb']:summary[metric]=max(float(r[metric]) for r in rows)
    summary['process_gpu_memory_peak_mib']=max(int(r['process_gpu_memory_mib'].split(',')[-1]) for r in rows)
    summary['model_fps']=1000/summary['model_ms']['mean'];summary['e2e_fps']=1000/summary['e2e_ms']['mean'];summary['grid_validation']={'all_shapes_correct':True,'all_classes_in_range':True}
    counts=[]
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap,BoundaryNorm
    colors=['#808080','#d08000','#d00000','#00c0c0','#0066ff','#ffcc00','#b00000','#ff00ff','#ff8000','#804080','#4040c0','#705070','#ff9696','#505050','#a06030','#d0d0d0','#008030','#ffffff']
    cmap=ListedColormap(colors);norm=BoundaryNorm(np.arange(-.5,18.5),18)
    for i,fr in enumerate(m['frames']):
        z=np.load(fr['occupancy_path'])['occupancy'];summary['grid_validation']['all_shapes_correct'] &= z.shape==(200,200,16);summary['grid_validation']['all_classes_in_range'] &= bool(((z>=0)&(z<=17)).all());counts.append({'frame_id':i,'sample_token':fr['sample_token'],**{name:int((z==c).sum()) for c,name in enumerate(m['classes'])}})
        if i in [0,len(m['frames'])//2,len(m['frames'])-1]:
            valid=(z>0)&(z<17);top=np.where(valid,np.arange(16)[None,None,:],-1).max(axis=2);bev=np.take_along_axis(z,np.maximum(top,0)[...,None],axis=2)[...,0];bev[top<0]=17
            fig,ax=plt.subplots(figsize=(11,8));im=ax.imshow(bev.T,origin='lower',extent=[-50,50,-50,50],cmap=cmap,norm=norm,interpolation='nearest');ax.set_xlabel('LIDAR_TOP x [m]');ax.set_ylabel('LIDAR_TOP y [m]');ax.set_title(f'GaussianWorld FP32 / scene-stream / frame {i}\n{fr["sample_token"]}\nHighest occupied voxel per XY column; prediction only');ax.plot(0,0,'k+',ms=10);cb=fig.colorbar(im,ax=ax,ticks=range(18));cb.ax.set_yticklabels(m['classes']);fig.tight_layout();fig.savefig(a.report/f'bev_{i:02d}.png',dpi=140);plt.close(fig)
    (a.report/'summary.json').write_text(json.dumps(summary,indent=2))
    with (a.report/'class_counts.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=counts[0]);w.writeheader();w.writerows(counts)
    (a.report/'timing.csv').write_text((a.run/'timing.csv').read_text())
    # Publish the newly measured grids, never leave a previous run behind.
    (a.report/'grids').mkdir(exist_ok=True);published=[]
    for fr in m['frames']:
        src=Path(fr['occupancy_path']);dst=a.report/'grids'/src.name;shutil.copy2(src,dst)
        if hashlib.sha256(dst.read_bytes()).hexdigest()!=fr['sha256']:raise ValueError('Grid hash mismatch')
        published.append({**{k:v for k,v in fr.items() if k!='inputs'},'published_grid':'grids/'+src.name})
    (a.report/'grids_manifest.json').write_text(json.dumps({'frames':published,'upstream_commit':m['upstream_commit'],'checkpoint_sha256':m['checkpoint_sha256'],'local_manifest':str(a.run/'manifest.json')},indent=2))
    provenance=a.report/'provenance.json'
    if provenance.exists():
        prov=json.loads(provenance.read_text())
        for art in prov.get('artifacts',[]):
            path=Path(art['path'])
            if not path.is_absolute():path=Path(__file__).resolve().parents[2]/path
            if path.exists():art.update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)
        prov['published_grids_count']=len(published);prov['published_grids_bytes']=sum(p.stat().st_size for p in (a.report/'grids').glob('*.npz'));provenance.write_text(json.dumps(prov,indent=2))
    report=a.report/'report.md'
    if report.exists():
        text=report.read_text();tim=summary['model_ms'];e2e=summary['e2e_ms']
        replacements={
            '模型平均 / P50 / P95':f"{tim['mean']:.3f} / {tim['p50']:.3f} / {tim['p95']:.3f} ms",
            '模型首帧（独立预热后）':f"{tim['first']:.3f} ms",
            '图像读取、哈希与官方预处理平均':f"{summary['preprocess_ms']['mean']:.3f} ms",
            'H2D / 后处理+D2H 平均':f"{summary['h2d_ms']['mean']:.3f} / {summary['postprocess_d2h_ms']['mean']:.3f} ms",
            '输入到 CPU 类别网格平均 / P50 / P95':f"{e2e['mean']:.3f} / {e2e['p50']:.3f} / {e2e['p95']:.3f} ms",
            '对应模型 / 输入到网格 FPS':f"{summary['model_fps']:.3f} / {summary['e2e_fps']:.3f}",
            'PyTorch allocated / reserved 峰值':f"{summary['allocated_peak_mb']:.3f} / {summary['reserved_peak_mb']:.3f} MB（十进制）",
            '帧边界 nvidia-smi 进程显存最大采样':f"{summary['process_gpu_memory_peak_mib']} MiB",
            '模型构建+严格加载+GPU搬运':f"{summary['cold_model_load_s']:.3f} s（不含 Python 导入）",
        }
        for label,value in replacements.items():text=re.sub(r'^\| '+re.escape(label)+r' \|.*$',f'| {label} | {value} |',text,flags=re.M)
        audit=summary.get('label_independence_audit',{}).get('full_network_repeat')
        if audit:
            text=re.sub(r'完整网络同输入重复则最大差[^，]+、平均差[^，]+，\d+/640000体素类别不同',f"完整网络同输入重复则最大差{audit['max_abs']:.6f}、平均差{audit['mean_abs']:.8f}，{audit['voxel_class_disagreements']}/640000体素类别不同",text)
        text=text.replace('独立三帧运行与完整运行首三帧有3、275、1128个体素类别差异（各640000体素）。','初次独立三帧运行与当时完整运行首三帧有3、275、1128个体素类别差异（各640000体素）；该项为保留的早期重复性审计。')
        report.write_text(text)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
