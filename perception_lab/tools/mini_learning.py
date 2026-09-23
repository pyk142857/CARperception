"""Verify recorded mini evidence and build a teaching replay; never run benchmark jobs."""
import html
import json
import os
from pathlib import Path
import cv2
import numpy as np
from evidence import reusable

ROOT = Path(__file__).resolve().parents[1]
STEPS = [
 ('M00','smoke','数据与坐标','六相机、点云、雷达及标定 → 按时间戳变换到参考坐标系 → 投影图。GT 图仅用于理解标注。','prepare_mini.py'),
 ('M01','smoke','2D 检测','六张图像 → resize / 归一化 → YOLOv8s → 类别、置信度、二维框。','vision_infer.py'),
 ('M02','smoke','语义分割','六张图像 → SegFormer B0 → 每个像素的类别编号与分割叠加图。','segformer_infer.py'),
 ('M03','smoke','深度','六张图像 → Metric Depth Anything V2 → 单位为米的深度数组与颜色图。颜色是展示，数值看 NPY。','vision_infer.py'),
 ('M04','history_smoke','PointPillars','当前点云与历史 sweeps → 坐标对齐与体素化 → 三维框。与 CenterPoint 是并列模型。','lidar_smoke.py'),
 ('M05','mini_scene','CenterPoint 连续检测','39 帧点云及历史 sweeps → 逐帧三维框与速度 → BEV 回放。','lidar_scene.py'),
 ('M11','mini_tracking_M05','跨帧跟踪','本次 CenterPoint 检测 → 转到全局坐标 → 作者跟踪器进行跨帧关联 → ID 与轨迹。','track_mini.py'),
 ('M12','mini_onnx_M01','导出与校验','YOLO 权重 → ONNX 图 → 对同一 mini 图像运行 PyTorch GPU / ORT GPU → 比较数值。','export_yolo.py'),
]


def main():
 status=json.loads((ROOT/'results/status.json').read_text())
 packets=json.loads((ROOT/'manifests/frame_packets.json').read_text())
 records={}
 for mid,stage,*_ in STEPS:
  record=status['modules'].get(mid,{}).get(stage,{})
  if not reusable(record,record.get('fingerprint')):
   raise RuntimeError('Missing or corrupt recorded evidence: '+mid+':'+stage)
  records[mid]=record
 def directory(mid):
  return Path(records[mid]['log']).parent
 detections=json.loads((directory('M05')/'predictions.json').read_text())
 tracks=json.loads((directory('M11')/'predictions.json').read_text())
 tokens=[p['sample_token'] for p in packets]
 assert len(tokens)==39 and len(set(tokens))==39
 assert tokens==[p['sample_token'] for p in detections]==[p['sample_token'] for p in tracks]
 check=json.loads((directory('M12')/'numerical_check.json').read_text())
 assert check['image']==packets[0]['sensors']['CAM_FRONT']['path']
 assert check['finite'] and 'CUDAExecutionProvider' in check['profile_providers']
 out=ROOT/'reports';out.mkdir(exist_ok=True)
 video=ROOT/'outputs/replay/mini_learning.mp4'
 video.parent.mkdir(parents=True,exist_ok=True)
 caps=[cv2.VideoCapture(str(directory('M05')/'centerpoint_bev.mp4')),cv2.VideoCapture(str(directory('M11')/'mini_tracking_M05.mp4'))]
 # Discover the detector video name from its recorded run when named differently.
 if not caps[0].isOpened():
  caps[0].release();caps[0]=cv2.VideoCapture(str(next(directory('M05').glob('*.mp4'))))
 writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),2,(1440,540))
 if not writer.isOpened():raise RuntimeError('Cannot open video writer')
 for i,packet in enumerate(packets):
  images=[cv2.imread(packet['sensors']['CAM_FRONT']['path'])]
  for cap in caps:
   ok,frame=cap.read()
   if not ok:raise RuntimeError('Replay has missing frame '+str(i))
   images.append(frame)
  canvas=np.full((540,1440,3),245,np.uint8)
  for j,(im,label) in enumerate(zip(images,['CAM_FRONT (raw)','CenterPoint (detections)','PubTracker (track IDs)'])):
   if im is None:raise RuntimeError('Missing camera image')
   h,w=im.shape[:2];scale=min(480/w,470/h);w,h=int(w*scale),int(h*scale)
   x=j*480+(480-w)//2;y=40+(470-h)//2
   canvas[y:y+h,x:x+w]=cv2.resize(im,(w,h))
   cv2.putText(canvas,label,(j*480+12,28),cv2.FONT_HERSHEY_SIMPLEX,.58,(20,30,40),1,cv2.LINE_AA)
  cv2.putText(canvas,f'mini scene-0061 | {i+1}/39 | timestamp {packet["timestamp_us"]} us | playback 2 fps',(15,530),cv2.FONT_HERSHEY_SIMPLEX,.55,(20,30,40),1,cv2.LINE_AA)
  writer.write(canvas)
 writer.release()
 for cap in caps:
  ok,_=cap.read();cap.release()
  if ok:raise RuntimeError('Replay has unexpected extra frames')
 cap=cv2.VideoCapture(str(video));count=0
 while True:
  ok,_=cap.read()
  if not ok:break
  count+=1
 cap.release()
 assert count==39
 def rel(path):return html.escape(os.path.relpath(path,out))
 sections=[]
 for mid,stage,title,explanation,source in STEPS:
  d=directory(mid)
  images=list(d.glob('*.jpg'))+list(d.glob('*.png'))
  picture=''.join(f'<img loading="lazy" src="{rel(p)}" alt="{mid} output">' for p in images[:2])
  artifacts=[p for p in d.glob('*.json') if p.name!='status.json']
  links=' · '.join(f'<a href="{rel(p)}">{html.escape(p.name)}</a>' for p in artifacts)
  sections.append(f'<section><h2>{mid} {title}</h2><p>{explanation}</p><p><a href="../tools/{source}">对应代码</a> · {links}</p>{picture}</section>')
 body='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>mini 感知流程学习</title>
 <style>body{max-width:1200px;margin:32px auto;padding:20px;font:17px/1.7 system-ui;color:#192c3c;background:#f6f8fa}section{background:white;padding:24px;margin:20px 0}img{max-width:48%;vertical-align:top}video{width:100%}a{color:#1766b3}</style>
 <h1>用 nuScenes mini 理解感知流程</h1><p>学习目标：读懂输入 → 预处理 → 模型 → 后处理 → 跟踪 / 导出 → 回放。下方展示已验证的真实输出。</p>
 <p>图像分别进入检测、分割、深度三个并列分支；点云进入三维检测；CenterPoint 检测再送入跟踪器。它们并非全部串联，也尚未进行多传感器模型融合。</p>
 <p>数据范围：scene-0061，连续 39 帧。2D 三个模型演示首个时刻的六路相机；连续视频展示原始前视相机、三维检测与跟踪。2 fps 是播放速率，不是推理速度。</p>
 <video controls preload="metadata" src="../outputs/replay/mini_learning.mp4"></video>
 <p>验收依据：八个必需阶段的输出哈希有效，检测 / 跟踪与传感器帧 token 一致，视频完整解码，mini 图像导出数值校验通过。复用的旧输出不代表当前代码已全部重新推理。</p>'''
 body+=''.join(sections)+'<p>M06–M10、TensorRT、微调训练为扩展。mini 结果用于理解流程；原全量 A/B/C 验收不适用于本学习目标。</p><p><a href="../LEARNING_GUIDE.md">学习说明与复现命令</a> · <a href="index.html">原始详细报告</a></p></html>'
 (out/'mini_learning.html').write_text(body)
 summary={'scope':'mini_learning','status':'passed','evidence_type':'verified_existing_runs_plus_mini_export','stages':[m+':'+s for m,s,*_ in STEPS],'run_ids':{m:r['run_id'] for m,r in records.items()},'frame_count':count,'sample_tokens':tokens,'video':str(video),'formal_benchmark':False}
 (ROOT/'results/mini_learning.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
 print(json.dumps({'status':'passed','stages':len(STEPS),'decoded_frames':count,'report':str(out/'mini_learning.html')}))

if __name__=='__main__':main()
