"""Check the target centering in the 1900x1250 browser_selection fixture."""
import argparse,json
from PIL import Image
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args()
im=Image.open(a.directory/'selected_frame2.png').convert('RGB')
assert im.size==(1900,1250)
measurements={}
for name,box in [('focus',(1059,181,1900,663)),('bev',(370,181,1058,510)),('overview',(1340,859,1624,999))]:
    crop=im.crop(box)
    points=[(i%crop.width,i//crop.width) for i,(r,g,b) in enumerate(crop.getdata()) if 150<=r<=170 and g>=245 and b>=245]
    assert points,name+' has no selected target'
    x,y=zip(*points)
    center=[(min(x)+max(x))/2,(min(y)+max(y))/2]
    if name!='overview':
        assert abs(center[0]-crop.width/2)<3 and abs(center[1]-crop.height/2)<3,(name,center)
    measurements[name]=dict(target_center=center,viewport_center=[crop.width/2,crop.height/2],height=max(y)-min(y))
assert measurements['focus']['height']>5*measurements['overview']['height']
(a.directory/'focus_verification.json').write_text(json.dumps(measurements,indent=2)+'\n')
print('Camera and BEV centered within 3px; close-up target height exceeds overview by 5x')
