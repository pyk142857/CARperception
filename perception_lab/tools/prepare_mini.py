"""Freeze mini manifests and check true asynchronous nuScenes calibration."""
import argparse
import json
from pathlib import Path
import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from geometry import sensor_to_sensor, transform_points, project_depth
from evidence import hash_files


def main():
    p = argparse.ArgumentParser(); p.add_argument('--out', required=True); args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    data = root / 'data/nuscenes'
    nusc = NuScenes(version='v1.0-mini', dataroot=str(data), verbose=True)
    scene = sorted(nusc.scene, key=lambda s:s['name'])[0]
    samples = []; token = scene['first_sample_token']
    while token:
        sample = nusc.get('sample',token);samples.append(sample);token=sample['next']
    assert len(samples) == scene['nbr_samples']
    assert len(set(s['token'] for s in samples)) == len(samples)
    assert all(a['timestamp'] < b['timestamp'] for a,b in zip(samples,samples[1:]))
    packets=[]; images=[]; paths=[]
    def sensor(sd_token):
        sd=nusc.get('sample_data',sd_token)
        cs=nusc.get('calibrated_sensor',sd['calibrated_sensor_token'])
        pose=nusc.get('ego_pose',sd['ego_pose_token'])
        path=data/sd['filename']
        if not path.is_file():raise FileNotFoundError(path)
        paths.append(path)
        return {'path':str(path),'sample_data_token':sd_token,'timestamp_us':sd['timestamp'],
                'K':cs['camera_intrinsic'],'width':sd['width'],'height':sd['height'],
                'T_sensor_to_ego':transform_matrix(cs['translation'],Quaternion(cs['rotation'])).tolist(),
                'T_ego_to_global':transform_matrix(pose['translation'],Quaternion(pose['rotation'])).tolist()}
    for sample in samples:
        sensors={channel:sensor(t) for channel,t in sample['data'].items()}
        ref=sensors['LIDAR_TOP']; sweeps=[]
        prev=nusc.get('sample_data',sample['data']['LIDAR_TOP'])['prev']
        for _ in range(10):
            if not prev:break
            sd=nusc.get('sample_data',prev); entry=sensor(prev)
            entry['dt_to_reference_seconds']=(ref['timestamp_us']-entry['timestamp_us'])/1e6
            entry['T_sensor_to_ego_ref']=sensor_to_sensor(np.array(entry['T_ego_to_global']),
                np.array(entry['T_sensor_to_ego']),np.array(ref['T_ego_to_global']),np.eye(4)).tolist()
            sweeps.append(entry);prev=sd['prev']
        packets.append({'sample_token':sample['token'],'scene_token':scene['token'],
            'timestamp_us':sample['timestamp'],'reference_timestamp_us':ref['timestamp_us'],
            'split':'mini','ego_ref':'LIDAR_TOP timestamp ego; x forward y left z up; metres',
            'sensors':sensors,'lidar_sweeps':sweeps,
            'lidar_raw_fields':['x','y','z','intensity','ring'],
            'radar_filters':'nuScenes devkit defaults; saved in geometry_summary.json'})
        for channel,entry in sensors.items():
            if channel.startswith('CAM'):
                images.append(dict(path=entry['path'],sample_token=sample['token'],scene_token=scene['token'],
                                   timestamp_us=sample['timestamp'],sensor_timestamp_us=entry['timestamp_us'],camera_channel=channel))
    manifest={'scope':'mini_scene_engineering_only','scene_name':scene['name'],'scene_token':scene['token'],
              'sample_tokens':[s['token'] for s in samples],'images':images}
    (out/'mini_scene.json').write_text(json.dumps(manifest,indent=2))
    (out/'mini_smoke.json').write_text(json.dumps(dict(manifest,scope='mini_single_frame_smoke',
        sample_tokens=[samples[0]['token']],images=images[:6]),indent=2))
    (out/'frame_packets.json').write_text(json.dumps(packets,indent=2))
    (out/'input_hashes.json').write_text(json.dumps(hash_files(set(paths)),indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    packet=packets[0];lidar=packet['sensors']['LIDAR_TOP']
    points=LidarPointCloud.from_file(lidar['path']).points[:3]
    details=[]
    fig,axes=plt.subplots(2,3,figsize=(21,8))
    for ax,(channel,cam) in zip(axes.flat,[(k,v) for k,v in packet['sensors'].items() if k.startswith('CAM')]):
        T=sensor_to_sensor(np.array(lidar['T_ego_to_global']),np.array(lidar['T_sensor_to_ego']),
                           np.array(cam['T_ego_to_global']),np.array(cam['T_sensor_to_ego']))
        inverse_error=float(np.max(np.abs(np.linalg.inv(T)@T-np.eye(4))))
        assert inverse_error < 1e-8
        depth,mask,uv,z=project_depth(transform_points(T,points),np.array(cam['K']),cam['height'],cam['width'])
        np.savez_compressed(out/(channel+'_sparse_depth.npz'),depth=depth,valid_mask=mask)
        ax.imshow(Image.open(cam['path']));ax.scatter(uv[0],uv[1],c=z,s=1,vmin=0,vmax=80,cmap='turbo')
        radar_count=0
        for rchannel,radar in packet['sensors'].items():
            if not rchannel.startswith('RADAR'):continue
            rpc=RadarPointCloud.from_file(radar['path'])
            RT=sensor_to_sensor(np.array(radar['T_ego_to_global']),np.array(radar['T_sensor_to_ego']),
                              np.array(cam['T_ego_to_global']),np.array(cam['T_sensor_to_ego']))
            _,_,ruv,_=project_depth(transform_points(RT,rpc.points[:3]),np.array(cam['K']),cam['height'],cam['width'])
            ax.scatter(ruv[0],ruv[1],s=10,facecolors='none',edgecolors='magenta');radar_count+=ruv.shape[1]
        _,boxes,K=nusc.get_sample_data(cam['sample_data_token'])
        for box in boxes:box.render(ax,view=K,normalize=True,colors=('lime','lime','lime'),linewidth=.5)
        ax.set_xlim(0,cam['width']);ax.set_ylim(cam['height'],0);ax.axis('off')
        ax.set_title(channel+' | LiDAR: depth colour; radar: magenta; GT boxes: green')
        details.append({'channel':channel,'roundtrip_max_error':inverse_error,'valid_depth_pixels':int(mask.sum()),
                        'radar_projected_points':radar_count,'camera_minus_lidar_us':cam['timestamp_us']-lidar['timestamp_us']})
    fig.tight_layout();fig.savefig(out/'six_camera_projection.jpg',dpi=130);plt.close(fig)
    nusc.render_sample_data(samples[0]['data']['LIDAR_TOP'],out_path=str(out/'bev_GT.png'),verbose=False)
    (out/'geometry_summary.json').write_text(json.dumps({'source':'measured','scope':'mini',
        'sample_count':len(samples),'scene_name':scene['name'],'cameras':details,
        'radar_filters':{'invalid_states':list(RadarPointCloud.invalid_states),'dynprop_states':list(RadarPointCloud.dynprop_states),
                         'ambig_states':list(RadarPointCloud.ambig_states)},
        'visual_review':'pending','GT':'Only visualizer accesses GT; FramePackets contain no annotations'},indent=2))


if __name__=='__main__':main()
