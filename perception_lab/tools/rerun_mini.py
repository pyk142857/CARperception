"""Export existing measured mini results to a calibrated, interactive Rerun recording."""
import argparse
import json
from pathlib import Path
import numpy as np
from pyquaternion import Quaternion
from evidence import reusable, sha256
from maptr_geometry import CLASSES, validate_frames
from camera_overlay import project_segments
from tracking_overlay import track_color, bev_outline, camera_box_segments
from lidarseg_utils import NAMES as SEG_NAMES, PALETTE as SEG_PALETTE

ROOT = Path(__file__).resolve().parents[1]
CAMERAS = ['CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT','CAM_BACK_LEFT','CAM_BACK','CAM_BACK_RIGHT']


def transform_points(points, matrix):
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def camera_to_reference(sensor, reference_pose):
    return np.linalg.inv(reference_pose) @ np.asarray(sensor['T_ego_to_global']) @ np.asarray(sensor['T_sensor_to_ego'])


def box_components(boxes):
    """Standardized ego center/wlh/wxyz -> Rerun center/lwh/xyzw."""
    centers = np.array([b['center_xyz'] for b in boxes], dtype=float).reshape(-1, 3)
    sizes = np.array([b['size_wlh'] for b in boxes], dtype=float).reshape(-1, 3)[:, [1, 0, 2]]
    rotations = np.array([b['rotation_wxyz'] for b in boxes], dtype=float).reshape(-1, 4)[:, [1, 2, 3, 0]]
    if not all(np.isfinite(x).all() for x in (centers, sizes, rotations)) or (sizes <= 0).any():
        raise ValueError('Invalid box geometry')
    if len(rotations) and not np.allclose(np.linalg.norm(rotations, axis=1), 1, atol=1e-4):
        raise ValueError('Quaternion must be normalized')
    return centers, sizes, rotations


def measured_results(status, module, stage, packets):
    record = status['modules'][module][stage]
    if not reusable(record, record.get('fingerprint')):
        raise ValueError('Missing or corrupt evidence: ' + module + ':' + stage)
    path = Path(record['log']).parent / 'predictions.json'
    results = json.loads(path.read_text())
    if [r['sample_token'] for r in results] != [p['sample_token'] for p in packets]:
        raise ValueError('Frame token/order mismatch: ' + module)
    if any(r['source'] != 'measured' or r['status'] != 'passed' for r in results):
        raise ValueError('Only measured predictions are accepted')
    return results, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'outputs/rerun/mini_scene.rrd')
    parser.add_argument('--score', type=float, default=.25)
    parser.add_argument('--connect', help='Stream to an existing local Rerun gRPC server instead of saving')
    parser.add_argument('--maptr', type=Path, default=ROOT/'outputs/maptr/predictions.json')
    parser.add_argument('--lane-score', type=float, default=.5)
    parser.add_argument('--lidarseg', type=Path, default=ROOT/'outputs/lidarseg/predictions.json')
    args = parser.parse_args()
    if not 0 <= args.score <= 1:
        parser.error('--score must be between 0 and 1')
    if not 0 <= args.lane_score <= 1:
        parser.error('--lane-score must be between 0 and 1')
    import rerun as rr
    import rerun.blueprint as rrb
    packets = json.loads((ROOT/'manifests/frame_packets.json').read_text())
    status = json.loads((ROOT/'results/status.json').read_text())
    detections, detection_path = measured_results(status, 'M05', 'mini_scene', packets)
    tracks, track_path = measured_results(status, 'M11', 'mini_tracking_M05', packets)
    lanes = None
    if args.maptr.exists():
        lane_evidence = json.loads(args.maptr.with_name('summary.json').read_text())
        if not reusable(lane_evidence, None):
            raise ValueError('Missing or corrupt MapTR evidence')
        if lane_evidence['artifacts'].get(str(args.maptr.resolve())) != sha256(args.maptr):
            raise ValueError('MapTR file is not covered by evidence')
        lanes = json.loads(args.maptr.read_text())
        validate_frames(lanes, packets)
    segmentation = None
    if args.lidarseg.exists():
        seg_evidence = json.loads(args.lidarseg.with_name('summary.json').read_text())
        if not reusable(seg_evidence, None) or seg_evidence['artifacts'].get(str(args.lidarseg.resolve())) != sha256(args.lidarseg):
            raise ValueError('Missing or corrupt LiDARSeg evidence')
        segmentation = json.loads(args.lidarseg.read_text())
        if [r['sample_token'] for r in segmentation] != [p['sample_token'] for p in packets]:
            raise ValueError('LiDARSeg frame order mismatch')
        for row, packet in zip(segmentation, packets):
            source = packet['sensors']['LIDAR_TOP']['path']
            if row['source'] != 'measured' or row['status'] != 'passed' or seg_evidence['source_sha256'].get(source) != sha256(source):
                raise ValueError('LiDARSeg measured source mismatch')
            if row['path'] not in seg_evidence['artifacts']:
                raise ValueError('Unverified LiDARSeg point labels')
    data = ROOT/'data/nuscenes/v1.0-mini'
    def table(name):
        return {x['token']: x for x in json.loads((data/(name+'.json')).read_text())}
    samples, annotations, instances, categories = (table(n) for n in ['sample','sample_annotation','instance','category'])
    by_sample = {}
    for ann in annotations.values():
        by_sample.setdefault(ann['sample_token'], []).append(ann)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rr.init('CARperception-mini', strict=True)
    if args.connect:
        rr.connect_grpc(args.connect)
    else:
        rr.save(str(args.out))
    rr.log('ego', rr.ViewCoordinates.FLU, static=True)
    if segmentation is not None:
        context = rr.AnnotationContext([rr.AnnotationInfo(id=i, label=name, color=SEG_PALETTE[i]) for i,name in enumerate(SEG_NAMES)])
        rr.log('ego', context, static=True)
        rr.log('lidarseg_gt', rr.ViewCoordinates.FLU, context, static=True)
    rr.log('guide', rr.TextDocument(
        '# CARperception: nuScenes mini\n'
        '39 keyframes. Camera images include projected MapTR lines; no separate 2D lane inference.\n'
        'GT = green, CenterPoint = orange, tracks = stable ID colors (labels enabled). Box filter: score >= '+str(args.score)+'.\n'
        'Use frame or elapsed timeline; rotate / zoom the 3D view. Toggle entities in the blueprint.\n'
        'Reference: ego at LiDAR time, metres. Camera poses use their own timestamps.\n'
        'Keyframe grouping is not exact simultaneous capture. Ground truth is a separate display layer.\n'
        'MapTR: divider=yellow, crossing=magenta, boundary=blue. Score >= '+str(args.lane_score)+'.\n'
        'MapTR is a camera prediction, not map GT. Camera overlays use approximate ground Z=0; no occlusion test.\n'
        'LiDARSeg: road=turquoise, vegetation=green, manmade=cream, car=orange. Compare the ground truth tab.\n'
        'Tracking: BEV boxes/trails and camera 3D box projections share ID colors. No object-motion or occlusion correction.\n'
        'BEV: forward=up, left=left. No new inference is performed by this viewer.', media_type='text/markdown'), static=True)
    camera_views = [rrb.Spatial2DView(name=c, origin='ego/cameras/'+c+'/image') for c in CAMERAS]
    rr.send_blueprint(rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(rrb.Spatial2DView(name='BEV / tracks + lanes (forward up)', origin='bev'),
                         rrb.Tabs(rrb.Spatial3DView(name='LiDARSeg prediction / detections', origin='ego'),
                                  rrb.Spatial3DView(name='LiDARSeg ground truth', origin='lidarseg_gt')),
                         rrb.TextDocumentView(name='Guide', origin='guide'), row_shares=[.45,.45,.10]),
            rrb.Grid(*camera_views, grid_columns=3), column_shares=[.5,.5]),
        rrb.TimePanel(state='expanded'), collapse_panels=True))
    history = {}
    counts = []
    def log_boxes(path, boxes, color, show_labels=False):
        centers, sizes, rotations = box_components(boxes)
        labels = [b.get('tracking_id', b.get('class_name','GT')) for b in boxes]
        rr.log(path, rr.Boxes3D(centers=centers, sizes=sizes, quaternions=rotations,
                               colors=color, labels=labels, show_labels=show_labels),
               rr.AnyValues(score=[b.get('score',1.) for b in boxes]))
    for i, (packet, pred, tracked) in enumerate(zip(packets, detections, tracks)):
        rr.set_time('frame', sequence=i)
        rr.set_time('elapsed', duration=(packet['timestamp_us']-packets[0]['timestamp_us'])/1e6)
        selected_tracks = [b for b in tracked['tracks3d'] if b['score'] >= args.score]
        camera_track_counts = {}
        pose = np.asarray(packet['sensors']['LIDAR_TOP']['T_ego_to_global'])
        inverse = np.linalg.inv(pose)
        rr.log('ego/frame', rr.AnyValues(sample_token=packet['sample_token'],
                                         timestamp_us=packet['timestamp_us']))
        sensor = packet['sensors']['LIDAR_TOP']
        points = np.fromfile(sensor['path'], dtype=np.float32).reshape(-1, 5)
        points = transform_points(points[:, :3], np.asarray(sensor['T_sensor_to_ego']))
        point_colors = np.tile([150,160,170], (len(points),1))
        if segmentation is not None:
            row = segmentation[i]
            with np.load(row['path']) as values:
                labels, gt_labels = values['prediction'], values['ground_truth']
                confidence = values['confidence']
            if (row['point_count'] != len(points) or labels.shape != (len(points),) or gt_labels.shape != labels.shape
                    or confidence.shape != labels.shape or not np.isfinite(confidence).all()
                    or ((confidence<0)|(confidence>1)).any() or labels.max()>16 or gt_labels.max()>16):
                raise ValueError('LiDARSeg point alignment/values invalid')
            point_colors = SEG_PALETTE[labels]
            rr.log('ego/lidar', rr.Points3D(points, colors=point_colors, radii=rr.Radius.ui_points(1),
                                           class_ids=labels, show_labels=False), rr.AnyValues(confidence=confidence))
            rr.log('lidarseg_gt/points', rr.Points3D(points, colors=SEG_PALETTE[gt_labels], radii=rr.Radius.ui_points(1),
                                                   class_ids=gt_labels, show_labels=False))
        else:
            rr.log('ego/lidar', rr.Points3D(points, colors=point_colors, radii=.035))
        near_mask = (np.abs(points[:,0]) < 32) & (np.abs(points[:,1]) < 18) & (points[:,2] < 2)
        near = points[near_mask]
        rr.log('bev/lidar', rr.Points2D(-near[:,[1,0]], colors=point_colors[near_mask], radii=.035))
        rr.log('ego/maptr', rr.Clear(recursive=True))
        rr.log('bev/maptr', rr.Clear(recursive=True))
        lane_count = 0
        if lanes is not None:
            palette = [[255,225,30], [255,80,190], [70,145,255]]
            for cls, color in zip(CLASSES, palette):
                selected_lines = [v for v in lanes[i]['vectors'] if v['class_name'] == cls and v['score'] >= args.lane_score]
                lane_count += len(selected_lines)
                if selected_lines:
                    xyz = [np.asarray(v['points_xyz']) for v in selected_lines]
                    scores = [v['score'] for v in selected_lines]
                    rr.log('ego/maptr/'+cls, rr.LineStrips3D(xyz, colors=color, radii=.10), rr.AnyValues(score=scores))
                    rr.log('bev/maptr/'+cls, rr.LineStrips2D([-p[:,[1,0]] for p in xyz], colors=color, radii=rr.Radius.ui_points(2.0)), rr.AnyValues(score=scores))
        for camera in CAMERAS:
            entry = packet['sensors'][camera]
            transform = camera_to_reference(entry, pose)
            path = 'ego/cameras/'+camera
            rr.log(path, rr.Transform3D(translation=transform[:3,3], mat3x3=transform[:3,:3]))
            rr.log(path+'/image', rr.Pinhole(image_from_camera=entry['K'], width=entry['width'],
                                            height=entry['height'], camera_xyz=rr.ViewCoordinates.RDF,
                                            image_plane_distance=1.5))
            rr.log(path+'/image', rr.EncodedImage(path=entry['path']),
                   rr.AnyValues(sensor_timestamp_us=entry['timestamp_us'],
                                offset_from_lidar_ms=(entry['timestamp_us']-packet['reference_timestamp_us'])/1000))
            track_overlay = path+'/image/tracks'
            rr.log(track_overlay, rr.Clear(recursive=True))
            visible = 0
            for box in selected_tracks:
                segments = camera_box_segments(box, transform, entry['K'], entry['width'], entry['height'])
                if not segments:
                    continue
                visible += 1
                ident = box['tracking_id']
                color = track_color(ident)
                rr.log(track_overlay+'/'+ident+'/box', rr.LineStrips2D(segments, colors=color,
                       radii=rr.Radius.ui_points(1.5), draw_order=20))
                pixels = np.asarray(segments).reshape(-1,2)
                anchor = pixels[np.argmin(pixels[:,1])]
                rr.log(track_overlay+'/'+ident+'/label', rr.Points2D([anchor], colors=color,
                       radii=rr.Radius.ui_points(1), labels=['#'+ident],
                       show_labels=True, draw_order=21))
            camera_track_counts[camera] = visible
            overlay = path+'/image/maptr'
            rr.log(overlay, rr.Clear(recursive=True))
            if lanes is not None:
                for cls, color in zip(CLASSES, palette):
                    segments = []
                    for vector in lanes[i]['vectors']:
                        if vector['class_name'] == cls and vector['score'] >= args.lane_score:
                            segments.extend(project_segments(vector['points_xyz'], transform,
                                                             entry['K'], entry['width'], entry['height']))
                    if segments:
                        rr.log(overlay+'/'+cls, rr.LineStrips2D(segments, colors=color,
                               radii=rr.Radius.ui_points(1.5), draw_order=10))
        gt = []
        for ann in by_sample.get(packet['sample_token'], []):
            category = categories[instances[ann['instance_token']]['category_token']]['name']
            gt.append(dict(center_xyz=transform_points([ann['translation']], inverse)[0],
                           size_wlh=ann['size'], rotation_wxyz=(Quaternion(matrix=inverse[:3,:3])*Quaternion(ann['rotation'])).elements,
                           class_name=category))
        rr.log('ego/boxes', rr.Clear(recursive=True))
        selected = [b for b in pred['boxes3d'] if b['score'] >= args.score]
        log_boxes('ego/boxes/ground_truth', gt, [80,220,100])
        log_boxes('ego/boxes/centerpoint', selected, [255,150,30])
        log_boxes('ego/boxes/tracks', selected_tracks,
                  [track_color(b['tracking_id']) for b in selected_tracks], show_labels=True)
        rr.log('bev/tracks', rr.Clear(recursive=True))
        rr.log('ego/trails', rr.Clear(recursive=True))
        for box in selected_tracks:
            ident = box['tracking_id']
            color = track_color(ident)
            rr.log('bev/tracks/'+ident+'/box', rr.LineStrips2D([bev_outline(box)], colors=color,
                   radii=rr.Radius.ui_points(2), draw_order=20))
            center = -np.asarray(box['center_xyz'])[[1,0]]
            rr.log('bev/tracks/'+ident+'/label', rr.Points2D([center], colors=color,
                   radii=rr.Radius.ui_points(1), labels=['#'+ident], show_labels=True, draw_order=21))
            history.setdefault(ident, []).append(transform_points([box['center_xyz']], pose)[0])
            history[ident] = history[ident][-20:]
            if len(history[ident]) > 1:
                trail = transform_points(history[ident], inverse)
                rr.log('ego/trails/'+ident, rr.LineStrips3D([trail], colors=color, radii=.06))
                rr.log('bev/tracks/'+ident+'/trail', rr.LineStrips2D([-trail[:,[1,0]]], colors=color,
                       radii=rr.Radius.ui_points(1.5), draw_order=15))
        counts.append(dict(frame=i, sample_token=packet['sample_token'], lidar_points=len(points),
                           cameras=len(CAMERAS), gt=len(gt), detections=len(selected), tracks=len(selected_tracks), camera_tracks=camera_track_counts, maptr_vectors=lane_count))
        print(f'frame {i+1}/{len(packets)}', flush=True)
    rr.get_global_data_recording().flush()
    rr.disconnect()
    if args.connect:
        print('Streamed all frames to '+args.connect)
        return
    summary = dict(scope='mini_scene_0061', rerun_version=rr.__version__, frame_count=len(counts),
                   camera_images=len(counts)*len(CAMERAS), score_threshold=args.score, lane_score_threshold=args.lane_score,
                   maptr_loaded=lanes is not None, lidarseg_loaded=segmentation is not None,
                   camera_lane_overlay=lanes is not None, tracking_overlay=True, track_history_positions=20,
                   coordinate_system='reference ego FLU, metres; camera asynchronous ego compensation',
                   source_sha256={str(p):sha256(p) for p in ([detection_path,track_path,ROOT/'manifests/frame_packets.json'] + ([args.maptr] if lanes is not None else []) + ([args.lidarseg] if segmentation is not None else []))},
                   recording=str(args.out.resolve()), recording_sha256=sha256(args.out), frames=counts)
    args.out.with_suffix('.json').write_text(json.dumps(summary,indent=2))
    print('Saved '+str(args.out))


if __name__ == '__main__':
    main()
