import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from aggregate_failure_events import aggregate,enrich,build_groups

def row(i, frame, **kw):
    r=dict(case_id=str(i),frame=frame,scene_token='s',module='detection',
           kind='false_negative',class_name='pedestrian',instance_token='a',
           timestamp_us=frame*500000,elapsed_seconds=frame*.5,center_global=[0,0,0],
           velocity_global=[0,0,0],size_wlh=[1,1,2],distance_m=12,
           sample_token=str(frame),tracking_id=None)
    r.update(kw);return r

class EventTests(unittest.TestCase):
    def test_gap_and_scene_boundary(self):
        es=aggregate([row(0,0),row(1,1),row(2,3),row(3,4,scene_token='other')])
        self.assertEqual([e['failure_frame_count'] for e in es],[2,1,1])
    def test_fp_one_to_one_velocity_and_id_gate(self):
        base=dict(kind='false_positive',instance_token=None)
        es=aggregate([row(0,0,velocity_global=[10,0,0],**base),
                      row(1,1,center_global=[5,0,0],**base),
                      row(2,1,center_global=[5.1,0,0],**base)])
        self.assertEqual(sorted(e['failure_frame_count'] for e in es),[1,2])
        es=aggregate([row(0,0,module='tracking',tracking_id='1',**base),
                      row(1,1,module='tracking',tracking_id='2',**base)])
        self.assertEqual(len(es),2)
    def test_switches_and_order_stable(self):
        rows=[row(0,0,kind='id_switch',previous_tracking_id='1',tracking_id='2'),
              row(1,1,kind='id_switch',previous_tracking_id='2',tracking_id='3'),
              row(2,1,kind='gap_recovery')]
        a=aggregate(rows)
        self.assertEqual(a,aggregate(list(reversed(rows))))
        self.assertEqual(a[0]['transition_count'],2)
        self.assertEqual(a[0]['case_ids'],['0','1'])
    def test_no_merge_large_time_gap_or_duplicate_frame(self):
        self.assertEqual(len(aggregate([row(0,0),row(1,1,timestamp_us=2000000)])),2)
        with self.assertRaises(ValueError):
            aggregate([row(0,0),row(1,0)])

class GeometryTests(unittest.TestCase):
    def test_ego_motion_compensation(self):
        packets=[];rows=[];preds=[]
        for i in range(2):
            pose=[[1,0,0,10*i],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
            packets.append(dict(scene_token='s',sample_token=str(i),timestamp_us=i*500000,
                sensors={'LIDAR_TOP':{'T_ego_to_global':pose}}))
            box=dict(center_xyz=[20-10*i,0,0],size_wlh=[1,1,2],velocity_xy=[0,0],class_name='pedestrian')
            preds.append(dict(sample_token=str(i),boxes3d=[box]))
            rows.append(row(i,i,kind='false_positive',instance_token=None,
                center_ego=box['center_xyz'],prediction_index=0))
        es=aggregate(enrich(rows,packets,{'detection':preds}))
        self.assertEqual(len(es),1)
        self.assertEqual(es[0]['failure_frame_count'],2)
    def test_size_and_class_separation(self):
        base=dict(kind='false_positive',instance_token=None)
        self.assertEqual(len(aggregate([row(0,0,**base),row(1,1,size_wlh=[5,5,5],**base)])),2)
        self.assertEqual(len(aggregate([row(0,0),row(1,1,class_name='car')])),2)
    def test_real_membership(self):
        import json
        root=Path(__file__).resolve().parents[1]
        data=json.loads((root/'reports/failure_events/events.json').read_text())
        rows=json.loads((root/'reports/mini_evaluation/cases.json').read_text())
        actual=[c for e in data['events'] for c in e['case_ids']]
        expected=[r['case_id'] for r in rows if r['kind']!='gap_recovery']
        self.assertEqual(sorted(actual),sorted(expected))
        self.assertEqual(sum(g['event_count'] for g in data['groups']),len(data['events']))
        self.assertEqual(sum(e['transition_count'] for e in data['events']),84)
