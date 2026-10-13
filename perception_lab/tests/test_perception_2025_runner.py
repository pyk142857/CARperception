"""Recovery must fail closed when an invocation or its evidence is stale."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import run_perception_2025 as runner


class RunnerRecovery(unittest.TestCase):
    def test_ignored_archive_inside_parent_git_still_hashes_source(self):
        import subprocess
        subprocess.run(['git','init','-q',str(self.root)],check=True)
        (self.root/'.gitignore').write_text('archive/\n')
        source=self.root/'archive';source.mkdir()
        code=source/'model.py';code.write_text('x=1')
        cache=source/'__pycache__';cache.mkdir();(cache/'model.pyc').write_bytes(b'cache')
        self.assertEqual(runner.source_paths(source),[code])
        cfg={'models':{'deimv2':{'source':'archive'}},'splits':{}}
        with mock.patch.object(runner,'environment_snapshot',return_value={}):
            before=runner.fingerprint(cfg,'deimv2',[['python','model.py']])
            code.write_text('x=2')
            self.assertNotEqual(before,runner.fingerprint(cfg,'deimv2',[['python','model.py']]))

    def test_raw_input_changes_invalidate_fingerprint_without_manifest_edit(self):
        folder=self.root/'report/inputs';folder.mkdir(parents=True)
        image=self.root/'image.jpg';image.write_bytes(b'image1')
        point=self.root/'point.bin';point.write_bytes(b'point1')
        (folder/'coco_files.json').write_text(json.dumps([{'file':str(image)}]))
        packet=self.root/'packets.json';packet.write_text(json.dumps([{'sensors':{'LIDAR_TOP':{'path':str(point)}}}]))
        cfg={'report':'report','splits':{'scene0061':'packets.json'},'models':{}}
        with mock.patch.object(runner,'environment_snapshot',return_value={}):
            first=runner.fingerprint(cfg,'deimv2',[['python','model.py']])
            image.write_bytes(b'image2')
            second=runner.fingerprint(cfg,'deimv2',[['python','model.py']])
            point.write_bytes(b'point2')
            third=runner.fingerprint(cfg,'deimv2',[['python','model.py']])
        self.assertNotEqual(first,second);self.assertNotEqual(second,third)

    def test_grae_runs_inference_then_separate_cpu_evaluation_and_audit(self):
        cmds=runner.commands({'environments':{'grae3dmot':'env/infer'}},'grae3dmot',Path('config'))
        self.assertEqual([cmd[3] for cmd in cmds],['run','evaluate','audit'])
        self.assertEqual(cmds[0][0],'env/infer')
        self.assertEqual(cmds[1][0],'envs/sparse4d/bin/python')
        self.assertEqual(runner.result_paths(Path('report'),'grae3dmot',1),[Path('report/grae3dmot/evaluate_status.json')])
        self.assertEqual(runner.result_paths(Path('report'),'grae3dmot',2),[Path('report/grae3dmot/audit_status.json')])
    def test_inventory_and_summary_require_actual_fresh_evidence(self):
        out=Path('report')
        self.assertEqual(runner.result_paths(out,'preflight',0),[out/'inputs/summary.json'])
        self.assertEqual(runner.result_paths(out,'preflight',1),[out/'preflight.json'])
        self.assertEqual(runner.result_paths(out,'summarize',0),[out/'summarize/status.json'])
        for stage in ['preflight','summarize','visualize']:
            expected=runner.result_paths(self.root/'report',stage,0)
            self.assertEqual(runner.classify_run(0,expected,{} )['status'],'failed')

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.patch=mock.patch.object(runner,'ROOT',self.root);self.patch.start()
    def tearDown(self):
        self.patch.stop();self.temp.cleanup()
    def status(self, name='passed'):
        p=self.root/'report/deimv2/status.json';p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps({'status':name}));return p
    def test_visualize_uses_configured_environment_with_fallback(self):
        config={'environments':{'visualize':'envs/rerun/bin/python'}}
        self.assertEqual(runner.commands(config,'visualize',Path('config'))[0][0], 'envs/rerun/bin/python')
        self.assertEqual(runner.commands({'environments':{}},'visualize',Path('config'))[0][0],sys.executable)
        self.assertEqual(runner.result_paths(Path('report'),'visualize',0),[Path('report/visualization/status.json')])

    def test_old_pass_and_missing_executable_fail(self):
        p=self.status();before={p:runner.snapshot(p)}
        result=runner.classify_run(127,[p],before)
        self.assertEqual(result['status'],'failed')
        self.assertFalse(result['fresh_result'])
    def test_exit_zero_cannot_reuse_old_pass(self):
        p=self.status();before={p:runner.snapshot(p)}
        self.assertEqual(runner.classify_run(0,[p],before)['status'],'failed')
    def test_block_code_requires_fresh_blocked_state(self):
        p=self.status('blocked_resource');before={p:runner.snapshot(p)}
        self.assertEqual(runner.classify_run(2,[p],before)['status'],'failed')
        p.write_text(json.dumps({'status':'blocked_resource','reason':'new attempt'}))
        self.assertEqual(runner.classify_run(2,[p],before)['status'],'blocked_resource')
    def test_fresh_pass_does_not_hide_nonzero_error(self):
        p=self.status();before={p:runner.snapshot(p)}
        p.write_text(json.dumps({'status':'passed','new':True}))
        self.assertEqual(runner.classify_run(1,[p],before)['status'],'failed')
    def test_complete_manifest_detects_corrupt_prediction_and_manifest(self):
        p=self.status();prediction=p.parent/'prediction.bin';prediction.write_bytes(b'prediction')
        manifest=p.parent/'artifact_manifest.json'
        manifest.write_text(json.dumps({'complete':True,'artifacts':[
            {'path':str(x),'sha256':runner.hash_file(x)} for x in [p,prediction]]}))
        proof=runner.output_proof(manifest,p)
        self.assertIsNotNone(proof)
        row={'status':'passed','fingerprint':'f','output_proof':proof}
        self.assertTrue(runner.can_resume(row,'f',manifest,p))
        prediction.write_bytes(b'corrupted!')
        self.assertFalse(runner.can_resume(row,'f',manifest,p))
        manifest.write_text('{}')
        self.assertFalse(runner.can_resume(row,'f',manifest,p))
    def test_no_complete_manifest_never_resumes(self):
        p=self.status();m=p.parent/'artifact_manifest.json';m.write_text(json.dumps({'artifacts':[]}))
        self.assertFalse(runner.can_resume({'status':'passed','fingerprint':'x'},'x',m,p))
    def test_main_missing_module_does_not_reuse_prior_pass(self):
        self.status()
        config={'report':'report','environments':{'deimv2':sys.executable},'models':{},'splits':{}}
        cfg=self.root/'config.json';cfg.write_text(json.dumps(config))
        with mock.patch.object(sys,'argv',['runner','--config',str(cfg),'--phase','deimv2']), mock.patch.object(runner,'fingerprint',return_value='test'):
            self.assertEqual(runner.main(),1)
        row=json.loads((self.root/'report/runner_state.json').read_text())['deimv2']
        self.assertEqual(row['status'],'failed')
        self.assertFalse(row['runs'][0]['fresh_result'])

    def test_main_preserves_new_resource_block(self):
        self.status()
        tool=self.root/'tools/deimv2_2025.py';tool.parent.mkdir()
        tool.write_text("from pathlib import Path; import json; Path('report/deimv2/status.json').write_text(json.dumps({'status':'blocked_resource','reason':'missing checkpoint'})); raise SystemExit(2)")
        config={'report':'report','environments':{'deimv2':sys.executable},'models':{},'splits':{}}
        cfg=self.root/'config.json';cfg.write_text(json.dumps(config))
        with mock.patch.object(sys,'argv',['runner','--config',str(cfg),'--phase','deimv2']), mock.patch.object(runner,'fingerprint',return_value='test'):
            self.assertEqual(runner.main(),0)
        row=json.loads((self.root/'report/runner_state.json').read_text())['deimv2']
        self.assertEqual(row['status'],'blocked_resource')
        self.assertTrue(row['runs'][0]['fresh_result'])

    def test_environment_hashes_installed_code_not_only_package_versions(self):
        site=self.root/'env/lib/site-packages';site.mkdir(parents=True)
        package=site/'module.py';package.write_text('x=1')
        response=mock.Mock(returncode=0,stdout=json.dumps({'version':'same','prefix':'env','paths':[str(site)],'packages':[['package','1.0']]}))
        with mock.patch.object(runner.subprocess,'run',return_value=response):
            before=runner.environment_snapshot('env/bin/python')
            package.write_text('x=2')
            after=runner.environment_snapshot('env/bin/python')
        self.assertNotEqual(before['code_and_metadata_sha256'],after['code_and_metadata_sha256'])

    def test_fingerprint_tracks_source_checkpoint_and_environment(self):
        source=self.root/'upstream';source.mkdir();code=source/'model.py';code.write_text('x=1')
        weight=self.root/'w.pth';weight.write_bytes(b'weights')
        config={'models':{'deimv2':{'source':'upstream','checkpoint':'w.pth'}},'splits':{},'environments':{}}
        with mock.patch.object(runner,'environment_snapshot',return_value={'version':'one'}):
            first=runner.fingerprint(config,'deimv2',[['python','model.py']])
            code.write_text('x=2')
            second=runner.fingerprint(config,'deimv2',[['python','model.py']])
            self.assertNotEqual(first,second)
            weight.write_bytes(b'new weights')
            third=runner.fingerprint(config,'deimv2',[['python','model.py']])
            self.assertNotEqual(second,third)
        with mock.patch.object(runner,'environment_snapshot',return_value={'version':'two'}):
            fourth=runner.fingerprint(config,'deimv2',[['python','model.py']])
            self.assertNotEqual(third,fourth)

if __name__=='__main__':unittest.main()
