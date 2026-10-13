"""Offline transport-integrity failures must never replace a valid checkpoint."""
import importlib.util
import io
import os
import tempfile
import subprocess
import sys
import unittest
import json
from unittest import mock
from pathlib import Path

SPEC = importlib.util.spec_from_file_location('recovery', Path(__file__).resolve().parents[1] / 'tools/recover_perception_weights_2025.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


class Response(io.BytesIO):
    def __init__(self, data, status=206, content_range=None, content_type='application/octet-stream'):
        super().__init__(data)
        self.status = status
        self.headers = {'Content-Type': content_type}
        if content_range:
            self.headers['Content-Range'] = content_range


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.output = self.directory / 'incoming.part'
        self.data = b'\x80\x02legacy payload'

    def test_exact_206_accepts_legacy_torch_header(self):
        size = len(self.data)
        event = {}
        self.assertFalse(m.receive(Response(self.data,content_range=f'bytes 0-{size-1}/{size}'), self.output, 0, size-1, size,event))
        self.assertEqual(self.output.read_bytes(),self.data)
        self.assertEqual(event['bytes_received'],size)

    def test_wrong_total_and_range_rejected(self):
        for header in ['bytes 0-3/999','bytes 1-4/4','bytes 0-2/4']:
            with self.subTest(header=header),self.assertRaises(ValueError):
                m.receive(Response(b'PK\x03\x04',content_range=header),self.output,0,3,4,{})

    def test_200_writes_fresh_whole_object_never_appends(self):
        self.output.write_bytes(b'old staging garbage')
        self.assertTrue(m.receive(Response(self.data,status=200),self.output,4,7,len(self.data),{}))
        self.assertEqual(self.output.read_bytes(), self.data)

    def test_200_partial_body_is_not_complete(self):
        with self.assertRaises(ValueError):
            m.receive(Response(self.data,status=200),self.output,4,7,len(self.data)+1,{})

    def test_416_never_success(self):
        with self.assertRaises(ValueError):
            m.receive(Response(b'',status=416),self.output,0,3,4,{})

    def test_disconnect_keeps_previously_verified_block(self):
        complete = self.directory / 'prior.block'
        complete.write_bytes(b'known good')
        with self.assertRaises(ValueError):
            m.receive(Response(b'ab',content_range='bytes 4-7/8'),self.output,4,7,8,{})
        self.assertEqual(complete.read_bytes(),b'known good')

    def test_corrupted_block_is_not_reused(self):
        block = self.directory / f'{0:012d}-{len(self.data)-1:012d}.block'
        block.write_bytes(self.data)
        record = {'filename':block.name,'start':0,'end':len(self.data)-1,'size':len(self.data),'sha256':m.digest(block)}
        self.assertEqual(m.verified_blocks(self.directory,[record]),[record])
        block.write_bytes(b'x'*len(self.data))
        self.assertEqual(m.verified_blocks(self.directory,[record]),[])

    def test_html_json_and_lfs_pointer_rejected(self):
        for data,ctype in [(b'<html>bad</html>','application/octet-stream'),(b'{"error":1}','application/octet-stream'),(b'PK\x03\x04','text/html'),(b'version https://git-lfs.github.com/spec/v1','application/octet-stream')]:
            with self.subTest(data=data),self.assertRaises(ValueError):
                m.receive(Response(data,status=200,content_type=ctype),self.output,0,len(data)-1,len(data),{})

    def test_atomic_install_refuses_size_hash_or_parse_failure(self):
        final = self.directory / 'final.pth'
        final.write_bytes(b'old checkpoint')
        for size,sha,parser in [(1,None,lambda p:{}),(len(self.data),'bad',lambda p:{}),(len(self.data),None,lambda p:(_ for _ in ()).throw(ValueError('parse')))]:
            self.output.write_bytes(self.data)
            with self.assertRaises(ValueError):
                m.install(self.output,final,size,sha,parser)
            self.assertEqual(final.read_bytes(),b'old checkpoint')
            self.assertTrue(self.output.exists())

    def test_imported_mamba_requires_source_hash_before_reading_file(self):
        result = subprocess.run([sys.executable,str(SPEC.origin),'--model','mambafusion','--install-from','nonexistent.pth'],capture_output=True,text=True)
        self.assertEqual(result.returncode,2)
        self.assertIn('requires --relay-sha256',result.stderr)

    def test_block_index_rejects_path_traversal_symlink_and_bad_ranges(self):
        filename = f'{0:012d}-{len(self.data)-1:012d}.block'
        block = self.directory / filename
        block.write_bytes(self.data)
        good = {'filename':filename,'start':0,'end':len(self.data)-1,'size':len(self.data),'sha256':m.digest(block)}
        for mutation in [{'filename':'../'+filename},{'filename':str(block)},{'size':1},{'start':-1},{'end':-1},{'start':True},{'sha256':'secret'}, {'filename':'other.block'}]:
            with self.subTest(mutation=mutation):
                self.assertEqual(m.verified_blocks(self.directory,[dict(good,**mutation)]),[])
        self.assertEqual(m.verified_blocks(self.directory,[good],total=1),[])
        self.assertEqual(m.verified_blocks(self.directory,[good,good]),[good])
        other = self.directory/'other'
        block.rename(other)
        block.symlink_to(other)
        self.assertEqual(m.verified_blocks(self.directory,[good]),[])

    def test_untrusted_headers_never_echo_in_event(self):
        secret = 'credential=DO_NOT_LOG'
        for header in ['bytes 0-3/'+secret,secret]:
            event = {}
            with self.assertRaises(ValueError):
                m.receive(Response(b'PK\x03\x04',content_range=header),self.output,0,3,4,event)
            self.assertNotIn(secret,str(event))
            self.assertNotIn('content_range',event)

    def test_malformed_length_exception_is_sanitized_in_attempt_log(self):
        secret = 'credential=DO_NOT_LOG'
        def response(*args,**kwargs):
            r = Response(b'PK\x03\x04',status=200)
            r.headers['Content-Length'] = secret
            return r
        opener = mock.Mock()
        opener.open.side_effect = response
        with mock.patch.object(m,'ROOT',self.directory), mock.patch.object(m,'REPORT',self.directory), mock.patch.object(m.urllib.request,'build_opener',return_value=opener):
            with self.assertRaises(RuntimeError):
                m.download('harpnext','direct')
        log = (self.directory/'harpnext_attempts.json').read_text()
        self.assertNotIn(secret,log)
        attempts = json.loads(log)
        self.assertEqual(len(attempts),2)
        self.assertTrue(all(a['reason']=='validation_error' for a in attempts))

    def test_disk_commit_failure_stops_after_two_attempts(self):
        class UnexpectedRetry(BaseException):
            pass
        for failure in ('rename', 'hash', 'index'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as td:
                directory = Path(td)
                calls = []
                def response(*args, **kwargs):
                    calls.append(1)
                    if len(calls) > 2:
                        raise UnexpectedRetry('retry limit exceeded')
                    return Response(b'PK\x03\x04', content_range='bytes 0-3/64865210')
                opener = mock.Mock()
                opener.open.side_effect = response
                real_replace, real_digest, real_save = m.os.replace, m.digest, m.save_json
                def replace(src, dst):
                    if failure == 'rename' and str(dst).endswith('.block'):
                        raise OSError('disk commit failed')
                    return real_replace(src, dst)
                def digest(path):
                    if failure == 'hash' and str(path).endswith('.block'):
                        raise OSError('disk read failed')
                    return real_digest(path)
                def save(path, value):
                    if failure == 'index' and Path(path).name == 'blocks.json' and value:
                        raise OSError('index commit failed')
                    return real_save(path, value)
                with mock.patch.object(m,'ROOT',directory), mock.patch.object(m,'REPORT',directory), mock.patch.object(m.urllib.request,'build_opener',return_value=opener), mock.patch.object(m.os,'replace',side_effect=replace), mock.patch.object(m,'digest',side_effect=digest), mock.patch.object(m,'save_json',side_effect=save):
                    with self.assertRaises(RuntimeError):
                        m.download('harpnext','direct',block_size=4)
                self.assertEqual(len(calls),2)
                index = directory/'perception_lab/checkpoints/harpnext_2025/recovery_staging/blocks.json'
                self.assertEqual(json.loads(index.read_text()),[])

    def test_cpu_structure_checks_survive_pythonoptimize(self):
        # Stub only deserialization; execute the actual parser child program under -O.
        python = self.directory/'perception_lab/envs/harpnext_2025/bin/python'
        python.parent.mkdir(parents=True)
        python.symlink_to(sys.executable)
        (self.directory/'torch.py').write_text("import json\ndef load(path, **kwargs):\n    return json.load(open(path))\n")
        checkpoint = self.directory/'stub.json'
        with mock.patch.object(m,'ROOT',self.directory), mock.patch.dict(os.environ,{'PYTHONOPTIMIZE':'1','PYTHONPATH':str(self.directory)}):
            for bad in ([], {}, {'net':{}}, {'net':[]}, {'net':[1]}):
                with self.subTest(bad=bad):
                    checkpoint.write_text(json.dumps(bad))
                    with self.assertRaises(ValueError):
                        m.cpu_parse(checkpoint,'harpnext')
            checkpoint.write_text(json.dumps({'net':{'weight':1}}))
            self.assertEqual(m.cpu_parse(checkpoint,'harpnext')['state_entries'],1)

    def test_atomic_install_success(self):
        self.output.write_bytes(self.data)
        final = self.directory / 'final.pth'
        receipt = m.install(self.output,final,len(self.data),m.digest(self.output),lambda p:{'state_entries':1})
        self.assertTrue(receipt['atomic_install'])
        self.assertEqual(final.read_bytes(),self.data)
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
