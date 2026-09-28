"""Serve the saved recording and Rerun web assets on loopback; start/stop owned processes."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.request
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT/'outputs/rerun'
STATE = RUNTIME/'server.json'


def alive(pid):
    try:
        cmd=(Path('/proc')/str(pid)/'cmdline').read_bytes().decode().replace('\x00',' ')
        return str(ROOT/'envs/rerun') in cmd
    except OSError:
        return False


class RecordingHandler(SimpleHTTPRequestHandler):
    def __init__(self,*a,**kw):super().__init__(*a,directory=str(RUNTIME),**kw)
    def do_GET(self):
        if self.path != '/mini_scene.rrd':self.send_error(404);return
        super().do_GET()
    def do_HEAD(self):
        if self.path != '/mini_scene.rrd':self.send_error(404);return
        super().do_HEAD()
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin','*')
        super().end_headers()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stop',action='store_true')
    p.add_argument('--no-browser',action='store_true')
    p.add_argument('--port',type=int,default=9090)
    p.add_argument('--grpc-port',type=int,default=9876)
    p.add_argument('--data-port',type=int,default=9091)
    p.add_argument('--serve-recording',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.serve_recording:
        ThreadingHTTPServer(('127.0.0.1',args.data_port),RecordingHandler).serve_forever()
        return
    state=json.loads(STATE.read_text()) if STATE.exists() else {}
    pids=state.get('pids', [state['pid']] if 'pid' in state else [])
    if args.stop:
        for pid in pids:
            if alive(pid):os.kill(pid,signal.SIGTERM)
        for _ in range(20):
            if not any(alive(pid) for pid in pids):break
            time.sleep(.1)
        STATE.unlink(missing_ok=True)
        print('Project Rerun services stopped.');return
    if pids and all(alive(pid) for pid in pids):
        print(state['url'])
        if not args.no_browser:webbrowser.open(state['url'])
        return
    if any(alive(pid) for pid in pids):raise SystemExit('Partial service; run --stop before restarting.')
    recording=RUNTIME/'mini_scene.rrd'
    binaries=list((ROOT/'envs/rerun/lib').glob('python*/site-packages/rerun_sdk/rerun_cli/rerun'))
    if not recording.is_file() or not binaries:
        raise SystemExit('Install envs/rerun and run tools/rerun_mini.py first; see RERUN.md')
    for port in (args.port,args.grpc_port,args.data_port):
        with socket.socket() as s:
            s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            try:s.bind(('127.0.0.1',port))
            except OSError:raise SystemExit(f'Port {port} busy; choose alternate ports.')
    commands=[
        [str(binaries[0]),'--serve-web','--bind','127.0.0.1','--web-viewer-port',str(args.port),
         '--port',str(args.grpc_port),'--server-memory-limit','256MB','--threads','2'],
        [str(ROOT/'envs/rerun/bin/python'),str(Path(__file__).resolve()),'--serve-recording','--data-port',str(args.data_port)],
    ]
    procs=[]
    try:
        with (RUNTIME/'server.log').open('a') as log:
            for command in commands:
                procs.append(subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL,cwd=ROOT,start_new_session=True))
        for port,path in [(args.port,'/'),(args.data_port,'/mini_scene.rrd')]:
            for _ in range(40):
                if any(proc.poll() is not None for proc in procs):raise RuntimeError('Service exited')
                try:
                    req=urllib.request.Request(f'http://127.0.0.1:{port}{path}',method='HEAD')
                    with urllib.request.urlopen(req,timeout=1) as response:
                        if response.status==200:break
                except OSError:time.sleep(.25)
            else:raise RuntimeError('Service startup timed out')
    except Exception:
        for proc in procs:proc.terminate()
        raise
    url=f'http://127.0.0.1:{args.port}/?url=http%3A%2F%2F127.0.0.1%3A{args.data_port}%2Fmini_scene.rrd&renderer=webgl'
    STATE.write_text(json.dumps(dict(pids=[p.pid for p in procs],url=url,commands=commands),indent=2))
    print(url)
    if not args.no_browser:webbrowser.open(url)


if __name__=='__main__':main()
