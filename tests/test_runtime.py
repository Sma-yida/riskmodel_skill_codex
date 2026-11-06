"""Portable standard-library acceptance: python -B -m unittest discover -s tests -v.

Uses a temporary Skill copy with a fake data-analysis backend, NEVER user models.
Run this on Windows to exercise real Job Objects, file locks and handoff.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(SCRIPTS))
from env_support import run_step
from platform_process import IS_WINDOWS

STUB = '''import json, subprocess, sys, time
from pathlib import Path
def run(config_path):
    if Path('sleep.flag').exists():
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
        Path('descendant.pid').write_text(str(child.pid), encoding='utf-8')
        time.sleep(120)
    cfg = json.loads(Path(config_path).read_text(encoding='utf-8'))
    report = Path(cfg['output_path']['dq_report_path'])
    report.parent.mkdir(parents=True,exist_ok=True)
    report.write_text('test artifact',encoding='utf-8')
    Path('results/data_analysis.json').write_text(json.dumps({
        '报告标题':'runtime fixture', '样本整体概述':{}, '特征分析':{}, '违约率分析':{}
    }),encoding='utf-8')
    return True
'''


def alive(pid):
    if IS_WINDOWS:
        import ctypes as c
        from ctypes import wintypes as w
        api = c.WinDLL('kernel32', use_last_error=True)
        api.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
        api.OpenProcess.restype = w.HANDLE
        api.WaitForSingleObject.argtypes = [w.HANDLE, w.DWORD]
        api.WaitForSingleObject.restype = w.DWORD
        api.CloseHandle.argtypes = [w.HANDLE]
        handle = api.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
        if not handle:
            if c.get_last_error() == 5:
                raise PermissionError('Cannot verify test process termination')
            return False
        try:
            return api.WaitForSingleObject(handle,0) == 258  # WAIT_TIMEOUT
        finally:
            api.CloseHandle(handle)
    result = subprocess.run(['ps','-p',str(pid),'-o','stat='],capture_output=True,text=True)
    return result.returncode == 0 and bool(result.stdout.strip()) and not result.stdout.strip().startswith('Z')


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='riskmodel-runtime-')
        self.base = Path(self.tmp.name)/'中文 空格 & test'
        self.skill = self.base/'skill'
        self.work = self.base/'work'
        (self.skill/'scripts/modeling').mkdir(parents=True)
        for path in SCRIPTS.glob('*.py'):
            shutil.copy2(path,self.skill/'scripts'/path.name)
        shutil.copytree(SCRIPTS.parent/'assets',self.skill/'assets')
        (self.skill/'scripts/modeling/data_analysis.py').write_text(STUB,encoding='utf-8')
        settings = self.base/'settings'; settings.mkdir()
        # Test double only: backend has no third-party dependencies.
        (settings/'settings.json').write_text(json.dumps({'python':sys.executable,'verification':{'full_passed':True}}),encoding='utf-8')
        self.env = dict(os.environ,RISKMODEL_HOME=str(settings),PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
        self.call('--workdir',str(self.work),script='init_workspace.py')
        (self.work/'data/data.csv').write_text('id,date,label,channel,f\n1,20240101,0,A,1\n2,20240102,1,A,2\n',encoding='utf-8')

    def tearDown(self):
        try:
            self.call('-w',str(self.work),'--stop')
        finally:
            self.tmp.cleanup()

    def command(self,*args,script='run_stage.py'):
        return [sys.executable,'-B',str(self.skill/'scripts'/script)] + list(args)

    def call(self,*args,script='run_stage.py'):
        result = subprocess.run(self.command(*args,script=script),env=self.env,
                                capture_output=True,text=True,encoding='utf-8',timeout=25)
        self.assertTrue(result.stdout,result.stderr)
        return json.loads(result.stdout)

    def stage(self,*args):
        return self.call('-w',str(self.work),*args)

    def wait_descendant(self):
        marker = self.work/'descendant.pid'
        deadline = time.monotonic()+10
        while not marker.exists() and time.monotonic()<deadline:
            time.sleep(.05)
        self.assertTrue(marker.exists(),self.stage('--status'))
        return int(marker.read_text(encoding='utf-8'))

    def test_detached_success_reconnect_and_unicode_paths(self):
        started = self.stage('--start','data_analysis')
        self.assertTrue(started['ok'],started)
        result = self.stage('--wait','--seconds','10')
        self.assertEqual(result['task']['status'],'succeeded',result)
        self.assertFalse(result['busy'])
        self.assertEqual(self.stage('--status')['stages']['data_analysis']['state'],'succeeded')
        self.assertTrue((self.work/'results/data_analysis.html').is_file())

    def test_stop_cleans_descendants_and_blocks_mutation(self):
        (self.work/'sleep.flag').touch()
        self.assertTrue(self.stage('--start','data_analysis')['ok'])
        pid = self.wait_descendant()
        self.assertTrue(alive(pid))
        self.assertEqual(self.stage('--start','data_analysis')['error_kind'],'workspace_busy')
        change = self.call('-w',str(self.work),'--set','{"feature_select":{"iv_thresh":0.03}}',script='make_config.py')
        self.assertEqual(change['error_kind'],'workspace_busy')
        result = self.stage('--stop')
        self.assertEqual(result['task']['status'],'stopped',result)
        self.assertFalse(result['busy'])
        self.assertFalse(alive(pid))

    def test_simultaneous_starts_have_one_owner(self):
        (self.work/'sleep.flag').touch()
        calls = [subprocess.Popen(self.command('-w',str(self.work),'--start','data_analysis'),env=self.env,
                                  stdout=subprocess.PIPE,stderr=subprocess.PIPE) for _ in range(2)]
        responses = [json.loads(p.communicate(timeout=20)[0]) for p in calls]
        self.assertEqual(sum(r['ok'] for r in responses),1,responses)
        self.assertEqual([r['error_kind'] for r in responses if not r['ok']],['workspace_busy'])
        self.wait_descendant()

    def test_probe_cleanup_after_successful_parent_exits(self):
        marker = self.base/'probe.pid'
        code = "import subprocess,sys,pathlib; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)']); pathlib.Path(sys.argv[1]).write_text(str(p.pid))"
        result = run_step('probe',[sys.executable,'-c',code,str(marker)],self.base/'probe-logs',timeout=10)
        self.assertTrue(result['ok'],result)
        self.assertFalse(alive(int(marker.read_text())))

    def test_probe_timeout_cleans_process_tree(self):
        marker = self.base/'timeout.pid'
        code = "import subprocess,sys,pathlib,time; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)']); pathlib.Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(120)"
        result = run_step('timeout',[sys.executable,'-c',code,str(marker)],self.base/'probe-logs',timeout=1)
        self.assertEqual(result['error_kind'],'timeout',result)
        self.assertFalse(alive(int(marker.read_text())))

    @unittest.skipUnless(IS_WINDOWS,'Windows Job crash semantics require a real Windows host')
    def test_crashed_worker_cleans_job_and_releases_lock(self):
        (self.work/'sleep.flag').touch()
        self.assertTrue(self.stage('--start','data_analysis')['ok'])
        pid = self.wait_descendant()
        worker_pid = self.stage('--status')['task']['pid']
        # Kill only our isolated test worker; its Job must clean its own descendants.
        os.kill(worker_pid,9)
        deadline = time.monotonic()+5
        while alive(pid) and time.monotonic()<deadline:
            time.sleep(.05)
        self.assertFalse(alive(pid))
        result = self.stage('--status')
        self.assertFalse(result['busy'])
        self.assertEqual(result['task']['status'],'interrupted')


if __name__ == '__main__':
    unittest.main()
