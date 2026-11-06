"""Process trees on POSIX and Windows; no third-party controller dependencies."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

IS_WINDOWS = sys.platform == "win32"


class ProcessControlError(OSError):
    kind = "process_control_unavailable"


class WindowsJob:
    """Own a kernel Job handle, so cleanup never targets a recycled PID."""
    def __init__(self):
        import ctypes as c
        from ctypes import wintypes as w

        class Limits(c.Structure):
            _fields_ = [("PerProcessUserTimeLimit", c.c_int64), ("PerJobUserTimeLimit", c.c_int64),
                        ("LimitFlags", w.DWORD), ("MinimumWorkingSetSize", c.c_size_t),
                        ("MaximumWorkingSetSize", c.c_size_t), ("ActiveProcessLimit", w.DWORD),
                        ("Affinity", c.c_size_t), ("PriorityClass", w.DWORD), ("SchedulingClass", w.DWORD)]

        class IO(c.Structure):
            _fields_ = [(name, c.c_uint64) for name in
                        ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                         "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        class Extended(c.Structure):
            _fields_ = [("BasicLimitInformation", Limits), ("IoInfo", IO)] + [
                (name, c.c_size_t) for name in ("ProcessMemoryLimit", "JobMemoryLimit",
                                               "PeakProcessMemoryUsed", "PeakJobMemoryUsed")]

        class Accounting(c.Structure):
            _fields_ = [(name, c.c_int64) for name in
                        ("TotalUserTime", "TotalKernelTime", "ThisPeriodTotalUserTime", "ThisPeriodTotalKernelTime")] + [
                (name, w.DWORD) for name in ("TotalPageFaultCount", "TotalProcesses", "ActiveProcesses", "TotalTerminatedProcesses")]

        self.c, self.Accounting = c, Accounting
        self.api = c.WinDLL("kernel32", use_last_error=True)
        signatures = {
            "CreateJobObjectW": ([c.c_void_p, w.LPCWSTR], w.HANDLE),
            "SetInformationJobObject": ([w.HANDLE, c.c_int, c.c_void_p, w.DWORD], w.BOOL),
            "AssignProcessToJobObject": ([w.HANDLE, w.HANDLE], w.BOOL),
            "TerminateJobObject": ([w.HANDLE, w.UINT], w.BOOL),
            "QueryInformationJobObject": ([w.HANDLE, c.c_int, c.c_void_p, w.DWORD, c.c_void_p], w.BOOL),
            "CloseHandle": ([w.HANDLE], w.BOOL),
        }
        for name, (args, result) in signatures.items():
            fn = getattr(self.api, name)
            fn.argtypes, fn.restype = args, result
        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle:
            raise c.WinError(c.get_last_error())
        limits = Extended()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE
        try:
            self.check(self.api.SetInformationJobObject(self.handle, 9, c.byref(limits), c.sizeof(limits)))
        except BaseException:
            self.close()
            raise

    def check(self, result):
        if not result:
            raise self.c.WinError(self.c.get_last_error())

    def attach(self, process):
        self.check(self.api.AssignProcessToJobObject(self.handle, int(process._handle)))

    def terminate(self, timeout=3):
        self.check(self.api.TerminateJobObject(self.handle, 1))
        deadline = time.monotonic() + timeout
        while True:
            info = self.Accounting()
            self.check(self.api.QueryInformationJobObject(self.handle, 1, self.c.byref(info), self.c.sizeof(info), None))
            if info.ActiveProcesses == 0:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError("Windows Job 中仍有进程，未确认清理完成")
            time.sleep(0.05)

    def close(self):
        if self.handle:
            self.api.CloseHandle(self.handle)
            self.handle = None


def spawn_tree(command, **kwargs):
    """Gate Windows payload until its bootstrap is assigned to a Job.

    Direct Popen then AssignProcessToJobObject races with immediate descendants.
    The bootstrap reads a private stdin pipe first; on parent death it sees EOF.
    """
    if not IS_WINDOWS:
        return subprocess.Popen(command, start_new_session=True, **kwargs)
    if kwargs.pop("pass_fds", ()):
        raise ValueError("Windows 不支持继承 POSIX 文件锁")
    kwargs.pop("stdin", None)
    try:
        job = WindowsJob()
    except OSError as exc:
        raise ProcessControlError("Windows Job 创建失败；未启动任务：" + str(exc)) from exc
    child = None
    try:
        child = subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()), "--bootstrap"],
                                 stdin=subprocess.PIPE, creationflags=0x08000000, **kwargs)
        child.riskmodel_job = job
        try:
            job.attach(child)
        except OSError as exc:
            raise ProcessControlError("Windows Job 接管失败；未启动业务命令：" + str(exc)) from exc
        child.stdin.write(json.dumps([str(x) for x in command], ensure_ascii=True).encode("ascii") + b"\n")
        child.stdin.close()
        return child
    except BaseException:
        # Payload has not been released if assignment failed.
        job.close()
        if child is not None:
            if child.stdin and not child.stdin.closed:
                child.stdin.close()
            if child.poll() is None:
                child.kill()
            child.wait(timeout=3)
        raise


def cleanup_windows(child):
    job = child.riskmodel_job
    try:
        job.terminate()
    finally:
        job.close()
        child.wait(timeout=3)


if __name__ == "__main__":
    if sys.argv[1:] != ["--bootstrap"]:
        sys.exit("Internal process bootstrap only")
    command = sys.stdin.buffer.readline()
    if not command:
        sys.exit(1)
    # Never use a shell for user paths or JSON; inherited Job contains descendants.
    sys.exit(subprocess.call(json.loads(command.decode("ascii")), stdin=subprocess.DEVNULL))
