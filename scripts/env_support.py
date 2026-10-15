"""Environment controller helpers. Standard library only; Python 3.8+."""
import hashlib
import json
import os
import platform
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

ERROR_OPTIONS = {
    "missing_openmp": ["用 --library-dir 指定同架构 OpenMP 动态库目录", "联网使用 conda 新建含 OpenMP 的独立环境"],
    "missing_java": ["用 --java-home 指定已有 JDK 17", "联网使用 conda 新建包含 Java 的独立环境"],
    "architecture_mismatch": ["选择与本机架构一致的原生 Python 3.9", "在目标系统和架构上重新准备离线包"],
    "architecture_unverified": ["在允许系统架构检测的终端重新运行检查", "Apple Silicon 机器可用原生 arm64 Python 启动控制脚本"],
    "network_error": ["检查连接或用 --index-url 指定可用的 HTTPS 镜像", "用 --wheel-dir 离线安装到新的 --prefix 路径"],
    "timeout": ["查看日志进度后增大 --timeout 或 --install-timeout", "改用已有环境或离线包，保留本次日志和部分环境"],
    "disk_full": ["释放目标磁盘空间", "用 --prefix 和 --log-dir 选择其他磁盘"],
    "wheel_unavailable": ["检查 wheel 包是否匹配目标系统、架构和 Python 3.9", "在目标同类机器重新执行 --download-wheels"],
    "version_conflict": ["为新版本新建独立环境，不升级旧模型环境", "根据日志核对固定依赖和所选 Python"],
    "cleanup_permission": ["在允许清理子进程的终端重新验证", "检查日志并确认本次 Python/Java 子进程已退出"],
}


class SetupError(Exception):
    def __init__(self, kind, message, options=None):
        super().__init__(message)
        self.kind = kind
        self.options = options or ERROR_OPTIONS.get(kind, ["查看日志修复后重试", "使用其他已有环境或新建独立环境"])


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def read_pins(path):
    result = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)(?:\[[A-Za-z0-9_,.-]+\])?==([A-Za-z0-9_.+!-]+)", line)
        if not match:
            raise SetupError("invalid_requirements", "依赖清单只允许固定版本 name==version：" + line)
        result[normalize(match[1])] = match[2]
    return result


def native_machine():
    machine = platform.machine().lower()
    if sys.platform == "darwin" and machine != "arm64":
        try:
            out = subprocess.run(["/usr/sbin/sysctl", "-n", "hw.optional.arm64"],
                                 capture_output=True, text=True, timeout=5)
            if out.returncode == 0 and out.stdout.strip() in ("0", "1"):
                machine = "arm64" if out.stdout.strip() == "1" else "x86_64"
            else:
                raise SetupError("architecture_unverified", "系统未允许确认本机硬件架构，不能用当前控制进程架构代替；请允许架构检测后重试")
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SetupError("architecture_unverified", "无法确认本机硬件架构：" + str(exc)) from exc
    return {"aarch64": "arm64", "amd64": "x86_64"}.get(machine, machine)


def runtime_env(python, library_dir=None, java_home=None, mapping_dir=None, cache_dir=None):
    env = os.environ.copy()
    # Prevent controller's conda/Python settings leaking into the chosen interpreter.
    for key in list(env):
        if key.startswith("PIP_") or key in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"):
            env.pop(key, None)
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", MPLBACKEND="Agg",
               PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
               OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PIP_CONFIG_FILE=os.devnull,
               PIP_DISABLE_PIP_VERSION_CHECK="1", PIP_NO_INPUT="1")
    env["PATH"] = str(Path(python).absolute().parent) + os.pathsep + env.get("PATH", "")
    if java_home:
        env["JAVA_HOME"] = str(java_home)
        env["PATH"] = str(Path(java_home) / "bin") + os.pathsep + env["PATH"]
    if library_dir:
        key = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
        env[key] = str(library_dir) + (os.pathsep + env[key] if env.get(key) else "")
    if mapping_dir:
        env["RISKMODEL_MAPPING_DIR"] = str(mapping_dir)
    else:
        env.pop("RISKMODEL_MAPPING_DIR", None)
    if cache_dir:
        env["MPLCONFIGDIR"] = str(cache_dir)
    return env


def diagnose(output, fallback="check_failed"):
    s = output.lower()
    patterns = [
        ("architecture_mismatch", ("wrong architecture", "incompatible architecture", "bad cpu type", "exec format error")),
        ("missing_openmp", ("libomp", "libgomp", "libiomp", "omp: error")),
        ("missing_java", ("java_home", "java executable", "no java runtime", "java: not found")),
        ("disk_full", ("no space left", "disk quota")),
        ("network_error", ("connectionerror", "connection refused", "proxyerror", "certificate_verify_failed", "could not resolve", "temporary failure in name resolution", "max retries exceeded")),
        ("version_conflict", ("resolutionimpossible", "conflicting dependencies", "dependency conflict", "unsatisfiableerror", "found to be incompatible")),
        ("missing_package", ("no module named", "packagenotfounderror")),
        ("wheel_unavailable", ("no matching distribution", "not a supported wheel", "could not find a version")),
    ]
    for kind, needles in patterns:
        if any(n in s for n in needles):
            return kind
    return fallback


def run_step(name, command, log_dir, env=None, cwd=None, timeout=180):
    """Bound the whole process group (including Java), keep logs, report progress."""
    path = Path(log_dir) / (name + ".log")
    path.parent.mkdir(parents=True, exist_ok=True)
    print("[%s] 开始；日志：%s" % (name, path), file=sys.stderr, flush=True)
    start = time.monotonic()
    process = None
    timed_out = False
    cleanup_warning = None
    try:
        with path.open("wb") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, env=env, cwd=cwd,
                                       start_new_session=True)
            next_progress = start + 15
            while process.poll() is None:
                now = time.monotonic()
                if now - start >= timeout:
                    timed_out = True
                    break
                if now >= next_progress:
                    print("[%s] 运行中，已用 %ds；日志 %d 字节" %
                          (name, now - start, path.stat().st_size), file=sys.stderr, flush=True)
                    next_progress = now + 15
                time.sleep(0.1)
    except OSError as exc:
        raise SetupError("executable_unavailable", str(exc)) from exc
    finally:
        if process is not None:
            # Also clean descendants after a successful parent exits.
            try:
                os.killpg(process.pid, signal.SIGTERM)
                time.sleep(0.2)
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except PermissionError as exc:
                # Some restricted hosts deny signalling an already exited process group.
                # Reap the parent and disclose that descendant cleanup was not confirmed.
                cleanup_warning = str(exc)
                if process.poll() is None:
                    process.kill()
            process.wait()
    output = path.read_text(encoding="utf-8", errors="replace")
    result = {"name": name, "ok": process.returncode == 0 and not timed_out,
              "returncode": process.returncode, "seconds": round(time.monotonic() - start, 2),
              "log": str(path), "output": output, "timed_out": timed_out}
    if cleanup_warning:
        result["cleanup_warning"] = cleanup_warning
        result["ok"] = False
        result["error_kind"] = "cleanup_permission"
    if timed_out:
        result["error_kind"] = "timeout"
    elif not result["ok"] and not cleanup_warning:
        result["error_kind"] = diagnose(output)
    print("[%s] %s" % (name, "通过" if result["ok"] else result["error_kind"]), file=sys.stderr, flush=True)
    return result


def require_step(result, fallback=None):
    if not result["ok"]:
        kind = result.get("error_kind", fallback or "check_failed")
        if kind == "check_failed" and fallback:
            kind = fallback
        raise SetupError(kind, "%s 失败；日志：%s\n%s" %
                         (result["name"], result["log"], result["output"][-2000:]))
    return result


def atomic_json(path, value):
    import tempfile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
