"""Discover, verify, create and remember a unified Python 3.9 modeling environment."""
import argparse
import datetime
import hashlib
import json
import os
import platform
import re
import shutil
import sys
import tempfile
import zipfile
from email.parser import BytesParser
from pathlib import Path
from urllib.parse import urlparse

from common import SETTINGS_PATH, emit, load_settings
from env_support import (SetupError, atomic_json, native_machine, normalize, read_pins,
                         require_step, run_step, runtime_env, sha256)

HERE = Path(__file__).resolve().parent
REQUIREMENTS = HERE / "requirements.txt"
CONSTRAINTS = HERE / "requirements" / "verified-macos-arm64-py39.txt"
BOOTSTRAP = HERE / "requirements" / "bootstrap.txt"
PROBE = HERE / "env_probe.py"
FULL_CHECKS = ["imports", "pmml", "scorecard", "autogluon"]


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def validate_profile(info):
    if info["python_minor"] != "3.9" or info["implementation"] != "CPython":
        raise SetupError("python_version", "需要 CPython 3.9，当前为 %s %s" %
                         (info["implementation"], info["version"]))
    if info["system"] not in ("Darwin", "Linux"):
        raise SetupError("unsupported_system", "目前支持 macOS 和 Linux")
    machine = {"aarch64": "arm64", "amd64": "x86_64"}.get(info["machine"], info["machine"])
    if machine != native_machine():
        raise SetupError("architecture_mismatch", "Python 架构 %s 与本机 %s 不一致；请选择本机原生 Python" %
                         (info["machine"], native_machine()))


def check_versions(info):
    missing, conflicts = [], []
    for name, required in read_pins(REQUIREMENTS).items():
        actual = info["packages"].get(name)
        if actual is None:
            missing.append(name + "==" + required)
        elif actual != required:
            conflicts.append("%s：当前 %s，需要 %s" % (name, actual, required))
    if missing or conflicts:
        raise SetupError("missing_package" if missing and not conflicts else "version_conflict",
                         "缺少：%s；版本不一致：%s" % (", ".join(missing) or "无", "; ".join(conflicts) or "无"),
                         ["新建 riskmodel-py39 环境", "选择符合本版本清单的已有环境；旧模型继续使用原环境"])


def target_key(info):
    return {key: info[key] for key in ("system", "machine", "python_minor", "implementation")}


def find_java_home(prefix):
    prefix = Path(prefix)
    for path in (prefix / "lib" / "jvm", prefix / "lib" / "jvm" / "Contents" / "Home", prefix):
        java = path / "bin" / "java"
        if java.is_file() and os.access(str(java), os.X_OK):
            return str(path)
    return None


def fingerprint(info, java_output, library_dir, java_home):
    files = [REQUIREMENTS, CONSTRAINTS, PROBE, HERE / "setup_env.py", HERE / "env_support.py"]
    files += sorted((HERE / "modeling").rglob("*.py"))
    libraries = []
    if library_dir:
        for pattern in ("*omp*.dylib", "*omp*.so*"):
            for path in sorted(Path(library_dir).glob(pattern)):
                libraries.append([str(path), path.stat().st_size, path.stat().st_mtime_ns])
    value = {"inventory": info, "java": java_output, "java_home": java_home,
             "library_dir": library_dir, "libraries": libraries,
             "code": [(str(p.relative_to(HERE)), sha256(p)) for p in files]}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def known_grpc_warning(result, info):
    lines = [line.strip() for line in result["output"].splitlines() if line.strip()]
    return (not result["timed_out"] and result["returncode"] == 1
            and info["system"] == "Darwin" and info["machine"] == "arm64"
            and info["packages"].get("grpcio") == "1.67.1"
            and lines == ["grpcio 1.67.1 is not supported on this platform"])


class Manager:
    def __init__(self, args):
        self.args = args
        state_dir = Path(SETTINGS_PATH).parent
        log_root = Path(args.log_dir).expanduser().absolute() if args.log_dir else state_dir / "logs"
        log_root.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(tempfile.mkdtemp(prefix="setup-", dir=str(log_root)))
        self.report = {"ok": False, "started_at": stamp(), "log_dir": str(self.log_dir), "checks": [], "warnings": []}

    def run(self, name, command, **kwargs):
        original_name = name
        index = 2
        while (self.log_dir / (name + ".log")).exists():
            name = original_name + "-%d" % index
            index += 1
        result = run_step(name, command, self.log_dir, timeout=kwargs.pop("timeout", self.args.timeout), **kwargs)
        self.report["checks"].append({k: v for k, v in result.items() if k != "output"})
        return result

    def probe(self, python, name, env):
        directory = Path(tempfile.mkdtemp(prefix=name + "-", dir=str(self.log_dir)))
        result_path = directory / "result.json"
        result = self.run(name, [python, "-B", str(PROBE), name, "--result", str(result_path)],
                          env=env, cwd=str(directory))
        require_step(result)
        try:
            data = json.loads(result_path.read_text(encoding="utf-8"))
            if not data["ok"]:
                raise ValueError(data.get("error"))
            return data["details"]
        except (OSError, ValueError, KeyError) as exc:
            raise SetupError("invalid_probe_result", str(exc)) from exc

    def settings(self):
        settings = load_settings()
        return settings if isinstance(settings, dict) else {}

    def candidates(self):
        choices = []
        saved = self.settings().get("python")
        if saved:
            choices.append((saved, "saved"))
        for directory in os.environ.get("PATH", "").split(os.pathsep):
            if directory:
                for name in ("python3.9", "python3", "python"):
                    path = Path(directory) / name
                    if path.is_file() and os.access(str(path), os.X_OK):
                        choices.append((str(path.absolute()), "PATH"))
        for path in ("/opt/homebrew/opt/python@3.9/bin/python3.9", "/usr/local/bin/python3.9"):
            if Path(path).is_file():
                choices.append((path, "system"))
        conda = self.args.conda or shutil.which("conda")
        if conda:
            result = self.run("conda-list", [conda, "env", "list", "--json"], timeout=30)
            if result["ok"]:
                try:
                    for prefix in json.loads(result["output"])["envs"]:
                        choices.append((str(Path(prefix) / "bin" / "python"), "conda:" + Path(prefix).name))
                except (ValueError, KeyError):
                    self.report["warnings"].append("无法解析 conda 环境列表，仍可指定 --python 绝对路径")
        unique = []
        for path, source in choices:
            # Do not resolve symlinks: that would lose virtualenv identity.
            path = os.path.abspath(os.path.expanduser(path))
            if path not in [p[0] for p in unique]:
                unique.append((path, source))
        return unique

    def resolve_python(self, explicit=None):
        selected = explicit or self.settings().get("python")
        if selected:
            if os.path.isfile(os.path.expanduser(selected)):
                return os.path.abspath(os.path.expanduser(selected))
            path = shutil.which(selected)
            if path:
                return os.path.abspath(path)
            for path, source in self.candidates():
                if source == "conda:" + selected and Path(path).is_file():
                    return path
            raise SetupError("python_unavailable", "找不到已指定的 Python：" + selected)
        raise SetupError("environment_not_selected", "尚未选择环境；运行 --list 后用 --check --python 路径，或 --create")

    def env(self, python, reuse=True):
        saved = self.settings() if reuse and self.settings().get("python") == python else {}
        prefix = Path(python).parent.parent
        library = self.args.library_dir or saved.get("library_dir")
        if not library and any((prefix / "lib").glob("lib*omp*")):
            library = str(prefix / "lib")
        java_home = self.args.java_home or saved.get("java_home") or find_java_home(prefix)
        mapping = self.args.mapping_dir if self.args.mapping_dir is not None else self.settings().get("mapping_dir")
        for field, path in (("library_dir", library), ("java_home", java_home), ("mapping_dir", mapping)):
            if path and not Path(path).expanduser().is_dir():
                raise SetupError("invalid_path", "%s 目录不存在：%s" % (field, path))
        if java_home and not os.access(str(Path(java_home).expanduser() / "bin" / "java"), os.X_OK):
            raise SetupError("missing_java", "指定 JDK 中不存在 bin/java：" + java_home)
        self.runtime = {"library_dir": str(Path(library).expanduser().absolute()) if library else None,
                        "java_home": str(Path(java_home).expanduser().absolute()) if java_home else None,
                        "mapping_dir": str(Path(mapping).expanduser().absolute()) if mapping else None}
        return runtime_env(python, cache_dir=self.log_dir / "mpl-cache", **self.runtime)

    def check(self, python):
        env = self.env(python)
        info = self.probe(python, "inventory", env)
        self.report["environment"] = info
        validate_profile(info)
        check_versions(info)
        java = self.java(env)
        native = self.probe(python, "native", env)
        if not native["openmp_libraries"]:
            raise SetupError("missing_openmp", "未检测到已加载的 OpenMP 库")
        self.report["native"] = native
        pip = self.run("pip-check", [python, "-m", "pip", "check"], env=env)
        if known_grpc_warning(pip, info):
            self.probe(python, "grpc", env)
            self.report["checks"][-2]["accepted_warning"] = "grpcio_platform_metadata"
            self.report["warnings"].append("grpcio 1.67.1 的 macOS ARM wheel 元数据不匹配；pip check 原始报警已保留，实际本机 RPC 通过。其他 pip check 错误仍会阻断。")
        else:
            require_step(pip, "dependency_check_failed")
        key = fingerprint(info, java["output"], self.runtime["library_dir"], self.runtime["java_home"])
        old = self.settings()
        reuse = (old.get("python") == python and old.get("verification", {}).get("fingerprint") == key
                 and old.get("verification", {}).get("full_passed") is True and not self.args.full)
        if not reuse:
            self.report["models"] = {name: self.probe(python, name, env) for name in FULL_CHECKS}
        now = stamp()
        verification = {"fingerprint": key, "full_passed": True,
                        "full_checked_at": old.get("verification", {}).get("full_checked_at") if reuse else now,
                        "report": old.get("verification", {}).get("report") if reuse else str(self.log_dir / "report.json")}
        if not self.args.no_save:
            old.update(python=python, checked_at=now, verification=verification, **self.runtime)
            atomic_json(SETTINGS_PATH, old)
        self.report.update(ok=True, reused=reuse, full_verified=True,
                           settings_saved=not self.args.no_save, python=python,
                           settings_path=SETTINGS_PATH, runtime=self.runtime)
        return self.report

    def java(self, env):
        java = shutil.which("java", path=env["PATH"])
        if not java:
            raise SetupError("missing_java", "未找到 Java；使用已有 JDK 17（--java-home），或用 conda 新建环境")
        result = self.run("java", [java, "-version"], env=env, timeout=min(30, self.args.timeout))
        require_step(result, "missing_java")
        match = re.search(r'version "(?:1\.)?(\d+)', result["output"])
        if not match or int(match[1]) < 11:
            raise SetupError("java_version", "需要 Java 11+，本次验证版本为 Java 17；当前：" + result["output"])
        return result

    def list(self):
        items = []
        for index, (python, source) in enumerate(self.candidates()):
            env = runtime_env(python)
            result = self.run("candidate-%d" % index,
                              [python, "-c", "import json,platform,sys;print(json.dumps({'version':platform.python_version(),'machine':platform.machine(),'python':sys.executable}))"],
                              env=env, timeout=15)
            item = {"python": python, "source": source, "available": result["ok"]}
            if result["ok"]:
                try:
                    item.update(json.loads(result["output"].strip().splitlines()[-1]))
                except (ValueError, IndexError):
                    item["available"] = False
            items.append(item)
        self.report.update(ok=True, candidates=items, native_machine=native_machine())
        return self.report

    def index_args(self):
        index = self.args.index_url
        if not index:
            return ["--index-url", "https://pypi.org/simple"]
        parsed = urlparse(index)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
            raise SetupError("invalid_index", "镜像必须是无内嵌账号密码的 HTTPS 地址")
        return ["--index-url", index]

    def bundle(self, path, info):
        path = Path(path).expanduser().absolute()
        manifest = path / "manifest.json"
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            if data["target"] != target_key(info):
                raise SetupError("wheel_target_mismatch", "离线包目标 %s 与当前 %s 不同" % (data["target"], target_key(info)))
            if data["requirements_sha256"] != sha256(REQUIREMENTS):
                raise SetupError("wheel_requirements_mismatch", "离线包不属于当前依赖版本，请重新准备")
            files = data["wheels"]
            if not files:
                raise ValueError("离线包中没有 wheel")
            if set(files) != {p.name for p in path.glob("*.whl")}:
                raise ValueError("目录中的 wheel 与校验清单不一致；请使用原始包目录")
            pins = {}
            for name, expected in files.items():
                if Path(name).name != name or not name.endswith(".whl"):
                    raise ValueError("非法 wheel 文件名")
                file = path / name
                if file.is_symlink() or sha256(file) != expected:
                    raise ValueError("wheel 校验失败：" + name)
                with zipfile.ZipFile(file) as z:
                    # Vendored packages (e.g. setuptools, ray) also carry nested metadata.
                    names = [n for n in z.namelist() if n.count("/") == 1 and n.endswith(".dist-info/METADATA")]
                    if len(names) != 1:
                        raise ValueError("wheel 元数据不唯一")
                    meta = BytesParser().parsebytes(z.read(names[0]))
                package = normalize(meta["Name"])
                version = meta["Version"]
                if package in pins and pins[package] != version:
                    raise ValueError("wheel 中存在同包不同版本：" + package)
                pins[package] = version
            for name, version in {**read_pins(REQUIREMENTS), **read_pins(BOOTSTRAP)}.items():
                if pins.get(name) != version:
                    raise ValueError("离线包缺少固定版本：" + name + "==" + version)
            lock = self.log_dir / "offline-requirements.txt"
            lock.write_text("\n".join(k + "==" + v for k, v in sorted(pins.items())) + "\n", encoding="utf-8")
            read_pins(lock)  # No URLs or arbitrary pip flags from archive metadata.
            return path, lock
        except SetupError:
            raise
        except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
            raise SetupError("invalid_wheel_bundle", str(exc)) from exc

    def download(self, python, destination):
        env = self.env(python)
        info = self.probe(python, "inventory", env)
        validate_profile(info)
        dest = Path(destination).expanduser().absolute()
        if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
            raise SetupError("destination_exists", "离线包输出目录必须不存在或为空：" + str(dest))
        dest.mkdir(parents=True, exist_ok=True)
        result = self.run("build-wheels", [python, "-m", "pip", "wheel", "--wheel-dir", str(dest),
                          "--prefer-binary", "-r", str(REQUIREMENTS), "-r", str(BOOTSTRAP),
                          "-c", str(CONSTRAINTS)] + self.index_args(), env=env, timeout=self.args.install_timeout)
        require_step(result, "installation_failed")
        manifest = {"schema_version": 1, "created_at": stamp(), "target": target_key(info),
                    "build_platform": info["platform"], "requirements_sha256": sha256(REQUIREMENTS),
                    "wheels": {p.name: sha256(p) for p in sorted(dest.glob("*.whl"))}}
        atomic_json(dest / "manifest.json", manifest)
        self.bundle(dest, info)
        self.report.update(ok=True, wheel_dir=str(dest), target=target_key(info),
                           wheel_count=len(manifest["wheels"]),
                           note="仅适用于同系统、架构、CPython 3.9；仍须在目标机器验收。Java/OpenMP 不在 wheel 包中。")
        return self.report

    def create(self):
        args = self.args
        prefix = Path(args.prefix or Path(SETTINGS_PATH).parent / "envs" / "riskmodel-py39").expanduser().absolute()
        if prefix.exists() or prefix.is_symlink():
            raise SetupError("environment_exists", "目标已存在，拒绝安装或升级已有环境：" + str(prefix),
                             ["用 --check --python 接入现有环境", "使用新的 --prefix 路径"])
        self.report["created_prefix"] = str(prefix)
        bundle = None
        if args.base_python or args.wheel_dir:
            if not args.base_python:
                raise SetupError("base_python_required", "离线创建需要 --base-python 指向已有原生 Python 3.9；不下载 Python 或 Java")
            base = self.resolve_python(args.base_python)
            env = self.env(base, reuse=False)
            info = self.probe(base, "inventory", env)
            validate_profile(info)
            self.java(env)
            if args.wheel_dir:
                bundle = self.bundle(args.wheel_dir, info)
            else:
                self.index_args()  # Validate before creating files.
            require_step(self.run("create-venv", [base, "-m", "venv", str(prefix)], env=env,
                                  timeout=args.install_timeout), "installation_failed")
        else:
            conda = args.conda or shutil.which("conda")
            if not conda:
                raise SetupError("missing_conda", "未找到 conda；可指定 --conda，或 --base-python 使用已有 Python 3.9 创建 venv")
            self.index_args()
            env = runtime_env(sys.executable)
            machine = native_machine()
            subdir = {("Darwin", "arm64"): "osx-arm64", ("Darwin", "x86_64"): "osx-64",
                      ("Linux", "arm64"): "linux-aarch64", ("Linux", "x86_64"): "linux-64"}.get((platform.system(), machine))
            if not subdir:
                raise SetupError("unsupported_system", "无法选择 conda 原生架构")
            env["CONDA_SUBDIR"] = subdir
            if platform.system() == "Darwin":
                # Older conda interpreters can otherwise see macOS 11+ as 10.16.
                env["SYSTEM_VERSION_COMPAT"] = "0"
            # Bootstrap tools are pinned by pip below; do not couple conda's build
            # variants to PyPI's version availability during Python creation.
            packages = ["python=3.9", "pip", "openjdk=17"]
            packages.append("llvm-openmp=18.1.8" if platform.system() == "Darwin" else "libgomp")
            require_step(self.run("create-conda", [conda, "create", "--yes", "--prefix", str(prefix),
                                     "--override-channels", "--channel", "conda-forge"] + packages,
                                  env=env, timeout=args.install_timeout), "installation_failed")
            # Native packages may need the environment's lib path at dynamic load time.
            if not args.library_dir:
                args.library_dir = str(prefix / "lib")
            if not args.java_home:
                args.java_home = find_java_home(prefix)
                if not args.java_home:
                    raise SetupError("missing_java", "新 conda 环境内未找到可执行的 Java：" + str(prefix))
        python = str(prefix / "bin" / "python")
        env = self.env(python, reuse=False)
        pip_base = [python, "-m", "pip", "install", "--no-input", "--disable-pip-version-check"]
        if bundle:
            directory, lock = bundle
            command = pip_base + ["--no-index", "--find-links", str(directory), "--only-binary=:all:",
                                  "--no-deps", "--no-cache-dir", "-r", str(lock)]
            # Explicit --no-index plus disabled pip configuration; all pins are local wheels.
            require_step(self.run("install-offline", command, env=env, timeout=args.install_timeout), "installation_failed")
        else:
            require_step(self.run("bootstrap-pip", pip_base + ["-r", str(BOOTSTRAP)] + self.index_args(),
                                  env=env, timeout=args.install_timeout), "installation_failed")
            require_step(self.run("install-online", pip_base + ["--prefer-binary", "-r", str(REQUIREMENTS),
                                  "-c", str(CONSTRAINTS)] + self.index_args(), env=env,
                                  timeout=args.install_timeout), "installation_failed")
        return self.check(python)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    actions = p.add_mutually_exclusive_group(required=True)
    actions.add_argument("--list", action="store_true", help="列出已有 Python 和 conda 环境")
    actions.add_argument("--check", action="store_true", help="检查并记住环境；记录未变化时快速复用")
    actions.add_argument("--create", action="store_true", help="新建独立 riskmodel-py39；绝不安装到已有路径")
    actions.add_argument("--download-wheels", metavar="DIR", help="在目标同类机器上准备带清单的离线 wheel 包")
    p.add_argument("--python", help="检查/下载所用 Python 路径、PATH 命令或 conda 环境名")
    p.add_argument("--base-python", help="使用已有 CPython 3.9 创建 venv；离线创建必填")
    p.add_argument("--prefix", help="新环境目录，默认 ~/.risk-modeling/envs/riskmodel-py39")
    p.add_argument("--conda", help="conda 可执行文件路径")
    p.add_argument("--wheel-dir", help="离线创建使用的包目录（含 manifest.json），仅配合 --create")
    p.add_argument("--index-url", help="在线安装/下载使用的 HTTPS 镜像，不修改全局设置")
    p.add_argument("--library-dir", help="已有原生 OpenMP 动态库目录")
    p.add_argument("--java-home", help="已有 JDK 目录（包含 bin/java）")
    p.add_argument("--mapping-dir", help="记住映射表目录")
    p.add_argument("--full", action="store_true", help="强制重跑所有小模型")
    p.add_argument("--no-save", action="store_true", help="仅验证，不修改环境选择记录")
    p.add_argument("--log-dir", help="日志根目录，每次运行创建独立子目录")
    p.add_argument("--timeout", type=int, default=180, help="每项检查超时秒数，默认 180")
    p.add_argument("--install-timeout", type=int, default=1800, help="每项安装/下载超时秒数，默认 1800")
    return p


def main():
    p = parser()
    args = p.parse_args()
    if args.timeout <= 0 or args.install_timeout <= 0:
        p.error("超时必须大于零")
    if args.wheel_dir and (not args.create or args.index_url):
        p.error("--wheel-dir 只用于 --create，不能同时指定 --index-url")
    if (args.base_python or args.prefix) and not args.create:
        p.error("--base-python 和 --prefix 只用于 --create")
    manager = None
    try:
        manager = Manager(args)
        if args.list:
            result = manager.list()
        elif args.create:
            result = manager.create()
        elif args.download_wheels:
            result = manager.download(manager.resolve_python(args.python), args.download_wheels)
        else:
            result = manager.check(manager.resolve_python(args.python))
    except (SetupError, OSError, ValueError) as exc:
        result = manager.report if manager else {}
        result.update(ok=False, error_kind=getattr(exc, "kind", "filesystem_error"), error=str(exc),
                      options=getattr(exc, "options", ["检查路径和访问权限后重试"]))
    except KeyboardInterrupt:
        result = manager.report if manager else {}
        result.update(ok=False, error_kind="interrupted", error="已中断，环境和日志保留，未记录为验证通过")
    if manager:
        result["finished_at"] = stamp()
        result["report_path"] = str(manager.log_dir / "report.json")
        atomic_json(result["report_path"], result)
    emit(result)


if __name__ == "__main__":
    main()
