"""Start, inspect, wait for or stop local modeling stages (Python standard library)."""
import argparse
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from common import (STAGE_IDS, MODELING_DIR, config_path, emit,
                    load_json, load_settings, resolve_workdir, save_json)
from env_support import runtime_env, sha256
from execution_state import (ACTIVE, TaskError, acquire, artifact_stamp, input_stamp,
                             lock_busy, read_record, receipt_path, required_artifacts,
                             result_path, stage_statuses, tail, upstream_stamp,
                             validate_config_paths, validate_plan, validate_result,
                             within, work_root)


def task_path(root):
    return within(root, "logs/task.json")


def write_task(root, task):
    task["heartbeat"] = time.time()
    save_json(str(task_path(root)), task)


def inspect(root):
    task = read_record(task_path(root))
    busy = lock_busy(root)
    if task.get("status") in ACTIVE:
        if not busy:
            task.update(status="interrupted", error="执行进程已退出，请查看日志后重跑")
        elif time.time() - task.get("heartbeat", 0) > 15:
            task.update(status="unresponsive", error="执行器超过 15 秒没有更新，仍有进程持锁；不启动重复任务")
    stage = task.get("stage")
    stages = stage_statuses(root)
    if stage and stage in stages and task.get("status") == "unresponsive":
        stages[stage]["state"] = "unresponsive"
    return {"ok": True, "workdir": str(root), "busy": busy, "task": task,
            "stages": stages, "log_tail": tail(task.get("log"), 4000),
            "elapsed_seconds": round((task.get("finished_at") or time.time()) - task.get("started_at", time.time()), 1)}


def preflight(root, stages):
    statuses = stage_statuses(root)
    validate_plan(stages, statuses)
    settings = load_settings()
    python = settings.get("python")
    if not python or not Path(python).is_file() or not settings.get("verification", {}).get("full_passed"):
        raise TaskError("environment_unchecked", "请先用 setup_env.py --check 验证并记录建模环境")
    configs = {s: validate_config_paths(root, s) for s in stages}
    # Reuse exactly the parameter validation used by make_config, without writing.
    from make_config import apply_changes, cross_check, user_values
    from common import template_path
    errors, warnings = [], []
    for s, cfg in configs.items():
        normalized, bad = apply_changes(cfg, s, user_values(s, cfg))
        errors.extend(bad)
        if not bad:
            # These are stored values, not a new request to change every parameter.
            checks = {}
            if s == "feature_select":
                checks["user_feature_list"] = normalized["settings"]["upload_fea_list"]
            if s == "model_train":
                checks["optimization_targets"] = normalized["settings"]["optimization_targets"]
            bad, warns, _ = cross_check(s, cfg, normalized, checks, load_json(template_path(s)), str(root),
                                        defer_data_range="data_analysis" in stages)
            errors.extend(bad)
            warnings.extend(warns)
    if errors:
        raise TaskError("invalid_config", str(errors))
    if "model_eval" in stages:
        train = load_json(config_path(str(root), "model_train"))
        eval_cfg = configs["model_eval"]
        algorithm = train["settings"]["model_select"].lower()
        requested = eval_cfg["settings"].get("eval_model_type")
        if requested and requested.lower() != algorithm:
            raise TaskError("model_mismatch", "评估模型必须与本次有效训练结果一致，请先训练所选模型或将 eval_model_type 设为 null")
        if algorithm == "autogluon" and not train["settings"]["save_autogluon_predictor"]:
            raise TaskError("model_not_saved", "AutoGluon 未保存模型，不能继续评估；请先确认开启保存并重新训练")
    return settings, warnings


def start(root, stages):
    lock = acquire(root)
    try:
        settings, warnings = preflight(root, stages)
        old = read_record(task_path(root))
        if old.get("status") in ACTIVE:
            old.update(status="interrupted", finished_at=time.time())
            save_json(str(within(root, "logs/previous_task.json")), old)
        task = {"status": "starting", "stages": stages, "stage": None,
                "started_at": time.time(), "finished_at": None,
                "settings": settings, "warnings": warnings,
                "configs": {s: sha256(config_path(str(root), s)) for s in stages},
                "inputs": {s: input_stamp(root, s) for s in stages}}
        write_task(root, task)
        within(root, "logs/stop.json").unlink(missing_ok=True)
        with within(root, "logs/runner.log").open("w", encoding="utf-8") as log:
            worker = subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()),
                                       "-w", str(root), "--worker", str(lock.fileno())],
                                      stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True, pass_fds=(lock.fileno(),), cwd=str(root))
        # Worker owns the inherited lock; closing this duplicate does not unlock it.
        return {"ok": True, "status": "starting", "pid": worker.pid, "stages": stages,
                "workdir": str(root), "warnings": warnings, "state_file": str(task_path(root))}
    finally:
        lock.close()


def clean_stage(root, stage):
    # Only standardized generated directories are owned by the runner. Data is untouched.
    out = within(root, "outputs/" + stage)
    keep = set()
    if stage == "feature_select":
        # Backend validates input hashes and required columns before using these caches.
        keep = {out / "outputs" / n for n in
                ("single_var.csv", "mono_var.csv", "feature_select_cache.json")}
    if out.exists():
        for p in sorted(out.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            within(root, str(p.relative_to(root)))
            if p in keep:
                continue
            if p.is_file():
                p.unlink()
            elif p.is_dir() and not any(p.iterdir()):
                p.rmdir()
    for suffix in ("json", "md", "html"):
        within(root, "results/" + stage + "." + suffix).unlink(missing_ok=True)
    within(root, "logs/" + stage + "_startup.txt").unlink(missing_ok=True)


def stop_requested(root, task):
    return read_record(within(root, "logs/stop.json")).get("started_at") == task["started_at"]


def terminate_child(child):
    """Signal this worker's group; refuse a PID that now identifies another process."""
    def verify_identity():
        # An unreaped child PID cannot be reused. Its command can legitimately
        # change during interpreter startup, so comparing ps command text races.
        if child.poll() is None:
            return
        try:
            os.kill(child.pid, 0)
        except ProcessLookupError:
            return
        raise TaskError("process_identity_changed", "已回收的子进程 PID 再次出现，拒绝向可能复用的进程组发送信号")
    try:
        verify_identity()
        os.killpg(child.pid, signal.SIGTERM)
        time.sleep(0.2)
        # macOS may report EPERM, not ESRCH, for an already vanished process group.
        members = subprocess.run(["ps", "-axo", "pid=,pgid=,stat="], capture_output=True, text=True, timeout=5)
        live = [line.split()[0] for line in members.stdout.splitlines()
                if len(line.split()) >= 3 and line.split()[1] == str(child.pid)
                and not line.split()[2].startswith("Z")]
        if members.returncode or live:
            verify_identity()
            os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError as exc:
        members = subprocess.run(["ps", "-axo", "pid=,pgid=,stat="], capture_output=True, text=True, timeout=5)
        live = [line for line in members.stdout.splitlines() if len(line.split()) >= 3
                and line.split()[1] == str(child.pid) and not line.split()[2].startswith("Z")]
        if members.returncode == 0 and not live and child.poll() is not None:
            return
        if child.poll() is None:
            child.kill()
        raise TaskError("cleanup_permission", "系统拒绝清理任务子进程组：" + str(exc))
    finally:
        # A rejected signal must never turn the cleanup path into an endless wait.
        try:
            child.wait(timeout=3)
        except subprocess.TimeoutExpired as exc:
            raise TaskError("cleanup_incomplete", "子进程尚未结束，不能报告停止成功；请查看 runner.log") from exc


def worker(root, fd):
    lock = os.fdopen(fd, "a+")
    task = read_record(task_path(root))
    task.update(status="running", pid=os.getpid())
    child, record, stage = None, None, None
    interrupted = [False]
    def on_signal(signum, frame):
        interrupted[0] = True
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    try:
        for stage in task["stages"]:
            record = None
            task.update(stage=stage, log=str(within(root, "logs/" + stage + ".log")))
            write_task(root, task)
            if interrupted[0] or stop_requested(root, task):
                raise TaskError("stopped", "用户停止了任务")
            preflight(root, [stage])
            cfg = validate_config_paths(root, stage)
            if sha256(config_path(str(root), stage)) != task["configs"][stage] or input_stamp(root, stage) != task["inputs"][stage]:
                raise TaskError("inputs_changed", "等待执行期间数据或配置发生变化，请重新确认")
            record = {"status": "running", "stage": stage, "started_at": time.time(),
                      "config_sha256": task["configs"][stage], "inputs": input_stamp(root, stage),
                      "upstream": upstream_stamp(root, stage),
                      "log": str(within(root, "logs/" + stage + ".log"))}
            save_json(str(receipt_path(root, stage)), record)
            task.update(stage=stage, log=record["log"])
            write_task(root, task)
            clean_stage(root, stage)
            env = runtime_env(task["settings"]["python"],
                              **{k: task["settings"].get(k) for k in ("java_home", "library_dir", "mapping_dir")},
                              cache_dir=str(within(root, "logs/matplotlib")))
            with Path(record["log"]).open("w", encoding="utf-8") as log:
                child = subprocess.Popen([task["settings"]["python"], "-B", "-u", str(Path(__file__).resolve()),
                                          "-w", str(root), "--backend", stage], cwd=str(root), env=env,
                                         stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True, pass_fds=(fd,))
                while child.poll() is None:
                    write_task(root, task)
                    if interrupted[0] or stop_requested(root, task):
                        terminate_child(child)
                        child = None
                        raise TaskError("stopped", "用户停止了任务；当前环节结果不能用于后续环节")
                    time.sleep(0.5)
                code = child.returncode
                terminate_child(child)
                child = None
            if code:
                raise TaskError("backend_failed", "后台执行失败；请查看日志末尾和完整日志")
            try:
                validate_result(result_path(root, stage), stage)
            except (OSError, ValueError) as exc:
                lines = tail(record["log"]).splitlines()
                errors = [line for line in lines if "Error:" in line or "Exception:" in line or "出错:" in line]
                detail = errors[-1] if errors else str(exc)
                raise TaskError("invalid_result", "后台未生成有效结果：" + detail) from exc
            required_artifacts(root, stage, cfg)
            if (sha256(config_path(str(root), stage)) != record["config_sha256"] or
                    input_stamp(root, stage) != record["inputs"] or upstream_stamp(root, stage) != record["upstream"]):
                raise TaskError("inputs_changed", "执行期间数据、配置或上游结果变化，本次结果不可用于后续环节")
            record.update(status="succeeded", completed_at=time.time(), completed_ns=time.time_ns(),
                          result_sha256=sha256(result_path(root, stage)), artifacts=artifact_stamp(root, stage))
            save_json(str(receipt_path(root, stage)), record)
            try:
                from make_report import generate
                reports = generate(root, stage)
                record.update(reports=reports, report_error=None)
                save_json(str(receipt_path(root, stage)), record)
            except Exception as exc:
                record["report_error"] = str(exc)
                save_json(str(receipt_path(root, stage)), record)
                task.update(status="report_failed", error=str(exc))
                break
        else:
            task["status"] = "succeeded"
    except BaseException as exc:
        if child is not None:
            try:
                terminate_child(child)
            except Exception as cleanup:
                exc = cleanup
        kind = getattr(exc, "kind", "execution_error")
        task.update(status="stopped" if kind == "stopped" else "failed", error_kind=kind, error=str(exc))
        if record and record.get("stage") == stage and record.get("status") == "running":
            record.update(status=task["status"], error_kind=kind, error=str(exc), completed_at=time.time())
            save_json(str(receipt_path(root, stage)), record)
        traceback.print_exc()
    finally:
        task["finished_at"] = time.time()
        write_task(root, task)
        lock.close()


def backend(root, stage):
    os.chdir(root)
    sys.path.insert(0, MODELING_DIR)
    spec = importlib.util.spec_from_file_location("modeling_entry", str(Path(MODELING_DIR) / (stage + ".py")))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Original CLI discards this return value. The adapter makes it observable.
    return 0 if module.run(config_path(str(root), stage)) is True else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-w", "--workdir")
    action = p.add_mutually_exclusive_group(required=True)
    action.add_argument("--start", nargs="+", choices=STAGE_IDS)
    action.add_argument("--status", action="store_true")
    action.add_argument("--wait", action="store_true")
    action.add_argument("--stop", action="store_true")
    action.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    action.add_argument("--backend", choices=STAGE_IDS, help=argparse.SUPPRESS)
    p.add_argument("--seconds", type=float, default=30, help="单次等待 0–60 秒，默认 30")
    p.add_argument("--check", action="store_true", help="配合 --start 只检查，不启动")
    args = p.parse_args()
    if args.check and not args.start:
        p.error("--check 必须与 --start 一起使用")
    try:
        root = work_root(resolve_workdir(args.workdir) or "riskmodel_work")
        if args.backend:
            sys.exit(backend(root, args.backend))
        if args.worker is not None:
            worker(root, args.worker)
            return
        if args.start:
            if args.check:
                _, warnings = preflight(root, args.start)
                emit({"ok": True, "mode": "check", "stages": args.start, "warnings": warnings})
            emit(start(root, args.start))
        if args.stop:
            current = inspect(root)
            if current["task"].get("status") in ACTIVE | {"unresponsive"} and current["busy"]:
                save_json(str(within(root, "logs/stop.json")), {"started_at": current["task"]["started_at"]})
            elif current["busy"]:
                raise TaskError("stop_unconfirmed", "执行器未响应或其他操作持锁，未向未知进程发送信号；请检查 runner.log")
        if args.wait or args.stop:
            deadline = time.monotonic() + min(60, max(0, args.seconds if args.wait else 5))
            while lock_busy(root) and time.monotonic() < deadline:
                time.sleep(0.25)
        emit(inspect(root))
    except (TaskError, OSError, ValueError, KeyError) as exc:
        emit({"ok": False, "error_kind": getattr(exc, "kind", "execution_error"), "error": str(exc)})


if __name__ == "__main__":
    main()
