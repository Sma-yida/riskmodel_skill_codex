"""Create or inspect a work directory without overwriting user files."""
import argparse
import json
import os
from pathlib import Path

from common import (BASE_COLUMNS, DEFAULT_WORKDIR, RESULT_KEYS, SKILL_DIR, STAGE_IDS,
                    STARTUP_KEYS, WORKDIR_SUBDIRS, emit, resolve_workdir, template_path)
from execution_state import TaskError, lock_busy, stage_statuses
from project_state import PROJECT_FILE, read_project, update_project, validate_labels


def plan_workspace(workdir=None):
    """Validate the proposed location and compute missing files without writing."""
    requested = Path(resolve_workdir(workdir) or Path.cwd() / DEFAULT_WORKDIR)
    if requested.is_symlink():
        raise ValueError("工作目录不能是符号链接：" + str(requested))
    root = requested.resolve()
    skill = Path(SKILL_DIR).resolve()
    if root == skill or skill in root.parents or root in skill.parents:
        raise ValueError("工作目录必须与 skill 目录分开，不能使用 skill 内部目录或其父目录")
    # Check all collisions and templates before creating anything.
    paths = [root] + [root / d for d in WORKDIR_SUBDIRS]
    for path in paths:
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            raise ValueError("目录位置被文件或链接占用：" + str(path))
    templates = {}
    for stage in STAGE_IDS:
        dest = root / "config" / (stage + "_config.json")
        if dest.is_symlink() or (dest.exists() and not dest.is_file()):
            raise ValueError("配置位置被目录或链接占用：" + str(dest))
        if not dest.exists():
            raw = Path(template_path(stage)).read_bytes()
            json.loads(raw)
            templates[dest] = raw
    read_project(root)  # A damaged project record must not be overwritten.
    return root, paths, templates


def initialize(workdir=None, check=False, label_meaning=None):
    root, paths, templates = plan_workspace(workdir)
    labels = validate_labels(label_meaning) if label_meaning is not None else None
    if check:
        return {"ok": True, "mode": "check", "workdir": str(root),
                "would_create": [str(p.relative_to(root)) if p != root else "."
                                 for p in paths if not p.exists()]
                                + [str(p.relative_to(root)) for p in templates],
                "project": read_project(root), "proposed_label_meaning": labels,
                "data_files": sorted(p.name for p in (root / "data").glob("*") if p.is_file()),
                "message": "仅预览，未创建目录或写文件；确认后去掉 --check"}
    created = []
    if (root / "logs").is_dir() and templates and lock_busy(root):
        raise ValueError("任务运行期间不能补写配置，请等待任务结束")
    for path in paths:
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            created.append(str(path.relative_to(root)) if path != root else ".")
    for path, raw in templates.items():
        try:
            with path.open("xb") as f:  # Never overwrite, even with concurrent initializers.
                f.write(raw)
            created.append(str(path.relative_to(root)))
        except FileExistsError:
            pass
    if labels is not None:
        update_project(root, label_meaning=labels)
    statuses = {}
    for stage in STAGE_IDS:
        config = root / "config" / (stage + "_config.json")
        status = {"config": str(config), "state": "not_started"}
        try:
            cfg = json.loads(config.read_text(encoding="utf-8"))
            output = cfg["output_path"]
            result = root / output[RESULT_KEYS[stage]]
            startup = root / output[STARTUP_KEYS[stage]]
            # Do not inspect paths outside the selected workspace.
            if any(root not in p.resolve().parents for p in (result, startup)):
                raise ValueError("结果或启动标记路径位于工作目录之外")
            if result.is_file():
                status.update(state="untracked", result=str(result),
                              modified_at=result.stat().st_mtime)
                try:
                    json.loads(result.read_text(encoding="utf-8"))
                except (ValueError, OSError):
                    status["state"] = "invalid_result"
            elif startup.exists():
                status["state"] = "startup_marker_present"
                status["note"] = "存在启动标记；仅凭标记不能判断进程仍在运行"
        except (ValueError, TypeError, KeyError, OSError) as exc:
            status.update(state="invalid_config", error=str(exc))
        statuses[stage] = status
    verified = stage_statuses(root)
    for stage, status in verified.items():
        if (root / "logs/stages" / (stage + ".json")).exists():
            statuses[stage].update(status)
    task = {}
    if (root / "logs/task.json").exists():
        from run_stage import inspect
        task = inspect(root)["task"]
    return {"ok": True, "workdir": str(root), "created": created,
            "project": read_project(root),
            "data_files": sorted(p.name for p in (root / "data").iterdir() if p.is_file()),
            "stages": statuses, "task": task,
            "guidance": {"data": "将主数据命名为 data.csv、data.parquet 或 data.feather，放入 data/",
                         "required_columns": BASE_COLUMNS,
                         "optional": "时间外样本使用 oot_*.csv / parquet / feather；映射表默认放 data/mapping/",
                         "next": "先确认数据与五份配置，再执行数据质量分析；本工具不会启动建模"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", help="默认当前目录下 riskmodel_work；已在工作目录中则复用")
    parser.add_argument("--check", action="store_true", help="预览创建位置和缺失文件，不写入")
    parser.add_argument("--label-zero", help="用户已说明的标签 0 含义，与 --label-one 同时提供")
    parser.add_argument("--label-one", help="用户已说明的标签 1 含义，与 --label-zero 同时提供")
    args = parser.parse_args()
    try:
        labels = None if args.label_zero is None and args.label_one is None else {"0": args.label_zero, "1": args.label_one}
        emit(initialize(args.workdir, check=args.check, label_meaning=labels))
    except (OSError, ValueError, TaskError) as exc:
        emit({"ok": False, "error_kind": "workspace_error", "error": str(exc)})


if __name__ == "__main__":
    main()
