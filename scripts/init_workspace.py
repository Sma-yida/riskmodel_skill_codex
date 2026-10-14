"""Create or inspect a work directory without overwriting user files."""
import argparse
import json
import os
from pathlib import Path

from common import (BASE_COLUMNS, DEFAULT_WORKDIR, RESULT_KEYS, SKILL_DIR, STAGE_IDS,
                    STARTUP_KEYS, WORKDIR_SUBDIRS, emit, resolve_workdir, template_path)


def initialize(workdir=None):
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
    created = []
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
                status.update(state="result_present", result=str(result),
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
    return {"ok": True, "workdir": str(root), "created": created,
            "data_files": sorted(p.name for p in (root / "data").iterdir() if p.is_file()),
            "stages": statuses,
            "guidance": {"data": "将主数据命名为 data.csv、data.parquet 或 data.feather，放入 data/",
                         "required_columns": BASE_COLUMNS,
                         "optional": "时间外样本使用 oot_*.csv / parquet / feather；映射表默认放 data/mapping/",
                         "next": "先确认数据与五份配置，再执行数据质量分析；本工具不会启动建模"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", help="默认当前目录下 riskmodel_work；已在工作目录中则复用")
    args = parser.parse_args()
    try:
        emit(initialize(args.workdir))
    except (OSError, ValueError) as exc:
        emit({"ok": False, "error_kind": "workspace_error", "error": str(exc)})


if __name__ == "__main__":
    main()
