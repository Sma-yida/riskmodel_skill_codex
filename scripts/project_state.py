"""Minimal project context: verified environment and user-supplied label meaning.

Execution progress stays in logs/task.json and logs/stages/. This file is not an
authorization token, a second task log, or a copy of the modeling configuration.
"""
from pathlib import Path

from common import load_json, save_json

PROJECT_FILE = "config/project.json"
ENVIRONMENT_KEYS = ("python", "java_home", "library_dir", "mapping_dir", "checked_at", "verification")


def validate_labels(value):
    if (not isinstance(value, dict) or set(value) != {"0", "1"}
            or any(not isinstance(v, str) or not v.strip() for v in value.values())
            or value["0"].strip() == value["1"].strip()):
        raise ValueError("标签含义必须同时提供不同的 0/1 含义；不能自动推断")
    return {key: value[key].strip() for key in ("0", "1")}


def read_project(root):
    path = Path(root) / PROJECT_FILE
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError("项目记录不能是符号链接：" + str(path))
    if not path.exists():
        return {}
    data = load_json(str(path))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("项目记录格式或版本无效：" + str(path))
    if "label_meaning" in data:
        validate_labels(data["label_meaning"])
    if "environment" in data:
        env = data["environment"]
        if (not isinstance(env, dict) or not isinstance(env.get("python"), str)
                or not env["python"] or not isinstance(env.get("verification"), dict)):
            raise ValueError("项目环境记录无效：" + str(path))
        for key in ("java_home", "library_dir", "mapping_dir", "checked_at"):
            if env.get(key) is not None and not isinstance(env[key], str):
                raise ValueError("项目环境字段无效：" + key)
    return data


def update_project(root, environment=None, label_meaning=None):
    from execution_state import mutation_lock, within, work_root
    root = work_root(root)
    labels = validate_labels(label_meaning) if label_meaning is not None else None
    if environment is not None:
        if (not isinstance(environment.get("python"), str) or not environment["python"]
                or environment.get("verification", {}).get("full_passed") is not True):
            raise ValueError("只能记录已完整验证的项目环境")
    with mutation_lock(root):
        data = read_project(root)
        data.setdefault("schema_version", 1)
        if labels is not None:
            data["label_meaning"] = labels
        if environment is not None:
            data["environment"] = {k: environment[k] for k in ENVIRONMENT_KEYS if k in environment}
        save_json(str(within(root, PROJECT_FILE)), data)
    return data
