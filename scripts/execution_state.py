"""Workspace locking, provenance and result validation (standard library only)."""
import contextlib
import fcntl
import json
import os
from pathlib import Path

from common import (STAGE_IDS, STAGE_NAMES, SKILL_DIR, config_path, load_json,
                    load_settings, template_path, find_data_file)
from env_support import sha256

ACTIVE = {"starting", "running", "stopping"}
SECTIONS = {
    "data_analysis": ["样本整体概述", "特征分析", "违约率分析"],
    "data_clean_split": ["数据清洗概览", "数据切分方式", "数据集分布"],
    "feature_select": ["特征筛选概览", "特征详情", "统计分析"],
    "model_eval": ["模型信息", "模型效果评估", "模型稳定性评估", "Lift分析结果"],
}


class TaskError(Exception):
    def __init__(self, kind, message):
        super().__init__(message)
        self.kind = kind


def work_root(path):
    requested = Path(path).expanduser()
    if requested.is_symlink():
        raise TaskError("invalid_workspace", "工作目录不能是符号链接")
    root = requested.resolve()
    skill = Path(SKILL_DIR).resolve()
    if root == skill or skill in root.parents or root in skill.parents:
        raise TaskError("invalid_workspace", "工作目录必须与 Skill 目录分开")
    for name in ("data", "config", "outputs", "results", "logs"):
        child = root / name
        if child.is_symlink() or not child.is_dir():
            raise TaskError("invalid_workspace", "请先初始化工作目录；缺少目录或目录是链接：" + str(child))
    return root


def within(root, relative):
    path = Path(root) / relative
    if Path(root).resolve() not in path.resolve().parents:
        raise TaskError("path_outside_workspace", "路径必须位于工作目录内：" + str(path))
    for part in (path,) + tuple(path.parents):
        if part == Path(root):
            break
        if part.is_symlink():
            raise TaskError("unsafe_link", "工作产物路径不能是符号链接：" + str(part))
    return path


def acquire(root):
    file = open(within(root, "logs/.task.lock"), "a+")
    try:
        fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        file.close()
        raise TaskError("workspace_busy", "工作目录已有任务或配置写入操作，请先查看状态或停止任务")
    return file


@contextlib.contextmanager
def mutation_lock(root):
    file = acquire(root)
    try:
        yield
    finally:
        file.close()


def read_record(path):
    try:
        data = load_json(str(path))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def receipt_path(root, stage):
    return within(root, "logs/stages/" + stage + ".json")


def result_path(root, stage):
    return within(root, "results/" + stage + ".json")


def validate_result(path, stage):
    data = load_json(str(path))
    if not isinstance(data, dict) or not isinstance(data.get("报告标题"), str) or not data["报告标题"].strip():
        raise ValueError("结果不是有效报告对象，或缺少报告标题")
    sections = SECTIONS.get(stage, [])
    if stage == "model_train":
        if "leaderboard" in data:
            sections = ["对比模型列表", "leaderboard"]
            if not data["leaderboard"]:
                raise ValueError("AutoGluon 未产生模型排名")
        elif "训练状态" in data:
            if data["训练状态"] != "成功":
                raise ValueError("评分卡训练状态不是成功")
            sections = ["算法信息", "特征流转", "模型性能指标"]
        else:
            sections = ["模型参数信息", "模型性能指标", "特征重要性分析"]
    for key in sections:
        if key not in data or data[key] is None:
            raise ValueError("结果缺少必需部分：" + key)
    return data


def file_stamp(path):
    stat = path.stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def input_stamp(root, stage):
    """Track only this stage's raw inputs; predecessor receipts cover intermediates."""
    root = Path(root)
    data = root / "data"
    paths = []
    if stage == "data_analysis":
        paths.extend(data.glob("data.*"))
    elif stage == "data_clean_split":
        paths.extend(data.glob("data.*"))
        paths.extend(data.glob("oot_*"))
    elif stage == "feature_select":
        paths.append(data / "null_importance_config.json")
    elif stage == "model_train":
        cfg = load_json(config_path(str(root), stage))
        algorithm = cfg["settings"]["model_select"].lower()
        paths += [data / "feature_dict.csv", data / ("model_train_spec_" + algorithm + ".json")]
    elif stage == "model_eval":
        mapping = Path(load_settings().get("mapping_dir") or data / "mapping")
        paths.extend(mapping.rglob("*"))
        paths.append(data / "feature_dict.csv")
    if stage in ("data_analysis", "data_clean_split"):
        paths.extend(p for p in data.iterdir() if p.suffix.lower() in (".pkl", ".pmml"))
        paths.append(data / "feature_dict.csv")
    return {str(p.absolute()): (dict(file_stamp(p), sha256=sha256(p))
                              if p.suffix == ".json" else file_stamp(p))
            for p in sorted(set(paths)) if p.is_file()}


def validate_config_paths(root, stage):
    cfg = load_json(config_path(str(root), stage))
    template = load_json(template_path(stage))
    for key in ("data_path", "output_path", "model_file", "report_all"):
        if cfg.get(key) != template.get(key):
            raise TaskError("unsupported_paths", "%s 的 %s 必须使用工作目录标准路径，请重新初始化缺失配置或通过 make_config 修改参数" % (stage, key))
    for section in ("output_path", "model_file", "report_all"):
        for value in cfg.get(section, {}).values():
            if isinstance(value, str):
                within(root, value)
    within(root, "config/" + stage + "_config.json")
    return cfg


def required_artifacts(root, stage, cfg):
    """Nonempty artifacts needed for this branch; optional report switches respected."""
    output, settings = cfg["output_path"], cfg["settings"]
    names = []
    if stage == "data_analysis":
        if settings["gen_report"]:
            names.append(output["dq_report_path"])
    elif stage == "data_clean_split":
        for name in ("train", "test", "psi"):
            found = find_data_file(str(Path(root) / "outputs/data_clean_split/data"), name)
            if not found:
                raise TaskError("missing_artifact", "缺少清洗数据集：" + name)
            names.append(str(Path(found).relative_to(root)))
    elif stage == "feature_select":
        names += [output[k] for k in ("use_feature_path", "single_var_path", "mono_var_path")]
        features = load_json(str(within(root, output["use_feature_path"])))
        if not isinstance(features, list) or not features:
            raise TaskError("empty_features", "筛选后没有可用特征，请检查数据或调整已支持的筛选参数")
    elif stage == "model_train":
        algorithm = settings["model_select"].lower()
        if algorithm == "lightgbm":
            names += [output[k] for k in ("pkl_model_path", "pmml_model_path", "model_fea_path")]
        elif algorithm == "lrscorecard":
            names += ["outputs/model_train/LR/model/scorecard_model.pkl",
                      "outputs/model_train/LR/report/评分卡模型报告.xlsx",
                      "outputs/model_train/LR/outputs/scorecard_points.csv"]
        else:
            names += [output["model_fea_path"], "outputs/model_train/outputs/autogluon_leaderboard.csv"]
            if settings["save_autogluon_predictor"]:
                names.append("outputs/model_train/model/autogluon_predictor/predictor.pkl")
    elif stage == "model_eval":
        train = load_json(config_path(str(root), "model_train"))
        algorithm = train["settings"]["model_select"].lower()
        if algorithm == "lrscorecard":
            names.append("outputs/model_train/LR/report/评分卡模型报告.xlsx")
        elif settings["gen_report"]:
            branch = "LightGBM"
            if algorithm == "autogluon":
                result = load_json(str(result_path(root, stage)))
                model = result.get("模型信息", {}).get("模型名称")
                if not isinstance(model, str) or not model or Path(model).name != model:
                    raise TaskError("invalid_result", "AutoGluon 评估结果缺少有效模型名称")
                branch = "Autogluon/" + model
            names.append(output["dst"] + "/" + branch + "/report/model_evaluation_report.xlsx")
    for name in names:
        p = within(root, name)
        if not p.is_file() or not p.stat().st_size:
            raise TaskError("missing_artifact", "缺少本次运行的非空产物：" + name)
    return names


def artifact_stamp(root, stage):
    """Includes model directory contents; no model loading needed for status."""
    dirs = [Path(root) / "outputs" / stage]
    if stage == "data_clean_split":
        dirs.append(Path(root) / "outputs/baseline_model")
    return {str(p.relative_to(root)): file_stamp(p)
            for d in dirs for p in sorted(d.rglob("*")) if p.is_file()}


def upstream_stamp(root, stage):
    return {s: {"result": sha256(result_path(root, s)),
                "completed_ns": read_record(receipt_path(root, s)).get("completed_ns")}
            for s in STAGE_IDS[:STAGE_IDS.index(stage)]}


def lock_busy(root):
    try:
        handle = acquire(root)
    except TaskError:
        return True
    handle.close()
    return False


def stage_statuses(root):
    root = Path(root).resolve()
    statuses = {}
    for stage in STAGE_IDS:
        result = result_path(root, stage)
        record = read_record(receipt_path(root, stage))
        item = {"stage": stage, "name": STAGE_NAMES[stage], "state": "not_started"}
        if record:
            item.update(state=record.get("status", "unknown"), log=record.get("log"),
                        error=record.get("error"), completed_at=record.get("completed_at"))
        elif result.exists():
            item.update(state="untracked", reason="存在历史结果，但没有本执行器的成功记录")
            try:
                load_json(str(result))
            except (OSError, ValueError):
                item.update(state="invalid_result", reason="历史结果不是有效 JSON")
        if record.get("status") in ACTIVE and not lock_busy(root):
            item.update(state="interrupted", reason="任务锁已释放，运行进程已退出；请查看日志后重跑")
        if record.get("status") == "succeeded":
            try:
                validate_result(result, stage)
                reasons = []
                if sha256(result) != record.get("result_sha256"):
                    reasons.append("结果文件已变更")
                if sha256(config_path(str(root), stage)) != record.get("config_sha256"):
                    reasons.append("当前配置与执行时不同")
                if input_stamp(root, stage) != record.get("inputs"):
                    reasons.append("原始数据或辅助输入已变更")
                for prior in STAGE_IDS[:STAGE_IDS.index(stage)]:
                    if statuses[prior]["state"] != "succeeded":
                        reasons.append("前置环节 %s 尚未成功或已过期" % prior)
                    elif {"result": sha256(result_path(root, prior)),
                          "completed_ns": read_record(receipt_path(root, prior)).get("completed_ns")} != record.get("upstream", {}).get(prior):
                        reasons.append("前置环节 %s 已重跑" % prior)
                    elif result_path(root, prior).stat().st_mtime_ns > result.stat().st_mtime_ns:
                        reasons.append("前置环节 %s 结果更新时间更晚" % prior)
                current_artifacts = artifact_stamp(root, stage)
                expected_artifacts = record.get("artifacts", {})
                for name in sorted(set(current_artifacts) - set(expected_artifacts)):
                    reasons.append("产物目录新增文件：" + name)
                for name, expected in expected_artifacts.items():
                    p = within(root, name)
                    if not p.is_file() or file_stamp(p) != expected:
                        reasons.append("产物缺失或已变更：" + name)
                if reasons:
                    item.update(state="stale", reasons=reasons)
                else:
                    item.update(result=str(result), reports=record.get("reports", {}),
                                report_error=record.get("report_error"))
                    if any(not Path(p).is_file() for p in record.get("reports", {}).values()):
                        item["report_error"] = "报告文件缺失，可单独补生成"
            except (OSError, ValueError, TaskError) as exc:
                item.update(state="invalid_result", reason=str(exc))
        statuses[stage] = item
    return statuses


def validate_plan(stages, statuses):
    if not stages or any(s not in STAGE_IDS for s in stages):
        raise TaskError("invalid_stages", "请指定五个环节中的一个或多个：" + ", ".join(STAGE_IDS))
    if stages != sorted(set(stages), key=STAGE_IDS.index):
        raise TaskError("invalid_order", "环节必须不重复，并按数据分析 → 清洗 → 筛选 → 训练 → 评估排序")
    scheduled = set()
    for stage in stages:
        for prior in STAGE_IDS[:STAGE_IDS.index(stage)]:
            if prior not in scheduled and statuses[prior]["state"] != "succeeded":
                raise TaskError("prerequisite_missing", "执行 %s 前需先成功运行 %s（当前 %s）" %
                                (stage, prior, statuses[prior]["state"]))
        scheduled.add(stage)


def tail(path, limit=6000):
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - limit))
            return f.read().decode("utf-8", errors="replace")
    except (OSError, TypeError):
        return ""
