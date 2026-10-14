"""把确认后的参数写成环节配置文件 config/<环节>_config.json。

用法：
  检查参数（不写文件，需求确认前使用）：
    python3 make_config.py -w <工作目录> --set '{"feature_select": {"iv_thresh": 0.03}}' --check
  用户确认后写入：
    python3 make_config.py -w <工作目录> --set '{"feature_select": {"iv_thresh": 0.03}}'
  查看某个环节的当前参数：
    python3 make_config.py -w <工作目录> --show feature_select

--set 是 {环节: {参数: 值}} 形式的 JSON，可以一次写多个环节；只写要改的参数，没写的保持当前值。
JSON 里引号不好处理时，可以写进文件，改用 --set-file <文件>。
可改的参数和取值范围见 references/ 下各环节文档的参数表，只认表里的参数。
"""
import argparse
import ast
import copy
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    BASE_COLUMNS, RESULT_KEYS, STAGE_IDS, STAGE_NAMES, STARTUP_KEYS,
    config_path, emit, find_data_file, list_oot_names, load_json,
    read_columns, resolve_workdir, save_json, template_path,
)


# ==================== 参数对照表 ====================
# 每个参数：kind = 取值类型，path = 写入配置文件的位置，label = 中文名，其余为取值限制。
# 写入位置与原 MCP 的参数更新逻辑（modeling_params_update）一致。

def P(kind, path, label, **opts):
    spec = {"kind": kind, "path": tuple(path.split(".")), "label": label}
    spec.update(opts)
    return spec


def G(label, params):
    return {"kind": "group", "label": label, "params": params}


def _lgb(key):
    return "settings.model_param." + key


def _hpo(key):
    return "settings.model_selection_params." + key


SPEC = {
    "data_analysis": {
        "missing_list": P("missing_list", "settings.missing_list", "缺失值标记"),
        "gen_data_quality_report": P("bool", "settings.gen_report", "是否生成数据质量报告"),
        "save_data_quality_output": P("bool", "settings.save_output", "是否保存中间结果"),
    },
    "data_clean_split": {
        "train_split_range": P("date_range", "settings.train_split_range", "训练集时间范围"),
        "oot_split_range": P("date_range", "settings.oot_split_range", "测试集(OOT)时间范围"),
        "psi_split_range": P("date_range", "settings.psi_split_range", "PSI数据集时间范围"),
        "channel_keep_list": P("str_list", "settings.product_list_keep", "保留渠道"),
        "fill_val": P("int", "settings.fill_val", "缺失值填充值"),
        "split_method": P("enum", "settings.method", "切分方式",
                          choices=["date_range", "date_proportion", "random"]),
        "test_size": P("number", "settings.test_size", "测试集占比", gt=0, lt=1),
        "random_state": P("int", "settings.random_state", "随机种子"),
    },
    "feature_select": {
        "iv_thresh": P("number", "settings.iv_thresh", "IV阈值", min=0, max=1),
        "psi_thresh": P("number", "settings.psi_thresh", "PSI阈值", min=0, max=1),
        "miss_thresh": P("number", "settings.miss_thresh", "缺失率阈值", min=0, max=1),
        "corr_thresh": P("number", "settings.corr_thresh", "相关性阈值", min=0, max=1),
        "use_null_importance": P("bool", "settings.use_null_importance", "是否启用Null Importance"),
        "top_n": P("int", "settings.top_n", "Null Importance保留Top N", min=1),
        "user_feature_list": P("str_list", "settings.upload_fea_list", "人工指定特征"),
    },
    "model_train": {
        "model_select": P("enum", "settings.model_select", "模型算法",
                          choices=["LightGBM", "AutoGluon", "LrScorecard"]),
        "n_trials": P("int", "settings.n_trials", "寻优次数", min=1),
        "ks_gap_threshold": P("number", "settings.ks_gap_threshold", "训练/测试KS差值上限", min=0, max=1),
        "run_hpo": P("bool", "settings.RUN_HPO", "是否超参寻优"),
        "optimization_mode": P("enum", "settings.optimization_mode", "寻优模式", choices=["single", "multi"]),
        "optimization_targets": P("targets", "settings.optimization_targets", "寻优目标"),
        "autogluon_params": G("AutoGluon参数", {
            "autogluon_time_limit": P("int", "settings.autogluon_time_limit", "AutoGluon时间限制(秒)", min=1),
            "save_autogluon_predictor": P("bool", "settings.save_autogluon_predictor", "是否保存AutoGluon模型文件"),
        }),
        "lr_scorecard_params": G("评分卡参数", {
            "binning_method": P("enum", "settings.lr_scorecard_params.binning_method", "分箱方法",
                                choices=["chi", "dt", "quantile", "step"]),
            "adj_rules": P("dict", "settings.lr_scorecard_params.adj_rules", "人工分箱规则"),
            "fine_selection_corr": P("number", "settings.lr_scorecard_params.fine_selection_corr",
                                     "细筛相关性阈值", gt=0, max=1),
            "stepwise_criterion": P("enum", "settings.lr_scorecard_params.stepwise_criterion", "逐步回归准则",
                                    choices=["aic", "bic"]),
            "score_scale": P("score_scale", "settings.lr_scorecard_params.score_scale", "分数刻度"),
        }),
        "lightgbm_params": G("LightGBM参数", {
            "random_state": P("int", _lgb("random_state"), "随机种子"),
            "n_jobs": P("int", _lgb("n_jobs"), "CPU线程数", min=1, also=[-1]),
            "min_split_gain": P("int", _lgb("min_split_gain"), "最小分割增益", min=0),
            "learning_rate": P("number", _lgb("learning_rate"), "学习率", gt=0, max=1),
            "num_leaves": P("int", _lgb("num_leaves"), "每棵树最大叶子数", min=2),
            "max_bin": P("int", _lgb("max_bin"), "最大分箱数", min=2),
            "min_child_samples": P("int", _lgb("min_child_samples"), "叶节点最少样本数", min=1),
            "reg_alpha": P("number", _lgb("reg_alpha"), "L1正则", min=0),
            "reg_lambda": P("number", _lgb("reg_lambda"), "L2正则", min=0),
            "subsample": P("number", _lgb("subsample"), "样本采样比例", gt=0, max=1),
            "subsample_freq": P("int", _lgb("subsample_freq"), "子采样频率", min=0),
            "colsample_bytree": P("number", _lgb("colsample_bytree"), "特征采样比例", gt=0, max=1),
            "min_child_weight": P("number", _lgb("min_child_weight"), "叶节点最小权重和", min=0),
            "scale_pos_weight": P("number", _lgb("scale_pos_weight"), "正负样本权重比", gt=0),
            "n_estimators": P("int", _lgb("n_estimators"), "树的数量", min=1),
            "max_depth": P("int", _lgb("max_depth"), "最大深度", min=1, also=[-1]),
        }),
        "hpo_params": G("寻优搜索范围", {
            "min_split_gain": P("hpo", _hpo("min_split_gain"), "最小分割增益范围", item="int", min=0),
            "learning_rate": P("hpo", _hpo("learning_rate"), "学习率范围", item="number", gt=0, max=1),
            "num_leaves": P("hpo", _hpo("num_leaves"), "叶子数范围", item="int", min=2),
            "max_bin": P("hpo", _hpo("max_bin"), "最大分箱数范围", item="int", min=2),
            "min_child_samples": P("hpo", _hpo("min_child_samples"), "叶节点最少样本数范围", item="int", min=1),
            "reg_alpha": P("hpo", _hpo("reg_alpha"), "L1正则范围", item="number", min=0),
            "reg_lambda": P("hpo", _hpo("reg_lambda"), "L2正则范围", item="number", min=0),
            "subsample": P("hpo", _hpo("subsample"), "样本采样比例范围", item="number", gt=0, max=1),
            "colsample_bytree": P("hpo", _hpo("colsample_bytree"), "特征采样比例范围", item="number", gt=0, max=1),
            "subsample_freq": P("hpo", _hpo("subsample_freq"), "子采样频率范围", item="int", min=0),
            "min_child_weight": P("hpo", _hpo("min_child_weight"), "叶节点最小权重和范围", item="number", min=0),
            "n_estimators": P("hpo", _hpo("n_estimators"), "树的数量范围", item="int", min=1),
            "max_depth": P("hpo", _hpo("max_depth"), "最大深度范围", item="int", min=1, also=[-1]),
            "scale_pos_weight": P("hpo", _hpo("scale_pos_weight"), "正负样本权重比范围", item="number", gt=0),
        }),
    },
    "model_eval": {
        "eval_model_type": P("enum", "settings.eval_model_type", "评估模型类型",
                             choices=["LightGBM", "AutoGluon", "LrScorecard"], nullable=True),
        "eval_model_name": P("str", "settings.eval_model_name", "AutoGluon评估模型名", nullable=True),
        "bin_num_lift": P("int", "settings.bin_num", "Lift分箱数", min=2),
        "pdo": P("int", "settings.pdo", "PDO(赔率翻倍分数)", gt=0),
        "theta": P("number", "settings.theta", "每倍赔率分数差(theta)", gt=0),
        "p0": P("int", "settings.p0", "基准分(p0)"),
        "cut_points": P("num_list", "settings.cut_points", "分数切分点"),
        "gen_eval_report": P("bool", "settings.gen_report", "是否生成评估报告"),
        "save_eval_output": P("bool", "settings.save_output", "是否保存评估结果"),
    },
}


def iter_params(stage):
    """依次给出 (参数名, spec)；分组参数展开为 '组.参数'。"""
    for name, spec in SPEC[stage].items():
        if spec["kind"] == "group":
            for sub, sub_spec in spec["params"].items():
                yield "%s.%s" % (name, sub), sub_spec
        else:
            yield name, spec


def get_path(cfg, path):
    cur = cfg
    for key in path:
        cur = cur[key]
    return cur


def set_path(cfg, path, value):
    cur = cfg
    for key in path[:-1]:
        cur = cur[key]
    cur[path[-1]] = value


# ==================== 单个值的检查 ====================

class Bad(Exception):
    """取值不合法；message 说明原因和允许的取值。"""


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _bounds_text(spec):
    parts = []
    if "gt" in spec:
        parts.append("> %s" % spec["gt"])
    if "min" in spec:
        parts.append(">= %s" % spec["min"])
    if "lt" in spec:
        parts.append("< %s" % spec["lt"])
    if "max" in spec:
        parts.append("<= %s" % spec["max"])
    text = "、".join(parts)
    if spec.get("also"):
        text = "%s，或 %s" % (text, " / ".join(str(a) for a in spec["also"])) if text else " / ".join(map(str, spec["also"]))
    return text


def _check_bounds(v, spec, what="取值"):
    if v in spec.get("also", []):
        return
    ok = True
    if "gt" in spec and not v > spec["gt"]:
        ok = False
    if "min" in spec and not v >= spec["min"]:
        ok = False
    if "lt" in spec and not v < spec["lt"]:
        ok = False
    if "max" in spec and not v <= spec["max"]:
        ok = False
    if not ok:
        raise Bad("%s %s 超出范围，应为 %s" % (what, json.dumps(v), _bounds_text(spec)))


def _to_int(v, what="取值"):
    if isinstance(v, bool) or not _is_num(v):
        raise Bad("%s %s 应为整数" % (what, json.dumps(v, ensure_ascii=False)))
    if isinstance(v, float):
        if not v.is_integer():
            raise Bad("%s %s 应为整数" % (what, json.dumps(v)))
        v = int(v)
    return v


def _to_yyyymmdd(v):
    if isinstance(v, str) and v.isdigit() and len(v) == 8:
        v = int(v)
    v = _to_int(v, "日期")
    try:
        datetime.datetime.strptime(str(v), "%Y%m%d")
    except ValueError:
        raise Bad("日期 %s 不是有效的 yyyyMMdd 日期（例如 20240301）" % v)
    return v


def check_value(spec, v):
    """检查并规范化一个取值；不合法时抛出 Bad。"""
    kind = spec["kind"]
    if v is None and spec.get("nullable"):
        return None
    if kind == "bool":
        if not isinstance(v, bool):
            raise Bad("应为 true 或 false，收到 %s" % json.dumps(v, ensure_ascii=False))
        return v
    if kind == "int":
        v = _to_int(v)
        _check_bounds(v, spec)
        return v
    if kind == "number":
        if not _is_num(v):
            raise Bad("应为数字，收到 %s" % json.dumps(v, ensure_ascii=False))
        _check_bounds(v, spec)
        return v
    if kind == "enum":
        if v not in spec["choices"]:
            allowed = " / ".join(spec["choices"]) + (" / null" if spec.get("nullable") else "")
            raise Bad("可选值只有 %s，收到 %s" % (allowed, json.dumps(v, ensure_ascii=False)))
        return v
    if kind == "str":
        if not isinstance(v, str) or not v.strip():
            raise Bad("应为非空字符串%s" % ("或 null" if spec.get("nullable") else ""))
        return v
    if kind == "str_list":
        if not isinstance(v, list) or not all(isinstance(x, str) and x.strip() for x in v):
            raise Bad("应为字符串列表，例如 [\"a\", \"b\"]；空列表 [] 表示不指定")
        bad = [x for x in v if "..." in x or "…" in x]
        if bad:
            raise Bad("列表中出现省略号 %s，请写出完整名称，不能省略" % json.dumps(bad, ensure_ascii=False))
        return v
    if kind == "num_list":
        if not isinstance(v, list) or not all(_is_num(x) for x in v):
            raise Bad("应为数字列表，例如 [300, 400, 500]；空列表 [] 表示按分数最小最大值等距十分箱")
        if any(b <= a for a, b in zip(v, v[1:])):
            raise Bad("切分点需从小到大排列且不重复，收到 %s" % json.dumps(v))
        return v
    if kind == "missing_list":
        if isinstance(v, str):
            try:
                v = ast.literal_eval(v)
            except (ValueError, SyntaxError):
                raise Bad("无法解析为列表，例如 [-9999, \"-9999\"]")
        if not isinstance(v, list) or not v or not all(isinstance(x, (int, float, str)) and not isinstance(x, bool) for x in v):
            raise Bad("应为非空列表，元素为数字或字符串，例如 [-9999, -9998, \"-9999\", \" \"]")
        return v
    if kind == "date_range":
        if not isinstance(v, list) or len(v) != 2:
            raise Bad("应为 [开始日期, 结束日期]，日期写成 yyyyMMdd 整数，例如 [20240301, 20240930]")
        start, end = _to_yyyymmdd(v[0]), _to_yyyymmdd(v[1])
        if start > end:
            raise Bad("开始日期 %s 晚于结束日期 %s" % (start, end))
        return [start, end]
    if kind == "targets":
        if not isinstance(v, list):
            raise Bad("应为列表，例如 [{\"dataset\": \"test\", \"metric\": \"ks\"}]；空列表 [] 表示使用默认目标")
        for item in v:
            if not isinstance(item, dict) or set(item) != {"dataset", "metric"}:
                raise Bad("每一项必须是 {\"dataset\": ..., \"metric\": ...}，收到 %s" % json.dumps(item, ensure_ascii=False))
            if item["metric"] not in ("ks", "lift_tail10", "lift_head10"):
                raise Bad("metric 只能是 ks / lift_tail10 / lift_head10，收到 %s" % json.dumps(item["metric"], ensure_ascii=False))
            if not isinstance(item["dataset"], str) or not item["dataset"]:
                raise Bad("dataset 应为 test 或 OOT 名称（如 oot_main）")
        return v
    if kind == "dict":
        if not isinstance(v, dict):
            raise Bad("应为 JSON 对象，例如 {}")
        return v
    if kind == "score_scale":
        allowed = {"base_score", "pdo", "base_odds"}
        if not isinstance(v, dict) or not v or not set(v) <= allowed:
            raise Bad("应为包含 base_score、pdo、base_odds 的对象，例如 {\"base_score\": 500, \"pdo\": 30, \"base_odds\": 20}")
        for k, x in v.items():
            if not _is_num(x):
                raise Bad("score_scale.%s 应为数字" % k)
            if k in ("pdo", "base_odds") and not x > 0:
                raise Bad("score_scale.%s 应 > 0" % k)
        return v
    if kind == "hpo":
        if not isinstance(v, list) or not 1 <= len(v) <= 3:
            raise Bad("应为 [下界, 上界] 或 [下界, 上界, 步长]，单个固定值写成 [值]")
        out = []
        for i, x in enumerate(v):
            if spec["item"] == "int":
                x = _to_int(x)
            elif not _is_num(x):
                raise Bad("应为数字列表，收到 %s" % json.dumps(v, ensure_ascii=False))
            if i < 2:
                _check_bounds(x, spec, "边界")
            elif not x > 0:
                raise Bad("步长应 > 0，收到 %s" % x)
            out.append(x)
        if len(out) >= 2 and out[0] > out[1] and out[0] not in spec.get("also", []):
            raise Bad("下界 %s 大于上界 %s" % (out[0], out[1]))
        return out
    raise Bad("未知类型 %s" % kind)


# ==================== 应用改动 ====================

def _unknown(stage, name, group=None):
    if group:
        names = list(SPEC[stage][group]["params"])
        return "%s 不是 %s 中的参数；可改的有：%s" % (name, group, "、".join(names))
    names = list(SPEC[stage])
    return "%s 不是「%s」的可改参数；可改的有：%s" % (name, STAGE_NAMES[stage], "、".join(names))


def apply_changes(cfg, stage, changes):
    """把用户侧参数改动写入配置副本。返回 (新配置, 错误列表)。只做格式和取值检查。"""
    new = copy.deepcopy(cfg)
    errors = []

    def err(param, msg):
        errors.append({"stage": stage, "param": param, "error": msg})

    if not isinstance(changes, dict):
        err("", "参数应为 JSON 对象，例如 {\"iv_thresh\": 0.03}")
        return new, errors
    for name, value in changes.items():
        spec = SPEC[stage].get(name)
        if spec is None:
            err(name, _unknown(stage, name))
            continue
        if spec["kind"] == "group":
            if not isinstance(value, dict):
                err(name, "应为 JSON 对象，只写要改的子参数，例如 {\"%s\": 值}" % next(iter(spec["params"])))
                continue
            for sub, sub_value in value.items():
                sub_spec = spec["params"].get(sub)
                if sub_spec is None:
                    err("%s.%s" % (name, sub), _unknown(stage, sub, name))
                    continue
                _apply_one(new, sub_spec, sub_value, "%s.%s" % (name, sub), err)
        else:
            _apply_one(new, spec, value, name, err)
    # 结果文件名固定为 results/<环节>.json，启动标记放在 logs/ 下
    new["output_path"][RESULT_KEYS[stage]] = "results/%s.json" % stage
    new["output_path"][STARTUP_KEYS[stage]] = "logs/%s_startup.txt" % stage
    return new, errors


def _apply_one(cfg, spec, value, param, err):
    try:
        value = check_value(spec, value)
    except Bad as e:
        err(param, "%s（%s）：%s" % (param, spec["label"], e))
        return
    if spec["kind"] == "score_scale":
        merged = dict(get_path(cfg, spec["path"]) or {})
        merged.update(value)
        value = merged
    set_path(cfg, spec["path"], value)


# ==================== 参数之间、参数与数据之间的检查 ====================

def _data_date_range(workdir):
    """从数据质量分析结果中取样本时间范围，返回 (开始, 结束) 的 yyyyMMdd 整数；没有则返回 None。"""
    path = os.path.join(workdir, "results", "data_analysis.json")
    if not os.path.exists(path):
        return None

    def find(node):
        if isinstance(node, dict):
            if "样本时间范围" in node and isinstance(node["样本时间范围"], dict):
                return node["样本时间范围"]
            for v in node.values():
                r = find(v)
                if r:
                    return r
        elif isinstance(node, list):
            for v in node:
                r = find(v)
                if r:
                    return r
        return None

    try:
        rng = find(load_json(path))
        if rng:
            return (int(rng["开始日期"].replace("-", "")), int(rng["结束日期"].replace("-", "")))
    except (ValueError, KeyError, AttributeError):
        pass
    return None


def cross_check(stage, old, new, changes, template, workdir):
    """检查参数之间的关系，并在有工作目录时对照数据。返回 (错误, 提示, 说明)。"""
    errors, warnings, notes = [], [], []
    s = new["settings"]

    def err(param, msg):
        errors.append({"stage": stage, "param": param, "error": msg})

    if stage == "data_clean_split":
        range_keys = ("train_split_range", "oot_split_range", "psi_split_range")
        given_ranges = [k for k in range_keys if k in changes]
        if given_ranges and s["method"] != "date_range":
            if "split_method" in changes:
                err("split_method", "提供了时间范围（%s）时，split_method 必须为 date_range" % "、".join(given_ranges))
            else:
                s["method"] = "date_range"
                notes.append("提供了时间范围，切分方式按规则改为 date_range（按指定时间范围切分）")
        if "oot_split_range" in changes and "psi_split_range" not in changes:
            s["psi_split_range"] = list(s["oot_split_range"])
            notes.append("未单独指定 PSI 数据集时间范围，按规则与测试集(OOT)时间范围相同")
        if s["method"] == "date_range":
            t = template["settings"]
            for key, label in (("train_split_range", "训练集时间范围"), ("oot_split_range", "测试集(OOT)时间范围")):
                if key not in changes and s[key] == t[key] and old["settings"]["method"] != "date_range":
                    err(key, "按时间范围切分（date_range）时必须指定 %s（%s），当前还是默认示例值 %s"
                        % (key, label, json.dumps(s[key])))
            if workdir:
                rng = _data_date_range(workdir)
                if rng is None:
                    warnings.append("还没有数据质量分析结果，无法核对时间范围是否落在数据日期内")
                else:
                    lo, hi = rng
                    for key in range_keys:
                        a, b = s[key]
                        if b < lo or a > hi:
                            err(key, "%s %s 与数据日期范围 [%s, %s] 没有交集" % (key, json.dumps([a, b]), lo, hi))
                        elif a < lo or b > hi:
                            warnings.append("%s %s 部分超出数据日期范围 [%s, %s]" % (key, json.dumps([a, b]), lo, hi))

    if stage == "feature_select" and workdir and s["upload_fea_list"] and "user_feature_list" in changes:
        feats = s["upload_fea_list"]
        base_used = [f for f in feats if f in BASE_COLUMNS]
        if base_used:
            err("user_feature_list", "%s 是主键/日期/标签/渠道列，不能作为入模特征" % "、".join(base_used))
        data_file = find_data_file(os.path.join(workdir, "data"), "data")
        cols = read_columns(data_file) if data_file else None
        if cols is None:
            warnings.append("无法读取数据列名，未核对 user_feature_list 中的特征是否存在")
        else:
            missing = [f for f in feats if f not in cols]
            if missing:
                err("user_feature_list", "数据中不存在这些特征（请核对名称，区分大小写）：%s"
                    % "、".join(missing[:30]) + (" 等 %d 个" % len(missing) if len(missing) > 30 else ""))
            dup = sorted({f for f in feats if feats.count(f) > 1})
            if dup:
                warnings.append("user_feature_list 中有重复特征：%s" % "、".join(dup))

    if stage == "model_train":
        targets = s["optimization_targets"]
        if targets:
            n = len(targets)
            if s["optimization_mode"] == "single" and n != 1:
                err("optimization_targets", "single 模式只能有 1 个寻优目标，当前 %d 个" % n)
            if s["optimization_mode"] == "multi" and not 2 <= n <= 3:
                err("optimization_targets", "multi 模式需要 2～3 个寻优目标，当前 %d 个" % n)
            if workdir and "optimization_targets" in changes:
                available = ["test"] + list_oot_names(workdir)
                for t in targets:
                    if t["dataset"] not in available:
                        err("optimization_targets", "数据集 %s 不存在；可用的有：%s（OOT 来自 data/ 下的 oot_*.csv 等文件）"
                            % (t["dataset"], "、".join(available)))
        mp = s["model_param"]
        if mp["max_depth"] > 0 and mp["num_leaves"] > 2 ** mp["max_depth"]:
            err("lightgbm_params.num_leaves", "num_leaves=%d 超过 2^max_depth=%d（max_depth=%d），请调小 num_leaves 或调大 max_depth"
                % (mp["num_leaves"], 2 ** mp["max_depth"], mp["max_depth"]))

    if stage == "model_eval":
        if s["eval_model_name"] is not None and s["eval_model_type"] != "AutoGluon":
            err("eval_model_name", "只有 eval_model_type 为 AutoGluon 时才能指定 eval_model_name")

    return errors, warnings, notes


# ==================== 改动列表与当前参数 ====================

def user_values(stage, cfg):
    """按用户侧参数名读出配置中的当前值。"""
    out = {}
    for name, spec in SPEC[stage].items():
        if spec["kind"] == "group":
            out[name] = {sub: get_path(cfg, s["path"]) for sub, s in spec["params"].items()}
        else:
            out[name] = get_path(cfg, spec["path"])
    return out


def diff(stage, old, new):
    changes = []
    for name, spec in iter_params(stage):
        a, b = get_path(old, spec["path"]), get_path(new, spec["path"])
        if a != b:
            changes.append({"param": name, "label": spec["label"], "from": a, "to": b})
    return changes


def load_current(workdir, stage):
    path = config_path(workdir, stage) if workdir else None
    if path and os.path.exists(path):
        return load_json(path), True
    return load_json(template_path(stage)), False


def prepare(workdir, all_changes):
    """检查全部环节的改动。返回 (每个环节的结果, 错误, 提示)。"""
    results, errors, warnings = [], [], []
    if not isinstance(all_changes, dict) or not all_changes:
        return results, [{"stage": "", "param": "", "error": "--set 应为 {环节: {参数: 值}} 形式的 JSON，例如 {\"feature_select\": {\"iv_thresh\": 0.03}}"}], warnings
    for stage in all_changes:
        if stage not in SPEC:
            errors.append({"stage": stage, "param": "", "error": "未知环节 %s；可选：%s" % (stage, "、".join(STAGE_IDS))})
    for stage in STAGE_IDS:
        if stage not in all_changes or stage not in SPEC:
            continue
        old, _ = load_current(workdir, stage)
        new, errs = apply_changes(old, stage, all_changes[stage])
        notes = []
        if not errs:
            errs, warns, notes = cross_check(stage, old, new, all_changes[stage], load_json(template_path(stage)), workdir)
            warnings.extend(warns)
        errors.extend(errs)
        results.append({"stage": stage, "name": STAGE_NAMES[stage], "changes": diff(stage, old, new),
                        "notes": notes, "_config": new})
    return results, errors, warnings


def main():
    ap = argparse.ArgumentParser(description="把确认后的参数写成环节配置文件", formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("-w", "--workdir", help="工作目录（默认当前目录下的 riskmodel_work）")
    ap.add_argument("--set", dest="set_json", help="{环节: {参数: 值}} 形式的 JSON")
    ap.add_argument("--set-file", help="从文件读取 --set 的 JSON")
    ap.add_argument("--check", action="store_true", help="只检查，不写文件")
    ap.add_argument("--show", metavar="STAGE", help="查看某个环节的当前参数")
    args = ap.parse_args()

    workdir = resolve_workdir(args.workdir)
    if workdir is None or not os.path.isdir(os.path.join(workdir, "config")):
        emit({"ok": False, "error": "找不到工作目录%s，请先用 init_workspace.py 创建" % ("（%s）" % args.workdir if args.workdir else "")})

    if args.show:
        if args.show not in SPEC:
            emit({"ok": False, "error": "未知环节 %s；可选：%s" % (args.show, "、".join(STAGE_IDS))})
        cfg, exists = load_current(workdir, args.show)
        emit({"ok": True, "stage": args.show, "name": STAGE_NAMES[args.show],
              "source": "工作目录配置" if exists else "默认配置（尚未修改过）",
              "params": user_values(args.show, cfg)})

    try:
        if args.set_file:
            with open(args.set_file, "r", encoding="utf-8") as f:
                all_changes = json.load(f)
        elif args.set_json is not None:
            all_changes = json.loads(args.set_json)
        else:
            emit({"ok": False, "error": "需要 --set、--set-file 或 --show 之一"}, 2)
    except ValueError as e:
        emit({"ok": False, "error": "JSON 解析失败：%s。引号不好处理时可改用 --set-file" % e})

    results, errors, warnings = prepare(workdir, all_changes)
    out = {"ok": not errors, "mode": "check" if args.check else "write", "workdir": workdir,
           "stages": [{k: v for k, v in r.items() if k != "_config"} for r in results]}
    if errors:
        out["errors"] = errors
        out["message"] = "参数有误，未写入任何配置。请按 errors 修改后重新检查。"
    if warnings:
        out["warnings"] = warnings
    if not errors and not args.check:
        written = []
        for r in results:
            save_json(config_path(workdir, r["stage"]), r["_config"])
            written.append("config/%s_config.json" % r["stage"])
        out["written"] = written
        out["message"] = "配置已写入"
    elif not errors:
        out["message"] = "检查通过（未写入）。用户确认后去掉 --check 再执行一次即可写入。"
    emit(out)


if __name__ == "__main__":
    main()
