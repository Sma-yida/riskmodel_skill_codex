"""WOE-LR 信用评分卡训练、导出和预测封装。

该模块只由 ``model_select=LrScorecard`` 的训练分支惰性导入，避免评分卡依赖影响
LightGBM 和 AutoGluon 的既有执行路径。
"""

from __future__ import annotations

import json
import math
import os
import pickle
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import roc_auc_score

from scorecardpipeline import (
    Combiner,
    FeatureSelection,
    ITLubberLogisticRegression,
    ScoreCard,
    StepwiseSelection,
    WOETransformer,
)
from utils.data_utils import get_data
from .utils1 import decile_analysis, ks_score
from .utils_model_train_scorecard_report import (
    build_scorecard_feature_summary,
    generate_scorecard_training_report,
    parse_report_dates,
    resolve_report_context_columns,
)


SCORECARD_PARAMETER_PATH = Path(__file__).with_name("lr_scorecard_params.json")


def _load_scorecard_parameters(path: Path = SCORECARD_PARAMETER_PATH) -> Dict[str, Any]:
    """Load the code-shipped LrScorecard defaults."""
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


SCORECARD_DEFAULTS = _load_scorecard_parameters()
SUPPORTED_BINNING_METHODS = frozenset({"chi", "dt", "quantile", "step"})
NUMERIC_MISSING_VALUES: Tuple[Any, ...] = tuple(SCORECARD_DEFAULTS["missing_values"]["numeric"])
CATEGORICAL_MISSING_VALUES: Tuple[str, ...] = tuple(
    str(value) for value in SCORECARD_DEFAULTS["missing_values"]["categorical"]
)
PUBLIC_SCORECARD_PARAMETER_KEYS = {
    "binning_method",
    "adj_rules",
    "fine_selection_corr",
    "stepwise_criterion",
    "score_scale",
}


def _resolve_scorecard_parameters(
    config: Mapping[str, Any],
    uploaded_model_params: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Resolve one training run's parameters without mutating module defaults."""
    parameters = deepcopy(SCORECARD_DEFAULTS)
    configured = config.get("settings", {}).get("lr_scorecard_params", {})
    if configured is None:
        configured = {}
    if not isinstance(configured, Mapping):
        raise ValueError("settings.lr_scorecard_params必须是JSON对象")

    uploaded = uploaded_model_params or {}
    if not isinstance(uploaded, Mapping):
        raise ValueError("data/model_train_spec_lrscorecard.json中的model_params必须是JSON对象")

    def apply(values: Mapping[str, Any], source: str) -> None:
        unknown = sorted(set(values) - PUBLIC_SCORECARD_PARAMETER_KEYS)
        if unknown:
            raise ValueError(f"{source}包含不支持的LrScorecard参数: {unknown}")
        if "binning_method" in values:
            method = values["binning_method"]
            if not isinstance(method, str) or method.strip().lower() not in SUPPORTED_BINNING_METHODS:
                supported = ", ".join(sorted(SUPPORTED_BINNING_METHODS))
                raise ValueError(f"{source}.binning_method只能为: {supported}")
            parameters["binning"]["method"] = method.strip().lower()
        if "adj_rules" in values:
            rules = values["adj_rules"]
            if not isinstance(rules, Mapping):
                raise ValueError(f"{source}.adj_rules必须是JSON对象")
            # Uploaded rules replace the configured dictionary as a whole. This makes
            # Deleting a rule from the LR-specific spec returns that feature to auto binning.
            parameters["binning"]["adj_rules"] = deepcopy(dict(rules))
        if "fine_selection_corr" in values:
            corr = values["fine_selection_corr"]
            if isinstance(corr, bool) or not isinstance(corr, (int, float)) or not 0 < corr <= 1:
                raise ValueError(f"{source}.fine_selection_corr必须是(0, 1]范围内的数值")
            parameters["fine_selection"]["corr"] = float(corr)
        if "stepwise_criterion" in values:
            criterion = values["stepwise_criterion"]
            if not isinstance(criterion, str) or criterion.strip().lower() not in {"aic", "bic"}:
                raise ValueError(f"{source}.stepwise_criterion只能为aic或bic")
            parameters["stepwise"]["criterion"] = criterion.strip().lower()
        if "score_scale" in values:
            scale = values["score_scale"]
            if not isinstance(scale, Mapping):
                raise ValueError(f"{source}.score_scale必须是JSON对象")
            unknown_scale = sorted(set(scale) - {"base_score", "pdo", "base_odds"})
            if unknown_scale:
                raise ValueError(f"{source}.score_scale包含不支持的参数: {unknown_scale}")
            for key in ("base_score", "pdo", "base_odds"):
                if key not in scale:
                    continue
                value = scale[key]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"{source}.score_scale.{key}必须是数值")
                number = float(value)
                if not math.isfinite(number):
                    raise ValueError(f"{source}.score_scale.{key}必须是有限数值")
                if key in {"pdo", "base_odds"} and number <= 0:
                    raise ValueError(f"{source}.score_scale.{key}必须大于0")
                parameters["score_scale"][key] = number
    apply(configured, "settings.lr_scorecard_params")
    apply(uploaded, "data/model_train_spec_lrscorecard.json中的model_params")
    return parameters


def _resolve_binning_parameters(binning: Mapping[str, Any]) -> Dict[str, Any]:
    """Flatten common and selected-method parameters for ``Combiner``."""
    if not isinstance(binning, Mapping):
        raise ValueError("binning配置必须是JSON对象")

    method = binning.get("method", "chi")
    if not isinstance(method, str) or method.strip().lower() not in SUPPORTED_BINNING_METHODS:
        supported = ", ".join(sorted(SUPPORTED_BINNING_METHODS))
        raise ValueError(f"binning.method只能为: {supported}")
    method = method.strip().lower()

    method_params = binning.get("method_params", {})
    if not isinstance(method_params, Mapping):
        raise ValueError("binning.method_params必须是JSON对象")
    selected = method_params.get(method, {})
    if not isinstance(selected, Mapping):
        raise ValueError(f"binning.method_params.{method}必须是JSON对象")
    duplicated_common = sorted(set(selected) & {"method", "adj_rules", "method_params"})
    if duplicated_common:
        raise ValueError(
            f"binning.method_params.{method}不能重复配置公共参数: {duplicated_common}"
        )
    if method != "chi" and "constraint_mode" in selected:
        raise ValueError(f"constraint_mode仅支持chi分箱，不能配置在{method}中")

    resolved = {
        key: deepcopy(value)
        for key, value in binning.items()
        if key != "method_params"
    }
    resolved["method"] = method
    resolved.update(deepcopy(dict(selected)))
    return resolved


def _json_value(value: Any) -> Any:
    """Convert numpy/pandas values into strict JSON values."""
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Index)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if math.isnan(number):
            return None
        if math.isinf(number):
            return "positive_infinity" if number > 0 else "negative_infinity"
        return number
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is pd.NA or value is None:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(_json_value(payload), stream, ensure_ascii=False, indent=2, allow_nan=False)


def _write_binning_review_csv(review: pd.DataFrame, path: Path) -> None:
    """Write user-requested ratio columns with an explicit four-decimal display."""
    exported = review.copy()
    for column in (
        "最小常规箱占比",
        "缺失箱占比",
        "缺失箱坏率",
        "主验证集未见类别占比",
    ):
        if column in exported.columns:
            exported[column] = exported[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.4f}"
            )
    exported.to_csv(path, index=False, encoding="utf-8-sig")


def _validate_binary_target(target: pd.Series, label_col: str) -> pd.Series:
    if target.isna().any():
        raise ValueError(f"标签字段 {label_col} 存在缺失值")
    try:
        normalized = pd.to_numeric(target, errors="raise").astype(int)
    except Exception as exc:
        raise ValueError(f"标签字段 {label_col} 必须只包含0和1") from exc
    values = set(normalized.unique().tolist())
    if not values or not values.issubset({0, 1}) or values != {0, 1}:
        raise ValueError(f"标签字段 {label_col} 必须同时包含0和1，当前取值为: {sorted(values)}")
    normalized.name = label_col
    return normalized


def normalize_scorecard_features(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> pd.DataFrame:
    """Apply the training-time schema and missing-value policy to raw features."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("评分卡模型输入必须是 pandas.DataFrame")
    missing = [feature for feature in feature_names if feature not in frame.columns]
    if missing:
        raise ValueError(f"评分卡模型输入缺少特征: {missing}")

    result = frame.loc[:, list(feature_names)].copy()
    numeric_set = set(numeric_features)
    categorical_set = set(categorical_features)
    for feature in feature_names:
        if feature in numeric_set:
            value = result[feature].replace(list(NUMERIC_MISSING_VALUES), np.nan)
            if not is_numeric_dtype(value.dtype):
                text = value.astype("string").str.strip()
                value = value.mask(text.isin(CATEGORICAL_MISSING_VALUES), np.nan)
                converted = pd.to_numeric(value, errors="coerce")
                invalid = value.notna() & converted.isna()
                if invalid.any():
                    examples = value[invalid].astype(str).drop_duplicates().head(5).tolist()
                    raise ValueError(f"数值特征 {feature} 包含无法转换的值: {examples}")
                value = converted
            result[feature] = value.astype(float)
        elif feature in categorical_set:
            value = result[feature].astype(object).copy()
            missing_mask = value.isna()
            for missing_value in NUMERIC_MISSING_VALUES:
                missing_mask |= value.eq(missing_value).fillna(False)
            text = value.astype("string").str.strip()
            missing_mask |= text.isin(CATEGORICAL_MISSING_VALUES).fillna(False)
            value.loc[missing_mask] = np.nan
            result[feature] = value
        else:
            raise ValueError(f"特征 {feature} 未记录训练期类型")
    return result


class LRScorecardModel:
    """Complete raw-data WOE-LR scorecard model saved in the PKL artifact."""

    def __init__(
        self,
        candidate_features: Sequence[str],
        final_features: Sequence[str],
        numeric_features: Sequence[str],
        categorical_features: Sequence[str],
        combiner: Combiner,
        woe_transformer: WOETransformer,
        fine_selector: FeatureSelection,
        stepwise_selector: StepwiseSelection,
        logistic_model: ITLubberLogisticRegression,
        scorecard: ScoreCard,
        parameters: Optional[Mapping[str, Any]] = None,
        classification_threshold: float = 0.5,
        unseen_category_policy: str = "min",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.candidate_features_ = list(candidate_features)
        self.feature_names_ = list(final_features)
        self.candidate_numeric_features_ = list(numeric_features)
        self.candidate_categorical_features_ = list(categorical_features)
        self.numeric_features_ = [
            feature for feature in self.feature_names_ if feature in self.candidate_numeric_features_
        ]
        self.categorical_features_ = [
            feature for feature in self.feature_names_ if feature in self.candidate_categorical_features_
        ]
        self.combiner = combiner
        self.woe_transformer = woe_transformer
        self.fine_selector = fine_selector
        self.stepwise_selector = stepwise_selector
        self.logistic_model = logistic_model
        self.scorecard = scorecard
        self.parameters_ = deepcopy(dict(parameters or SCORECARD_DEFAULTS))
        self.classification_threshold = float(classification_threshold)
        self.unseen_category_policy = unseen_category_policy
        self.metadata = metadata or {}
        self.missing_values = {
            "numeric": list(NUMERIC_MISSING_VALUES),
            "categorical": list(CATEGORICAL_MISSING_VALUES),
        }

    def _normalize(self, raw_dataframe: pd.DataFrame) -> pd.DataFrame:
        return normalize_scorecard_features(
            raw_dataframe,
            self.feature_names_,
            self.numeric_features_,
            self.categorical_features_,
        )

    def predict_proba(self, raw_dataframe: pd.DataFrame) -> np.ndarray:
        normalized = self._normalize(raw_dataframe)
        # Keep probability and score consistent with the configured unseen-category
        # policy. ``toad.ScoreCard.predict_proba`` always uses its own "min" default.
        score = np.asarray(
            self.scorecard.predict(normalized, default=self.unseen_category_policy),
            dtype=float,
        )
        bad_probability = np.asarray(self.scorecard.score_to_proba(score), dtype=float)
        return np.column_stack((1.0 - bad_probability, bad_probability))

    def predict_score(self, raw_dataframe: pd.DataFrame) -> np.ndarray:
        normalized = self._normalize(raw_dataframe)
        return np.asarray(
            self.scorecard.predict(normalized, default=self.unseen_category_policy),
            dtype=float,
        )

    def predict_score_components(self, raw_dataframe: pd.DataFrame) -> pd.DataFrame:
        normalized = self._normalize(raw_dataframe)
        _, components = self.scorecard.predict(
            normalized,
            return_sub=True,
            default=self.unseen_category_policy,
        )
        return components.astype(float)

    def predict(self, raw_dataframe: pd.DataFrame) -> np.ndarray:
        bad_probability = self.predict_proba(raw_dataframe)[:, 1]
        return (bad_probability >= self.classification_threshold).astype(int)


def _feature_map(feature_dic_path: Optional[str]) -> Dict[str, str]:
    if not feature_dic_path or not os.path.exists(feature_dic_path):
        return {}
    frame = None
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            frame = pd.read_csv(feature_dic_path, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
        except Exception as exc:
            print(f"评分卡特征字典读取失败，变量含义将留空: {exc}")
            return {}
    if frame is None or "Feature" not in frame.columns:
        return {}
    description_col = "feature_cn" if "feature_cn" in frame.columns else None
    if description_col is None:
        return {}
    return {
        str(row["Feature"]): "" if pd.isna(row[description_col]) else str(row[description_col])
        for _, row in frame.iterrows()
    }


def _is_missing_bin(raw_bin: Any, label: str) -> bool:
    if label == "缺失值":
        return True
    values = raw_bin if isinstance(raw_bin, (list, tuple, set, np.ndarray)) else [raw_bin]
    return any(value is None or (isinstance(value, float) and math.isnan(value)) for value in values)


def _bin_labels(rule: Sequence[Any]) -> List[str]:
    bins = np.empty(len(rule), dtype=object)
    bins[:] = list(rule)
    return [str(value) for value in ScoreCard.format_bins(bins, decimal=12)]


def build_scorecard_points(model: LRScorecardModel, feature_map: Mapping[str, str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    card = model.scorecard
    intercept = float(card.intercept_)
    factor = float(card.factor)
    offset = float(card.offset)
    final_feature_count = int(card.n_features_)
    base_allocation = (offset - factor * intercept) / final_feature_count
    for feature in model.scorecard.features_:
        rule = model.scorecard.rules[feature]
        labels = _bin_labels(list(rule["bins"]))
        scores = np.asarray(rule["scores"], dtype=float)
        woes = np.asarray(rule["woes"], dtype=float)
        if not (len(labels) == len(scores) == len(woes)):
            raise ValueError(f"评分卡规则长度不一致: {feature}")
        feature_type = "categorical" if feature in model.categorical_features_ else "numeric"
        for index, (label, woe, score) in enumerate(zip(labels, woes, scores)):
            calculated_score = base_allocation - factor * float(rule["weight"]) * float(woe)
            if not np.isclose(calculated_score, score, atol=1e-9, rtol=1e-10):
                raise RuntimeError(
                    f"评分卡箱分公式校验失败: {feature}/{label}: {calculated_score} != {score}"
                )
            rows.append(
                {
                    "变量名称": feature,
                    "变量含义": feature_map.get(feature, ""),
                    "变量类型": feature_type,
                    "分箱序号": index,
                    "变量分箱": label,
                    "WOE": float(woe),
                    "LR系数": float(rule["weight"]),
                    # 策略交付表只展示整数箱分；PKL仍保存精确浮点分数。
                    "对应分数": int(np.rint(score)),
                }
            )
    return pd.DataFrame(rows)


def _population_psi(expected: pd.Series, actual: pd.Series) -> float:
    bins = sorted(set(expected.index.tolist()) | set(actual.index.tolist()), key=str)
    expected_values = expected.reindex(bins, fill_value=0.0).to_numpy(dtype=float)
    actual_values = actual.reindex(bins, fill_value=0.0).to_numpy(dtype=float)
    epsilon = 1e-6
    expected_values = np.clip(expected_values, epsilon, None)
    actual_values = np.clip(actual_values, epsilon, None)
    return float(np.sum((actual_values - expected_values) * np.log(actual_values / expected_values)))


def _unseen_category_rate(
    feature: str,
    frame: pd.DataFrame,
    raw_rule: Sequence[Any],
) -> float:
    seen = set()
    for group in raw_rule:
        values = group if isinstance(group, (list, tuple, set, np.ndarray)) else [group]
        for value in values:
            if value is not None and not (isinstance(value, float) and math.isnan(value)):
                seen.add(value)
    values = frame[feature]
    nonmissing = values.notna()
    if not nonmissing.any():
        return 0.0
    return float((~values[nonmissing].isin(seen)).mean())


def _bad_rate_shape(bad_rates: Sequence[float], tolerance: float = 1e-12) -> str:
    values = np.asarray(list(bad_rates), dtype=float)
    values = values[np.isfinite(values)]
    # Missing bins are removed by the caller. One regular bin has no direction;
    # two regular bins can already be classified as increasing/decreasing/flat.
    if len(values) < 2:
        return "箱数不足"
    differences = np.diff(values)
    signs = np.sign(differences[np.abs(differences) > tolerance])
    if len(signs) == 0:
        return "坏率持平"
    if np.all(signs > 0):
        return "单调递增"
    if np.all(signs < 0):
        return "单调递减"
    changes = np.flatnonzero(signs[1:] != signs[:-1])
    if len(changes) == 1 and signs[0] < 0 < signs[-1]:
        return "U形"
    if len(changes) == 1 and signs[0] > 0 > signs[-1]:
        return "倒U形"
    return "多次转折"


def _selector_drop_map(selector: Any) -> Dict[str, str]:
    dropped = getattr(selector, "dropped", None)
    if not isinstance(dropped, pd.DataFrame) or dropped.empty:
        return {}
    if not {"variable", "rm_reason"}.issubset(dropped.columns):
        return {}
    return {
        str(row["variable"]): str(row["rm_reason"])
        for _, row in dropped.iterrows()
    }


def _training_bin_statistics(
    bin_values: pd.Series,
    target: pd.Series,
    labels: Sequence[str],
    woe_mapping: Mapping[Any, Any],
) -> pd.DataFrame:
    grouped = pd.DataFrame({"bin": bin_values.to_numpy(), "target": target.to_numpy()})
    stats = grouped.groupby("bin", dropna=False)["target"].agg(["count", "sum"])
    all_indexes = sorted(set(range(len(labels))) | set(stats.index.tolist()), key=lambda value: (value == -1, value))
    stats = stats.reindex(all_indexes, fill_value=0)
    stats["bad"] = stats["sum"].astype(int)
    stats["good"] = (stats["count"] - stats["sum"]).astype(int)
    total = float(stats["count"].sum())
    stats["share"] = stats["count"] / total if total else 0.0
    stats["bad_rate"] = np.where(stats["count"] > 0, stats["bad"] / stats["count"], np.nan)
    total_bad = float(stats["bad"].sum())
    total_good = float(stats["good"].sum())
    stats["bad_distribution"] = stats["bad"] / total_bad if total_bad else 0.0
    stats["good_distribution"] = stats["good"] / total_good if total_good else 0.0
    stats["woe"] = [float(woe_mapping.get(index, np.nan)) for index in stats.index]
    stats["iv_contribution"] = (
        (stats["bad_distribution"] - stats["good_distribution"]) * stats["woe"]
    )
    stats["label"] = [labels[index] if isinstance(index, (int, np.integer)) and 0 <= index < len(labels) else "未见类别/默认箱" for index in stats.index]
    display_order = sorted(
        stats.index,
        key=lambda index: (
            2 if stats.loc[index, "label"] == "缺失值" else (1 if index == -1 else 0),
            index,
        ),
    )
    return stats.loc[display_order]


def build_binning_review(
    model: LRScorecardModel,
    train_features: pd.DataFrame,
    train_target: pd.Series,
    validation_features: Mapping[str, pd.DataFrame],
    primary_validation: str,
    feature_map: Mapping[str, str],
) -> pd.DataFrame:
    train_bins = model.combiner.transform(train_features)
    primary_bins = model.combiner.transform(validation_features[primary_validation])
    raw_rules = model.combiner.export()
    woe_rules = model.woe_transformer.transformer.export()
    fine_drops = _selector_drop_map(model.fine_selector)
    stepwise_drops = _selector_drop_map(model.stepwise_selector)
    final_features = set(model.feature_names_)
    lr_summary = model.logistic_model.summary()
    rows: List[Dict[str, Any]] = []

    for feature in model.candidate_features_:
        labels = _bin_labels(list(raw_rules[feature]))
        stats = _training_bin_statistics(
            train_bins[feature], train_target, labels, woe_rules[feature]
        )
        missing_indexes = {index for index, label in enumerate(labels) if label == "缺失值"}
        observed = stats.loc[stats["count"] > 0]
        regular = observed.loc[[index for index in observed.index if index not in missing_indexes and index != -1]]
        shape = _bad_rate_shape(regular["bad_rate"].tolist())
        min_regular_share = float(regular["share"].min()) if not regular.empty else float("nan")
        pure_good = bool((regular["bad"] == 0).any()) if not regular.empty else False
        pure_bad = bool((regular["good"] == 0).any()) if not regular.empty else False
        max_abs_woe = float(stats["woe"].abs().max()) if stats["woe"].notna().any() else float("nan")
        training_iv = float(stats["iv_contribution"].sum(skipna=True))

        missing_stats = stats.loc[[index for index in stats.index if index in missing_indexes]]
        missing_share = float(missing_stats["share"].sum()) if not missing_stats.empty else 0.0
        missing_bad = float(missing_stats["bad"].sum()) if not missing_stats.empty else 0.0
        missing_count = float(missing_stats["count"].sum()) if not missing_stats.empty else 0.0
        missing_bad_rate = missing_bad / missing_count if missing_count else float("nan")

        train_distribution = train_bins[feature].value_counts(normalize=True, dropna=False)
        primary_distribution = primary_bins[feature].value_counts(normalize=True, dropna=False)
        primary_psi = _population_psi(train_distribution, primary_distribution)
        unseen_rate = 0.0
        if feature in model.candidate_categorical_features_:
            unseen_rate = _unseen_category_rate(
                feature, validation_features[primary_validation], raw_rules[feature]
            )

        fine_reason = fine_drops.get(feature)
        fine_reason_labels = {
            "iv": f"删除：IV<{model.parameters_['fine_selection']['iv']}",
            "empty": f"删除：缺失率>{model.parameters_['fine_selection']['empty']}",
            "corr": f"删除：相关性>{model.parameters_['fine_selection']['corr']}时被淘汰",
            "identical": f"删除：单一值占比>{model.parameters_['fine_selection']['identical']}",
        }
        fine_result = (
            "保留"
            if fine_reason is None
            else fine_reason_labels.get(fine_reason, f"删除：{fine_reason}")
        )
        if fine_result != "保留":
            final_result = f"未入模：{fine_result.removeprefix('删除：')}"
        elif feature in stepwise_drops:
            final_result = "未入模：双向逐步回归剔除"
        else:
            final_result = "入模"

        coefficient = float(model.logistic_model.coef_[0][list(model.logistic_model.feature_names_in_).index(feature)]) if feature in final_features else float("nan")
        p_value = float(lr_summary.loc[feature, "P>|z|"]) if feature in final_features and feature in lr_summary.index else float("nan")

        rows.append(
            {
                "变量名称": feature,
                "变量含义": feature_map.get(feature, ""),
                "变量类型": "categorical" if feature in model.candidate_categorical_features_ else "numeric",
                "总箱数": int(len(observed)),
                "常规箱数": int(len(regular)),
                "最小常规箱占比": round(min_regular_share, 4) if np.isfinite(min_regular_share) else float("nan"),
                "训练IV": training_iv,
                "坏率形状": shape,
                "常规箱是否存在纯好箱": pure_good,
                "常规箱是否存在纯坏箱": pure_bad,
                "最大绝对WOE": max_abs_woe,
                "缺失箱占比": round(missing_share, 4),
                "缺失箱坏率": round(missing_bad_rate, 4) if np.isfinite(missing_bad_rate) else float("nan"),
                "主验证集名称": primary_validation,
                "主验证集分箱PSI": primary_psi,
                "主验证集未见类别占比": round(unseen_rate, 4),
                "最终是否入模": final_result,
                "LR系数": coefficient,
                "LR p值": p_value,
            }
        )
    return pd.DataFrame(rows)


def build_binning_detail(
    model: LRScorecardModel,
    datasets: Mapping[str, Tuple[pd.DataFrame, pd.Series]],
    primary_validation: str,
    feature_map: Mapping[str, str],
) -> pd.DataFrame:
    raw_rules = model.combiner.export()
    woe_rules = model.woe_transformer.transformer.export()
    binned = {name: model.combiner.transform(frame) for name, (frame, _) in datasets.items()}
    rows: List[Dict[str, Any]] = []

    for feature in model.candidate_features_:
        labels = _bin_labels(list(raw_rules[feature]))
        train_stats = _training_bin_statistics(
            binned["train"][feature], datasets["train"][1], labels, woe_rules[feature]
        )
        train_iv = train_stats["iv_contribution"].to_dict()
        for dataset_name, (_, target) in datasets.items():
            stats = _training_bin_statistics(
                binned[dataset_name][feature], target, labels, woe_rules[feature]
            )
            for bin_index, stat in stats.iterrows():
                rows.append(
                    {
                        "数据集名称": dataset_name,
                        "变量名称": feature,
                        "变量含义": feature_map.get(feature, ""),
                        "变量类型": "categorical" if feature in model.candidate_categorical_features_ else "numeric",
                        "分箱序号": int(bin_index),
                        "训练期分箱": stat["label"],
                        "样本数": int(stat["count"]),
                        "样本占比": float(stat["share"]),
                        "好样本数": int(stat["good"]),
                        "坏样本数": int(stat["bad"]),
                        "坏率": float(stat["bad_rate"]) if np.isfinite(stat["bad_rate"]) else float("nan"),
                        "训练WOE": float(stat["woe"]) if np.isfinite(stat["woe"]) else float("nan"),
                        "训练IV贡献": float(train_iv.get(bin_index, np.nan)),
                    }
                )
    return pd.DataFrame(rows)


def _dataset_metrics(model: LRScorecardModel, datasets: Mapping[str, Tuple[pd.DataFrame, pd.Series]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for name, (features, target) in datasets.items():
        probability = model.predict_proba(features)[:, 1]
        score = model.predict_score(features)
        (
            _,
            _,
            lift_tail10,
            lift_head10,
            _,
            _,
            _,
            _,
            _,
        ) = decile_analysis(target, probability, n_bins=10)
        result[name] = {
            "samples": len(target),
            "good": int((target == 0).sum()),
            "bad": int((target == 1).sum()),
            "bad_rate": float(target.mean()),
            "auc": float(roc_auc_score(target, probability)),
            "ks": float(ks_score(target, probability)),
            "lift_tail10": float(lift_tail10),
            "lift_head10": float(lift_head10),
            "score_min": float(np.min(score)),
            "score_max": float(np.max(score)),
            "score_mean": float(np.mean(score)),
            "score_median": float(np.median(score)),
            "score_std": float(np.std(score)),
            "score_p01": float(np.quantile(score, 0.01)),
            "score_p99": float(np.quantile(score, 0.99)),
        }
    return result


def _save_pickle(model: LRScorecardModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(model, stream, protocol=pickle.HIGHEST_PROTOCOL)


def _reload_validation(
    model: LRScorecardModel,
    path: Path,
    sample: pd.DataFrame,
) -> Tuple[LRScorecardModel, Dict[str, Any]]:
    before_probability = model.predict_proba(sample)
    before_score = model.predict_score(sample)
    before_class = model.predict(sample)
    components = model.predict_score_components(sample)
    component_difference = float(np.max(np.abs(before_score - components.sum(axis=1).to_numpy())))

    with path.open("rb") as stream:
        loaded: LRScorecardModel = pickle.load(stream)
    after_probability = loaded.predict_proba(sample)
    after_score = loaded.predict_score(sample)
    after_class = loaded.predict(sample)

    probability_difference = float(np.max(np.abs(before_probability - after_probability)))
    score_difference = float(np.max(np.abs(before_score - after_score)))
    class_equal = bool(np.array_equal(before_class, after_class))
    sorted_index = np.argsort(before_probability[:, 1], kind="mergesort")
    monotonic_score = bool(np.all(np.diff(before_score[sorted_index]) <= 1e-9))
    passed = bool(
        probability_difference <= 1e-12
        and score_difference <= 1e-12
        and class_equal
        and component_difference <= 1e-9
        and monotonic_score
    )
    result = {
        "passed": passed,
        "rows_checked": len(sample),
        "max_probability_difference": probability_difference,
        "max_score_difference": score_difference,
        "classification_equal": class_equal,
        "max_score_component_difference": component_difference,
        "higher_bad_probability_means_lower_score": monotonic_score,
    }
    if not passed:
        raise RuntimeError(f"评分卡模型保存重载一致性验证失败: {result}")
    return loaded, result




def _fit_fine_selector(
    selector: FeatureSelection,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Fit the package selector while supporting explicitly disabled rules.

    scorecardpipeline 0.1.39 iterates every drop-reason value when
    ``return_drop=True``.  Its toad engine returns ``None`` for a disabled
    rule such as ``empty=False``, which raises ``TypeError`` inside the
    package wrapper.  Calling the same package ``select`` function here and
    normalising ``None`` to an empty list preserves all other drop reasons.
    """
    if selector.engine != "toad" or not selector.return_drop:
        return selector.fit_transform(frame)

    from scorecardpipeline.processing import select as scorecard_select

    selected, dropped = scorecard_select(
        frame,
        target=selector.target,
        empty=selector.empty,
        iv=selector.iv,
        corr=selector.corr,
        identical=selector.identical,
        return_drop=True,
        exclude=selector.exclude,
    )
    dropped = {
        reason: [] if features is None else list(features)
        for reason, features in dropped.items()
    }
    selector.dropped = pd.DataFrame(
        [
            (feature, reason)
            for reason, features in dropped.items()
            for feature in features
        ],
        columns=["variable", "rm_reason"],
    )
    selector.select_columns = list(selected.columns)
    if selector.target_rm and selector.target in selector.select_columns:
        selector.select_columns.remove(selector.target)
    selector.fitted_ = True
    return selector.transform(frame)




def _normalize_adj_rules(
    adjusted_rules: Mapping[str, Any],
    feature_names: Sequence[str],
    numeric_features: Sequence[str],
) -> Dict[str, Any]:
    """Validate public manual rules and convert JSON nulls for Combiner."""
    candidates = set(feature_names)
    numeric = set(numeric_features)
    normalized: Dict[str, Any] = {}
    for feature, rule in adjusted_rules.items():
        if feature not in candidates:
            raise ValueError(f"adj_rules包含非候选特征: {feature}")
        if not isinstance(rule, list):
            raise ValueError(f"adj_rules[{feature}]必须是数组")
        if feature in numeric:
            missing_indexes = [index for index, value in enumerate(rule) if value is None]
            if len(missing_indexes) > 1 or (missing_indexes and missing_indexes[0] != len(rule) - 1):
                raise ValueError(f"adj_rules[{feature}]的null缺失箱最多一个且必须放在最后")
            cut_points: List[float] = []
            for value in rule:
                if value is None:
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
                    raise ValueError(f"adj_rules[{feature}]的数值切点必须是有限数值或null")
                number = float(value)
                if not math.isfinite(number):
                    raise ValueError(f"adj_rules[{feature}]的数值切点必须是有限数值")
                cut_points.append(number)
            if any(left >= right for left, right in zip(cut_points, cut_points[1:])):
                raise ValueError(f"adj_rules[{feature}]的数值切点必须严格递增且不能重复")
            normalized[feature] = [np.nan if value is None else value for value in rule]
            continue

        groups: List[List[Any]] = []
        seen: set[str] = set()
        for group_index, group in enumerate(rule):
            if not isinstance(group, list) or not group:
                raise ValueError(f"adj_rules[{feature}][{group_index}]必须是非空类别数组")
            normalized_group: List[Any] = []
            for value in group:
                if isinstance(value, (list, dict, tuple, set)):
                    raise ValueError(f"adj_rules[{feature}]的类别值必须是JSON标量或null")
                normalized_value = "nan" if value is None else value
                identity = f"{type(normalized_value).__name__}:{normalized_value!r}"
                if identity in seen:
                    raise ValueError(f"adj_rules[{feature}]的类别值重复出现: {value!r}")
                seen.add(identity)
                normalized_group.append(normalized_value)
            groups.append(normalized_group)
        normalized[feature] = groups
    return normalized


def _export_adj_rules(model: LRScorecardModel) -> Dict[str, Any]:
    """Export every candidate rule in an editable strict-JSON representation."""
    exported = model.combiner.export()
    result: Dict[str, Any] = {}
    categorical = set(model.candidate_categorical_features_)
    for feature in model.candidate_features_:
        rule = deepcopy(exported[feature])
        if feature in categorical:
            rule = [
                [None if value == "nan" else value for value in group]
                for group in rule
            ]
        result[feature] = _json_value(rule)
    return result


def _fit_scorecard(
    train_features: pd.DataFrame,
    train_target: pd.Series,
    label_col: str,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    feature_names: Sequence[str],
    parameters: Mapping[str, Any],
) -> LRScorecardModel:
    train = train_features.copy()
    train[label_col] = train_target.to_numpy()

    binning_parameters = _resolve_binning_parameters(parameters["binning"])
    adjusted_rules = binning_parameters.get("adj_rules") or {}
    binning_parameters["adj_rules"] = _normalize_adj_rules(
        adjusted_rules,
        feature_names,
        numeric_features,
    )

    combiner = Combiner(target=label_col, **binning_parameters)
    binned_train = combiner.fit_transform(train)

    woe_transformer = WOETransformer(target=label_col, **parameters["woe"])
    woe_train = woe_transformer.fit_transform(binned_train)

    fine_selector = FeatureSelection(target=label_col, **parameters["fine_selection"])
    fine_train = _fit_fine_selector(fine_selector, woe_train)
    if label_col not in fine_train.columns:
        raise RuntimeError("WOE后细筛意外删除标签字段")
    fine_features = [column for column in fine_train.columns if column != label_col]
    if not fine_features:
        raise RuntimeError("WOE后细筛未保留任何特征")

    stepwise_selector = StepwiseSelection(target=label_col, **parameters["stepwise"])
    stepwise_train = stepwise_selector.fit_transform(fine_train)
    if label_col not in stepwise_train.columns:
        raise RuntimeError("逐步回归意外删除标签字段")
    final_features = [column for column in stepwise_train.columns if column != label_col]
    if not final_features:
        raise RuntimeError("逐步回归未保留任何特征")

    logistic_model = ITLubberLogisticRegression(
        target=label_col,
        **parameters["logistic"],
    )
    logistic_model.fit(stepwise_train)
    if getattr(logistic_model, "n_iter_", np.array([0])).max() >= parameters["logistic"]["max_iter"]:
        raise RuntimeError("逻辑回归达到最大迭代次数，可能尚未收敛")

    scorecard = ScoreCard(
        target=label_col,
        combiner=combiner,
        transer=woe_transformer,
        pretrain_lr=logistic_model,
        **parameters["score_scale"],
    )
    scorecard.fit(stepwise_train)

    return LRScorecardModel(
        candidate_features=feature_names,
        final_features=final_features,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        combiner=combiner,
        woe_transformer=woe_transformer,
        fine_selector=fine_selector,
        stepwise_selector=stepwise_selector,
        logistic_model=logistic_model,
        scorecard=scorecard,
        parameters=parameters,
        classification_threshold=parameters["prediction"]["classification_threshold"],
        unseen_category_policy=parameters["prediction"]["unseen_category_policy"],
    )


def train_scorecard_model(
    dev_data: pd.DataFrame,
    test_data: pd.DataFrame,
    config: Mapping[str, Any],
    use_features: Sequence[str],
    feature_source: str,
    uploaded_model_params: Optional[Mapping[str, Any]] = None,
    oot_files: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Train and export the complete v1 WOE-LR scorecard workflow."""
    label_col = config["settings"]["label"]
    id_col = config["settings"].get("id_col", "id")
    date_col, group_col = resolve_report_context_columns(config)
    excluded_columns = {
        label_col,
        id_col,
        "baseline_prob",
        "baseline_score",
    }
    excluded_columns.update(column for column in (date_col, group_col) if column)
    original_features = list(dict.fromkeys(str(feature).strip() for feature in use_features))
    actual_features = [feature for feature in original_features if feature not in excluded_columns]
    if not actual_features:
        raise ValueError("候选特征剔除基础字段后为空")

    missing_train = [feature for feature in actual_features if feature not in dev_data.columns]
    if missing_train:
        raise ValueError(f"训练数据缺少指定候选特征: {missing_train}")
    missing_test = [feature for feature in actual_features if feature not in test_data.columns]
    if missing_test:
        raise ValueError(f"测试数据缺少指定候选特征: {missing_test}")
    if label_col not in dev_data.columns or label_col not in test_data.columns:
        raise ValueError(f"训练或测试数据缺少标签字段: {label_col}")

    effective_parameters = _resolve_scorecard_parameters(config, uploaded_model_params)

    numeric_features = [feature for feature in actual_features if is_numeric_dtype(dev_data[feature].dtype)]
    categorical_features = [feature for feature in actual_features if feature not in numeric_features]
    train_features = normalize_scorecard_features(
        dev_data, actual_features, numeric_features, categorical_features
    )
    test_features = normalize_scorecard_features(
        test_data, actual_features, numeric_features, categorical_features
    )
    train_target = _validate_binary_target(dev_data[label_col], label_col)
    test_target = _validate_binary_target(test_data[label_col], label_col)

    def _add_report_context(
        features: pd.DataFrame,
        raw_frame: pd.DataFrame,
    ) -> pd.DataFrame:
        report_features = features.copy()
        for column in (date_col, group_col):
            if column and column in raw_frame.columns and column not in report_features.columns:
                report_features[column] = (
                    parse_report_dates(raw_frame[column]).to_numpy()
                    if column == date_col
                    else raw_frame[column].to_numpy()
                )
        return report_features

    validation_features: Dict[str, pd.DataFrame] = {"test": test_features}
    datasets: Dict[str, Tuple[pd.DataFrame, pd.Series]] = {
        "train": (train_features, train_target),
        "test": (test_features, test_target),
    }
    report_datasets: Dict[str, Tuple[pd.DataFrame, pd.Series]] = {
        "train": (_add_report_context(train_features, dev_data), train_target),
        "test": (_add_report_context(test_features, test_data), test_target),
    }
    for oot_file in oot_files or []:
        dataset_name = Path(oot_file).stem
        oot_frame = get_data(oot_file)
        missing_oot = [feature for feature in actual_features if feature not in oot_frame.columns]
        if missing_oot:
            raise ValueError(f"OOT数据集 {dataset_name} 缺少候选特征: {missing_oot}")
        if label_col not in oot_frame.columns:
            raise ValueError(f"OOT数据集 {dataset_name} 缺少标签字段: {label_col}")
        oot_features = normalize_scorecard_features(
            oot_frame, actual_features, numeric_features, categorical_features
        )
        oot_target = _validate_binary_target(oot_frame[label_col], label_col)
        validation_features[dataset_name] = oot_features
        datasets[dataset_name] = (oot_features, oot_target)
        report_datasets[dataset_name] = (
            _add_report_context(oot_features, oot_frame),
            oot_target,
        )

    print("\n>>> WOE-LR信用评分卡训练开始 ...")
    print(f"候选特征数量: {len(actual_features)}（数值 {len(numeric_features)}，类别 {len(categorical_features)}）")
    model = _fit_scorecard(
        train_features,
        train_target,
        label_col,
        numeric_features,
        categorical_features,
        actual_features,
        effective_parameters,
    )
    final_features = list(model.scorecard.features_)
    print(f"WOE后细筛及逐步回归最终保留 {len(final_features)} 个特征: {final_features}")

    root = Path(config["output_path"].get("dst", "outputs/model_train")) / "LR"
    model_dir = root / "model"
    outputs_dir = root / "outputs"
    report_dir = root / "report"
    model_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Retired files must not survive as stale artifacts from an older run.
    for obsolete_name in ("training_manifest.json", "binning_rules.json", "scorecard_scale.csv"):
        obsolete_path = outputs_dir / obsolete_name
        if obsolete_path.exists():
            obsolete_path.unlink()

    pkl_path = model_dir / "scorecard_model.pkl"
    points_path = outputs_dir / "scorecard_points.csv"
    review_path = outputs_dir / "binning_review.csv"
    detail_path = outputs_dir / "binning_detail.csv"
    model_train_spec_path = outputs_dir / "model_train_spec_lrscorecard.json"
    report_path = report_dir / "评分卡模型报告.xlsx"
    result_json_path = Path(
        config.get("output_path", {}).get("model_json_path", "results/model_train.json")
    )

    model.metadata = {
        "model_type": "woe_lr_scorecard",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "label": {"column": label_col, "bad_value": 1, "good_value": 0},
        "feature_source": feature_source,
    }
    _save_pickle(model, pkl_path)
    reload_sample = test_data.head(min(len(test_data), 500)).copy()
    deployed_model, reload_validation = _reload_validation(model, pkl_path, reload_sample)

    descriptions = _feature_map(config.get("data_path", {}).get("feature_dic_path"))
    points = build_scorecard_points(deployed_model, descriptions)
    primary_validation = "oot_main" if "oot_main" in validation_features else "test"
    review = build_binning_review(
        deployed_model,
        train_features,
        train_target,
        validation_features,
        primary_validation,
        descriptions,
    )
    detail = build_binning_detail(
        deployed_model,
        datasets,
        primary_validation,
        descriptions,
    )
    points.to_csv(points_path, index=False, encoding="utf-8-sig")
    _write_binning_review_csv(review, review_path)
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    _write_json(
        model_train_spec_path,
        {
            "use_features": deployed_model.candidate_features_,
            "model_params": {
                "binning_method": deployed_model.parameters_["binning"]["method"],
                "adj_rules": _export_adj_rules(deployed_model),
                "fine_selection_corr": deployed_model.parameters_["fine_selection"]["corr"],
                "stepwise_criterion": deployed_model.parameters_["stepwise"]["criterion"],
                "score_scale": {
                    key: deployed_model.parameters_["score_scale"][key]
                    for key in ("base_score", "pdo", "base_odds")
                },
            },
        },
    )

    excel_report = generate_scorecard_training_report(
        model=deployed_model,
        datasets=report_datasets,
        report_path=report_path,
        feature_map=descriptions,
        detail=detail,
        review=review,
        points=points,
        config=config,
        raw_prediction_sample=test_data,
    )
    metrics = _dataset_metrics(deployed_model, datasets)
    feature_summary = build_scorecard_feature_summary(
        deployed_model,
        review,
        descriptions,
    )

    from utils.generate_json_reports import generate_lr_scorecard_train_json

    fine_drop_count = len(_selector_drop_map(deployed_model.fine_selector))
    fine_selection_count = len(deployed_model.candidate_features_) - fine_drop_count
    summary_result = generate_lr_scorecard_train_json(
        metrics=metrics,
        primary_validation=primary_validation,
        feature_source=feature_source,
        candidate_features=deployed_model.candidate_features_,
        fine_selection_count=fine_selection_count,
        stepwise_count=len(deployed_model.feature_names_),
        final_features=deployed_model.feature_names_,
        feature_summary=feature_summary,
        parameter_source=(
            "用户上传的model_train_spec_lrscorecard.json"
            if uploaded_model_params
            else "模型训练配置"
        ),
        output_path=str(result_json_path),
    )
    if summary_result is None:
        raise RuntimeError("LrScorecard模型训练JSON生成失败")

    print(f"完整评分卡模型已保存: {pkl_path}")
    print(f"评分卡实验复现配置已保存: {model_train_spec_path}")
    print(f"策略分箱分值表已保存: {points_path}")
    print(f"评分卡Excel报告已保存: {excel_report}")
    print(f"评分卡训练总结JSON已保存: {result_json_path}")
    print(">>> WOE-LR信用评分卡训练完成！\n")
    return {
        "model": str(pkl_path),
        "model_train_spec": str(model_train_spec_path),
        "scorecard_points": str(points_path),
        "binning_review": str(review_path),
        "binning_detail": str(detail_path),
        "excel_report": str(excel_report),
        "model_train_summary": str(result_json_path),
        "reload_validation": reload_validation,
    }
