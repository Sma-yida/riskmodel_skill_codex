"""
基线模型相关工具函数
用于处理基线模型加载、特征映射、兼容性检查和评估（支持 pkl -> pmml 回退）
"""
import os
import pickle
import shutil
from typing import Dict, List, Optional, Tuple

import pandas as pd

from utils.data_utils import identify_types
from utils.utils_model_evaluation import get_final_result
from utils.utils1 import compute_psi, evaluate_on_dataset

# ==================== 基线模型配置 ====================
BASELINE_MODEL_DIR = "data"
FEATURE_MAPPING_PATH = "data/feature_dict.csv"  # CSV需包含 Feature / old_feature 列
LABEL_COL = "label"
CHANNEL_COL = "channel"
BASELINE_PROB_COL = "baseline_prob"
BASELINE_SCORE_COL = "baseline_score"
BASELINE_MODEL_OUTPUT_DIR = "outputs/baseline_model"

# 兼容旧代码引用，当前仅作保留
OLD_MODEL_PKL_PATH = "data/old_model.pkl"


def _format_attempts(attempts: List[str]) -> str:
    return "；".join(attempts) if attempts else "无"


def clear_baseline_output_dir() -> None:
    """清理旧的基线评估产物，避免历史结果污染当前报告。"""
    if not os.path.exists(BASELINE_MODEL_OUTPUT_DIR):
        return
    for name in os.listdir(BASELINE_MODEL_OUTPUT_DIR):
        target = os.path.join(BASELINE_MODEL_OUTPUT_DIR, name)
        try:
            if os.path.isdir(target):
                shutil.rmtree(target)
            else:
                os.remove(target)
        except Exception as e:
            print(f"警告: 清理基线产物失败 '{target}': {str(e)}")


def _discover_baseline_candidates(model_dir: str = BASELINE_MODEL_DIR) -> Dict[str, List[str]]:
    """
    扫描 data 目录顶层模型文件（不递归），按修改时间倒序返回。
    """
    candidates = {"pkl": [], "pmml": []}
    if not os.path.isdir(model_dir):
        return candidates

    for entry in os.listdir(model_dir):
        full_path = os.path.join(model_dir, entry)
        if not os.path.isfile(full_path):
            continue
        ext = os.path.splitext(entry)[1].lower()
        if ext == ".pkl":
            candidates["pkl"].append(full_path)
        elif ext == ".pmml":
            candidates["pmml"].append(full_path)

    candidates["pkl"].sort(key=lambda p: os.path.getmtime(p), reverse=True)
    candidates["pmml"].sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates


def load_feature_mapping(feature_mapping_path: str = FEATURE_MAPPING_PATH) -> Dict[str, str]:
    """
    加载特征名称映射表，返回 {基线模型特征名: 当前数据集特征名}
    """
    if not feature_mapping_path or not os.path.exists(feature_mapping_path):
        return {}

    try:
        mapping_df = pd.read_csv(feature_mapping_path, encoding="utf-8")
    except UnicodeDecodeError:
        mapping_df = pd.read_csv(feature_mapping_path, encoding="gbk")

    if "old_feature" not in mapping_df.columns or "Feature" not in mapping_df.columns:
        print(f"警告: 特征映射表缺少必需列 old_feature/Feature，将忽略映射: {feature_mapping_path}")
        return {}

    mapping_dict = {}
    for _, row in mapping_df.iterrows():
        old_feat = str(row["old_feature"]).strip()
        current_feat = str(row["Feature"]).strip()
        if old_feat and current_feat and old_feat != "nan" and current_feat != "nan":
            mapping_dict[old_feat] = current_feat

    if mapping_dict:
        print(f"成功加载特征映射表 '{feature_mapping_path}': {len(mapping_dict)} 个映射关系")
    return mapping_dict


def _extract_pkl_model_features(model) -> List[str]:
    if hasattr(model, "booster_") and hasattr(model.booster_, "feature_name"):
        return list(model.booster_.feature_name())
    if hasattr(model, "feature_name_"):
        return list(model.feature_name_)
    return []


def _extract_pmml_model_features(pmml_model) -> List[str]:
    """
    从 PMML 模型中尽量提取入模特征列表，兼容不同 pypmml 版本接口。
    """
    # 常见接口：model.inputNames (属性) / model.inputNames() (方法)
    if hasattr(pmml_model, "inputNames"):
        input_names = pmml_model.inputNames
        if callable(input_names):
            input_names = input_names()
        if input_names is not None:
            return [str(x) for x in list(input_names)]

    # 兜底：inputFields (对象列表，可能有 name 字段)
    if hasattr(pmml_model, "inputFields"):
        input_fields = pmml_model.inputFields
        if callable(input_fields):
            input_fields = input_fields()
        if input_fields is not None:
            features = []
            for field in list(input_fields):
                if hasattr(field, "name"):
                    features.append(str(field.name))
                elif isinstance(field, dict) and "name" in field:
                    features.append(str(field["name"]))
                else:
                    features.append(str(field))
            return features

    return []


def _load_pkl_model(model_path: str) -> Tuple[Optional[object], List[str], Optional[str]]:
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        model_features = _extract_pkl_model_features(model)
        if not model_features:
            return None, [], "无法从PKL模型中提取入模特征"
        return model, model_features, None
    except Exception as e:
        return None, [], f"加载PKL失败: {str(e)}"


def _load_pmml_model(model_path: str) -> Tuple[Optional[object], List[str], Optional[str]]:
    try:
        import pypmml
    except Exception as e:
        return None, [], f"导入pypmml失败: {str(e)}"

    try:
        if hasattr(pypmml, "Model"):
            pmml_model = pypmml.Model.load(model_path)
        elif hasattr(pypmml, "load"):
            pmml_model = pypmml.load(model_path)
        else:
            return None, [], "pypmml接口不可用，未找到 Model.load/load"
    except Exception as e:
        return None, [], f"加载PMML失败: {str(e)}"

    model_features = _extract_pmml_model_features(pmml_model)
    if not model_features:
        return None, [], "无法从PMML模型中提取入模特征"
    return pmml_model, model_features, None


def _load_baseline_model_with_fallback() -> Dict[str, object]:
    """
    按策略加载基线模型：优先最新 pkl，失败回退最新 pmml。
    """
    attempts: List[str] = []
    candidates = _discover_baseline_candidates(BASELINE_MODEL_DIR)

    latest_pkl = candidates["pkl"][0] if candidates["pkl"] else None
    latest_pmml = candidates["pmml"][0] if candidates["pmml"] else None

    if not latest_pkl and not latest_pmml:
        return {
            "loaded": False,
            "状态": "未配置",
            "说明": "未发现基线模型文件（data目录下无 .pkl/.pmml）",
            "尝试记录": attempts,
            "失败原因": [],
        }

    # 1) 先尝试 pkl
    if latest_pkl:
        model, features, err = _load_pkl_model(latest_pkl)
        if err is None:
            attempts.append(f"PKL成功: {latest_pkl}")
            return {
                "loaded": True,
                "model": model,
                "model_type": "pkl",
                "model_path": latest_pkl,
                "model_features": features,
                "状态": "加载成功",
                "说明": f"基线模型加载成功（PKL）: {latest_pkl}",
                "尝试记录": attempts,
                "失败原因": [],
            }
        attempts.append(f"PKL失败: {latest_pkl}，原因: {err}")

    # 2) pkl 失败后回退 pmml
    if latest_pmml:
        model, features, err = _load_pmml_model(latest_pmml)
        if err is None:
            attempts.append(f"PMML成功: {latest_pmml}")
            return {
                "loaded": True,
                "model": model,
                "model_type": "pmml",
                "model_path": latest_pmml,
                "model_features": features,
                "状态": "加载成功",
                "说明": f"基线模型PKL加载失败后，已回退并成功加载PMML: {latest_pmml}",
                "尝试记录": attempts,
                "失败原因": [x for x in attempts if "失败" in x],
            }
        attempts.append(f"PMML失败: {latest_pmml}，原因: {err}")

    return {
        "loaded": False,
        "状态": "加载失败",
        "说明": "基线模型加载失败（PKL与PMML均不可用）",
        "尝试记录": attempts,
        "失败原因": [x for x in attempts if "失败" in x],
    }


def _detect_pmml_positive_prob_col(pred_df: pd.DataFrame) -> Optional[str]:
    """
    智能识别 PMML 输出中的正类概率列。
    """
    if pred_df is None or pred_df.empty:
        return None

    cols = [str(c) for c in pred_df.columns]
    col_map = {str(c): c for c in pred_df.columns}

    # 优先精确匹配（pypmml/LightGBM 等常输出 probability(1.0)，probability(1)）
    for key in ["probability(1)", "probability(1.0)", "probability_1", "prob_1", "1"]:
        if key in col_map:
            return col_map[key]

    # 次优：包含概率且指向正类（注意：(1) 不是 probability(1.0) 的子串，需单独判断）
    for c in cols:
        cl = c.lower().replace(" ", "")
        if "probability" in cl and (
            "(1)" in cl or "(1.0)" in cl or "_1" in cl or cl.endswith("1")
        ):
            return col_map[c]

    # 再次兜底：名称本身就是常见正类标记
    for c in cols:
        cl = c.lower().strip()
        if cl in {"true", "yes", "positive"}:
            return col_map[c]

    # 最后兜底：仅当只有一个数值列时使用该列
    numeric_cols = pred_df.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric_cols) == 1:
        return numeric_cols[0]

    return None


def _predict_baseline_proba(model, model_type: str, data_df: pd.DataFrame) -> pd.Series:
    if model_type == "pkl":
        return pd.Series(model.predict_proba(data_df)[:, 1], index=data_df.index)

    pred_df = model.predict(data_df)
    if isinstance(pred_df, pd.Series):
        return pred_df.astype(float)
    if not isinstance(pred_df, pd.DataFrame):
        pred_df = pd.DataFrame(pred_df)

    prob_col = _detect_pmml_positive_prob_col(pred_df)
    if prob_col is None:
        raise ValueError(f"无法识别PMML正类概率列，输出列为: {list(pred_df.columns)}")
    return pred_df[prob_col].astype(float)


def _series_from_column(dataset_df: pd.DataFrame, column: str, reverse_score: bool = False) -> pd.Series:
    values = pd.to_numeric(dataset_df[column], errors="coerce")
    if reverse_score:
        values = -values
    return pd.Series(values.values, index=dataset_df.index)


def _build_metric_row(
    dataset_name: str,
    dataset_df: pd.DataFrame,
    metrics: Optional[dict] = None,
    psi: Optional[float] = None,
    ks_gap_available: bool = True,
    status: str = "评估成功",
    reason: str = "",
    source_info: Optional[dict] = None,
) -> dict:
    y_true = dataset_df[LABEL_COL].values if LABEL_COL in dataset_df.columns else []
    total_samples = len(dataset_df)
    bad_samples = int(pd.Series(y_true).sum()) if total_samples else 0
    good_samples = total_samples - bad_samples
    bad_rate = float(pd.Series(y_true).mean()) if total_samples else 0.0

    source_info = source_info or {}
    dataset_sources = source_info.get("dataset_sources", {})
    source_desc = dataset_sources.get(dataset_name, source_info.get("基线来源", "none"))
    row = {
        "数据集": dataset_name,
        "样本数": total_samples,
        "好样本数": good_samples,
        "坏样本数": bad_samples,
        "坏率": f"{bad_rate:.4f}",
        "KS": "-",
        "AUC": "-",
        "PSI": "-",
        "KS_Gap": "-",
        "Lift_tail10": "-",
        "Lift_head10": "-",
        "Lift_head1": "-",
        "Lift_head2": "-",
        "Lift_head5": "-",
        "mono_violation": "-",
        "mono_jump_count": "-",
        "评估状态": status,
        "异常说明": reason,
        "基线来源": source_desc,
    }

    if metrics is not None:
        row.update({
            "KS": f"{metrics['ks']:.6f}",
            "AUC": f"{metrics['auc']:.6f}",
            "PSI": f"{psi:.6f}" if psi is not None else "-",
            "KS_Gap": f"{metrics['ks_gap']:.6f}" if ks_gap_available else "-",
            "Lift_tail10": f"{metrics['lift_tail10']:.4f}",
            "Lift_head10": f"{metrics['lift_head10']:.4f}",
            "Lift_head1": f"{metrics['lift_head1']:.4f}",
            "Lift_head2": f"{metrics['lift_head2']:.4f}",
            "Lift_head5": f"{metrics['lift_head5']:.4f}",
            "mono_violation": f"{metrics['mono_violation']:.6f}",
            "mono_jump_count": metrics["mono_jump_count"],
        })
    return row


def _save_baseline_metrics(metrics_df: pd.DataFrame) -> None:
    metrics_path = os.path.join(BASELINE_MODEL_OUTPUT_DIR, "baseline_model_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")


def _resolve_baseline_predictions(datasets: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.Series], dict]:
    """
    按数据集逐个解析基线预测来源：baseline_prob > baseline_score > pkl > pmml。
    """
    predictions: Dict[str, pd.Series] = {}
    dataset_sources: Dict[str, str] = {}
    dataset_errors: Dict[str, str] = {}
    model_needed = any(
        BASELINE_PROB_COL not in dataset_df.columns and BASELINE_SCORE_COL not in dataset_df.columns
        for dataset_df in datasets.values()
    )

    model_ctx = None
    baseline_model = None
    model_type = None
    mapped_features: List[str] = []
    model_source = "none"
    model_source_desc = ""
    attempts = "无"
    model_failures = "无"

    if model_needed:
        model_ctx = _load_baseline_model_with_fallback()
        attempts = _format_attempts(model_ctx.get("尝试记录", []))
        model_failures = _format_attempts(model_ctx.get("失败原因", []))
        if model_ctx.get("loaded"):
            baseline_model = model_ctx["model"]
            model_type = model_ctx["model_type"]
            model_source = "pkl_model" if model_type == "pkl" else "pmml_model"
            model_source_desc = f"使用用户上传的基线模型文件打分: {model_ctx.get('model_path')}"
            mapping_dict = load_feature_mapping(FEATURE_MAPPING_PATH) if FEATURE_MAPPING_PATH else {}
            mapped_features = [mapping_dict.get(old_feat, old_feat) for old_feat in model_ctx["model_features"]]

            if model_type == "pkl" and "train" in datasets:
                train_missing = [f for f in mapped_features if f not in datasets["train"].columns]
                if not train_missing:
                    _, cat_fea = identify_types(datasets["train"][mapped_features], remove_fea=[])
                    for feat in cat_fea:
                        for dataset_df in datasets.values():
                            if feat in dataset_df.columns:
                                dataset_df[feat] = dataset_df[feat].astype("category")
        else:
            model_source_desc = model_ctx.get("说明", "未发现可用基线模型")

    for dataset_name, dataset_df in datasets.items():
        if BASELINE_PROB_COL in dataset_df.columns:
            predictions[dataset_name] = _series_from_column(dataset_df, BASELINE_PROB_COL)
            dataset_sources[dataset_name] = f"使用上传数据列 {BASELINE_PROB_COL} 作为基线风险概率"
            continue

        if BASELINE_SCORE_COL in dataset_df.columns:
            predictions[dataset_name] = _series_from_column(dataset_df, BASELINE_SCORE_COL, reverse_score=True)
            dataset_sources[dataset_name] = f"使用上传数据列 {BASELINE_SCORE_COL}，按高分低风险转换为风险方向后评估"
            continue

        if baseline_model is None or model_type is None:
            reason = model_source_desc or "未发现可用基线预测列或基线模型"
            dataset_errors[dataset_name] = reason
            dataset_sources[dataset_name] = "none"
            continue

        missing_features = [f for f in mapped_features if f not in dataset_df.columns]
        if missing_features:
            reason = f"基线模型特征缺失: {', '.join(missing_features)}"
            dataset_errors[dataset_name] = reason
            dataset_sources[dataset_name] = model_source_desc
            continue

        try:
            predictions[dataset_name] = _predict_baseline_proba(
                baseline_model,
                model_type,
                dataset_df[mapped_features],
            )
            dataset_sources[dataset_name] = model_source_desc
        except Exception as e:
            reason = str(e)
            dataset_errors[dataset_name] = reason
            dataset_sources[dataset_name] = model_source_desc

    source_types = []
    for source_text in dataset_sources.values():
        if source_text.startswith(f"使用上传数据列 {BASELINE_PROB_COL}"):
            source_types.append("baseline_prob")
        elif source_text.startswith(f"使用上传数据列 {BASELINE_SCORE_COL}"):
            source_types.append("baseline_score")
        elif source_text.startswith("使用用户上传的基线模型文件打分"):
            source_types.append(model_source)
        else:
            source_types.append("none")

    unique_source_types = [source for source in dict.fromkeys(source_types) if source != "none"]
    if len(unique_source_types) == 1:
        source_summary = unique_source_types[0]
    elif unique_source_types:
        source_summary = "mixed"
    else:
        source_summary = "none"

    source_info = {
        "基线来源": source_summary,
        "dataset_sources": dataset_sources,
        "dataset_errors": dataset_errors,
        "模型类型": model_type or ("uploaded_column" if source_summary in {"baseline_prob", "baseline_score", "mixed"} else None),
        "模型路径": model_ctx.get("model_path") if model_ctx else None,
        "尝试记录": attempts,
        "失败原因": _format_attempts([f"{name}: {reason}" for name, reason in dataset_errors.items()]) or model_failures,
    }
    if not predictions:
        source_info.update({
            "状态": "未配置" if source_summary == "none" else "评估失败",
            "说明": "未发现可用基线预测列或基线模型，跳过基线模型评估",
        })
    return predictions, source_info


def _evaluate_baseline_predictions(
    datasets: Dict[str, pd.DataFrame],
    predictions: Dict[str, pd.Series],
    source_info: dict,
) -> Tuple[pd.DataFrame, dict]:
    all_metrics = []
    failed_datasets = []
    already_recorded_failures = set()
    dataset_errors = source_info.get("dataset_errors", {})
    train_pred = predictions.get("train")
    train_ks = None

    if train_pred is not None:
        try:
            train_metrics = evaluate_on_dataset(datasets["train"][LABEL_COL].values, train_pred.values, 0.0)
            train_ks = train_metrics["ks"]
        except Exception as e:
            failed_datasets.append(f"train: {str(e)}")
            all_metrics.append(_build_metric_row(
                "train",
                datasets["train"],
                metrics=None,
                status="评估失败",
                reason=str(e),
                source_info=source_info,
            ))
            already_recorded_failures.add("train")

    for dataset_name, dataset_df in datasets.items():
        if dataset_name in already_recorded_failures:
            print(f"  {dataset_name:10s}: 评估失败 - 已记录失败原因")
            continue
        if dataset_name not in predictions:
            reason = dataset_errors.get(dataset_name, "未获得基线预测结果")
            all_metrics.append(_build_metric_row(
                dataset_name,
                dataset_df,
                metrics=None,
                status="评估失败",
                reason=reason,
                source_info=source_info,
            ))
            failed_datasets.append(f"{dataset_name}: {reason}")
            print(f"  {dataset_name:10s}: 评估失败 - {reason}")
            continue

        pred = predictions[dataset_name]
        dataset_df = datasets[dataset_name]
        try:
            y_true = dataset_df[LABEL_COL].values
            if dataset_name == "train" and train_ks is not None:
                metrics = train_metrics
            else:
                metrics = evaluate_on_dataset(y_true, pred.values, train_ks if train_ks is not None else 0.0)
            psi = compute_psi(train_pred.values, pred.values) if train_pred is not None else None
            ks_gap_available = train_ks is not None
            if dataset_name == "train":
                psi = None
                ks_gap_available = False

            all_metrics.append(_build_metric_row(
                dataset_name=dataset_name,
                dataset_df=dataset_df,
                metrics=metrics,
                psi=psi,
                ks_gap_available=ks_gap_available,
                source_info=source_info,
            ))

            if CHANNEL_COL in dataset_df.columns:
                dataset_df_with_pred = dataset_df.copy()
                dataset_df_with_pred["pred_prob"] = pred.values
                get_final_result(
                    dataset_type=dataset_name,
                    df=dataset_df_with_pred,
                    target=LABEL_COL,
                    pred="pred_prob",
                    col=CHANNEL_COL,
                    dst=BASELINE_MODEL_OUTPUT_DIR,
                )

            if psi is None:
                print(f"  {dataset_name:10s}: KS={metrics['ks']:.4f}, AUC={metrics['auc']:.4f}, PSI=-")
            else:
                print(f"  {dataset_name:10s}: KS={metrics['ks']:.4f}, AUC={metrics['auc']:.4f}, PSI={psi:.4f}")
        except Exception as e:
            reason = str(e)
            all_metrics.append(_build_metric_row(
                dataset_name=dataset_name,
                dataset_df=dataset_df,
                metrics=None,
                status="评估失败",
                reason=reason,
                source_info=source_info,
            ))
            print(f"  {dataset_name:10s}: 评估失败 - {str(e)}")
            failed_datasets.append(f"{dataset_name}: {str(e)}")

    dataset_order = {name: i for i, name in enumerate(datasets.keys())}
    all_metrics.sort(key=lambda row: dataset_order.get(row["数据集"], 999))
    metrics_df = pd.DataFrame(all_metrics)
    _save_baseline_metrics(metrics_df)

    success_count = int((metrics_df["评估状态"] == "评估成功").sum()) if "评估状态" in metrics_df.columns else 0
    failed_desc = _format_attempts(failed_datasets)
    dataset_sources = source_info.get("dataset_sources", {})
    success_sources = {
        name: source
        for name, source in dataset_sources.items()
        if name in predictions and source and source != "none"
    }
    unique_sources = list(dict.fromkeys(success_sources.values()))
    if len(unique_sources) == 1:
        source_desc = unique_sources[0]
    elif unique_sources:
        source_desc = "；".join([f"{name}: {source}" for name, source in success_sources.items()])
    else:
        source_desc = source_info.get("基线来源", "none")
    status_desc = f"基线模型已基于{source_desc}完成 {success_count} 个数据集评估"
    if failed_datasets:
        status_desc += f"，部分数据集失败: {failed_desc}"

    status_info = {
        "状态": "评估成功" if success_count > 0 else "评估失败",
        "说明": status_desc,
        "模型类型": source_info.get("模型类型"),
        "模型路径": source_info.get("模型路径"),
        "基线来源": source_info.get("基线来源", "none"),
        "尝试记录": source_info.get("尝试记录", "无"),
        "失败原因": failed_desc if failed_datasets else source_info.get("失败原因", "无"),
    }
    return metrics_df, status_info


def load_old_model_features(model_path: str) -> List[str]:
    """
    兼容旧接口：从指定路径加载基线模型特征（支持 pkl/pmml）。
    """
    if not model_path or not os.path.exists(model_path):
        return []
    ext = os.path.splitext(model_path)[1].lower()
    if ext == ".pkl":
        _, features, err = _load_pkl_model(model_path)
    elif ext == ".pmml":
        _, features, err = _load_pmml_model(model_path)
    else:
        return []
    if err:
        print(f"警告: 从模型文件加载特征失败: {err}")
        return []
    return features


def check_old_model_features_with_status(current_data_columns: List[str]) -> Dict[str, object]:
    """
    检查基线模型特征兼容性，返回统一状态信息（供 JSON 报告使用）。
    """
    current_columns_set = set(current_data_columns) if current_data_columns else set()
    uploaded_columns = [col for col in [BASELINE_PROB_COL, BASELINE_SCORE_COL] if col in current_columns_set]
    if uploaded_columns:
        return {
            "状态": "检查通过",
            "说明": f"检测到上传基线列 {', '.join(uploaded_columns)}，将优先使用上传列进行基线模型评估，无需基线模型特征兼容性检查。",
            "模型类型": "uploaded_column",
            "模型路径": None,
            "尝试记录": [],
            "失败原因": [],
            "all_exist": True,
            "mapped_features": uploaded_columns,
            "missing_features": [],
            "missing_old_features": [],
        }

    model_ctx = _load_baseline_model_with_fallback()
    result = {
        "状态": model_ctx.get("状态", "未知"),
        "说明": model_ctx.get("说明", ""),
        "模型类型": model_ctx.get("model_type"),
        "模型路径": model_ctx.get("model_path"),
        "尝试记录": model_ctx.get("尝试记录", []),
        "失败原因": model_ctx.get("失败原因", []),
        "all_exist": False,
        "mapped_features": [],
        "missing_features": [],
        "missing_old_features": [],
    }

    if not model_ctx.get("loaded"):
        return result

    old_features = model_ctx["model_features"]
    mapping_dict = load_feature_mapping(FEATURE_MAPPING_PATH) if FEATURE_MAPPING_PATH else {}
    mapped_features: List[str] = []
    missing_features: List[str] = []
    missing_old_features: List[str] = []

    for old_feat in old_features:
        new_feat = mapping_dict.get(old_feat, old_feat)
        if new_feat in current_columns_set:
            mapped_features.append(new_feat)
        else:
            missing_features.append(new_feat)
            missing_old_features.append(old_feat)

    all_exist = len(missing_features) == 0
    result.update({
        "all_exist": all_exist,
        "mapped_features": mapped_features,
        "missing_features": missing_features,
        "missing_old_features": missing_old_features,
    })

    if all_exist:
        result["状态"] = "检查通过"
        result["说明"] = (
            f"基线模型特征兼容性检查通过。模型类型: {model_ctx.get('model_type')}，"
            f"入模特征数量: {len(mapped_features)}"
        )
    else:
        result["状态"] = "特征缺失"
        result["说明"] = (
            "数据集中缺少基线模型入模特征，无法进行准确模型对比。"
            f"缺失特征（映射后）: {', '.join(missing_features)}"
        )

    return result


def check_old_model_features(current_data_columns: List[str]) -> Tuple[bool, List[str], List[str], List[str]]:
    """
    兼容旧接口：返回 (是否全部存在, 映射后的特征列表, 缺失特征(映射后), 缺失特征(旧名))
    """
    info = check_old_model_features_with_status(current_data_columns)
    return (
        bool(info.get("all_exist", False)),
        list(info.get("mapped_features", [])),
        list(info.get("missing_features", [])),
        list(info.get("missing_old_features", [])),
    )


def evaluate_baseline_model(
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    oot_datasets: Optional[List[Tuple[pd.DataFrame, str]]] = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    加载基线模型并在所有数据集上评估（优先 pkl，失败回退 pmml）。
    """
    os.makedirs(BASELINE_MODEL_OUTPUT_DIR, exist_ok=True)

    print("\n" + "=" * 60)
    print("基线模型评估")

    datasets = {
        "train": dev_df.copy(),
        "test": test_df.copy(),
    }
    if oot_datasets:
        for oot_df, dataset_name in oot_datasets:
            datasets[dataset_name.lower()] = oot_df.copy()

    predictions, source_info = _resolve_baseline_predictions(datasets)
    if not predictions:
        status_info = {
            "状态": source_info.get("状态", "未配置"),
            "说明": source_info.get("说明", "未发现可用基线预测列或基线模型，跳过基线模型评估"),
            "模型类型": source_info.get("模型类型"),
            "模型路径": source_info.get("模型路径"),
            "基线来源": source_info.get("基线来源", "none"),
            "尝试记录": source_info.get("尝试记录", "无"),
            "失败原因": source_info.get("失败原因", "无"),
        }
        print(f"提示: {status_info['说明']}")
        print("=" * 60 + "\n")
        return pd.DataFrame(), status_info

    print(f"基线来源: {source_info.get('基线来源', '未知')}")
    metrics_df, status_info = _evaluate_baseline_predictions(datasets, predictions, source_info)

    print("=" * 60 + "\n")
    return metrics_df, status_info
