"""LrScorecard训练阶段Excel报告适配。

本模块只负责报告所需的统计、绘图和Excel布局处理。评分卡拟合、PKL和策略规则
仍由 ``utils_model_train_scorecard`` 负责。
"""

from __future__ import annotations

import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from scorecardpipeline import QuickModelReport
from sklearn.metrics import roc_auc_score

from .utils1 import Psi, ks_score

if TYPE_CHECKING:
    from .utils_model_train_scorecard import LRScorecardModel


def _effective_binning_parameters(parameters: Mapping[str, Any]) -> Dict[str, Any]:
    """Resolve the selected method without duplicating the training logic."""
    from .utils_model_train_scorecard import _resolve_binning_parameters

    resolved = _resolve_binning_parameters(parameters["binning"])
    resolved.setdefault("empty_separate", True)
    resolved.setdefault("max_n_bins", None)
    resolved.setdefault("min_bin_size", 0.05)
    return resolved


def _population_psi(expected: pd.Series, actual: pd.Series) -> float:
    """Calculate one PSI value from two aligned bin distributions."""
    bins = sorted(set(expected.index.tolist()) | set(actual.index.tolist()), key=str)
    expected_values = expected.reindex(bins, fill_value=0.0).to_numpy(dtype=float)
    actual_values = actual.reindex(bins, fill_value=0.0).to_numpy(dtype=float)
    epsilon = 1e-6
    expected_values = np.clip(expected_values, epsilon, None)
    actual_values = np.clip(actual_values, epsilon, None)
    return float(np.sum((actual_values - expected_values) * np.log(actual_values / expected_values)))


def _build_scorecard_scale(model: LRScorecardModel) -> pd.DataFrame:
    """Build the compact score-scale table consumed only by the Excel report."""
    card = model.scorecard
    intercept = float(card.intercept_)
    factor = float(card.factor)
    offset = float(card.offset)
    final_feature_count = int(card.n_features_)
    base_allocation = (offset - factor * intercept) / final_feature_count
    rows = [
        ("factor", factor, "概率到评分的比例系数"),
        ("ScoreCard offset", offset, "概率到评分的全局偏移"),
        ("LR截距", intercept, "最终WOE-LR模型截距"),
        ("最终变量数", final_feature_count, "最终进入评分卡的原始变量数量"),
        ("每变量基础分摊", base_allocation, "(offset - factor × LR截距) / 最终变量数"),
    ]
    return pd.DataFrame(rows, columns=["参数", "取值", "说明"])


def build_scorecard_feature_summary(
    model: LRScorecardModel,
    review: pd.DataFrame,
    feature_map: Mapping[str, str],
) -> pd.DataFrame:
    """Build the compact final-feature statistics shared by Excel and training JSON."""
    feature_names = list(model.feature_names_)
    final_review = review.loc[review["变量名称"].isin(feature_names)].copy()
    final_review = final_review.set_index("变量名称").reindex(feature_names)
    coefficients = pd.Series(
        np.abs(model.logistic_model.coef_[0]),
        index=list(model.logistic_model.feature_names_in_),
    )
    importance = coefficients / coefficients.sum() if coefficients.sum() else coefficients
    lr_summary = model.logistic_model.summary()
    return pd.DataFrame(
        {
            "特征名": feature_names,
            "变量含义": [feature_map.get(feature, "") for feature in feature_names],
            "特征重要性": [float(importance.get(feature, np.nan)) for feature in feature_names],
            "IV": [float(final_review.loc[feature, "训练IV"]) for feature in feature_names],
            "PSI": [float(final_review.loc[feature, "主验证集分箱PSI"]) for feature in feature_names],
            "缺失率": [float(final_review.loc[feature, "缺失箱占比"]) for feature in feature_names],
            "坏率形状": [str(final_review.loc[feature, "坏率形状"]) for feature in feature_names],
            "Coef.": [float(lr_summary.loc[feature, "Coef."]) for feature in feature_names],
            "P>|z|": [float(lr_summary.loc[feature, "P>|z|"]) for feature in feature_names],
            "VIF": [float(lr_summary.loc[feature, "VIF"]) for feature in feature_names],
        }
    )


class _ReportScorecardProxy:
    """Expose final-feature metadata and concise parameters to QuickModelReport."""

    def __init__(
        self,
        model: LRScorecardModel,
        points: pd.DataFrame,
        scale: pd.DataFrame,
    ):
        self.source = model.scorecard
        self.feature_names_ = np.asarray(model.feature_names_, dtype=object)
        self.feature_names_in_ = self.feature_names_.copy()
        self.pretrain_lr = model.logistic_model
        self._points = points
        self._scale = scale
        self.parameters_ = deepcopy(model.parameters_)

    @property
    def coef_(self) -> np.ndarray:
        return np.asarray(self.source.coef_).reshape(1, -1)

    def predict(self, frame: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        return self.source.predict(frame, **kwargs)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return self.source.predict_proba(frame)

    def scorecard_points(self, feature_map: Optional[Mapping[str, str]] = None) -> pd.DataFrame:
        return self._points.copy()

    def scorecard_scale(self) -> pd.DataFrame:
        return self._scale.copy()

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        binning = _effective_binning_parameters(self.parameters_)
        return {
            "候选特征数": len(getattr(self, "candidate_features_", [])),
            "最终入模特征数": len(self.feature_names_),
            "分箱": {
                key: binning.get(key)
                for key in ("method", "max_n_bins", "min_bin_size", "empty_separate", "constraint_mode")
            },
            "WOE后细筛": {
                key: self.parameters_["fine_selection"].get(key)
                for key in ("iv", "corr", "empty", "identical")
            },
            "逐步回归": {
                key: self.parameters_["stepwise"].get(key)
                for key in ("direction", "criterion", "p_value_enter", "p_remove", "p_enter")
            },
            "逻辑回归": {
                key: self.parameters_["logistic"].get(key)
                for key in ("penalty", "C", "solver", "max_iter", "class_weight")
            },
            "评分刻度": dict(self.parameters_["score_scale"]),
        }


class _ScorecardQuickModelReport(QuickModelReport):
    """QuickModelReport adapter that always reuses the trained scorecard bins."""

    def __init__(
        self,
        scorecard_model: LRScorecardModel,
        datasets: Mapping[str, Tuple[pd.DataFrame, pd.Series]],
        detail: pd.DataFrame,
        review: pd.DataFrame,
        points: pd.DataFrame,
        scale: pd.DataFrame,
        feature_map: Mapping[str, str],
    ):
        self.scorecard_model = scorecard_model
        self.binning_detail = detail
        self.binning_review = review
        self.feature_map = dict(feature_map)
        proxy = _ReportScorecardProxy(scorecard_model, points, scale)
        proxy.candidate_features_ = list(scorecard_model.candidate_features_)
        super().__init__(
            model=proxy,
            datasets=dict(datasets),
            target=scorecard_model.scorecard.target,
            feature_names=list(scorecard_model.feature_names_),
        )
        self.feature_names = list(scorecard_model.feature_names_)
        labels = {
            "train": "训练集",
            "test": "测试集",
            "oot_main": "主OOT",
        }
        for key, dataset in self._datasets.items():
            dataset.label = labels.get(key, key)
            self._datasets_info[key] = dataset.label

    def get_features_describe(self) -> pd.DataFrame:
        summary = build_scorecard_feature_summary(
            self.scorecard_model,
            self.binning_review,
            self.feature_map,
        )
        return summary.drop(columns=["VIF"]).rename(
            columns={"Coef.": "LR系数", "P>|z|": "LR p值"}
        )

    def get_bin_table(
        self,
        dataset: str = "train",
        method: str = "quantile",
        max_n_bins: int = 10,
        amount_col: Optional[str] = None,
        margins: bool = True,
        rules: Optional[List[Any]] = None,
    ) -> pd.DataFrame:
        """Build score bands with the correct direction: a higher score means lower risk."""
        from scorecardpipeline.processing import feature_bin_stats

        report_dataset = self._datasets[dataset]
        target_col = "__target__"
        score_col = "__score__"
        frame = report_dataset.X.copy()
        frame[target_col] = report_dataset.y.values
        frame[score_col] = report_dataset.score
        parameters: Dict[str, Any] = {
            "data": frame,
            "feature": score_col,
            "target": target_col,
            "method": method,
            "desc": "模型评分",
            "max_n_bins": max_n_bins,
            "margins": margins,
            "greater_is_better": True,
            "return_cols": [
                "分箱",
                "样本总数",
                "好样本数",
                "坏样本数",
                "样本占比",
                "好样本占比",
                "坏样本占比",
                "坏样本率",
                "LIFT值",
                "累积LIFT值",
                "坏账改善",
                "累积坏账改善",
                "分档KS值",
            ],
        }
        if amount_col and amount_col in frame.columns:
            parameters["amount"] = amount_col
        if rules is not None:
            parameters["rules"] = rules
        table = feature_bin_stats(**parameters)
        return table[0] if isinstance(table, tuple) else table

    def get_bin_table_rules(
        self,
        dataset: str = "train",
        method: str = "quantile",
        max_n_bins: int = 10,
    ) -> Tuple[pd.DataFrame, List[Any]]:
        """Return train score-band rules while keeping the score KS direction explicit."""
        from scorecardpipeline.processing import feature_bin_stats

        report_dataset = self._datasets[dataset]
        target_col = "__target__"
        score_col = "__score__"
        frame = report_dataset.X.copy()
        frame[target_col] = report_dataset.y.values
        frame[score_col] = report_dataset.score
        table, rules = feature_bin_stats(
            data=frame,
            feature=score_col,
            target=target_col,
            method=method,
            desc="模型评分",
            max_n_bins=max_n_bins,
            margins=False,
            greater_is_better=True,
            return_cols=[
                "分箱",
                "样本总数",
                "好样本数",
                "坏样本数",
                "样本占比",
                "好样本占比",
                "坏样本占比",
                "坏样本率",
                "LIFT值",
                "累积LIFT值",
                "坏账改善",
                "累积坏账改善",
                "分档KS值",
            ],
            return_rules=True,
        )
        return table, rules

    def get_feature_bin_table(
        self,
        feature: str,
        dataset: str = "train",
        max_n_bins: int = 10,
        method: str = "quantile",
        margins: bool = True,
        amount_col: Optional[str] = None,
    ) -> pd.DataFrame:
        subset = self.binning_detail.loc[
            (self.binning_detail["变量名称"] == feature)
            & (self.binning_detail["数据集名称"] == dataset)
        ].copy()
        if subset.empty:
            raise KeyError(f"未找到评分卡分箱明细: {feature}/{dataset}")
        table = subset[
            [
                "训练期分箱",
                "样本数",
                "样本占比",
                "好样本数",
                "坏样本数",
                "坏率",
                "训练WOE",
                "训练IV贡献",
            ]
        ].rename(
            columns={
                "训练期分箱": "分箱",
                "样本数": "样本总数",
                "坏率": "坏样本率",
                "训练WOE": "WOE",
                "训练IV贡献": "IV值",
            }
        )
        if margins:
            total_count = int(table["样本总数"].sum())
            total_bad = int(table["坏样本数"].sum())
            total_good = int(table["好样本数"].sum())
            total = pd.DataFrame(
                [
                    {
                        "分箱": "合计",
                        "样本总数": total_count,
                        "样本占比": 1.0,
                        "好样本数": total_good,
                        "坏样本数": total_bad,
                        "坏样本率": total_bad / total_count if total_count else np.nan,
                        "WOE": np.nan,
                        "IV值": float(table["IV值"].sum(skipna=True)),
                    }
                ]
            )
            table = pd.concat([table, total], ignore_index=True)
        return table

    def _get_monthly_psi_matrix(self, date_col: str) -> pd.DataFrame:
        """评分卡报告不展示难以解释且重复的分月两两PSI矩阵。"""
        _ = date_col
        return pd.DataFrame()

    def _get_top_n_lift_table(
        self,
        percentiles: Tuple[float, ...] = (0.01, 0.03, 0.05, 0.10),
        amount_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """沿用包的高坏概率TOP-N口径，并把容易误解的名称改得更准确。"""
        table = super()._get_top_n_lift_table(percentiles=percentiles, amount_col=amount_col)
        if table.shape[1] >= 2:
            metric_col = table.columns[1]
            table[metric_col] = table[metric_col].replace(
                {"坏账改善": "坏样本率相对提升"}
            )
        return table

    def _plot_composite_feature(self, feature: str, output_path: Path) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter

        subset = self.binning_detail.loc[
            self.binning_detail["变量名称"] == feature
        ].copy()
        train = subset.loc[subset["数据集名称"] == "train"].copy()
        if train.empty:
            return
        bin_indexes = train["分箱序号"].tolist()
        bin_labels = train["训练期分箱"].astype(str).tolist()
        x = np.arange(len(bin_indexes))
        width = min(24.0, max(10.0, 1.5 * len(bin_indexes) + 5.0))
        fig, left_axis = plt.subplots(figsize=(width, 6.5))
        right_axis = left_axis.twinx()
        bars = right_axis.bar(
            x,
            train["样本占比"].to_numpy(dtype=float),
            width=0.62,
            color="#D9E2F3",
            edgecolor="#4472C4",
            alpha=0.65,
            label="训练集样本占比",
            zorder=1,
        )
        colors = ["#C00000", "#ED7D31", "#70AD47", "#7030A0", "#00B0F0", "#A5A5A5"]
        for color, dataset_name in zip(colors, self._datasets.keys()):
            dataset_rows = subset.loc[subset["数据集名称"] == dataset_name].set_index("分箱序号")
            rates = [dataset_rows.loc[index, "坏率"] if index in dataset_rows.index else np.nan for index in bin_indexes]
            label = self._datasets[dataset_name].label
            left_axis.plot(x, rates, marker="o", linewidth=2, color=color, label=f"{label}坏率", zorder=3)
        left_axis.set_ylabel("坏率")
        right_axis.set_ylabel("训练集样本占比")
        left_axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        right_axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        left_axis.set_xticks(x)
        left_axis.set_xticklabels(bin_labels, rotation=35, ha="right")
        left_axis.set_xlabel("训练期评分卡分箱")
        left_axis.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
        title = feature
        if self.feature_map.get(feature):
            title += f"（{self.feature_map[feature]}）"
        left_axis.set_title(f"{title} 分箱表现")
        left_handles, left_labels = left_axis.get_legend_handles_labels()
        right_handles, right_labels = right_axis.get_legend_handles_labels()
        left_axis.legend(
            left_handles + right_handles,
            left_labels + right_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.22),
            ncol=min(4, len(left_labels) + len(right_labels)),
            frameon=False,
        )
        for bar, share in zip(bars, train["样本占比"]):
            right_axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{share:.1%}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _export_plots(
        self,
        output_dir: Path,
        n_bins: int = 10,
        bin_method: str = "quantile",
        amount_col: Optional[str] = None,
    ) -> Tuple[Dict[str, List[str]], Dict[str, pd.DataFrame]]:
        from scorecardpipeline.utils import bin_plot, corr_plot, ks_plot
        import matplotlib.pyplot as plt

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths: Dict[str, List[str]] = {}
        for dataset_name, dataset in self._datasets.items():
            figures: List[str] = []
            try:
                table = self.get_bin_table(dataset_name, method=bin_method, max_n_bins=n_bins, margins=True)
                plot_table = table.iloc[:-1].reset_index(drop=True) if len(table) > 1 else table
                path = str(output_dir / f"score_bins_{dataset_name}.png")
                figure = bin_plot(
                    plot_table,
                    desc="模型评分",
                    ending=f" {dataset.label}",
                    save=None,
                    figsize=(14, 8),
                    orientation="vertical",
                )
                for axis in figure.axes:
                    bottom, top = axis.get_ylim()
                    if np.isfinite(bottom) and np.isfinite(top) and top > bottom:
                        axis.set_ylim(bottom, bottom + (top - bottom) * 1.12)
                figure.savefig(path, dpi=150, bbox_inches="tight")
                plt.close("all")
                figures.append(path)
            except Exception as exc:
                print(f"评分分箱图生成失败但继续生成报告: {dataset_name}: {exc}")
            try:
                path = str(output_dir / f"ks_{dataset_name}.png")
                ks_plot(dataset.score, dataset.y, title=f"{dataset.label} KS曲线", save=path, figsize=(14, 8))
                plt.close("all")
                figures.append(path)
            except Exception as exc:
                print(f"KS图生成失败但继续生成报告: {dataset_name}: {exc}")
            if figures:
                paths[f"model_{dataset_name}"] = figures

        try:
            train_dataset = self._datasets.get("train") or next(iter(self._datasets.values()))
            correlation_data = train_dataset.X[self.feature_names].select_dtypes(include=[np.number])
            if correlation_data.shape[1] >= 2:
                path = str(output_dir / "feature_corr.png")
                corr_plot(correlation_data, annot=False, save=path)
                plt.close("all")
                paths["feature_corr"] = [path]
        except Exception as exc:
            print(f"相关性图生成失败但继续生成报告: {exc}")

        for feature in self.feature_names:
            path = output_dir / f"scorecard_bin_{feature}.png"
            try:
                self._plot_composite_feature(feature, path)
                if path.exists():
                    paths[f"feat_bin_{feature}"] = [str(path)]
            except Exception as exc:
                print(f"综合分箱图生成失败但继续生成报告: {feature}: {exc}")

        try:
            path = str(output_dir / "plot_weights.png")
            self.scorecard_model.logistic_model.plot_weights(save=path)
            plt.close("all")
            paths["model_weights"] = [path]
        except Exception as exc:
            print(f"LR权重图生成失败但继续生成报告: {exc}")
        return paths, {}


def resolve_report_context_columns(config: Mapping[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Inherit report date/channel columns from the cleaning/split configuration."""
    _ = config
    fallback = Path("config") / "data_clean_split_config.json"
    if not fallback.exists():
        print(f"未找到清洗切分配置，报告将跳过日期/渠道分析: {fallback}")
        return None, None
    try:
        with fallback.open("r", encoding="utf-8") as stream:
            fallback_settings = json.load(stream).get("settings", {})
        return fallback_settings.get("date_col"), fallback_settings.get("channel_col")
    except Exception as exc:
        print(f"清洗配置中的日期/渠道字段读取失败，报告将跳过相应分析: {exc}")
        return None, None


def parse_report_dates(values: pd.Series) -> pd.Series:
    """Parse common business date encodings without treating YYYYMMDD integers as nanoseconds."""
    if pd.api.types.is_datetime64_any_dtype(values.dtype):
        return pd.to_datetime(values, errors="coerce")

    text = values.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    compact_day = text.str.fullmatch(r"\d{8}", na=False)
    compact_month = text.str.fullmatch(r"\d{6}", na=False)
    if compact_day.any():
        parsed.loc[compact_day] = pd.to_datetime(
            text.loc[compact_day], format="%Y%m%d", errors="coerce"
        )
    if compact_month.any():
        parsed.loc[compact_month] = pd.to_datetime(
            text.loc[compact_month], format="%Y%m", errors="coerce"
        )
    remaining = values.notna() & ~compact_day & ~compact_month
    if remaining.any():
        parsed.loc[remaining] = pd.to_datetime(text.loc[remaining], errors="coerce")
    return parsed


def _report_feature_info(
    model: LRScorecardModel,
    feature_map: Mapping[str, str],
) -> pd.DataFrame:
    rows = []
    for index, feature in enumerate(model.feature_names_, start=1):
        categorical = feature in model.categorical_features_
        rows.append(
            {
                "序号": index,
                "特征名称": feature,
                "特征含义": feature_map.get(feature, ""),
                "字段类型": "类别型/string" if categorical else "数值型/float",
                "缺失值处理": (
                    "空值、空字符串及特殊缺失码恢复为缺失后命中训练期缺失箱"
                    if categorical
                    else "空值及特殊缺失码恢复为NaN后命中训练期缺失箱"
                ),
                "未见类别处理": (
                    "训练未见类别按该变量最低箱分保守兜底" if categorical else "不适用"
                ),
            }
        )
    return pd.DataFrame(rows)


def _prediction_examples(
    model: LRScorecardModel,
    raw_frame: pd.DataFrame,
    sample_size: int = 5,
) -> pd.DataFrame:
    """Create deployment examples by calling the public model API on raw feature values."""
    target = getattr(model.scorecard, "target", None)
    columns = list(model.feature_names_)
    if target and target in raw_frame.columns:
        columns.append(target)
    sample = raw_frame.loc[:, columns].head(min(sample_size, len(raw_frame))).copy()
    if sample.empty:
        return sample
    sample["预测坏概率"] = model.predict_proba(sample)[:, 1]
    sample["预测评分"] = model.predict_score(sample)
    sample["预测分类"] = model.predict(sample)
    return sample.reset_index(drop=True)


def _report_channel_metrics(
    report: _ScorecardQuickModelReport,
    group_col: Optional[str],
) -> pd.DataFrame:
    """Calculate per-channel performance from the report's model probabilities."""
    if not group_col:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for dataset in report._datasets.values():
        if group_col not in dataset.X.columns:
            continue
        groups = dataset.X[group_col].astype(object).where(dataset.X[group_col].notna(), "缺失值")
        target = pd.Series(dataset.y.to_numpy(), index=dataset.X.index)
        probability = pd.Series(dataset.y_proba, index=dataset.X.index)
        for group_value in pd.unique(groups):
            mask = groups == group_value
            group_target = target.loc[mask]
            group_probability = probability.loc[mask].to_numpy(dtype=float)
            has_both_classes = group_target.nunique() == 2
            rows.append(
                {
                    "数据集": dataset.label,
                    "渠道": str(group_value),
                    "样本数": int(mask.sum()),
                    "坏样本数": int(group_target.sum()),
                    "坏样本率": float(group_target.mean()),
                    "AUC": float(roc_auc_score(group_target, group_probability)) if has_both_classes else np.nan,
                    "KS": ks_score(group_target, group_probability) if has_both_classes else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _scorecard_feature_psi_table(
    model: LRScorecardModel,
    datasets: Mapping[str, Tuple[pd.DataFrame, pd.Series]],
) -> pd.DataFrame:
    """Calculate feature PSI with Train-fitted scorecard bins for every validation set."""
    if "train" not in datasets or len(datasets) < 2:
        return pd.DataFrame()
    labels = {"test": "测试集", "oot_main": "主OOT"}
    binned = {
        name: model.combiner.transform(frame.loc[:, model.candidate_features_])
        for name, (frame, _) in datasets.items()
    }
    rows: List[Dict[str, Any]] = []
    for feature in model.feature_names_:
        train_distribution = binned["train"][feature].value_counts(normalize=True, dropna=False)
        row: Dict[str, Any] = {"特征": feature}
        for name in datasets:
            if name == "train":
                continue
            actual_distribution = binned[name][feature].value_counts(normalize=True, dropna=False)
            row[f"PSI({labels.get(name, name)})"] = _population_psi(
                train_distribution, actual_distribution
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _build_score_psi_tables(
    report: _ScorecardQuickModelReport,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Build score PSI tables with the LightGBM Train-fitted probability bands.

    Train-vs-validation values come directly from ``utils1.Psi`` so that the
    scorecard and LightGBM branches share the same qcut, smoothing and clipping
    behaviour.  The remaining matrix cells compare distributions after all
    datasets have been mapped to those same Train bands.
    """
    if "train" not in report._datasets or len(report._datasets) < 2:
        return {"matrix": pd.DataFrame(), "train_totals": {}, "rules": [], "tables": {}}

    train_table, rules = report.get_bin_table_rules(
        "train", method="quantile", max_n_bins=n_bins
    )
    tables: Dict[str, pd.DataFrame] = {"train": train_table.copy()}
    for dataset_name in report._datasets:
        if dataset_name == "train":
            continue
        tables[dataset_name] = report.get_bin_table(
            dataset_name,
            method="quantile",
            max_n_bins=n_bins,
            margins=False,
            rules=rules,
        )

    train_probability = np.asarray(report._datasets["train"].y_proba, dtype=float)
    lightgbm_psi = Psi(q=n_bins)
    lightgbm_psi.fit(train_probability)
    probability_bins = np.asarray(lightgbm_psi.bins, dtype=float)

    distributions: Dict[str, pd.Series] = {}
    for dataset_name, dataset in report._datasets.items():
        probability = np.asarray(dataset.y_proba, dtype=float)
        binned = pd.cut(probability, bins=probability_bins, include_lowest=True)
        counts = pd.Series(binned).value_counts(sort=False).astype(float)
        distributions[dataset_name] = counts / counts.sum()

    dataset_names = list(report._datasets)
    labels = [report._datasets[name].label for name in dataset_names]
    matrix = pd.DataFrame(0.0, index=labels, columns=labels, dtype=float)
    for left_index, left_name in enumerate(dataset_names):
        for right_index, right_name in enumerate(dataset_names):
            if left_index == right_index:
                continue
            matrix.iloc[left_index, right_index] = _population_psi(
                distributions[left_name], distributions[right_name]
            )

    train_label = report._datasets["train"].label
    train_totals: Dict[str, float] = {}
    extreme_ratios: Dict[str, float] = {}
    for dataset_name in dataset_names:
        if dataset_name == "train":
            continue
        probability = np.asarray(report._datasets[dataset_name].y_proba, dtype=float)
        psi_value, extreme_ratio = lightgbm_psi.transform(probability)
        label = report._datasets[dataset_name].label
        train_totals[dataset_name] = float(psi_value)
        extreme_ratios[dataset_name] = float(extreme_ratio)
        matrix.loc[train_label, label] = float(psi_value)
        matrix.loc[label, train_label] = float(psi_value)
    return {
        "matrix": matrix,
        "train_totals": train_totals,
        "train_totals_by_label": {
            report._datasets[name].label: value for name, value in train_totals.items()
        },
        "extreme_ratios": extreme_ratios,
        "probability_bins": probability_bins.tolist(),
        "rules": rules,
        "tables": tables,
    }


def _build_training_score_ranking_summary(
    report: _ScorecardQuickModelReport,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Build the exact and grouped KS values displayed in the training report."""
    train_dataset = report._datasets["train"]
    target = pd.Series(train_dataset.y).reset_index(drop=True)
    bad_probability = np.asarray(train_dataset.y_proba, dtype=float)
    score = np.asarray(train_dataset.score, dtype=float)
    probability_ks = ks_score(target, bad_probability)
    score_ks = ks_score(target, -score)
    quantile_table = report.get_bin_table(
        "train", method="quantile", max_n_bins=n_bins, margins=True
    )
    def _grouped_ks(table: pd.DataFrame) -> float:
        values = pd.to_numeric(
            table.loc[table["分箱"].astype(str) != "合计", "分档KS值"], errors="coerce"
        )
        return float(values.max())

    return {
        "exact_probability_ks": probability_ks,
        "exact_score_ks": score_ks,
        "quantile_grouped_ks": _grouped_ks(quantile_table),
    }


def _report_parameter_table(model: LRScorecardModel) -> pd.DataFrame:
    """Show the user-facing controls and the key fixed modeling defaults."""
    adjusted_rules = model.parameters_["binning"].get("adj_rules") or {}
    binning = _effective_binning_parameters(model.parameters_)
    fine = model.parameters_["fine_selection"]
    stepwise = model.parameters_["stepwise"]
    logistic = model.parameters_["logistic"]
    scale = model.parameters_["score_scale"]

    rows = [
        ("人工分箱", "adj_rules", f"{len(adjusted_rules)}个变量", "可修改；覆盖指定变量的自动分箱，完整规则见model_train_spec_lrscorecard.json。"),
        ("自动分箱", "method", binning["method"], "分箱算法；可通过binning_method选择chi、dt、quantile或step。"),
        ("自动分箱", "max_n_bins", binning["max_n_bins"], "当前方法传入的最大箱数；None表示沿用该方法的包默认行为。"),
        ("自动分箱", "min_bin_size", binning["min_bin_size"], "当前Combiner默认的常规箱最小样本占比；quantile和step不使用该参数。"),
        ("WOE后细筛", "iv", fine["iv"], "训练IV下限，低于阈值的变量删除；系统默认，不支持用户修改。"),
        ("WOE后细筛", "corr", fine["corr"], "WOE特征相关性上限；可通过fine_selection_corr修改。"),
        ("WOE后细筛", "identical", fine["identical"], "单一WOE值占比上限；系统默认，不支持用户修改。"),
        ("双向逐步回归", "direction", stepwise["direction"], "both表示前向加入并回看已选变量；系统默认，不支持用户修改。"),
        ("双向逐步回归", "criterion", stepwise["criterion"], "信息准则，可通过stepwise_criterion选择AIC或BIC。"),
        ("双向逐步回归", "p_value_enter", stepwise["p_value_enter"], "每次加入变量后，用该p值上限回看并剔除已选变量；系统默认，不支持用户修改。"),
        ("双向逐步回归", "p_enter", stepwise["p_enter"], "新变量带来的信息准则改善须超过该阈值；系统默认，不支持用户修改。"),
        ("逻辑回归", "penalty", logistic["penalty"], "正则化类型，l2表示L2正则；系统默认，不支持用户修改。"),
        ("逻辑回归", "C", logistic["C"], "正则强度的倒数，越小正则越强；系统默认，不支持用户修改。"),
        ("逻辑回归", "solver", logistic["solver"], "逻辑回归数值求解器；系统默认，不支持用户修改。"),
        ("评分刻度", "base_score", scale["base_score"], "可修改；基准分，只改变评分刻度，不改变坏概率、AUC和KS。"),
        ("评分刻度", "pdo", scale["pdo"], "可修改；赔率翻倍时变化的分值，越小评分跨度通常越紧凑。"),
        ("评分刻度", "base_odds", scale["base_odds"], "可修改；基准分对应的好坏赔率，用于移动评分基准位置。"),
    ]
    return pd.DataFrame(rows, columns=["参数组", "参数", "当前值", "备注"])


def _annotate_scorecard_report(
    report_path: Path,
    prediction_examples: pd.DataFrame,
    channel_metrics: pd.DataFrame,
    feature_psi: pd.DataFrame,
    score_psi_data: Mapping[str, Any],
    score_ranking_summary: Mapping[str, Any],
    parameter_table: pd.DataFrame,
) -> None:
    """Apply project-specific content and layout corrections to the package report."""
    from copy import copy
    import re
    from openpyxl.styles import Alignment, Font
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.cell_range import CellRange
    from scorecardpipeline.excel_writer import ExcelWriter, dataframe2excel

    writer = ExcelWriter(style_excel=str(report_path), style_sheet_name="目录")
    temporary_style_sheet = writer.workbook.create_sheet("__scorecard_report_style__")
    writer.style_sheet = temporary_style_sheet

    def _find_text_row(sheet: Any, text: str, start_row: int = 1) -> Optional[int]:
        for row in sheet.iter_rows(min_row=start_row):
            for cell in row:
                if cell.value is not None and text in str(cell.value):
                    return cell.row
        return None

    def _shift_rows_with_layout(sheet: Any, index: int, amount: int) -> None:
        merged = [CellRange(str(value)) for value in list(sheet.merged_cells.ranges)]
        for value in list(sheet.merged_cells.ranges):
            sheet.unmerge_cells(str(value))
        sheet.insert_rows(index, amount)
        for cell_range in merged:
            if cell_range.min_row >= index:
                cell_range.shift(row_shift=amount)
            elif cell_range.max_row >= index:
                cell_range.max_row += amount
            sheet.merge_cells(str(cell_range))
        for image in getattr(sheet, "_images", []):
            anchor = getattr(image, "anchor", None)
            if hasattr(anchor, "_from") and anchor._from.row + 1 >= index:
                anchor._from.row += amount
                if hasattr(anchor, "to"):
                    anchor.to.row += amount

    def _delete_rows_with_layout(sheet: Any, index: int, amount: int) -> None:
        end_index = index + amount - 1
        merged = [CellRange(str(value)) for value in list(sheet.merged_cells.ranges)]
        for value in list(sheet.merged_cells.ranges):
            sheet.unmerge_cells(str(value))
        sheet.delete_rows(index, amount)
        for cell_range in merged:
            if cell_range.max_row < index:
                sheet.merge_cells(str(cell_range))
            elif cell_range.min_row > end_index:
                cell_range.shift(row_shift=-amount)
                sheet.merge_cells(str(cell_range))
        kept_images = []
        for image in getattr(sheet, "_images", []):
            anchor = getattr(image, "anchor", None)
            if not hasattr(anchor, "_from"):
                kept_images.append(image)
                continue
            anchor_row = anchor._from.row + 1
            if index <= anchor_row <= end_index:
                continue
            if anchor_row > end_index:
                anchor._from.row -= amount
                if hasattr(anchor, "to"):
                    anchor.to.row -= amount
            kept_images.append(image)
        sheet._images = kept_images

    def _renumber_sections(sheet: Any) -> None:
        section = 1
        pattern = re.compile(r"^\d+、")
        for row in range(1, sheet.max_row + 1):
            if sheet.row_dimensions[row].hidden:
                continue
            cell = sheet.cell(row, 2)
            if isinstance(cell.value, str) and pattern.match(cell.value):
                cell.value = pattern.sub(f"{section}、", cell.value)
                section += 1

    def _shrink_title_band(sheet: Any, end_column: int) -> None:
        """Limit the package's B:AJ page-title merge without touching wide data tables."""
        for value in list(sheet.merged_cells.ranges):
            cell_range = CellRange(str(value))
            if (
                cell_range.min_row == 2
                and cell_range.max_row == 2
                and cell_range.min_col == 2
                and cell_range.max_col > end_column
            ):
                sheet.unmerge_cells(str(value))
                sheet.merge_cells(
                    start_row=2,
                    start_column=2,
                    end_row=2,
                    end_column=end_column,
                )

    def _unmerge_overlapping(
        sheet: Any,
        row: int,
        min_column: int,
        max_column: int,
    ) -> None:
        for value in list(sheet.merged_cells.ranges):
            cell_range = CellRange(str(value))
            if (
                cell_range.min_row <= row <= cell_range.max_row
                and cell_range.max_col >= min_column
                and cell_range.min_col <= max_column
            ):
                sheet.unmerge_cells(str(value))

    # 目录不再声称报告包含CSI；当前报告没有计算或展示CSI。
    contents_sheet = writer.get_sheet_by_name("目录")
    for row in contents_sheet.iter_rows():
        for cell in row:
            if isinstance(cell.value, str) and "CSI" in cell.value:
                cell.value = cell.value.replace("、CSI", "").replace("CSI、", "")

    # 基本信息只使用少量列，收紧标题，并把长统计内容分配到三列宽度。
    basic_sheet = writer.get_sheet_by_name("1-基本信息")
    _shrink_title_band(basic_sheet, 8)
    for value in list(basic_sheet.merged_cells.ranges):
        cell_range = CellRange(str(value))
        if cell_range.min_row != 2 and cell_range.min_col == 2 and cell_range.max_col > 8:
            basic_sheet.unmerge_cells(str(value))
            basic_sheet.merge_cells(
                start_row=cell_range.min_row,
                start_column=2,
                end_row=cell_range.max_row,
                end_column=8,
            )
    for column in range(9, max(basic_sheet.max_column, 35) + 1):
        basic_sheet.column_dimensions[get_column_letter(column)].hidden = True

    stats_header_row = None
    for row in range(1, basic_sheet.max_row + 1):
        if (
            basic_sheet.cell(row, 2).value == "统计项"
            and basic_sheet.cell(row, 3).value == "统计内容"
        ):
            stats_header_row = row
            break
    if stats_header_row is not None:
        stats_end_row = stats_header_row
        while (
            stats_end_row + 1 <= basic_sheet.max_row
            and basic_sheet.cell(stats_end_row + 1, 2).value not in (None, "")
        ):
            stats_end_row += 1
        role_map = {
            "样本时间范围": "全部已加载数据集的最早至最晚业务日期。",
            "整体样本概况": "全部已加载数据集合并汇总，仅用于总体描述。",
            "训练集": "唯一用于拟合分箱、WOE、细筛、逐步回归和LR参数。",
            "测试集": "清洗切分固定留出的内部测试集；只复用Train规则，用于检验泛化能力。",
            "主OOT": "可选的主要独立/时间外验证集；存在时作为主验证集，不参与拟合。",
            "标签口径": "后续所有坏率、AUC、KS和分箱统计均使用同一标签口径。",
        }
        for row in range(stats_header_row, stats_end_row + 1):
            _unmerge_overlapping(basic_sheet, row, 3, 8)
            item_cell = basic_sheet.cell(row, 2)
            content_cell = basic_sheet.cell(row, 3)
            if row == stats_header_row:
                role = "数据集作用/口径说明"
            else:
                item = str(item_cell.value or "")
                content = str(content_cell.value or "")
                if item == "样本区间":
                    item = "样本时间范围"
                elif item == "整体样本":
                    item = "整体样本概况"
                    content = re.sub(
                        r"^\d{4}-\d{2}-\d{2}\s*~\s*\d{4}-\d{2}-\d{2}\s*",
                        "",
                        content,
                    )
                elif item == "数据源":
                    item = "标签口径"
                    content = "1=坏客户，0=好客户；坏率=坏样本数/样本数。"
                content = content.replace("TARGET:", "坏率:")
                item_cell.value = item
                content_cell.value = content
                if item.startswith("oot_"):
                    role = "可选辅助OOT；只展示效果和稳定性，不参与拟合、筛选或主验证判断。"
                else:
                    role = role_map.get(item, "")
            basic_sheet.merge_cells(
                start_row=row, start_column=3, end_row=row, end_column=5
            )
            basic_sheet.merge_cells(
                start_row=row, start_column=6, end_row=row, end_column=8
            )
            role_cell = basic_sheet.cell(row, 6)
            role_cell.value = role
            role_cell._style = copy(content_cell._style)
            role_cell.alignment = copy(content_cell.alignment)
            content_cell.alignment = copy(content_cell.alignment)
            content_cell.alignment = content_cell.alignment.copy(
                horizontal="left", wrap_text=True
            )
            role_cell.alignment = role_cell.alignment.copy(
                horizontal="left", wrap_text=True
            )
            if row > stats_header_row:
                basic_sheet.row_dimensions[row].height = 30
        for column in range(3, 9):
            basic_sheet.column_dimensions[get_column_letter(column)].width = 15

    performance_sheet = writer.get_sheet_by_name("2-模型性能")
    _shrink_title_band(performance_sheet, 10)
    score_psi_matrix = score_psi_data.get("matrix", pd.DataFrame())
    if isinstance(score_psi_matrix, pd.DataFrame) and not score_psi_matrix.empty:
        metrics_psi_row = _find_text_row(performance_sheet, "PSI")
        if metrics_psi_row is not None and performance_sheet.cell(metrics_psi_row, 2).value == "PSI":
            header_row = metrics_psi_row - 5
            for column in range(3, performance_sheet.max_column + 1):
                label = performance_sheet.cell(header_row, column).value
                if label in score_psi_matrix.columns:
                    value = (
                        "\\"
                        if label == "训练集"
                        else float(score_psi_matrix.loc["训练集", label])
                    )
                    performance_sheet.cell(metrics_psi_row, column).value = value
                    if value != "\\":
                        performance_sheet.cell(metrics_psi_row, column).number_format = "0.0000"

    top_row = _find_text_row(performance_sheet, "模型尾部区分能力")
    if top_row is not None and not channel_metrics.empty:
        inserted_rows = len(channel_metrics) + 6
        _shift_rows_with_layout(performance_sheet, top_row, inserted_rows)
        writer.insert_value2sheet(
            performance_sheet,
            (top_row, 2),
            value="分渠道模型效果",
            style="header_middle",
            align={"horizontal": "left"},
        )
        channel_end_row, _ = dataframe2excel(
            channel_metrics,
            writer,
            sheet_name=performance_sheet,
            start_row=top_row + 2,
            percent_cols=["坏样本率"],
            left_cols=["数据集", "渠道"],
        )
        writer.insert_value2sheet(
            performance_sheet,
            (channel_end_row + 1, 2),
            value="说明：渠道内同时存在好、坏样本时才计算AUC和KS；单一标签渠道显示为空。",
            style="middle",
            end_space=(channel_end_row + 1, 10),
            align={"horizontal": "left", "wrap_text": True},
        )
        top_row += inserted_rows

    if top_row is not None:
        next_section_row = None
        for row in range(top_row + 1, performance_sheet.max_row + 1):
            value = performance_sheet.cell(row, 2).value
            if isinstance(value, str) and re.match(r"^\d+、", value):
                next_section_row = row
                break
        if next_section_row:
            note_row = next_section_row - 1
            writer.insert_value2sheet(
                performance_sheet,
                (note_row, 2),
                value=(
                    "TOP n%说明：按预测坏概率从高到低排序（等价于评分从低到高），截取最高风险的前n%；"
                    "坏样本率为该子集实际坏样本占比；LIFT=该子集坏率/全体坏率；"
                    "坏样本率相对提升=LIFT-1，表示高风险子集相对整体多聚集了多少坏样本。"
                ),
                style="middle",
                end_space=(note_row, 12),
                align={"horizontal": "left", "wrap_text": True},
            )
            performance_sheet.row_dimensions[note_row].height = 36

    train_ranking_row = _find_text_row(performance_sheet, "训练集评分排序性")
    if train_ranking_row is not None:
        performance_sheet.cell(train_ranking_row, 2).value = re.sub(
            r"（等频10组）$", "", str(performance_sheet.cell(train_ranking_row, 2).value)
        ) + "（等频10组）"
        next_section_row = None
        for row in range(train_ranking_row + 1, performance_sheet.max_row + 1):
            value = performance_sheet.cell(row, 2).value
            if isinstance(value, str) and re.match(r"^\d+、", value):
                next_section_row = row
                break
        if next_section_row is not None:
            quantile_note_row = next_section_row - 1
            writer.insert_value2sheet(
                performance_sheet,
                (quantile_note_row, 2),
                value=(
                    "KS口径说明：左侧分箱图及下表为等频10组后的分档KS，最大值"
                    f"{float(score_ranking_summary['quantile_grouped_ks']):.2%}；右侧KS曲线和总指标为逐样本精确KS，"
                    f"坏概率与评分反向排序后的结果均为{float(score_ranking_summary['exact_probability_ks']):.2%}。"
                    "分组会损失部分排序细节，因此分档KS可以略低于精确KS。"
                ),
                style="middle",
                end_space=(quantile_note_row, 12),
                align={"horizontal": "left", "wrap_text": True},
            )
            performance_sheet.row_dimensions[quantile_note_row].height = 42
    _renumber_sections(performance_sheet)

    feature_sheet = writer.get_sheet_by_name("3-入模变量分析")
    _shrink_title_band(feature_sheet, 10)
    feature_header_row = None
    feature_headers: Dict[str, int] = {}
    for row in feature_sheet.iter_rows():
        values = {str(cell.value): cell.column for cell in row if cell.value is not None}
        if {"特征名", "IV", "PSI"}.issubset(values):
            feature_header_row = row[0].row
            feature_headers = values
            break
    if feature_header_row is not None:
        data_end_row = feature_header_row
        while (
            data_end_row + 1 <= feature_sheet.max_row
            and feature_sheet.cell(data_end_row + 1, feature_headers["特征名"]).value not in (None, "")
        ):
            data_end_row += 1
        for row in range(feature_header_row + 1, data_end_row + 1):
            feature_sheet.cell(row, feature_headers["IV"]).number_format = "0.0000"
            feature_sheet.cell(row, feature_headers["PSI"]).number_format = "0.0000"
        feature_note_row = data_end_row + 1
        writer.insert_value2sheet(
            feature_sheet,
            (feature_note_row, 2),
            value=(
                "口径说明：本表仅展示最终入模变量。特征重要性、IV、缺失率、坏率形状、LR系数和p值均基于Train；"
                "IV是评分卡卡方训练箱的IV，也是LrScorecard内部细筛使用的IV，不等同于第三阶段等频分箱IV；"
                "PSI为主验证集（优先oot_main，否则Test）相对Train、复用Train评分卡切点计算的分箱PSI，"
                "也不等同于第三阶段特征筛选PSI。"
            ),
            style="middle",
            end_space=(feature_note_row, 10),
            align={"horizontal": "left", "wrap_text": True},
        )
        feature_sheet.row_dimensions[feature_note_row].height = 48

    stability_sheet = writer.get_sheet_by_name("4-稳定性分析")
    _shrink_title_band(stability_sheet, 10)
    score_psi_title_row = _find_text_row(stability_sheet, "评分PSI对比矩阵")
    if (
        score_psi_title_row is not None
        and isinstance(score_psi_matrix, pd.DataFrame)
        and not score_psi_matrix.empty
    ):
        stability_sheet.cell(score_psi_title_row, 2).value = re.sub(
            r"（统一Train等频10组切点）$",
            "",
            str(stability_sheet.cell(score_psi_title_row, 2).value),
        ) + "（统一Train等频10组切点）"
        matrix_header_row = None
        matrix_columns: Dict[str, int] = {}
        for row in range(score_psi_title_row + 1, min(score_psi_title_row + 8, stability_sheet.max_row) + 1):
            values = {
                str(stability_sheet.cell(row, column).value): column
                for column in range(2, stability_sheet.max_column + 1)
                if stability_sheet.cell(row, column).value is not None
            }
            if set(score_psi_matrix.columns).issubset(values):
                matrix_header_row = row
                matrix_columns = values
                break
        if matrix_header_row is not None:
            matrix_rows: Dict[str, int] = {}
            for row in range(matrix_header_row + 1, matrix_header_row + 1 + len(score_psi_matrix)):
                label = stability_sheet.cell(row, 2).value
                if label in score_psi_matrix.index:
                    matrix_rows[str(label)] = row
            for row_label in score_psi_matrix.index:
                excel_row = matrix_rows[row_label]
                for column_label in score_psi_matrix.columns:
                    cell = stability_sheet.cell(excel_row, matrix_columns[column_label])
                    cell.value = float(score_psi_matrix.loc[row_label, column_label])
                    cell.number_format = "0.0000"
            reference_row = _find_text_row(
                stability_sheet, "PSI参考标准", start_row=matrix_header_row
            )
            if reference_row is not None:
                writer.insert_value2sheet(
                    stability_sheet,
                    (reference_row, 2),
                    value=(
                        "说明：全部数据集先复用Train等频10组评分切点统计箱占比，再计算PSI；"
                        "矩阵中Train行/列与后文“模型评分PSI分析”的总体PSI完全一致。"
                        "PSI参考标准：<0.1 稳定 | 0.1~0.25 略变 | >0.25 不稳定。"
                    ),
                    style="middle",
                    end_space=(reference_row, 10),
                    align={"horizontal": "left", "wrap_text": True},
                )
                stability_sheet.row_dimensions[reference_row].height = 42

    drift_row = _find_text_row(stability_sheet, "评分漂移分析")
    feature_psi_title_row = _find_text_row(stability_sheet, "入模特征PSI稳定性")
    if drift_row is not None and feature_psi_title_row is not None and feature_psi_title_row > drift_row:
        _delete_rows_with_layout(
            stability_sheet,
            drift_row,
            feature_psi_title_row - drift_row,
        )
        feature_psi_title_row = drift_row

    if feature_psi_title_row is not None and not feature_psi.empty:
        header_row = None
        for row in range(feature_psi_title_row + 1, min(stability_sheet.max_row, feature_psi_title_row + 10) + 1):
            if stability_sheet.cell(row, 2).value == "特征":
                header_row = row
                break
        if header_row is not None:
            old_end_row = header_row
            while old_end_row + 1 <= stability_sheet.max_row and stability_sheet.cell(old_end_row + 1, 2).value not in (None, ""):
                old_end_row += 1
            last_column = 1 + len(feature_psi.columns)
            for column_offset, column_name in enumerate(feature_psi.columns, start=2):
                stability_sheet.cell(header_row, column_offset).value = column_name
            for row_offset, (_, record) in enumerate(feature_psi.iterrows(), start=1):
                excel_row = header_row + row_offset
                for column_offset, column_name in enumerate(feature_psi.columns, start=2):
                    value = record[column_name]
                    stability_sheet.cell(excel_row, column_offset).value = (
                        None if pd.isna(value) else (value.item() if isinstance(value, np.generic) else value)
                    )
                    if column_name != "特征" and not pd.isna(value):
                        stability_sheet.cell(excel_row, column_offset).number_format = "0.0000"
            new_end_row = header_row + len(feature_psi)
            for row in range(new_end_row + 1, old_end_row + 1):
                for column in range(2, max(stability_sheet.max_column, last_column) + 1):
                    stability_sheet.cell(row, column).value = None
            note_row = new_end_row + 1
            writer.insert_value2sheet(
                stability_sheet,
                (note_row, 2),
                value=(
                    "说明：所有验证集均复用Train评分卡分箱规则计算特征PSI；"
                    "主验证集列与“3-入模变量分析”的PSI口径完全一致。"
                ),
                style="middle",
                end_space=(note_row, max(last_column, 8)),
                align={"horizontal": "left", "wrap_text": True},
            )

    # “模型评分PSI分析”沿用同一Train评分切点，并把展示精度统一为四位小数。
    expected_psi_by_label = score_psi_data.get("train_totals_by_label", {})
    for row in range(1, stability_sheet.max_row + 1):
        if not (
            isinstance(stability_sheet.cell(row, 2).value, str)
            and "评分PSI(" in stability_sheet.cell(row, 2).value
            and "vs 训练集" in stability_sheet.cell(row, 2).value
        ):
            continue
        table_header_row = None
        for candidate in range(row + 1, min(row + 8, stability_sheet.max_row) + 1):
            headers = {
                str(stability_sheet.cell(candidate, column).value): column
                for column in range(2, stability_sheet.max_column + 1)
                if stability_sheet.cell(candidate, column).value is not None
            }
            if "总体PSI值" in headers:
                table_header_row = candidate
                break
        if table_header_row is None:
            continue
        headers = {
            str(stability_sheet.cell(table_header_row, column).value): column
            for column in range(2, stability_sheet.max_column + 1)
            if stability_sheet.cell(table_header_row, column).value is not None
        }
        title = str(stability_sheet.cell(row, 2).value)
        label_match = re.search(r"评分PSI\((.+?)\s+vs\s+训练集\)", title)
        expected_total = (
            expected_psi_by_label.get(label_match.group(1)) if label_match else None
        )
        total_cell = stability_sheet.cell(
            table_header_row + 1, headers["总体PSI值"]
        )
        if expected_total is not None and total_cell.value not in (None, ""):
            total_cell.value = float(expected_total)
        data_row = table_header_row + 1
        while data_row <= stability_sheet.max_row and stability_sheet.cell(data_row, 2).value not in (None, ""):
            for column_name, column in headers.items():
                if "PSI" in column_name or "占比" in column_name:
                    stability_sheet.cell(data_row, column).number_format = "0.0000"
            data_row += 1
    _renumber_sections(stability_sheet)

    parameter_sheet = writer.get_sheet_by_name("5-模型参数")
    _shrink_title_band(parameter_sheet, 10)
    parameter_title_row = _find_text_row(parameter_sheet, "2、模型参数")
    feature_list_row = _find_text_row(parameter_sheet, "3、入模特征列表")
    if (
        parameter_title_row is not None
        and feature_list_row is not None
        and feature_list_row > parameter_title_row
        and not parameter_table.empty
    ):
        content_start_row = parameter_title_row + 1
        _delete_rows_with_layout(
            parameter_sheet,
            content_start_row,
            feature_list_row - content_start_row,
        )
        rows_to_insert = len(parameter_table) + 4
        _shift_rows_with_layout(parameter_sheet, content_start_row, rows_to_insert)
        parameter_end_row, _ = dataframe2excel(
            parameter_table,
            writer,
            sheet_name=parameter_sheet,
            start_row=parameter_title_row + 2,
            left_cols=["参数组", "参数", "当前值", "备注"],
        )
        for column, width in ((2, 18), (3, 26), (4, 18), (5, 72)):
            parameter_sheet.column_dimensions[get_column_letter(column)].width = width
        for row in range(parameter_title_row + 2, parameter_end_row + 1):
            parameter_sheet.row_dimensions[row].height = 22

    # 入模变量已在“3-入模变量分析”完整展示，参数页不再重复列出。
    feature_list_row = _find_text_row(parameter_sheet, "入模特征列表")
    logistic_result_row = _find_text_row(parameter_sheet, "逻辑回归拟合结果")
    if (
        feature_list_row is not None
        and logistic_result_row is not None
        and logistic_result_row > feature_list_row
    ):
        _delete_rows_with_layout(
            parameter_sheet,
            feature_list_row,
            logistic_result_row - feature_list_row,
        )
    _renumber_sections(parameter_sheet)

    # 明确策略表对未见类别的生产兜底，避免人工按CSV部署时把未见值当成报错或缺失值。
    points_title_row = _find_text_row(parameter_sheet, "评分卡分值表")
    if points_title_row is not None:
        points_header_row = None
        for row in range(points_title_row + 1, parameter_sheet.max_row + 1):
            values = [parameter_sheet.cell(row, column).value for column in range(1, 11)]
            if "变量名称" in values and "对应分数" in values:
                points_header_row = row
                break
        if points_header_row is not None:
            points_last_row = points_header_row
            for row in range(points_header_row + 1, parameter_sheet.max_row + 1):
                if parameter_sheet.cell(row, 2).value is None:
                    break
                points_last_row = row
            note_row = points_last_row + 2
            _unmerge_overlapping(parameter_sheet, note_row, 2, 9)
            parameter_sheet.merge_cells(
                start_row=note_row,
                start_column=2,
                end_row=note_row,
                end_column=9,
            )
            note_cell = parameter_sheet.cell(note_row, 2)
            note_cell.value = (
                "说明：数值变量的区间覆盖负无穷至正无穷，缺失值命中缺失箱；"
                "类别变量出现训练期未见值时，PKL按该变量最低精确箱分兜底。"
                "按scorecard_points.csv部署时，应取该类别变量已列分箱中的最低“对应分数”兜底，"
                "不得报错，也不得自动当作缺失值。"
            )
            note_cell.alignment = Alignment(wrap_text=True, vertical="top")
            note_cell.font = Font(color="C00000", italic=True)
            parameter_sheet.row_dimensions[note_row].height = 42

    deployment_sheet = writer.get_sheet_by_name("6-模型部署需求")
    _shrink_title_band(deployment_sheet, 10)
    test_header_row = None
    test_headers: Dict[str, int] = {}
    for row in deployment_sheet.iter_rows():
        values = {str(cell.value): cell.column for cell in row if cell.value is not None}
        if "序号" in values and "模型分数" in values:
            test_header_row = row[0].row
            test_headers = values
            break
    if test_header_row is not None and not prediction_examples.empty:
        output_columns = {"预测坏概率", "预测评分", "预测分类"}
        label_candidates = [
            column
            for column in prediction_examples.columns
            if column not in test_headers and column not in output_columns
        ]
        label_name = label_candidates[0] if label_candidates else "label"
        label_column = test_headers["模型分数"]
        score_column = label_column + 1
        probability_column = label_column + 2
        classification_column = label_column + 3
        template_header = deployment_sheet.cell(test_header_row, label_column)
        template_header.value = label_name
        for column, title in (
            (score_column, "预测评分"),
            (probability_column, "预测坏概率"),
            (classification_column, "预测分类"),
        ):
            header_cell = deployment_sheet.cell(test_header_row, column)
            header_cell.value = title
            header_cell._style = copy(template_header._style)
            header_cell.alignment = copy(template_header.alignment)
        for row_offset, (_, example) in enumerate(prediction_examples.iterrows(), start=1):
            excel_row = test_header_row + row_offset
            for feature in (column for column in prediction_examples.columns if column in test_headers):
                value = example[feature]
                deployment_sheet.cell(excel_row, test_headers[feature]).value = (
                    None if pd.isna(value) else (value.item() if isinstance(value, np.generic) else value)
                )
            label_value = example.get(label_name)
            label_cell = deployment_sheet.cell(excel_row, label_column)
            label_cell.value = (
                None
                if pd.isna(label_value)
                else (label_value.item() if isinstance(label_value, np.generic) else label_value)
            )
            score_cell = deployment_sheet.cell(excel_row, score_column)
            score_cell.value = float(example["预测评分"])
            score_cell._style = copy(label_cell._style)
            probability_cell = deployment_sheet.cell(excel_row, probability_column)
            probability_cell.value = float(example["预测坏概率"])
            probability_cell.number_format = "0.000000"
            probability_cell._style = copy(score_cell._style)
            probability_cell.number_format = "0.000000"
            classification_cell = deployment_sheet.cell(excel_row, classification_column)
            classification_cell.value = int(example["预测分类"])
            classification_cell._style = copy(score_cell._style)
    writer.save(str(report_path))


def generate_scorecard_training_report(
    model: LRScorecardModel,
    datasets: Mapping[str, Tuple[pd.DataFrame, pd.Series]],
    report_path: Path,
    feature_map: Mapping[str, str],
    detail: pd.DataFrame,
    review: pd.DataFrame,
    points: pd.DataFrame,
    config: Mapping[str, Any],
    raw_prediction_sample: pd.DataFrame,
) -> Path:
    from scorecardpipeline.utils import init_setting
    import scorecardpipeline.model_report as model_report_module

    report_path.parent.mkdir(parents=True, exist_ok=True)
    init_setting()
    scale = _build_scorecard_scale(model)
    report = _ScorecardQuickModelReport(
        scorecard_model=model,
        datasets=datasets,
        detail=detail,
        review=review,
        points=points,
        scale=scale,
        feature_map=feature_map,
    )
    date_col, group_col = resolve_report_context_columns(config)
    original_period_freq = model_report_module._period_freq

    def _pandas_period_compatible_freq(freq: str) -> str:
        converted = str(original_period_freq(freq))
        if converted.endswith("ME"):
            return converted[:-2] + "M"
        return converted

    def _write_excel(path: Path) -> None:
        report.to_excel(
            str(path),
            n_bins=10,
            bin_method="quantile",
            date_col=date_col,
            date_freq="M",
            group_col=group_col,
            with_plots=True,
            model_name="WOE-LR信用评分卡",
            project_desc="自动分箱、WOE、严格细筛、双向逐步回归和逻辑回归评分卡",
            feature_map=dict(feature_map),
            feature_info=_report_feature_info(model, feature_map),
        )

    def _unused_report_path() -> Path:
        """Return a new fallback path without overwriting an earlier report."""
        index = 1
        while True:
            suffix = "_新" if index == 1 else f"_新_{index}"
            candidate = report_path.with_name(f"{report_path.stem}{suffix}.xlsx")
            if not candidate.exists():
                return candidate
            index += 1

    def _remove_report_assets(path: Path) -> None:
        assets_dir = path.parent / f"{path.stem}_assets"
        if assets_dir.exists():
            shutil.rmtree(assets_dir)

    actual_report_path = report_path
    if report_path.exists():
        try:
            with report_path.open("a+b"):
                pass
        except PermissionError:
            actual_report_path = _unused_report_path()
            print(
                f"目标评分卡报告正在被其他程序占用，改为生成新文件: {actual_report_path}"
            )
    model_report_module._period_freq = _pandas_period_compatible_freq
    try:
        for _ in range(20):
            try:
                _write_excel(actual_report_path)
                break
            except PermissionError:
                _remove_report_assets(actual_report_path)
                actual_report_path = _unused_report_path()
                print(
                    f"目标评分卡报告正在被其他程序占用，改为生成新文件: {actual_report_path}"
                )
        else:
            raise PermissionError("连续20个评分卡报告文件名均不可写")
    finally:
        model_report_module._period_freq = original_period_freq
    if not actual_report_path.exists() or actual_report_path.stat().st_size == 0:
        raise RuntimeError("评分卡Excel报告未成功生成")
    examples = _prediction_examples(model, raw_prediction_sample)
    channel_metrics = _report_channel_metrics(report, group_col)
    feature_psi = _scorecard_feature_psi_table(model, datasets)
    score_psi_results = _build_score_psi_tables(report, n_bins=10)
    score_ranking_summary = _build_training_score_ranking_summary(report, n_bins=10)
    parameter_table = _report_parameter_table(model)
    _annotate_scorecard_report(
        actual_report_path,
        examples,
        channel_metrics,
        feature_psi,
        score_psi_results,
        score_ranking_summary,
        parameter_table,
    )
    for path in {report_path, actual_report_path}:
        assets_dir = path.parent / f"{path.stem}_assets"
        if assets_dir.exists():
            shutil.rmtree(assets_dir)
    return actual_report_path
