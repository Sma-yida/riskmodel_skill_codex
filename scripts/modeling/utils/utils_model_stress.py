"""模型评估的独立压测模块。

本文件只做三件事：
1. 计算基准集的样本、坏样本、KS 和最坏10%累计 Lift；
2. 对一个压测场景重新预测或筛选子群，然后计算同一套指标；
3. 输出简洁的 CSV、图和总览 Excel。

正常分箱直接复用 utils1.sta_groups，保证与原模型评估口径一致。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.api.types import is_numeric_dtype

from .utils1 import Transfer2Score, ks_score, sta_groups


# ==================== 固定配置：无需修改 model_eval_config.json ====================
STRESS_TEST_ENABLED = True
BIN_NUM = 10
MISSING_VALUE = -9999

# Top2变量来源：默认从模型训练产出的特征重要性文件中，按下列字段降序取前2个。
# 如需强制指定变量，可改为例如 ["feature_a", "feature_b"]；设为 None 时自动读取重要性。
TOP2_FEATURES = None
IMPORTANCE_COLUMN = "importance_score_split"
IMPORTANCE_FEATURE_COLUMN = "Feature"

AGE_COLUMN = "age"
EDUCATION_COLUMN = "rhzx_edubackground"
CARD_LIMIT_COLUMN = "rhzx_card_max_credlimit"
EDUCATION_BACHELOR_CODES = {10.0, 20.0}
CARD_LIMIT_THRESHOLD = 30000.0

AGE_SCENARIOS = [
    ("年龄30岁以下", lambda age: age < 30),
    ("年龄30岁及以上", lambda age: age >= 30),
    ("年龄40岁以下", lambda age: age < 40),
    ("年龄40岁及以上", lambda age: age >= 40),
    ("年龄30至40岁", lambda age: (age >= 30) & (age < 40)),
    ("排除30至40岁", lambda age: (age < 30) | (age >= 40)),
]

# 学历：字符串或数值形式的 10 / 20 均归为本科及以上；缺失归入其它学历。
EDUCATION_SCENARIOS = [
    ("本科及以上", lambda education: education.isin(EDUCATION_BACHELOR_CODES)),
    ("其他学历", lambda education: ~education.isin(EDUCATION_BACHELOR_CODES)),
]

# 信用卡额度：缺失值与低于3万元均归入第一组，3万元及以上归入第二组。
CARD_LIMIT_SCENARIOS = [
    ("无信用卡或额度3万以下", lambda card: card.isna() | (card < CARD_LIMIT_THRESHOLD)),
    ("信用卡额度3万及以上", lambda card: card >= CARD_LIMIT_THRESHOLD),
]


def _make_dir(path):
    """创建并返回产物目录。"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _valid_frame(y_true, probability):
    """将标签和预测概率对齐，并去除缺失值。"""
    frame = pd.DataFrame({
        "label": pd.Series(y_true).reset_index(drop=True),
        "probability": pd.Series(probability, dtype=float).reset_index(drop=True),
    })
    return frame.dropna(subset=["label", "probability"])


def _fallback_bins(frame):
    """旧分箱函数无法处理单一标签/单一概率时的最小降级实现。"""
    if frame.empty:
        return pd.DataFrame(columns=["风险组", "样本总数", "坏样本数", "累计Lift"]), 0, "没有可用样本"

    if frame["probability"].nunique() == 1:
        grouped = frame.assign(风险组=1).groupby("风险组", as_index=False).agg(
            样本总数=("label", "size"), 坏样本数=("label", "sum")
        )
        actual_bins = 1
    else:
        groups = pd.qcut(frame["probability"], q=BIN_NUM, duplicates="drop")
        actual_bins = len(groups.cat.categories)
        frame = frame.assign(风险组=actual_bins - groups.cat.codes)
        grouped = frame.groupby("风险组", as_index=False, observed=True).agg(
            样本总数=("label", "size"), 坏样本数=("label", "sum")
        )

    grouped = grouped.sort_values("风险组").reset_index(drop=True)
    overall_bad_rate = frame["label"].mean()
    if overall_bad_rate > 0:
        grouped["累计Lift"] = (
            grouped["坏样本数"].cumsum() / grouped["样本总数"].cumsum() / overall_bad_rate
        )
    else:
        grouped["累计Lift"] = np.nan
    return grouped, actual_bins, "已使用降级分箱"


def build_decile_table(y_true, probability):
    """按风险从高到低做等频十分箱，返回截图所需的三项分箱指标。

    优先调用原模型评估使用的 sta_groups。重复概率时该函数会自动减少实际箱数；
    若它无法处理极端数据，则降级为 qcut，仍尽可能给出结果和提示。
    """
    frame = _valid_frame(y_true, probability)
    if frame.empty:
        return pd.DataFrame(columns=["风险组", "样本总数", "坏样本数", "累计Lift"]), 0, "没有可用样本"

    try:
        source = sta_groups(
            frame["label"],
            frame["probability"],
            bins_num=BIN_NUM,
            labels=list(range(BIN_NUM, 0, -1)),
        )
        result = source.rename(columns={
            "level": "风险组",
            "num": "样本总数",
            "bad": "坏样本数",
            "lift": "累计Lift",
        })[["风险组", "样本总数", "坏样本数", "累计Lift"]]
        result = result.sort_values("风险组").reset_index(drop=True)
        # sta_groups 在实际箱数不足10时会保留 2~10 或 4~10 等原标签。
        # 分箱与 Lift 已经算好，只把展示序号重编为 1~实际箱数，保证风险组1最高。
        result["风险组"] = range(1, len(result) + 1)
        actual_bins = len(result)
        warning = "" if actual_bins == BIN_NUM else f"等频分箱实际为 {actual_bins} 箱，少于预期 {BIN_NUM} 箱（概率重复导致）"
        return result, actual_bins, warning
    except Exception as error:
        result, actual_bins, fallback_warning = _fallback_bins(frame)
        warning = f"原分箱函数不可用：{error}；{fallback_warning}"
        if actual_bins < BIN_NUM:
            warning += f"，实际为 {actual_bins} 箱"
        return result, actual_bins, warning


def calculate_metrics(y_true, probability):
    """计算报告所需的总体指标；KS 复用原评估 ks_score，Lift 取最高风险组累计 Lift。"""
    frame = _valid_frame(y_true, probability)
    bins, actual_bins, bin_warning = build_decile_table(frame["label"], frame["probability"])
    metrics = {
        "样本总数": len(frame),
        "坏样本数": int(frame["label"].sum()) if len(frame) else 0,
        "平均坏率": frame["label"].mean() if len(frame) else np.nan,
        "KS": np.nan,
        "最坏10%累计Lift": bins.iloc[0]["累计Lift"] if not bins.empty else np.nan,
        "实际分箱数": actual_bins,
        "分箱提示": bin_warning,
    }
    if frame["label"].nunique() == 2:
        metrics["KS"] = ks_score(frame["label"], frame["probability"]) * 100
    elif len(frame):
        metrics["分箱提示"] = (bin_warning + "；" if bin_warning else "") + "标签只有一个类别，KS不可计算"
    return metrics, bins


def _compare_bins(before_bins, after_bins):
    """将前后独立等频分箱按风险序号并排，最后追加样本汇总行。"""
    levels = pd.DataFrame({"风险组": range(1, BIN_NUM + 1)})
    before = before_bins.rename(columns={
        "样本总数": "压测前_样本总数",
        "坏样本数": "压测前_坏样本数",
        "累计Lift": "压测前_累计Lift",
    })
    after = after_bins.rename(columns={
        "样本总数": "压测后_样本总数",
        "坏样本数": "压测后_坏样本数",
        "累计Lift": "压测后_累计Lift",
    })
    result = levels.merge(before, on="风险组", how="left").merge(after, on="风险组", how="left")
    total = {"风险组": "汇总"}
    for column in ["压测前_样本总数", "压测前_坏样本数", "压测后_样本总数", "压测后_坏样本数"]:
        total[column] = result[column].sum(min_count=1)
    total["压测前_累计Lift"] = np.nan
    total["压测后_累计Lift"] = np.nan
    return pd.concat([result, pd.DataFrame([total])], ignore_index=True)


def _score_series(probability, score_params):
    """复用原评估 Transfer2Score，将概率转为展示分数。"""
    return pd.Series(probability, dtype=float).apply(
        lambda value: Transfer2Score(value, p0=score_params["p0"], PDO=score_params["pdo"], theta=score_params["theta"])
    )


def _plot_score_distribution(before_prob, after_prob, score_params, output_path):
    """Draw score distribution; fall back to a histogram when KDE is unavailable."""
    before_score = _score_series(before_prob, score_params)
    after_score = _score_series(after_prob, score_params)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    try:
        if before_score.nunique() < 2 or after_score.nunique() < 2:
            raise ValueError("分数取值不足")
        sns.kdeplot(before_score, fill=True, alpha=0.30, color="#d62728", label="Before", ax=ax)
        sns.kdeplot(after_score, fill=True, alpha=0.30, color="#1f77b4", label="After", ax=ax)
    except Exception:
        ax.hist(before_score, bins=20, density=True, alpha=0.35, color="#d62728", label="Before")
        ax.hist(after_score, bins=20, density=True, alpha=0.35, color="#1f77b4", label="After")
    ax.set_title("Score Distribution")
    ax.set_xlabel("Score")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_change(before_value, after_value, ylabel, output_path, title):
    """Draw the before/after change for KS or Lift."""
    fig, ax = plt.subplots(figsize=(5.5, 4), dpi=120)
    values = [before_value, after_value]
    ax.plot(["Before", "After"], values, marker="o", linewidth=2, color="#2f5597")
    for index, value in enumerate(values):
        if pd.notna(value):
            ax.annotate(f"{value:.2f}", (index, value), xytext=(0, 8), textcoords="offset points", ha="center")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _scenario_summary(category, name, baseline_metrics, after_metrics):
    """场景总览只记录压测后指标及相对基准变化，避免重复写基准数据。"""
    return {
        "压测类别": category,
        "压测场景": name,
        "样本总数": after_metrics["样本总数"],
        "坏样本数": after_metrics["坏样本数"],
        "平均坏率(%)": round(after_metrics["平均坏率"] * 100, 2),
        "KS": round(after_metrics["KS"], 2),
        "KS变化_百分点": round(after_metrics["KS"] - baseline_metrics["KS"], 2),
        "最坏10%累计Lift": round(after_metrics["最坏10%累计Lift"], 2),
        "Lift变化": round(after_metrics["最坏10%累计Lift"] - baseline_metrics["最坏10%累计Lift"], 2),
        "实际分箱数": after_metrics["实际分箱数"],
        "分箱提示": after_metrics["分箱提示"],
    }


def _baseline_summary(dataset_name, metrics):
    """场景汇总首行：压测前完整评估集，变化列固定为0。"""
    return {
        "压测类别": "压测前",
        "压测场景": f"原始数据集_{dataset_name}",
        "样本总数": metrics["样本总数"],
        "坏样本数": metrics["坏样本数"],
        "平均坏率(%)": round(metrics["平均坏率"] * 100, 2),
        "KS": round(metrics["KS"], 2),
        "KS变化_百分点": 0.00,
        "最坏10%累计Lift": round(metrics["最坏10%累计Lift"], 2),
        "Lift变化": 0.00,
        "实际分箱数": metrics["实际分箱数"],
        "分箱提示": metrics["分箱提示"],
    }


def _write_scenario(output_dir, category, name, baseline_metrics, baseline_bins, baseline_prob, after_y, after_prob, score_params):
    """输出一个场景的图；指标和分箱明细统一写入根目录 Excel。"""
    output_dir = _make_dir(output_dir)
    after_metrics, after_bins = calculate_metrics(after_y, after_prob)
    summary = _scenario_summary(category, name, baseline_metrics, after_metrics)
    bin_table = _compare_bins(baseline_bins, after_bins)
    _plot_score_distribution(baseline_prob, after_prob, score_params, output_dir / "score_distribution.png")
    _plot_change(baseline_metrics["KS"], after_metrics["KS"], "KS", output_dir / "ks_change.png", "KS Change")
    _plot_change(
        baseline_metrics["最坏10%累计Lift"], after_metrics["最坏10%累计Lift"], "Lift",
        output_dir / "lift_change.png", "Worst 10% Lift Change"
    )
    return summary, bin_table


def _write_skip(category_dir, scenario_names, reason, skipped_rows):
    """字段缺失或空子群时仅记录到总览 Excel 的“跳过场景”页。"""
    rows = [{"压测场景": name, "跳过原因": reason} for name in scenario_names]
    skipped_rows.extend(rows)


def _set_missing_value(frame, feature):
    """按字段类型写入模型可识别的缺失值，类别字段保留原类别类型。"""
    series = frame[feature]
    if isinstance(series.dtype, pd.CategoricalDtype):
        category = str(MISSING_VALUE)
        # LightGBM 会校验类别列表。不能为了压测临时新增类别；训练时不存在
        # "-9999" 时，保留原 categories 列表写入 Categorical。此时值会自然成为
        # NaN，但 dtype 及类别列表与训练时完全一致。
        frame[feature] = pd.Series(
            pd.Categorical(
                [category] * len(series),
                categories=series.cat.categories,
                ordered=series.cat.ordered,
            ),
            index=series.index,
        )
    elif is_numeric_dtype(series.dtype):
        frame[feature] = MISSING_VALUE
    else:
        frame[feature] = str(MISSING_VALUE)


def _run_subset_group(category_dir, category, name, mask, eval_df, baseline_prob, label_col, baseline_metrics, baseline_bins, score_params, summaries, all_bins, skipped_rows):
    """执行年龄、学历、额度等子群场景；压测前始终是完整评估集。"""
    if not mask.any():
        _write_skip(category_dir, [name], "筛选后无样本", skipped_rows)
        return
    after_df = eval_df.loc[mask]
    summary, bins = _write_scenario(
        Path(category_dir) / name, category, name, baseline_metrics, baseline_bins, baseline_prob,
        after_df[label_col], baseline_prob.loc[mask], score_params
    )
    summaries.append(summary)
    all_bins.append(bins.assign(压测场景=name)[["压测场景", *bins.columns]])


def _write_overview_workbook(workbook_path, summary_df, all_bins_df, skipped_df, top2_df):
    """写入唯一表格产物：场景按压测类别合并，分箱使用压测前/后两层表头。"""
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    bin_columns = [
        "压测场景", "风险组", "压测前_样本总数", "压测前_坏样本数", "压测前_累计Lift",
        "压测后_样本总数", "压测后_坏样本数", "压测后_累计Lift",
    ]
    bins_for_excel = all_bins_df.reindex(columns=bin_columns).rename(columns={
        "压测前_样本总数": "样本总数",
        "压测前_坏样本数": "坏样本数",
        "压测前_累计Lift": "累计Lift",
        "压测后_样本总数": "样本总数.1",
        "压测后_坏样本数": "坏样本数.1",
        "压测后_累计Lift": "累计Lift.1",
    })
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="压测汇总", index=False)
        bins_for_excel.to_excel(writer, sheet_name="分箱明细", index=False, startrow=1)
        skipped_df.to_excel(writer, sheet_name="跳过场景", index=False)
        top2_df.to_excel(writer, sheet_name="Top2变量", index=False)

        header_fill = PatternFill("solid", fgColor="1F4E78")
        header_font = Font(color="FFFFFF", bold=True)
        header_alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        two_decimal_headers = {
            "平均坏率(%)", "KS", "KS变化_百分点", "最坏10%累计Lift", "Lift变化",
            "压测前_累计Lift", "压测后_累计Lift", "累计Lift", IMPORTANCE_COLUMN,
        }
        count_headers = {"样本总数", "坏样本数", "压测前_样本总数", "压测前_坏样本数", "压测后_样本总数", "压测后_坏样本数"}
        for worksheet in writer.book.worksheets:
            header_row = 2 if worksheet.title == "分箱明细" else 1
            headers = {cell.column: cell.value for cell in worksheet[header_row]}
            worksheet.freeze_panes = "A3" if worksheet.title == "分箱明细" else "A2"
            for column, header in headers.items():
                for row in range(header_row + 1, worksheet.max_row + 1):
                    if header in two_decimal_headers:
                        worksheet.cell(row, column).number_format = "0.00"
                    elif header in count_headers:
                        worksheet.cell(row, column).number_format = "#,##0"
                worksheet.column_dimensions[get_column_letter(column)].width = min(max(len(str(header or "")) + 4, 14), 42)

            for cell in worksheet[header_row]:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = header_alignment

        # 场景汇总：同一类场景的首列纵向合并，例如六个年龄场景只显示一次类别。
        summary_sheet = writer.book["压测汇总"]
        start_row = 2
        while start_row <= summary_sheet.max_row:
            category = summary_sheet.cell(start_row, 1).value
            end_row = start_row
            while end_row + 1 <= summary_sheet.max_row and summary_sheet.cell(end_row + 1, 1).value == category:
                end_row += 1
            if end_row > start_row:
                summary_sheet.merge_cells(start_row=start_row, start_column=1, end_row=end_row, end_column=1)
            summary_sheet.cell(start_row, 1).alignment = header_alignment
            start_row = end_row + 1

        # 分箱明细：第一行按“压测前 / 压测后”分组，第二行只保留指标名称。
        bin_sheet = writer.book["分箱明细"]
        bin_sheet.merge_cells("A1:A2")
        bin_sheet.merge_cells("B1:B2")
        bin_sheet.merge_cells("C1:E1")
        bin_sheet.merge_cells("F1:H1")
        bin_sheet["A1"] = "压测场景"
        bin_sheet["B1"] = "风险组"
        bin_sheet["C1"] = "压测前"
        bin_sheet["F1"] = "压测后"
        for cell in bin_sheet[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = header_alignment


def run_model_stress_tests(
    eval_df,
    baseline_prob,
    predict_func,
    model_features,
    label_col,
    feature_importance_path,
    output_dir,
    score_params,
    dataset_name,
):
    """执行四类压测并写入 stress_test 目录。

    eval_df 必须是完整评估数据，除入模特征和标签外可带年龄、学历、额度等非入模字段。
    预测时仅使用 model_features；不重新读取模型或数据。
    """
    # 历史版本会产生 CSV；本版本只保留 Excel 和图片，因此只清理这些旧 CSV。
    # 不删除整个目录，避免用户打开 Excel 时 Windows 因文件锁导致压测提前失败。
    root_dir = _make_dir(output_dir)
    for old_csv in root_dir.rglob("*.csv"):
        try:
            old_csv.unlink()
        except PermissionError:
            print(f"无法删除已打开的历史CSV，已跳过：{old_csv}")
    eval_df = eval_df.copy()
    baseline_prob = pd.Series(baseline_prob, index=eval_df.index, dtype=float)
    baseline_metrics, baseline_bins = calculate_metrics(eval_df[label_col], baseline_prob)

    summaries, all_bins, skipped_rows = [], [], []
    category_dirs = {
        "top2": _make_dir(root_dir / "01_top2_features"),
        "age": _make_dir(root_dir / "02_age"),
        "education": _make_dir(root_dir / "03_education"),
        "card": _make_dir(root_dir / "04_card_limit"),
    }

    # 场景1：按配置手工指定，或按特征重要性文件中的 importance_score_split 取前两个变量。
    top2_df = pd.DataFrame(columns=["选择方式", "特征重要性文件", "变量", IMPORTANCE_COLUMN])
    try:
        if TOP2_FEATURES:
            top2 = pd.DataFrame({IMPORTANCE_FEATURE_COLUMN: TOP2_FEATURES, IMPORTANCE_COLUMN: np.nan})
            selection_method = "配置文件顶部手工指定"
        else:
            importance = pd.read_csv(feature_importance_path)
            required_columns = {IMPORTANCE_FEATURE_COLUMN, IMPORTANCE_COLUMN}
            if not required_columns.issubset(importance.columns):
                raise ValueError(f"特征重要性文件缺少字段：{sorted(required_columns - set(importance.columns))}")
            top2 = importance.loc[
                importance[IMPORTANCE_FEATURE_COLUMN].isin(model_features)
                & importance[IMPORTANCE_FEATURE_COLUMN].isin(eval_df.columns)
            ].sort_values(IMPORTANCE_COLUMN, ascending=False).head(2)
            selection_method = f"按 {IMPORTANCE_COLUMN} 前2名自动选择"
        if len(top2) < 2:
            raise ValueError("可用于压测的入模变量不足2个")
        stressed_features = top2[IMPORTANCE_FEATURE_COLUMN].tolist()
        top2_df = top2[[IMPORTANCE_FEATURE_COLUMN, IMPORTANCE_COLUMN]].rename(columns={IMPORTANCE_FEATURE_COLUMN: "变量"})
        top2_df.insert(0, "特征重要性文件", feature_importance_path if not TOP2_FEATURES else "手工指定，不读取文件")
        top2_df.insert(0, "选择方式", selection_method)
        stressed_df = eval_df.copy()
        for feature in stressed_features:
            _set_missing_value(stressed_df, feature)
        stressed_prob = pd.Series(predict_func(stressed_df), index=eval_df.index, dtype=float)
        name = "Top2重要变量置缺失"
        summary, bins = _write_scenario(
            category_dirs["top2"], "压测1-敏感性分析", name, baseline_metrics, baseline_bins, baseline_prob,
            eval_df[label_col], stressed_prob, score_params
        )
        summaries.append(summary)
        all_bins.append(bins.assign(压测场景=name)[["压测场景", *bins.columns]])
    except Exception as error:
        top2_df = pd.DataFrame([{
            "选择方式": "未执行",
            "特征重要性文件": feature_importance_path,
            "变量": "",
            IMPORTANCE_COLUMN: np.nan,
            "说明": str(error),
        }])
        _write_skip(category_dirs["top2"], ["Top2重要变量置缺失"], str(error), skipped_rows)

    # 场景2：年龄六个子群；年龄为空的样本不归入任何年龄子群。
    if AGE_COLUMN not in eval_df.columns:
        _write_skip(category_dirs["age"], [name for name, _ in AGE_SCENARIOS], f"评估数据缺少字段：{AGE_COLUMN}", skipped_rows)
    else:
        age = pd.to_numeric(eval_df[AGE_COLUMN], errors="coerce")
        for name, rule in AGE_SCENARIOS:
            _run_subset_group(category_dirs["age"], "压测2-年龄情景", name, age.notna() & rule(age), eval_df, baseline_prob, label_col, baseline_metrics, baseline_bins, score_params, summaries, all_bins, skipped_rows)

    # 场景3：10/20（数值或字符串均可转换）视为本科及以上，缺失进入其它。
    if EDUCATION_COLUMN not in eval_df.columns:
        _write_skip(category_dirs["education"], ["本科及以上", "其他学历"], f"评估数据缺少字段：{EDUCATION_COLUMN}", skipped_rows)
    else:
        education = pd.to_numeric(eval_df[EDUCATION_COLUMN], errors="coerce")
        for name, rule in EDUCATION_SCENARIOS:
            mask = rule(education)
            _run_subset_group(category_dirs["education"], "压测3-学历情景", name, mask, eval_df, baseline_prob, label_col, baseline_metrics, baseline_bins, score_params, summaries, all_bins, skipped_rows)

    # 场景4：缺失或低于3万归第一组，3万及以上归第二组。
    if CARD_LIMIT_COLUMN not in eval_df.columns:
        _write_skip(category_dirs["card"], ["无信用卡或额度3万以下", "信用卡额度3万及以上"], f"评估数据缺少字段：{CARD_LIMIT_COLUMN}", skipped_rows)
    else:
        card_limit = pd.to_numeric(eval_df[CARD_LIMIT_COLUMN], errors="coerce")
        for name, rule in CARD_LIMIT_SCENARIOS:
            mask = rule(card_limit)
            _run_subset_group(category_dirs["card"], "压测4-信用卡额度情景", name, mask, eval_df, baseline_prob, label_col, baseline_metrics, baseline_bins, score_params, summaries, all_bins, skipped_rows)

    # 唯一表格产物：首行是完整评估集，随后是各个压测场景。
    summary_df = pd.DataFrame([_baseline_summary(dataset_name, baseline_metrics), *summaries])
    all_bins_df = pd.concat(all_bins, ignore_index=True) if all_bins else pd.DataFrame(columns=["压测场景", "风险组"])
    skipped_df = pd.DataFrame(skipped_rows, columns=["压测场景", "跳过原因"])
    _write_overview_workbook(root_dir / "stress_test_summary.xlsx", summary_df, all_bins_df, skipped_df, top2_df)

    print(f"压测评估完成：{root_dir}")
    return {"baseline": baseline_metrics, "summary": summary_df, "bins": all_bins_df, "skipped": skipped_df}
