"""
JSON报告生成模块
统一管理所有环节的JSON报告生成函数
"""

import os
import json
import re
from pathlib import Path
import pandas as pd
import numpy as np
from utils.data_utils import identify_types


def _to_float_or_none(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        stripped = value.strip()
        if stripped in {"", "-", "None", "none", "nan", "NaN"}:
            return None
        value = stripped
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_optional_float(value, fmt=".4f"):
    numeric_value = _to_float_or_none(value)
    return format(numeric_value, fmt) if numeric_value is not None else "-"


def generate_data_quality_json(missing_sta, num_sta, cat_sta, dup_sta, con_sta, default_sta=None, current_data_columns=None, output_path=None):
    """
    生成数据质量分析报告的JSON文件【用于界面展示】
    
    参数:
        missing_sta: DataFrame, 缺失值统计结果
        num_sta: DataFrame, 数值型变量统计结果
        cat_sta: DataFrame, 类别型变量统计结果
        dup_sta: DataFrame, 样本重复ID检查结果【整体统计】
        con_sta: DataFrame, 集中度分析结果
        default_sta: DataFrame, 违约率统计结果
        output_path: str, JSON文件保存路径，如果为None则返回JSON字符串
    
    返回:
        str: 如果output_path为None，返回JSON字符串；否则返回保存路径
    """
    try:
        # 1. 样本整体概述（包含重复ID、渠道、时间范围等）
        sample_check = {
            "数据总量": int(dup_sta['count'].iloc[0]),
            "唯一ID数量": int(dup_sta['unique_cnt'].iloc[0]),
            "最大重复次数": int(dup_sta['max_dup'].iloc[0])
        }
        # 加入渠道统计
        if 'channel_count' in dup_sta.columns and pd.notna(dup_sta['channel_count'].iloc[0]):
            sample_check["样本渠道数量"] = int(dup_sta['channel_count'].iloc[0])
            sample_check["渠道列表"] = dup_sta['channel_list'].iloc[0]
        # 加入时间范围统计
        if 'date_range' in dup_sta.columns and dup_sta['date_range'].iloc[0]:
            sample_check["样本时间范围"] = dup_sta['date_range'].iloc[0]

            
        # 2. 特征分析
        # 2.1 数值型特征（前10个）
        # 从num_sta取前10个，补上缺失率missing_tot和集中度信息
        numeric_table = []
        if len(num_sta) > 0:
            # 合并缺失率信息
            num_sta_with_missing = pd.merge(
                num_sta,
                missing_sta[['feature_name', 'missing_tot']],
                on='feature_name',
                how='left'
            )
            # 合并集中度信息
            num_sta_with_all = pd.merge(
                num_sta_with_missing,
                con_sta[['feature_name', 'unique_cnt']],
                on='feature_name',
                how='left'
            )
            # 取前10个（按特征名称排序，保证结果稳定）
            top_numeric = num_sta_with_all.head(10)
            
            for _, row in top_numeric.iterrows():
                feature_row = {
                    "特征名称": row['feature_name'],
                    "数据类型": "数值型",
                    "缺失率": f"{float(row['missing_tot']):.1%}" if pd.notna(row['missing_tot']) else "0.0%",
                    "非空值数量": int(row['count']) if pd.notna(row['count']) else 0,
                    "唯一值数量": int(row['unique_cnt']) if pd.notna(row['unique_cnt']) else 0,
                    "平均值": f"{float(row['mean']):.1f}" if pd.notna(row['mean']) else "-",
                    "标准差": f"{float(row['std']):.1f}" if pd.notna(row['std']) else "-",
                    "最大值": f"{float(row['max']):.1f}" if pd.notna(row['max']) else "-",
                    "最小值": f"{float(row['min']):.1f}" if pd.notna(row['min']) else "-",
                    "中位数": f"{float(row['p50']):.1f}" if pd.notna(row['p50']) else "-",
                    "变异系数": f"{float(row['cv']):.3f}" if pd.notna(row['cv']) and row['cv'] != '-' else "-"
                }
                numeric_table.append(feature_row)
        
        # 2.2 类别型特征（前10个）
        # 从cat_sta取feature_name, count, unique_cnt, value1, cnt1, ratio1，补上缺失率missing_tot
        categorical_table = []
        if len(cat_sta) > 0:
            # 合并缺失率信息
            cat_sta_with_missing = pd.merge(
                cat_sta[['feature_name', 'count', 'unique_cnt', 'value1', 'ratio1']],
                missing_sta[['feature_name', 'missing_tot']],
                on='feature_name',
                how='left'
            )
            # 取前10个（按特征名称排序，保证结果稳定）
            top_categorical = cat_sta_with_missing.head(10)
            
            for _, row in top_categorical.iterrows():
                feature_row = {
                    "特征名称": row['feature_name'],
                    "数据类型": "类别型",
                    "缺失率": f"{float(row['missing_tot']):.1%}" if pd.notna(row['missing_tot']) else "0.0%",
                    "非空值数量": int(row['count']) if pd.notna(row['count']) else 0,
                    "唯一值数量": int(row['unique_cnt']) if pd.notna(row['unique_cnt']) else 0,
                    "最常见值": str(row['value1']) if pd.notna(row['value1']) else "-",
                    "最常见值频率": f"{float(row['ratio1']):.1%}" if pd.notna(row['ratio1']) else "-"
                }
                categorical_table.append(feature_row)
        
        # 统计总数（用于说明）
        total_features = len(missing_sta) if len(missing_sta) > 0 else 0
        total_numeric = len(num_sta) if len(num_sta) > 0 else 0
        total_categorical = len(cat_sta) if len(cat_sta) > 0 else 0
        
        feature_analysis = {
            "特征总数": total_features,
            "数值型特征总数": total_numeric,
            "类别型特征总数": total_categorical,
            "数值型特征明细（前10个，部分展示）": numeric_table,
            "类别型特征明细（前10个，部分展示）": categorical_table
        }
        
        # 3. 违约率分析
        default_analysis = {}
        if default_sta is not None:
            # 总体情况
            overall_stats = default_sta[default_sta['group'] == 'all'].iloc[0]
            default_analysis["总体情况"] = {
                "样本总量": int(overall_stats['total_samples']),
                "好样本数量": int(overall_stats['good']),
                "坏样本数量": int(overall_stats['bad']),
                "灰样本数量": int(overall_stats['grey']),
                "平均违约率": f"{float(overall_stats['bad_rate']):.1%}"
            }
            
            # 按渠道的违约率
            channel_stats = default_sta[(~default_sta['group'].str.match(r'\d{4}-\d{2}', na=False)) & (default_sta['group'] != 'all')]
            if len(channel_stats) > 0:
                default_analysis["按渠道"] = []
                for _, row in channel_stats.iterrows():
                    channel_dict = {
                        "渠道": str(row['group']),
                        "样本总量": int(row['total_samples']),
                        "好样本数量": int(row['good']),
                        "坏样本数量": int(row['bad']),
                        "灰样本数量": int(row['grey']),
                        "违约率": f"{float(row['bad_rate']):.1%}"
                    }
                    default_analysis["按渠道"].append(channel_dict)
            
            # 按月份的违约率
            monthly_stats = default_sta[default_sta['group'].str.match(r'\d{4}-\d{2}', na=False)]
            if len(monthly_stats) > 0:
                default_analysis["按月份"] = []
                for _, row in monthly_stats.iterrows():
                    month_dict = {
                        "月份": str(row['group']),
                        "样本总量": int(row['total_samples']),
                        "好样本数量": int(row['good']),
                        "坏样本数量": int(row['bad']),
                        "灰样本数量": int(row['grey']),
                        "违约率": f"{float(row['bad_rate']):.1%}"
                    }
                    default_analysis["按月份"].append(month_dict)
        
         # 4. 基线模型兼容性检查（可选功能）
        baseline_check_info = None
        # 延迟导入，避免循环导入（utils_data_quality -> utils_baseline_model -> utils_model_train -> utils_data_quality）
        from utils.utils_baseline_model import check_old_model_features_with_status
        baseline_check_status = check_old_model_features_with_status(
            current_data_columns=current_data_columns
        )

        status = baseline_check_status.get("状态")
        if status == "检查通过":
            mapped_features = baseline_check_status.get("mapped_features", [])
            baseline_check_info = (
                "基线模型特征兼容性检查通过，可以进行模型对比评估。"
                f"模型类型：{baseline_check_status.get('模型类型')}，"
                f"入模特征数量：{len(mapped_features)}，"
                f"映射后的特征名称列表：{', '.join(mapped_features)}"
            )
            print("=" * 60)
            print("已上传基线模型，基线模型特征兼容性检查通过")
            print("=" * 60)
        elif status == "特征缺失":
            baseline_check_info = baseline_check_status.get("说明")
            print("=" * 60)
            print(f"✗ 警告: {baseline_check_info}")
            print("=" * 60)
        elif status == "加载失败":
            baseline_check_info = (
                f"基线模型加载失败，无法进行模型对比评估。原因：{baseline_check_status.get('说明')}。"
                f"尝试记录：{'; '.join(baseline_check_status.get('尝试记录', []))}"
            )
            print("=" * 60)
            print(f"✗ 警告: {baseline_check_info}")
            print("=" * 60)
        
        # 整合所有结果
        report_data = {
            "报告标题": "数据质量分析报告",
            "报告说明": "本报告对上传的数据集进行全面的质量分析，涵盖：样本整体概述（数据量、唯一ID、渠道、时间范围）、特征分析（数值型和类别型特征的统计信息）、违约率分析（总体、按渠道、按月份的违约情况）。",
            "样本整体概述": sample_check,
            "特征分析": feature_analysis,
            "违约率分析": default_analysis
        }
        
        # 如果有基线模型检查结果，添加到报告中
        if baseline_check_info:
            report_data["报告说明"] = report_data["报告说明"] + "同时检查了基线模型的特征兼容性。"
            report_data["基线模型兼容性检查"] = baseline_check_info
        
        # 转换为JSON
        json_str = json.dumps(report_data, ensure_ascii=False, indent=2)
        
        # 保存或返回
        # output_path = os.path.join(output_path, 'qa_show.json')
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(json_str)
            return output_path
        else:
            return json_str
            
    except Exception as e:
        print(f"Error generating JSON report: {str(e)}")
        return None


def generate_clear_split_json(data, data_cleared, method, dev_datasets, oot_datasets, df_all, output_path, channel_col, date_col, baseline_evaluation_info=None, baseline_metrics_df=None):
    """
    生成数据清洗和切分的JSON报告
    
    参数:
        data: DataFrame, 原始数据
        data_cleared: DataFrame, 清洗后的数据
        method: str, 切分方法
        dev_datasets: 开发数据集列表，每个元素为 (df, name) 元组，如 [(train_df, 'DEV'), (test_df, 'TEST')]
        oot_datasets: OOT数据集列表，每个元素为 (df, name) 元组，可能为空，如 [(oot_df, 'oot_name')]
        df_all: DataFrame, 数据集分布情况
        output_path: str, 输出路径
        channel_col: str, 渠道列名
        date_col: str, 日期列名
        baseline_evaluation_info: dict, 基线模型评估状态信息（可选）
        baseline_metrics_df: DataFrame, 基线模型评估结果（可选）
    """
    try:
        # 从dev_datasets中提取DEV和TEST的DataFrame
        train_df = None
        test_df = None
        for df, name in dev_datasets:
            if name == 'DEV':
                train_df = df
            elif name == 'TEST':
                test_df = df
        
        # 查找oot_main数据集
        oot_main_df = None
        oot_main_name = None
        for df, name in oot_datasets:
            if 'oot_main' in name.lower():
                oot_main_df = df
                oot_main_name = name
                break
        
        # 1. 数据清洗概览
        # 渠道值可能混有 None/数值/字符串，先去空并统一转字符串后再排序，避免比较报错
        channel_values = (
            pd.Series(data_cleared[channel_col])
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        cleaning_overview = {
            "原始样本数量": len(data),
            "清洗后样本数量": len(data_cleared),
            "适用渠道": sorted(channel_values)
        }
        
        # 2. 数据切分方式
        if method == "date_range":
            split_method = {
                "切分方法": "按指定时间范围切分训练集和测试集",
                "训练集(DEV)时间范围": f"{train_df[date_col].min()}-{train_df[date_col].max()}",
                "测试集(TEST)时间范围": f"{test_df[date_col].min()}-{test_df[date_col].max()}"
            }
            # 如果有oot_main，添加验证集信息
            if oot_main_df is not None:
                split_method["验证集(OOT_MAIN)时间范围"] = f"{oot_main_df[date_col].min()}-{oot_main_df[date_col].max()}"
        elif method == "date_proportion":
            split_method = {
                "切分方法": "按时间排序比例切分训练集和测试集",
                "训练集(DEV)时间范围": f"{train_df[date_col].min()}-{train_df[date_col].max()}",
                "测试集(TEST)时间范围": f"{test_df[date_col].min()}-{test_df[date_col].max()}"
            }
            if oot_main_df is not None:
                split_method["验证集(OOT_MAIN)时间范围"] = f"{oot_main_df[date_col].min()}-{oot_main_df[date_col].max()}"
        else:  # random
            split_method = {
                "切分方法": "随机切分训练集和测试集",
                "训练集(DEV)时间范围": "随机划分",
                "测试集(TEST)时间范围": "随机划分"
            }
            if oot_main_df is not None:
                split_method["验证集(OOT_MAIN)时间范围"] = "随机划分"
        
        # 3. 数据集分布情况
        # 整体分布
        overall_stats = df_all[df_all['month'] == 'Overall']
        overall_distribution = []
        
        # 创建数据集名称到DataFrame的映射（简化版）
        dataset_map = {name: df for df, name in dev_datasets + oot_datasets}
        
        for _, row in overall_stats.iterrows():
            dataset_name = row['dataset']
            df_current = dataset_map.get(dataset_name)
            time_range = f"{df_current[date_col].min()}-{df_current[date_col].max()}" if df_current is not None else "未知"
            
            overall_distribution.append({
                "数据集": dataset_name,
                "时间范围": time_range,
                "样本量": int(row['n_samples']),
                "好样本": int(row['good']),
                "坏样本": int(row['bad']),
                "灰样本": int(row['grey']),
                "违约率": f"{float(row['bad_rate']):.1%}"
            })
        
        # 月度分布
        monthly_stats = df_all[df_all['month'] != 'Overall']
        monthly_distribution = {
            "训练集(DEV)": [],
            "测试集(TEST)": []
        }
        
        # 如果有oot_main，添加验证集分布
        if oot_main_name:
            monthly_distribution["验证集(OOT_MAIN)"] = []
        
        for _, row in monthly_stats.iterrows():
            month_stats = {
                "月份": row['month'],
                "样本量": int(row['n_samples']),
                "好样本": int(row['good']),
                "坏样本": int(row['bad']),
                "灰样本": int(row['grey']),
                "违约率": f"{float(row['bad_rate']):.1%}"
            }
            dataset_name = row['dataset']
            if dataset_name == 'DEV':
                monthly_distribution["训练集(DEV)"].append(month_stats)
            elif dataset_name == 'TEST':
                monthly_distribution["测试集(TEST)"].append(month_stats)
            elif oot_main_name and dataset_name == oot_main_name:
                monthly_distribution["验证集(OOT_MAIN)"].append(month_stats)
        
        # 4. 基线模型评估结果（可选）
        baseline_evaluation_section = None
        if baseline_evaluation_info is not None:
            if baseline_metrics_df is not None and len(baseline_metrics_df) > 0:
                # 构建各数据集评估结果；预打分列模式下可能同时包含成功、跳过、失败。
                baseline_results = []
                for _, row in baseline_metrics_df.iterrows():
                    dataset_name = row['数据集']
                    baseline_result = {
                        "数据集": dataset_name,
                        "KS": _format_optional_float(row.get('KS')),
                        "AUC": _format_optional_float(row.get('AUC')),
                        "PSI": _format_optional_float(row.get('PSI')),
                        "尾部10%Lift": _format_optional_float(row.get('Lift_tail10')),
                        "单调性": "-"
                    }
                    mono_violation = _to_float_or_none(row.get("mono_violation"))
                    if mono_violation is not None:
                        baseline_result["单调性"] = "是" if mono_violation == 0 else "否"
                    baseline_results.append(baseline_result)
                baseline_evaluation_section = {
                    "状态": baseline_evaluation_info.get("状态", "评估成功"),
                    "说明": baseline_evaluation_info.get("说明", "基线模型已在各数据集上完成评估"),
                    "基线来源": baseline_evaluation_info.get("基线来源"),
                    "评估结果": baseline_results
                }
            else:
                # 评估失败或其他状态
                baseline_evaluation_section = baseline_evaluation_info.copy()
        
        # 整合所有结果
        report_data = {
            "报告标题": "数据清洗与切分报告",
            "报告说明": "本报告展示数据清洗与切分的完整过程，涵盖：数据清洗概览（原始与清洗后的样本数量）、数据切分方式（训练集、测试集、验证集的划分方法和时间范围）、数据集分布情况（各数据集的整体和按月份的样本分布及违约率）。",
            "数据清洗概览": cleaning_overview,
            "数据切分方式": split_method,
            "数据集分布": {
                "整体分布": overall_distribution,
                "按月份分布": monthly_distribution
            }
        }
        
        # 如果有基线模型评估信息，添加到报告中
        if baseline_evaluation_section:
            report_data["基线模型评估"] = baseline_evaluation_section
        
        # 转换为JSON
        json_str = json.dumps(report_data, ensure_ascii=False, indent=2)
        
        # 保存
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json_str)
        return output_path
            
    except Exception as e:
        print(f"Error generating JSON report: {str(e)}")
        return None


def generate_feature_select_json(dev_data, use_fea, single_var, output_path, base_fea=None):
    """
    生成特征筛选报告的JSON文件（详细版，用于大模型理解生成报告）
    
    参数:
        dev_data: DataFrame, 原始数据
        use_fea: list, 筛选后的特征列表
        single_var: DataFrame, 特征统计信息
        output_path: str, 输出路径
        base_fea: list, 基础特征列表
    """
    try:
        # 如果没有传入base_fea，使用默认值[]
        if base_fea is None:
            base_fea = []
        
        # 识别数值型和类别型特征
        num_fea_all, cat_fea_all = identify_types(df=dev_data, remove_fea=base_fea)
        
        # 从筛选后的特征中分离数值特征和类别特征
        use_fea_num = [f for f in use_fea if f in num_fea_all]
        use_fea_cat = [f for f in use_fea if f in cat_fea_all]
            
        # 1. 特征筛选概览
        total_features = len(dev_data.columns) - len(base_fea)
        selected_features = len(use_fea)
        removed_features = total_features - selected_features
        
        # 将特征列表转换为可直接复制的字符串格式
        use_fea_str = str(use_fea)  # 带方括号的字符串，如 ['fea_1', 'fea_2', ...]
        use_fea_cat_str = str(use_fea_cat) if len(use_fea_cat) > 0 else "[]"
        
        overview = {
            "总特征数": total_features,
            "筛选后特征数": selected_features,
            "移除特征数": removed_features,
            "筛选后特征列表": use_fea_str,
            "特征类型说明": {
                "数值特征数量": len(use_fea_num),
                "类别特征数量": len(use_fea_cat),
                "类别特征列表": use_fea_cat_str
            }
        }
        
        # 2. 特征详情（IV值最高的前10个筛选后特征）
        # 从single_var中筛选出use_fea中的特征，并按IV值排序
        use_fea_df = single_var[single_var['Feature'].isin(use_fea)].copy()
        top_features = use_fea_df.sort_values('IV', ascending=False).head(10)
        
        # 格式化特征详情
        feature_details = []
        for idx, (_, row) in enumerate(top_features.iterrows(), 1):
            feature_name = row['Feature']
            is_cat = feature_name in use_fea_cat        
            feature_info = {
                "特征名称": feature_name,
                "特征类型": "类别型" if is_cat else "数值型",
                "IV值": f"{float(row['IV']):.3f}",
                "PSI值": f"{float(row['PSI']):.3f}",
                "缺失率": row['missing'],
                "均值": f"{float(row['mean']):.1f}" if not pd.isna(row['mean']) else "-",
                "最大值": f"{float(row['max']):.1f}" if not pd.isna(row['max']) else "-",
                "最小值": f"{float(row['min']):.1f}" if not pd.isna(row['min']) else "-",
                "单一值占比": row['单一值占比'],
            }
            
            feature_details.append(feature_info)
        
        # 3. 统计信息
        statistics = {}
        
        # 3.1 IV值统计
        if len(use_fea_df) > 0:
            iv_sorted = use_fea_df.sort_values('IV', ascending=False)
            if len(iv_sorted) >= 2:
                statistics["IV值统计"] = {
                    "IV值最高的两个特征": [
                        {"特征名称": iv_sorted.iloc[0]['Feature'], "IV值": f"{float(iv_sorted.iloc[0]['IV']):.3f}"},
                        {"特征名称": iv_sorted.iloc[1]['Feature'], "IV值": f"{float(iv_sorted.iloc[1]['IV']):.3f}"}
                    ],
                    "IV值最低的两个特征": [
                        {"特征名称": iv_sorted.iloc[-1]['Feature'], "IV值": f"{float(iv_sorted.iloc[-1]['IV']):.3f}"},
                        {"特征名称": iv_sorted.iloc[-2]['Feature'], "IV值": f"{float(iv_sorted.iloc[-2]['IV']):.3f}"}
                    ]
                }
            else:
                statistics["IV值统计"] = {
                    "最高IV值特征": {
                        "特征名称": iv_sorted.iloc[0]['Feature'],
                        "IV值": f"{float(iv_sorted.iloc[0]['IV']):.3f}"
                    }
                }
        
        # 通用函数：将百分比字符串转换为数值（0-1之间）
        def parse_percent_to_float(val):
            """将百分比字符串转换为float（0-1之间），支持字符串和数值格式"""
            if pd.isna(val) or val == '-':
                return 0.0
            try:
                if isinstance(val, str):
                    return float(val.rstrip('%')) / 100
                return float(val)  # 如果已经是数值，直接返回
            except:
                return 0.0
        
        # 3.2 缺失率统计
        if len(use_fea_df) > 0:
            use_fea_df['missing_num'] = use_fea_df['missing'].apply(parse_percent_to_float)
            missing_sorted = use_fea_df.sort_values('missing_num', ascending=False)
            
            if len(missing_sorted) >= 2:
                statistics["缺失率统计"] = {
                    "缺失率最高的两个特征": [
                        {"特征名称": missing_sorted.iloc[0]['Feature'], "缺失率": f"{float(missing_sorted.iloc[0]['missing_num']):.3f}"},
                        {"特征名称": missing_sorted.iloc[1]['Feature'], "缺失率": f"{float(missing_sorted.iloc[1]['missing_num']):.3f}"}
                    ]
                }
            else:
                statistics["缺失率统计"] = {
                    "最高缺失率特征": {
                        "特征名称": missing_sorted.iloc[0]['Feature'],
                        "缺失率": f"{float(missing_sorted.iloc[0]['missing_num']):.3f}"
                    }
                }
        
        # 3.3 PSI值统计
        if len(use_fea_df) > 0:
            psi_sorted = use_fea_df.sort_values('PSI', ascending=False)
            
            if len(psi_sorted) >= 2:
                statistics["PSI值统计"] = {
                    "PSI值最高的两个特征": [
                        {"特征名称": psi_sorted.iloc[0]['Feature'], "PSI值": f"{float(psi_sorted.iloc[0]['PSI']):.3f}"},
                        {"特征名称": psi_sorted.iloc[1]['Feature'], "PSI值": f"{float(psi_sorted.iloc[1]['PSI']):.3f}"}
                    ]
                }
            else:
                statistics["PSI值统计"] = {
                    "最高PSI值特征": {
                        "特征名称": psi_sorted.iloc[0]['Feature'],
                        "PSI值": f"{float(psi_sorted.iloc[0]['PSI']):.3f}"
                    }
                }
        
        # 3.4 单一值占比统计
        if len(use_fea_df) > 0:
            use_fea_df['unique_rate_num'] = use_fea_df['单一值占比'].apply(parse_percent_to_float)
            unique_sorted = use_fea_df.sort_values('unique_rate_num', ascending=False)
            
            # 统计单一值占比情况
            high_unique = unique_sorted[unique_sorted['unique_rate_num'] > 0.8]
            medium_unique = unique_sorted[(unique_sorted['unique_rate_num'] > 0.5) & (unique_sorted['unique_rate_num'] <= 0.8)]
            low_unique = unique_sorted[unique_sorted['unique_rate_num'] <= 0.5]
            
            statistics["单一值占比统计"] = {
                "单一值占比最高特征": {
                    "特征名称": unique_sorted.iloc[0]['Feature'],
                    "单一值占比": unique_sorted.iloc[0]['单一值占比']
                },
                "单一值占比最低特征": {
                    "特征名称": unique_sorted.iloc[-1]['Feature'],
                    "单一值占比": unique_sorted.iloc[-1]['单一值占比']
                },
                "单一值占比情况分布": {
                    "单一值占比>80%的特征数量": len(high_unique),
                    "50%<单一值占比<=80%的特征数量": len(medium_unique),
                    "单一值占比<=50%的特征数量": len(low_unique),
                }
            }
        
        # 整合所有结果
        report_data = {
            "报告标题": "特征筛选报告",
            "报告说明": "本报告详细展示了特征筛选的结果，涵盖：特征筛选概览（总特征数、筛选后特征数、移除特征数，筛选后的特征列表（必须全部展示），特征类型分布）、特征详情（IV值最高的前10个特征的详细信息）、统计分析（IV值、缺失率、PSI值、单一值占比等统计指标）。",
            "特征筛选概览": overview,
            "特征详情": {
                "说明": "展示筛选后特征中IV值最高的前10个特征的详细信息",
                "IV值TOP10特征": feature_details
            },
            "统计分析": statistics
        }
        
        # 转换为JSON
        json_str = json.dumps(report_data, ensure_ascii=False, indent=2)
        
        # 保存
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json_str)
        return output_path
            
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error generating feature selection JSON report: {str(e)}")
        return None


def generate_model_train_json(
    model_metrics,
    best_params,
    feature_imp,
    output_path,
    is_optuna,
    use_fea=None,
    use_fea_source=None,
    model_params_source=None,
    optimization_info=None,
):
    """
    生成模型训练的JSON报告
    
    参数:
        model_metrics: dict, 包含模型评估指标
        best_params: dict, 模型最优参数
        feature_imp: DataFrame, 特征重要性数据
        output_path: str, 输出路径
        is_optuna: str, Optuna优化状态
    """
    try:
        # 定义需要展示的关键参数列表
        key_params_list = [
            "importance_type",
            "min_split_gain",
            "learning_rate",
            "num_leaves",
            "max_bin",
            "reg_alpha",
            "reg_lambda",
            "subsample",
            "colsample_bytree",
            "subsample_freq",
            "min_child_weight",
            "n_estimators"
        ]
        
        # 从 best_params 中筛选出关键参数
        key_params = {k: v for k, v in best_params.items() if k in key_params_list}
        
        # 0. 入模特征信息（可选）
        feature_info = None
        if use_fea is not None:
            try:
                use_fea_list = list(use_fea)
            except Exception:
                use_fea_list = []
            feature_info = {
                "入模特征来源": use_fea_source or "未知",
                "入模特征数量": int(len(use_fea_list))
            }

        model_params_info = {
            "模型参数来源": model_params_source or "模型训练配置",
            "模型参数数量": int(len(best_params)),
            "是否执行超参数寻优": optimization_info is not None,
        }

        optimization_report = None
        if optimization_info is not None:
            optimization_report = {
                "优化模式": optimization_info["optimization_mode"],
                "寻优次数": int(optimization_info["n_trials"]),
                "优化目标": [
                    {
                        "数据集": target["dataset"],
                        "指标": target["metric"],
                        "方向": target["direction"],
                    }
                    for target in optimization_info["optimization_targets"]
                ],
                "最终选解数据集": optimization_info["selection_dataset"],
            }

        # 1. 模型基本信息
        eval_dataset_name = model_metrics['eval_dataset_name']
        model_search_desc = is_optuna or "使用默认参数"
        
        # 2. 新模型性能指标（扁平化结构）
        new_model_performance = {
            "评估数据集": eval_dataset_name,
            "模型寻优方式": model_search_desc,
            "数据集指标对比": model_metrics.get("dataset_metrics_compare", []),
            "模型关键参数": key_params
        }
        
        # 3. 等频十分箱结果分析（嵌套结构）
        raw_lift = model_metrics.get("df_lift", None)
        lift_df = pd.DataFrame(raw_lift) if raw_lift is not None else pd.DataFrame()
        mono_violation = float(model_metrics.get("mono_violation", -1.0))
        overall_bad_rate = float(model_metrics['overall_bad_rate'])
        
        # Lift结果分析
        lift_result_analysis = {
            "尾部10%样本Lift值": f"{float(model_metrics['lift_tail10']):.4f}",
            "头部10%样本Lift值": f"{float(model_metrics['lift_head10']):.4f}",
            "头部5%样本Lift值": f"{float(model_metrics['lift_head5']):.4f}",
            "头部2%样本Lift值": f"{float(model_metrics['lift_head2']):.4f}",
            "头部1%样本Lift值": f"{float(model_metrics['lift_head1']):.4f}",
            "整体坏样本率": f"{float(overall_bad_rate):.2%}"
        }
        
        # 坏率单调性分析
        monotonicity_analysis = {
            "坏样本率是否单调": "是" if mono_violation == 0 else "否",
            "最大单调性违背幅度": float(mono_violation),
            "单调性跳动次数": int(model_metrics.get('mono_jump_count_test', 0))
        }
        
        # 十分箱明细
        decile_details = []
        for _, row in lift_df.iterrows():
            decile_details.append({
                "分箱编号": int(row['level']),
                "分数区间": str(row['range']),
                "样本数": int(row['num']),
                "好样本数": int(row['good']),
                "坏样本数": int(row['bad']),
                "坏率": f"{float(row['bad_ration']):.4f}",
                "基准坏率": f"{float(overall_bad_rate):.4f}",
                "Lift值": f"{float(row['lift']):.4f}"
            })  
        
        # 等频十分箱结果分析（嵌套结构）
        lift_analysis = {
            "Lift结果分析": lift_result_analysis,
            "十分箱单调性分析": monotonicity_analysis,
            "十分箱明细": decile_details
        }
        
        # 4. 特征重要性分析（扁平化结构）
        total_features = len(feature_imp)
        non_zero_features = len(feature_imp[feature_imp['importance_score_split'] > 0])
        
        feature_importance = {
            "总特征数": total_features,
            "非零重要性特征数": non_zero_features,
            "特征重要性TOP15": []
        }
        
        # 获取TOP15特征
        top_features = feature_imp.head(15).copy()
        if len(top_features) > 0 and top_features['importance_score_split'].sum() > 0:
            top_features['importance_ratio'] = top_features['importance_score_split'] / top_features['importance_score_split'].sum()
            top_features['importance_ratio'] = top_features['importance_ratio'].apply(lambda x: f"{x:.1%}")
            top_features = top_features.where(~top_features.isna(), None)
            
            for _, row in top_features.iterrows():
                feature_importance["特征重要性TOP15"].append({
                    "特征名称": str(row['Feature']),
                    "中文名称": str(row.get('feature_cn', '未知')),
                    "数据源": str(row.get('data_source', '未知')),
                    "分裂重要性": int(row['importance_score_split']),
                    "增益重要性": f"{row['importance_score_gain']:.3f}" if pd.notna(row.get('importance_score_gain')) else "N/A",
                    "重要性占比": str(row['importance_ratio'])
                })
        
        # 5. 基线模型指标（只展示评估数据集对应的结果）
        baseline_model_info = None
        baseline_metrics_path = "outputs/baseline_model/baseline_model_metrics.csv"
        if os.path.exists(baseline_metrics_path):
            try:
                baseline_df = pd.read_csv(baseline_metrics_path, encoding='utf-8-sig')
                
                # 映射评估数据集名称到CSV中的数据集名称
                dataset_name_mapping = {
                    "验证集(oot_main)": "oot_main",
                    "测试集(test)": "test"
                }
                current_dataset_key = dataset_name_mapping.get(eval_dataset_name, "test")
                
                # 只获取评估数据集对应的基线模型指标
                baseline_row = baseline_df[baseline_df['数据集'] == current_dataset_key]
                if len(baseline_row) > 0:
                    baseline_row = baseline_row.iloc[0]
                    baseline_model_info = {
                        "说明": f"以下为基线模型在{eval_dataset_name}上的性能指标，如需更详细的基线模型对比分析，请执行模型评估环节。"
                    }
                    exception_note = baseline_row.get("异常说明", baseline_row.get("说明", ""))
                    if pd.notna(exception_note) and exception_note:
                        baseline_model_info["异常说明"] = str(exception_note)
                    if baseline_row.get("基线来源"):
                        baseline_model_info["基线来源"] = str(baseline_row.get("基线来源"))

                    optional_fields = [
                        ("KS值", "KS"),
                        ("AUC值", "AUC"),
                        ("KS差值", "KS_Gap"),
                        ("PSI值", "PSI"),
                        ("尾部10%样本Lift值", "Lift_tail10"),
                        ("头部10%样本Lift值", "Lift_head10"),
                        ("头部5%样本Lift值", "Lift_head5"),
                        ("头部2%样本Lift值", "Lift_head2"),
                        ("头部1%样本Lift值", "Lift_head1"),
                        ("单调性违背幅度", "mono_violation"),
                    ]
                    for output_key, source_key in optional_fields:
                        numeric_value = _to_float_or_none(baseline_row.get(source_key))
                        if numeric_value is not None:
                            baseline_model_info[output_key] = numeric_value

                    mono_jump_count = _to_float_or_none(baseline_row.get('mono_jump_count'))
                    if mono_jump_count is not None:
                        baseline_model_info["单调性跳动次数"] = int(mono_jump_count)

                    mono_violation = _to_float_or_none(baseline_row.get('mono_violation'))
                    if mono_violation is not None:
                        baseline_model_info["坏样本率是否单调"] = "是" if mono_violation == 0 else "否"

                    if len(baseline_model_info) <= 2:
                        baseline_model_info = None
                    
            except Exception as e:
                print(f"警告: 加载基线模型指标失败: {str(e)}")
                baseline_model_info = None
        
        # 6. 整合所有结果（扁平化结构）
        # 根据是否有基线模型生成报告说明
        if baseline_model_info:
            report_description = "本报告展示LightGBM模型的训练结果，涵盖：模型性能指标（训练集、测试集和主评估集的KS、AUC、PSI等）、等频十分箱结果分析（Lift值和单调性检测）、特征重要性分析，并与基线模型进行对比。"
        else:
            report_description = "本报告展示LightGBM模型的训练结果，涵盖：模型性能指标（训练集、测试集和主评估集的KS、AUC、PSI等）、等频十分箱结果分析（Lift值和单调性检测）、特征重要性分析。"
        
        report_data = {
            "报告标题": "模型训练报告",
            "报告说明": report_description,
            **({"入模特征信息": feature_info} if feature_info is not None else {}),
            "模型参数信息": model_params_info,
            **({"超参数寻优信息": optimization_report} if optimization_report is not None else {}),
            "模型性能指标": new_model_performance,
            "等频十分箱结果分析": lift_analysis,
            "特征重要性分析": feature_importance
        }
        
        # 如果有基线模型信息，添加到报告中
        if baseline_model_info:
            report_data["基线模型指标"] = baseline_model_info
        
        # 转换为JSON
        json_str = json.dumps(report_data, ensure_ascii=False, indent=2)
        
        # 保存
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json_str)
        return output_path
            
    except Exception as e:
        print(f"Error generating model training JSON report: {str(e)}")
        return None


def generate_lr_scorecard_train_json(
    metrics,
    primary_validation,
    feature_source,
    candidate_features,
    fine_selection_count,
    stepwise_count,
    final_features,
    feature_summary,
    parameter_source,
    output_path,
):
    """生成LrScorecard模型训练阶段的用户摘要。"""
    try:
        train_metrics = metrics["train"]
        final_features = list(final_features)

        dataset_labels = {"train": "训练集", "test": "测试集", "oot_main": "主OOT"}
        dataset_rows = []
        for name, values in metrics.items():
            current_ks = float(values["ks"])
            current_gap = float(train_metrics["ks"]) - current_ks
            if abs(current_gap) < 0.005:
                current_gap = 0.0
            dataset_rows.append(
                {
                    "数据集": dataset_labels.get(name, name),
                    "数据集标识": name,
                    "样本数": int(values["samples"]),
                    "坏样本率": f"{float(values['bad_rate']):.2%}",
                    "AUC": f"{float(values['auc']):.2f}",
                    "KS": f"{current_ks:.2f}",
                    "KS差值": "-" if name == "train" else f"{current_gap:.2f}",
                    "尾部10% Lift": f"{float(values['lift_tail10']):.2f}",
                    "头部10% Lift": f"{float(values['lift_head10']):.2f}",
                }
            )

        summary_rows = []
        for record in feature_summary.to_dict(orient="records"):
            summary_rows.append(
                {
                    "特征名": str(record["特征名"]),
                    "变量含义": "" if pd.isna(record["变量含义"]) else str(record["变量含义"]),
                    "特征重要性": _format_optional_float(record["特征重要性"], ".4f"),
                    "IV": _format_optional_float(record["IV"], ".4f"),
                    "PSI": _format_optional_float(record["PSI"], ".4f"),
                    "缺失率": _format_optional_float(record["缺失率"], ".4f"),
                    "坏率形状": str(record["坏率形状"]),
                    "Coef.": _format_optional_float(record["Coef."], ".4f"),
                    "P>|z|": _format_optional_float(record["P>|z|"], ".4f"),
                    "VIF": _format_optional_float(record["VIF"], ".4f"),
                }
            )

        report_data = {
            "报告标题": "LrScorecard模型训练报告",
            "报告说明": "本报告展示本次WOE-LR信用评分卡的特征流转、数据集指标和最终入模变量明细。",
            "训练状态": "成功",
            "算法信息": {
                "算法选择": "LrScorecard",
                "算法说明": "训练期自动分箱 → WOE → 细筛 → 双向逐步回归 → L2逻辑回归 → 评分卡",
                "标签口径": "1=坏客户，0=好客户",
                "特征来源": feature_source,
                "模型参数来源": parameter_source,
            },
            "特征流转": {
                "粗筛候选变量数": int(len(candidate_features)),
                "WOE后细筛保留数": int(fine_selection_count),
                "逐步回归保留数": int(stepwise_count),
                "最终入模变量数": int(len(final_features)),
                "最终入模变量明细": summary_rows,
                "变量指标口径说明": (
                    "本表仅展示最终入模变量。特征重要性、IV、缺失率、坏率形状、Coef.和P>|z|均基于Train；"
                    "IV是评分卡卡方训练箱的IV，也是LrScorecard内部细筛使用的IV，不等同于第三阶段等频分箱IV；"
                    "PSI为主验证集（优先oot_main，否则Test）相对Train、复用Train评分卡切点计算的分箱PSI，"
                    "也不等同于第三阶段特征筛选PSI；P>|z|就是LR p值，VIF用于观察多重共线性。"
                ),
            },
            "模型性能指标": {
                "主验证集": primary_validation,
                "数据集指标对比": dataset_rows,
            },
            "用户指引": {
                "分箱复核": "请查看 binning_review.csv，结合坏率形状、纯好/纯坏箱、WOE、主验证集PSI和未见类别占比复核或调整分箱。",
                "策略部署": "确认分箱和模型后，策略人员可直接使用 scorecard_points.csv 部署评分卡策略；类别变量出现训练期未见值时，按该变量最低箱分兜底。",
                "完整报告": "完整分析请查看 评分卡模型报告.xlsx。",
            },
        }

        baseline_metrics_path = "outputs/baseline_model/baseline_model_metrics.csv"
        if os.path.exists(baseline_metrics_path):
            try:
                baseline_df = pd.read_csv(baseline_metrics_path, encoding="utf-8-sig")
                baseline_row = baseline_df[
                    baseline_df["数据集"].astype(str) == str(primary_validation)
                ]
                if not baseline_row.empty:
                    source = baseline_row.iloc[0]
                    baseline = {
                        "说明": f"以下为基线模型在{primary_validation}上的性能指标，供本次模型训练阶段初步对比。",
                        "数据集": primary_validation,
                    }
                    baseline_source = source.get("基线来源")
                    if pd.notna(baseline_source) and str(baseline_source).strip():
                        baseline["基线来源"] = str(baseline_source)
                    for output_key, source_key in (
                        ("AUC", "AUC"),
                        ("KS", "KS"),
                        ("PSI", "PSI"),
                        ("尾部10% Lift", "Lift_tail10"),
                        ("头部10% Lift", "Lift_head10"),
                    ):
                        value = _to_float_or_none(source.get(source_key))
                        if value is not None:
                            baseline[output_key] = f"{value:.2f}"
                    report_data["基线模型指标"] = baseline
            except Exception as exc:
                print(f"警告: 加载LrScorecard基线指标失败: {exc}")

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as stream:
            json.dump(report_data, stream, ensure_ascii=False, indent=2)
        return output_path
    except Exception as exc:
        print(f"Error generating LrScorecard training JSON report: {exc}")
        return None


def _scorecard_excel_table(sheet, title, required_headers, max_header_scan=80):
    """Read one formatted scorecard table by title and header labels."""
    title_row = None
    for row_index, row in enumerate(sheet.iter_rows(), start=1):
        if any(title in str(cell.value) for cell in row if cell.value is not None):
            title_row = row_index
            break
    if title_row is None:
        raise ValueError(f"评分卡Excel缺少章节: {title}")

    header_row = None
    header_map = None
    scan_end = min(sheet.max_row, title_row + max_header_scan)
    for row_index in range(title_row + 1, scan_end + 1):
        values = [sheet.cell(row_index, column).value for column in range(1, sheet.max_column + 1)]
        positions = {str(value).strip(): index + 1 for index, value in enumerate(values) if value is not None}
        if all(header in positions for header in required_headers):
            header_row = row_index
            header_map = positions
            break
    if header_row is None:
        raise ValueError(f"评分卡Excel章节 {title} 缺少表头: {required_headers}")

    records = []
    for row_index in range(header_row + 1, sheet.max_row + 1):
        record = {
            header: sheet.cell(row_index, header_map[header]).value
            for header in required_headers
        }
        if all(value is None for value in record.values()):
            if records:
                break
            continue
        first_value = record[required_headers[0]]
        if records and isinstance(first_value, str) and (
            first_value.startswith("说明：")
            or first_value.startswith("口径说明：")
            or re.match(r"^\d+、", first_value)
        ):
            break
        records.append(record)
    return records


def _integer_score_range(value):
    text = str(value)
    return re.sub(
        r"(?<![\w])[-+]?\d+(?:\.\d+)?",
        lambda match: str(int(round(float(match.group(0))))),
        text,
    )


def generate_lr_scorecard_evaluation_json(config):
    """Generate stage-5 LrScorecard JSON by reading the stage-4 report only."""
    from openpyxl import load_workbook

    model_train_config_path = Path(config["model_file"]["model_train_config_path"])
    with model_train_config_path.open("r", encoding="utf-8") as stream:
        train_config = json.load(stream)
    train_json_path = Path(
        train_config.get("output_path", {}).get("model_json_path", "results/model_train.json")
    )
    with train_json_path.open("r", encoding="utf-8") as stream:
        train_summary = json.load(stream)
    if train_summary.get("算法信息", {}).get("算法选择") != "LrScorecard":
        raise ValueError("模型训练JSON不是LrScorecard报告，不能生成LrScorecard评估JSON")

    train_root = Path(train_config.get("output_path", {}).get("dst", "outputs/model_train"))
    report_candidates = [
        path
        for path in (train_root / "LR" / "report").glob("评分卡模型报告*.xlsx")
        if path.is_file() and path.stat().st_size > 0
    ]
    if not report_candidates:
        raise FileNotFoundError("未找到非空的LrScorecard评分卡模型报告.xlsx")
    report_path = max(report_candidates, key=lambda path: path.stat().st_mtime)
    workbook = load_workbook(report_path, read_only=True, data_only=True)
    required_sheets = {"2-模型性能", "3-入模变量分析"}
    missing_sheets = sorted(required_sheets - set(workbook.sheetnames))
    if missing_sheets:
        raise ValueError(f"评分卡Excel缺少工作表: {missing_sheets}")

    performance_sheet = workbook["2-模型性能"]
    feature_sheet = workbook["3-入模变量分析"]
    primary_validation = train_summary["模型性能指标"]["主验证集"]
    dataset_display = {
        "train": "训练集",
        "test": "测试集",
        "oot_main": "主OOT",
    }
    primary_display = dataset_display.get(primary_validation, primary_validation)

    # Verify that the selected workbook belongs to the same trained feature set.
    feature_records = _scorecard_excel_table(
        feature_sheet,
        "入模变量重要性及分布情况",
        ["特征名", "变量含义"],
        max_header_scan=10,
    )
    excel_features = [str(record["特征名"]) for record in feature_records]
    expected_features = [
        str(record["特征名"])
        for record in train_summary["特征流转"]["最终入模变量明细"]
    ]
    if excel_features != expected_features:
        raise ValueError("评分卡Excel与模型训练JSON的最终入模变量不一致")

    # The first performance table is transposed: metrics are rows and datasets are columns.
    title_row = None
    for row_index, row in enumerate(performance_sheet.iter_rows(), start=1):
        if any("模型性能验证指标" in str(cell.value) for cell in row if cell.value is not None):
            title_row = row_index
            break
    if title_row is None:
        raise ValueError("评分卡Excel缺少模型性能验证指标")
    header_row = None
    for row_index in range(title_row + 1, min(performance_sheet.max_row, title_row + 15) + 1):
        values = [performance_sheet.cell(row_index, column).value for column in range(1, performance_sheet.max_column + 1)]
        if "统计项" in values:
            header_row = row_index
            break
    if header_row is None:
        raise ValueError("评分卡Excel缺少模型性能验证指标表头")
    header_values = [performance_sheet.cell(header_row, column).value for column in range(1, performance_sheet.max_column + 1)]
    metric_column = header_values.index("统计项") + 1
    dataset_columns = {
        str(value): column
        for column, value in enumerate(header_values, start=1)
        if value is not None and str(value) != "统计项"
    }
    metric_rows = {}
    for row_index in range(header_row + 1, performance_sheet.max_row + 1):
        metric = performance_sheet.cell(row_index, metric_column).value
        if metric is None:
            break
        metric_rows[str(metric)] = row_index
    for metric in ("样本数", "坏样本率", "AUC", "KS", "PSI"):
        if metric not in metric_rows:
            raise ValueError(f"评分卡Excel模型性能表缺少指标: {metric}")

    performance_rows = []
    stability_rows = []
    raw_performance_lookup = {}
    for display_name, column in dataset_columns.items():
        psi_value = performance_sheet.cell(metric_rows["PSI"], column).value
        raw_auc = float(performance_sheet.cell(metric_rows["AUC"], column).value)
        raw_ks = float(performance_sheet.cell(metric_rows["KS"], column).value)
        raw_performance_lookup[display_name] = {"AUC": raw_auc, "KS": raw_ks}
        performance_rows.append(
            {
                "数据集": display_name,
                "样本数": int(performance_sheet.cell(metric_rows["样本数"], column).value),
                "坏样本率": f"{float(performance_sheet.cell(metric_rows['坏样本率'], column).value):.2%}",
                "AUC": f"{raw_auc:.2f}",
                "KS": f"{raw_ks:.2f}",
                "评分PSI": "-" if psi_value in (None, "\\") else f"{float(psi_value):.4f}",
            }
        )
        if psi_value not in (None, "\\"):
            numeric_psi = float(psi_value)
            stability_rows.append(
                {
                    "数据集": display_name,
                    "相对Train评分PSI": f"{numeric_psi:.4f}",
                    "稳定性判断": "稳定" if numeric_psi < 0.1 else ("略有变化" if numeric_psi <= 0.25 else "不稳定"),
                }
            )

    train_metric_lookup = {
        str(record["数据集"]): record
        for record in train_summary["模型性能指标"]["数据集指标对比"]
    }
    excel_metric_lookup = {record["数据集"]: record for record in performance_rows}
    if set(train_metric_lookup) != set(excel_metric_lookup):
        raise ValueError("评分卡Excel与模型训练JSON的数据集名单不一致")
    for display_name, train_record in train_metric_lookup.items():
        excel_record = excel_metric_lookup[display_name]
        comparable_fields = ("样本数", "坏样本率", "AUC", "KS")
        if any(train_record[field] != excel_record[field] for field in comparable_fields):
            raise ValueError(
                f"评分卡Excel与模型训练JSON的{display_name}样本数、坏率、AUC或KS不一致"
            )
    if primary_display not in excel_metric_lookup:
        raise ValueError(f"评分卡Excel缺少主验证集指标: {primary_display}")

    channel_records = _scorecard_excel_table(
        performance_sheet,
        "分渠道模型效果",
        ["数据集", "渠道", "样本数", "坏样本数", "坏样本率", "AUC", "KS"],
        max_header_scan=10,
    )
    channel_rows = []
    for record in channel_records:
        if str(record["数据集"]) != primary_display:
            continue
        channel_rows.append(
            {
                "渠道": str(record["渠道"]),
                "样本数": int(record["样本数"]),
                "坏样本数": int(record["坏样本数"]),
                "坏样本率": f"{float(record['坏样本率']):.2%}",
                "AUC": _format_optional_float(record["AUC"], ".2f"),
                "KS": _format_optional_float(record["KS"], ".2f"),
            }
        )

    lift_records = _scorecard_excel_table(
        performance_sheet,
        "模型尾部区分能力",
        ["数据集", "统计项", "TOP 1%", "TOP 3%", "TOP 5%", "TOP 10%", "TOTAL"],
        max_header_scan=10,
    )
    lift_lookup = {
        str(record["统计项"]): record
        for record in lift_records
        if str(record["数据集"]) == primary_display
    }
    if not {"坏样本率", "LIFT值"}.issubset(lift_lookup):
        raise ValueError(f"评分卡Excel缺少{primary_display}的Lift统计")
    lift_summary = []
    for label in ("TOP 1%", "TOP 3%", "TOP 5%", "TOP 10%"):
        lift_summary.append(
            {
                "范围": label,
                "坏样本率": f"{float(lift_lookup['坏样本率'][label]):.2%}",
                "Lift": f"{float(lift_lookup['LIFT值'][label]):.2f}",
            }
        )

    ranking_records = _scorecard_excel_table(
        performance_sheet,
        f"{primary_display}评分排序性",
        ["分箱", "样本总数", "坏样本数", "坏样本率", "LIFT值", "累积LIFT值", "分档KS值"],
        max_header_scan=60,
    )
    score_bins = []
    for record in ranking_records:
        if str(record["分箱"]) == "合计":
            continue
        score_bins.append(
            {
                "评分区间": _integer_score_range(record["分箱"]),
                "样本数": int(record["样本总数"]),
                "坏样本数": int(record["坏样本数"]),
                "坏样本率": f"{float(record['坏样本率']):.2%}",
                "Lift": f"{float(record['LIFT值']):.2f}",
                "累积Lift": f"{float(record['累积LIFT值']):.2f}",
                "分档KS": f"{float(record['分档KS值']):.2f}",
            }
        )

    has_baseline = "基线模型指标" in train_summary
    baseline_metric_lookup = {}
    if has_baseline:
        baseline_metrics_path = Path("outputs/baseline_model/baseline_model_metrics.csv")
        if baseline_metrics_path.exists():
            baseline_df = pd.read_csv(baseline_metrics_path, encoding="utf-8-sig")
            if {"数据集", "KS"}.issubset(baseline_df.columns):
                baseline_metric_lookup = {
                    str(record["数据集"]): record
                    for record in baseline_df.to_dict("records")
                }

        display_to_dataset = {
            "训练集": "train",
            "测试集": "test",
            "主OOT": "oot_main",
        }
        for record in performance_rows:
            dataset_key = display_to_dataset.get(str(record["数据集"]), str(record["数据集"]))
            baseline_record = baseline_metric_lookup.get(dataset_key, {})
            baseline_ks = _to_float_or_none(baseline_record.get("KS"))
            current_ks = raw_performance_lookup[str(record["数据集"])]["KS"]
            record["基线模型KS"] = "-" if baseline_ks is None else f"{baseline_ks:.2f}"
            record["KS Gap"] = (
                "-"
                if baseline_ks is None or current_ks is None
                else f"{current_ks - baseline_ks:.2f}"
            )
    report_description = (
        "本报告复用LrScorecard训练阶段的完整评分卡Excel结果，展示各数据集模型效果、"
        "评分稳定性、主验证集分渠道效果、Lift和评分等频10组。"
    )
    if has_baseline:
        report_description += "同时展示训练阶段已加载的基线模型指标。"
    report_data = {
        "报告标题": "模型评估报告",
        "报告说明": report_description,
        "评估数据集": "验证集(oot_main)" if primary_validation == "oot_main" else "测试集(test)",
        "模型信息": {
            "模型类型": "传统WOE-LR信用评分卡",
            "模型名称": "LrScorecard",
            "模型描述": "基于自动或人工分箱、WOE、细筛、双向逐步回归和逻辑回归的传统信用评分卡模型。",
        },
        "模型效果评估": performance_rows,
        "模型稳定性评估": {
            "PSI判断标准": "<0.1稳定；0.1~0.25略有变化；>0.25不稳定",
            "各数据集相对Train评分PSI": stability_rows,
        },
        "分渠道模型效果": channel_rows,
        "Lift分析结果": lift_summary,
        "分数分箱统计": score_bins,
    }
    output_path = Path(config["output_path"]["eval_json_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(report_data, stream, ensure_ascii=False, indent=2)
    workbook.close()
    print(f"LrScorecard模型评估JSON已从训练报告加载生成: {output_path}")
    return str(output_path)


def generate_autogluon_train_json(leaderboard, output_path, autogluon_supported, use_fea=None, use_fea_source=None):
    """
    生成AutoGluon多模型对比的JSON报告
    参数:
        leaderboard: DataFrame, AutoGluon输出的leaderboard
        output_path: str, 输出路径
        autogluon_supported: list, 对比的模型类型列表
    """
    try:
        # 保留所有float列4位小数
        float_cols = leaderboard.select_dtypes(include=['float', 'float64']).columns
        leaderboard[float_cols] = leaderboard[float_cols].round(4)
        # 只展示前15行
        if len(leaderboard) > 15:
            leaderboard_show = leaderboard.head(15)
        else:
            leaderboard_show = leaderboard
        feature_info = None
        if use_fea is not None:
            try:
                use_fea_list = list(use_fea)
            except Exception:
                use_fea_list = []
            feature_info = {
                "入模特征来源": use_fea_source or "未知",
                "入模特征数量": int(len(use_fea_list))
            }

        report_data = {
            "报告标题": "模型训练报告",
            "报告说明": f"本报告展示AutoGluon多模型对比的训练结果，对比模型包括：{', '.join(autogluon_supported)}。报告涵盖各模型在验证集上的性能排名（leaderboard），便于选择最优模型。",
            **({"入模特征信息": feature_info} if feature_info is not None else {}),
            "对比模型列表": autogluon_supported,
            "leaderboard": leaderboard_show.to_dict(orient='records')
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        return output_path
    except Exception as e:
        print(f"Error generating AutoGluon training JSON report: {str(e)}")
        return None


def generate_evaluation_json(train_combined_df, all_eval_results, psi_value, lift_res, output_path, train_date_range, eval_dataset_key, date_col, model_info=None, score_transform_result=None, psi_details=None):
    """
    生成模型评估的JSON报告
    
    参数:
        train_combined_df: DataFrame, 训练集合并的模型效果数据（包含overall和channel）
        all_eval_results: dict, 所有评估数据集的结果 {dataset_name: (combined_df, display_name, date_range)}
        psi_value: float, 模型PSI值
        lift_res: dict, Lift分析结果
        output_path: str, 输出路径
        train_date_range: str, 训练集时间范围
        eval_dataset_key: str, 评估数据集key（如 'oot_main', 'test'）
        date_col: str, 日期列名
        model_info: dict, 模型信息（可选），包含model_type, model_name等
        score_transform_result: dict, 分数转换结果（可选）
    """
    try:
        # 0. 检查基线模型数据（两个文件都存在才进行对比）
        baseline_metrics_path = "outputs/baseline_model/baseline_model_metrics.csv"
        baseline_channel_path = f"outputs/baseline_model/{eval_dataset_key}_channel_auc_ks.csv"
        
        baseline_df, baseline_channel_df = None, None
        if os.path.exists(baseline_metrics_path):
            try:
                baseline_df = pd.read_csv(baseline_metrics_path, encoding='utf-8-sig')
            except Exception:
                baseline_df = None
        if os.path.exists(baseline_channel_path):
            try:
                baseline_channel_df = pd.read_csv(baseline_channel_path, encoding='utf-8-sig')
            except Exception:
                baseline_channel_df = None
        
        # 1. 判断是否有基线模型 + 报告说明
        if baseline_df is not None:
            has_baseline = True
            report_description = "本报告包含新模型的完整评估结果，并与基线模型进行对比分析。报告涵盖：模型信息、各数据集模型效果对比、评估集的模型稳定性和分渠道效果对比、Lift分析结果及分数转换评估。"
        else:
            has_baseline = False
            report_description = "本报告包含新模型的完整评估结果，涵盖：模型信息、各数据集模型效果、评估集的模型稳定性和分渠道效果、Lift分析结果及分数转换评估。"
        has_baseline_channel = baseline_channel_df is not None
        
        # 2. 模型信息
        if model_info and model_info.get('model_type') == 'autogluon':
            model_name = model_info.get('model_name')
            from utils.utils_model_evaluation import get_model_description
            desc = get_model_description(model_name)
            model_info_section = {
                "模型类型": "AutoGluon多模型对比",
                "模型名称": model_name,
                "模型描述": desc
            }
        else:
            model_info_section = {
                "模型类型": "传统的LightGBM",
                "模型名称": "LightGBM",
                "模型描述": "该LightGBM模型是常规风控模型算法，是微软开发的高效梯度提升框架，采用基于直方图的算法，训练速度快，内存占用低。支持类别特征直接输入，适合大规模数据，支持并行和GPU加速，在保持高精度的同时训练效率高。"
            }
        
        # 3. 模型效果评估（全部数据集）
        model_performance_list = []
        
        # 处理训练集数据
        train_overall_row = train_combined_df[train_combined_df['col'] == 'overall'].iloc[0]
        train_perf = {
            "数据集": "训练集",
            "时间范围": train_date_range,
            "样本总数": int(train_overall_row['total']),
            "好样本": int(train_overall_row['good']),
            "坏样本": int(train_overall_row['bad']),
            "坏率": f"{train_overall_row['bad_ratio']:.2%}",
            "AUC": f"{train_overall_row['auc']:.4f}",
            "KS": f"{train_overall_row['ks']:.4f}"
        }
        if has_baseline:
            rows = baseline_df[baseline_df['数据集'] == 'train']
            if len(rows) > 0:
                row = rows.iloc[0]
                baseline_ks = _to_float_or_none(row.get('KS'))
                baseline_auc = _to_float_or_none(row.get('AUC'))
                if baseline_ks is not None:
                    train_perf["基线KS"] = f"{baseline_ks:.4f}"
                    ks_diff = float(train_overall_row['ks']) - baseline_ks
                    train_perf["KS提升"] = f"{ks_diff:.4f}" if abs(ks_diff) > 1e-6 else "0.0000"
                if baseline_auc is not None:
                    train_perf["基线AUC"] = f"{baseline_auc:.4f}"
        model_performance_list.append(train_perf)
        
        # 处理所有评估数据集（test + 所有OOT数据集）
        for dataset_name, (combined_df, display_name, date_range) in all_eval_results.items():
            dataset_overall_row = combined_df[combined_df['col'] == 'overall'].iloc[0]
            dataset_perf = {
                "数据集": display_name,
                "时间范围": date_range,
                "样本总数": int(dataset_overall_row['total']),
                "好样本": int(dataset_overall_row['good']),
                "坏样本": int(dataset_overall_row['bad']),
                "坏率": f"{dataset_overall_row['bad_ratio']:.2%}",
                "AUC": f"{dataset_overall_row['auc']:.4f}",
                "KS": f"{dataset_overall_row['ks']:.4f}"
            }
            if has_baseline:
                rows = baseline_df[baseline_df['数据集'] == dataset_name]
                if len(rows) > 0:
                    row = rows.iloc[0]
                    baseline_ks = _to_float_or_none(row.get('KS'))
                    baseline_auc = _to_float_or_none(row.get('AUC'))
                    if baseline_ks is not None:
                        dataset_perf["基线KS"] = f"{baseline_ks:.4f}"
                        ks_diff = float(dataset_overall_row['ks']) - baseline_ks
                        dataset_perf["KS提升"] = f"{ks_diff:.4f}" if abs(ks_diff) > 1e-6 else "0.0000"
                    if baseline_auc is not None:
                        dataset_perf["基线AUC"] = f"{baseline_auc:.4f}"
            model_performance_list.append(dataset_perf)
        
        # 4. 模型稳定性评估（评估数据集）
        stability_section = {
            "PSI值": f"{psi_value:.4f}",
            "稳定性评估": "模型稳定性良好" if psi_value < 0.1 else "模型稳定性较差"
        }
        if psi_details:
            stability_section["PSI明细"] = {
                str(name): f"{float(value):.4f}"
                for name, value in psi_details.items()
            }
        
        # 如果有基线模型，添加基线PSI
        if has_baseline:
            baseline_row = baseline_df[baseline_df['数据集'] == eval_dataset_key]
            if len(baseline_row) > 0:
                baseline_psi = baseline_row.iloc[0]['PSI']
                baseline_psi_value = _to_float_or_none(baseline_psi)
                if baseline_psi_value is not None:
                    stability_section.setdefault("PSI明细", {})
                    stability_section["PSI明细"]["基线PSI"] = f"{baseline_psi_value:.4f}"
        
        # 5. 分渠道模型效果（评估数据集）
        eval_combined_df = all_eval_results[eval_dataset_key][0]
        eval_channel_df = eval_combined_df[eval_combined_df['col'] != 'overall']
        
        channel_list = []
        for _, row in eval_channel_df.iterrows():
            channel_perf = {
                "渠道": row['col'],
                "样本总数": int(row['total']),
                "好样本": int(row['good']),
                "坏样本": int(row['bad']),
                "坏率": f"{row['bad_ratio']:.2%}",
                "AUC": f"{row['auc']:.4f}",
                "KS": f"{row['ks']:.4f}"
            }
            # 添加基线对比（分渠道）
            if has_baseline_channel:
                ch_rows = baseline_channel_df[baseline_channel_df['col'] == row['col']]
                if len(ch_rows) > 0:
                    ch_row = ch_rows.iloc[0]
                    baseline_ks = _to_float_or_none(ch_row.get('ks'))
                    baseline_auc = _to_float_or_none(ch_row.get('auc'))
                    if baseline_ks is not None:
                        channel_perf["基线KS"] = f"{baseline_ks:.4f}"
                        ks_diff = float(row['ks']) - baseline_ks
                        channel_perf["KS提升"] = f"{ks_diff:.4f}" if abs(ks_diff) > 1e-9 else "0.0000"
                    if baseline_auc is not None:
                        channel_perf["基线AUC"] = f"{baseline_auc:.4f}"
            channel_list.append(channel_perf)

        # 6. Lift分析结果（评估数据集）
        lift_df = lift_res['lift_top_20']
        base_rate = lift_res['base_rate']
        
        # Lift概览（直接使用传入的计算结果）
            # Lift对比（统一格式：指标名 + 当前模型/基线模型）
        if has_baseline:
            baseline_eval_rows = baseline_df[baseline_df['数据集'] == eval_dataset_key]
            b_row = baseline_eval_rows.iloc[0] if len(baseline_eval_rows) > 0 else None
            lift_summary = [
                {"指标": "尾部10%Lift", "当前模型": f"{lift_res.get('lift_tail10', 0):.4f}", "基线模型": _format_optional_float(b_row.get('Lift_tail10') if b_row is not None else None)},
                {"指标": "头部10%Lift", "当前模型": f"{lift_res.get('lift_head10', 0):.4f}", "基线模型": _format_optional_float(b_row.get('Lift_head10') if b_row is not None else None)},
                {"指标": "头部5%Lift", "当前模型": f"{lift_res.get('lift_head5', 0):.4f}", "基线模型": _format_optional_float(b_row.get('Lift_head5') if b_row is not None else None)},
                {"指标": "头部2%Lift", "当前模型": f"{lift_res.get('lift_head2', 0):.4f}", "基线模型": _format_optional_float(b_row.get('Lift_head2') if b_row is not None else None)},
                {"指标": "头部1%Lift", "当前模型": f"{lift_res.get('lift_head1', 0):.4f}", "基线模型": _format_optional_float(b_row.get('Lift_head1') if b_row is not None else None)},
            ]
        else:
            lift_summary = [
                {"指标": "尾部10%Lift", "当前模型": f"{lift_res.get('lift_tail10', 0):.4f}"},
                {"指标": "头部10%Lift", "当前模型": f"{lift_res.get('lift_head10', 0):.4f}"},
                {"指标": "头部5%Lift", "当前模型": f"{lift_res.get('lift_head5', 0):.4f}"},
                {"指标": "头部2%Lift", "当前模型": f"{lift_res.get('lift_head2', 0):.4f}"},
                {"指标": "头部1%Lift", "当前模型": f"{lift_res.get('lift_head1', 0):.4f}"},
            ]
        
        # 分箱明细（展示全部分箱）
        lift_details = []
        for _, row in lift_df.iterrows():
            lift_details.append({
                "分箱": int(row['level']),
                "概率区间": row['range'],
                "样本数": int(row['num']),
                "好样本": int(row['good']),
                "坏样本": int(row['bad']),
                "坏率": f"{row['bad_ration']:.2%}",
                "基准坏率": f"{base_rate:.2%}",
                "Lift值": f"{row['lift']:.2f}"
            })
        
        # 7. 分数转换结果（如果有）
        score_transform_section = None
        if score_transform_result is not None:
            df_score = score_transform_result['df_score']
            score_bins = []
            for _, row in df_score.iterrows():
                score_bins.append({
                    "分数区间": str(row['range']),
                    "样本数": int(row['num']),
                    "好样本": int(row['good']),
                    "坏样本": int(row['bad']),
                    "坏率": f"{row['bad_ration']:.2%}",
                    "样本占比": f"{row['num_ration']:.2%}"
                })
            
            score_transform_section = {
                "转换参数": score_transform_result['params'],
                "分数切点": score_transform_result['cut_points'],
                "坏率是否单调": "是" if score_transform_result['is_bad_rate_monotonic'] else "否",
                "样本分布是否正态": "是" if score_transform_result['is_num_normal'] else "否",
                "分数分箱统计": score_bins
            }
        
        # 整合报告（扁平化结构，中文标题）
        report_data = {
            "报告标题": "模型评估报告",
            "报告说明": report_description,
            "评估数据集": all_eval_results[eval_dataset_key][1],
            "模型信息": model_info_section,
            "模型效果评估": model_performance_list,
            "模型稳定性评估": stability_section,
            "分渠道模型效果": channel_list,
            "Lift分析结果": {
                "Lift概览": lift_summary,
                "分箱明细": lift_details
            }
        }
        
        # 添加分数转换结果（如果有）
        if score_transform_section is not None:
            report_data["分数转换结果"] = score_transform_section
        
        # 转换为JSON并保存
        json_str = json.dumps(report_data, ensure_ascii=False, indent=2)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json_str)
        return output_path
            
    except Exception as e:
        print(f"Error generating model evaluation JSON report: {str(e)}")
        return None
