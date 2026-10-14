import pandas as pd
import warnings
from utils.data_utils import identify_types
from sklearn.model_selection import train_test_split
import os
import json
import traceback
import copy
import glob
from utils.data_utils import get_data, save_data

BASELINE_OPTIONAL_COLS = {"baseline_prob", "baseline_score"}

def drop_baseline_optional_cols(df):
    return df.drop(columns=list(BASELINE_OPTIONAL_COLS), errors='ignore')

def compute_stats(df, label_col):
    # 处理缺失值：缺失值不算灰样本，直接drop掉
    valid_data = df.dropna(subset=[label_col])
    return {
        'n_samples': len(valid_data),
        'good': (valid_data[label_col] == 0).sum(),
        'bad': (valid_data[label_col] == 1).sum(),
        # 修改：除了0、1之外的所有值都算灰样本
        'grey': (~valid_data[label_col].isin([0, 1])).sum(),
        # 统一口径：仅 label==1 计为坏样本，其余（含灰样本）计为0
        'bad_rate': (valid_data[label_col] == 1).mean()
    }

def compute_monthly_stats(df, label_col, time_col):
    df['month'] = pd.to_datetime(df[time_col], format='%Y%m%d').dt.to_period('M').astype(str)
    grouped = df.groupby('month')
    results = []
    for month, group in grouped:
        stats = compute_stats(group, label_col)
        stats['month'] = month
        results.append(stats)
    return pd.DataFrame(results)

def spliter_report(dev_datasets, oot_datasets, time_col, label_col, save_path):
    """
    生成数据集分布报告
    
    参数:
        dev_datasets: 开发数据集列表，每个元素为 (df, name) 元组，如 [(train_df, 'DEV'), (test_df, 'TEST')]
        oot_datasets: OOT数据集列表，每个元素为 (df, name) 元组，可能为空，如 [(oot_df, 'oot_name')]
        time_col: 时间列名
        label_col: 标签列名
        save_path: 保存路径
    
    返回:
        df_all: 数据集分布统计DataFrame
    """
    result = []

    # 处理开发数据集
    for df, name in dev_datasets:
        # Overall stats
        result.append(pd.DataFrame([compute_stats(df.copy(), label_col)]).assign(dataset=name, month='Overall'))
        # Monthly stats
        result.append(compute_monthly_stats(df.copy(), label_col, time_col).assign(dataset=name))
    
    # 处理OOT数据集
    for df, name in oot_datasets:
        # Overall stats
        result.append(pd.DataFrame([compute_stats(df.copy(), label_col)]).assign(dataset=name, month='Overall'))
        # Monthly stats
        result.append(compute_monthly_stats(df.copy(), label_col, time_col).assign(dataset=name))

    df_all = pd.concat(result, ignore_index=True)
    df_all = df_all[['dataset', 'month', 'n_samples', 'good', 'bad', 'grey', 'bad_rate']]
    print("=" * 60)
    print('数据集分布统计')
    print(df_all[df_all['month'] == 'Overall'])
    
    df_all.to_csv(save_path, index=False)
    
    return df_all

#数据清洗-样本选择
def clean_input_data(data,label,channel_col,num_fea,cat_fea, product_list_keep=None, label_list_keep=None,fill_val=-9999,min_sample_size=500):
    """
    模型训练前的数据清洗函数

    参数说明：
    - data: 原始DataFrame
    - num_fea: 数值特征-转换为数值
    - cat_fea: 类别特征-转换为类别
    - product_list_keep: 保留的产品渠道列表（为空则不筛选）
    - label_list_keep: 保留的标签列表（为空则默认保留[0, 1]）
    - fill_val: 缺失值填充值
    - start_month/end_month: 月份筛选（字符串格式 "202301"）
    - min_sample_size: 筛选后最小样本数，默认500

    返回：
    - 清洗后的数据
    """
    df_clear = copy.deepcopy(data)
    # 渠道字段标准化：统一非空字符串，避免后续分组/排序时出现类型比较异常
    if channel_col in df_clear.columns:
        df_clear[channel_col] = (
            df_clear[channel_col]
            .fillna('MISSING')
            .astype(str)
            .str.strip()
            .replace('', 'MISSING')
        )

    # 渠道筛选
    if len(product_list_keep) !=0:
        df_clear = df_clear[df_clear[channel_col].isin(product_list_keep)]

    # 标签筛选
    if len(label_list_keep)==0:
        label_list_keep = [0, 1]
    df_clear = df_clear[df_clear[label].isin(label_list_keep)]

    # # 时间筛选
    # if start_date and end_date:
    #     df_clear[date_col] = df_clear[date_col].astype(str)
    #     df_clear = df_clear[(df_clear[date_col] >= start_date) & (df_clear[date_col] <= end_date)]

    # 缺失值填充
    for i in num_fea:
        df_clear[i] = df_clear[i].fillna(fill_val)
        df_clear[i] = df_clear[i].replace('', fill_val)
    
    str_fill_val = str(fill_val)
    for i in cat_fea:
        df_clear[i] = df_clear[i].fillna(str_fill_val)
        df_clear[i] = df_clear[i].replace('', str_fill_val)
    
    df_clear.reset_index(drop=True, inplace=True)

    # 样本量校验
    if len(df_clear) < min_sample_size:
        warnings.warn(f"样本量仅为 {len(df_clear)}，小于设定阈值 {min_sample_size}，可能不适合建模！")

    df_clear = df_clear.reset_index(drop=True)
    # print(f"清洗后大小为 {df_clear.shape[0]} 行 {df_clear.shape[1]} 列")    
    return df_clear

# 时间范围校验
def check_range_valid(name, date_range):
    if len(date_range) != 2 or date_range[0] > date_range[1]:
        raise ValueError(f"{name} 的时间范围设置不合理: {date_range}")
    
###  按照指定的时间范围划分
def split_by_date_range(data,date_col,train_range,oot_range,psi_range):
    """
    按照指定的时间范围划分数据集，要求所有范围都不为空，否则抛出异常。
    只在 oot_range != psi_range 时生成 psi_df，否则返回 None。
    """
    if not train_range or not oot_range or not psi_range:
        raise ValueError(f"[split_by_date_range] train_range/oot_range/psi_range 不能有空值，当前值：train_range={train_range}, oot_range={oot_range}, psi_range={psi_range}")
    check_range_valid('DEV_SPLIT', train_range)
    check_range_valid('OOT_SPLIT', oot_range)
    check_range_valid('PSI_SPLIT', psi_range)
    # 切分数据 -类型转为int，保证比较正确
    data[date_col] = data[date_col].astype(int)
    train_df = data[(data[date_col] >= train_range[0]) & (data[date_col] <= train_range[1])].copy()
    test_df = data[(data[date_col] >= oot_range[0]) & (data[date_col] <= oot_range[1])].copy()
    # 只在 oot_range != psi_range 时生成 psi_df
    if oot_range != psi_range:
        psi_df = data[(data[date_col] >= psi_range[0]) & (data[date_col] <= psi_range[1])].copy()
    else:
        psi_df = None
    return train_df, test_df, psi_df
    
# ... 按照时间排序-比例划分 ...
def split_by_date_proportion(data, date_col, test_size=0.2):
    """
    按时间排序后用比例切分，前面为训练集，后面为测试集
    返回train_df, test_df, None（psi_df始终等于test_df，不需要单独生成）
    """
    data_sorted = data.sort_values(by=date_col)
    n = len(data_sorted)
    split_idx = int(n * (1 - test_size))
    train_df = data_sorted.iloc[:split_idx].copy()
    test_df = data_sorted.iloc[split_idx:].copy()
    # psi_df始终等于test_df，不需要单独生成，返回None
    return train_df, test_df, None

def split_randomly(data,test_size = 0.2,random_state = 42):
    """
    随机切分数据集
    返回train_df, test_df, None（psi_df始终等于test_df，不需要单独生成）
    """
    train_df, test_df = train_test_split(data, test_size=test_size, random_state=random_state)
    # psi_df始终等于test_df，不需要单独生成，返回None
    return train_df, test_df, None

    
#============================数据清洗主程序==========================================    
def data_clear_func(df, config, oot_files=None):
    """
    数据清洗主函数
    
    参数:
        df: 原始数据DataFrame
        config: 配置字典
        oot_files: OOT文件路径列表（可选）
    """
    try:
        save_dir=config['output_path']['dst_data']
        data_path = config['data_path']
        base_fea = config['settings']['base_fea']
        date_col = config['settings']['date_col']
        channel_col = config['settings']['channel_col']
        label = config['settings']['label']
        product_list_keep = config['settings']['product_list_keep']
        label_list_keep = config['settings']['label_list_keep']
        fill_val = config['settings']['fill_val']
        
        os.makedirs(save_dir, exist_ok=True)
        # 数据识别；上传的基线预测列只用于基线对比，不作为普通建模特征处理。
        remove_fea = list(dict.fromkeys(list(base_fea) + list(BASELINE_OPTIONAL_COLS)))
        num_fea, cat_fea = identify_types(df=df,remove_fea=remove_fea)
        # 样本筛选和缺失值填充
        data = clean_input_data(data=df, channel_col =channel_col, label =label,num_fea =num_fea, cat_fea=cat_fea,
                                product_list_keep =product_list_keep,label_list_keep =label_list_keep,fill_val =fill_val,min_sample_size=500)
        
        # 保存清洗后的所有数据到指定路径，不保存索引
        cleaned_data_path = config['output_path']['cleaned_data_path']
        # data.to_csv(cleaned_data_path, index=False) 
        
        
        # 清洗 oot_*.csv 文件
        oot_datasets = []
        if oot_files:
            print(f"找到 {len(oot_files)} 个OOT文件，开始清洗...")
            expected_columns = list(data.columns)
            expected_required_columns = set(expected_columns) - BASELINE_OPTIONAL_COLS
            
            for oot_file in oot_files:
                filename = os.path.basename(oot_file)
                dataset_name = os.path.splitext(filename)[0]
                
                # 使用 get_data 统一读取，并检查列名
                try:
                    oot_df = get_data(oot_file, required_columns=None)
                    oot_columns = set(oot_df.columns)
                    
                    # 检查列名是否一致；baseline_prob/baseline_score 是可选基线列，不要求所有数据集都有。
                    oot_required_columns = oot_columns - BASELINE_OPTIONAL_COLS
                    if oot_required_columns != expected_required_columns:
                        missing_cols = expected_required_columns - oot_required_columns
                        extra_cols = oot_required_columns - expected_required_columns
                        if missing_cols:
                            error_msg = f"失败：{filename} 缺少列 {missing_cols}"
                            print(error_msg)
                            raise ValueError(error_msg)
                        if extra_cols:
                            # 只保留与data.csv相同的业务列，同时保留OOT自身携带的可选基线列。
                            keep_columns = [col for col in oot_df.columns if col in expected_required_columns or col in BASELINE_OPTIONAL_COLS]
                            oot_df = oot_df[keep_columns]
                    
                    # 使用相同的清洗逻辑
                    oot_cleaned = clean_input_data(
                        data=oot_df,
                        channel_col=channel_col,
                        label=label,
                        num_fea=num_fea,
                        cat_fea=cat_fea,
                        product_list_keep=product_list_keep,
                        label_list_keep=label_list_keep,
                        fill_val=fill_val,
                        min_sample_size=500
                    )

                    output_path = os.path.join(save_dir, dataset_name)  # 不带扩展名，由save_data决定
                    actual_path = save_data(drop_baseline_optional_cols(oot_cleaned), output_path)
                    
                    # 添加到oot_datasets列表（名称转换为大写）
                    oot_datasets.append((oot_cleaned, dataset_name.upper()))
                    
                except Exception as e:
                    error_msg = f"OOT文件 {filename} 清洗失败: {str(e)}"
                    print(error_msg)
                    raise ValueError(error_msg)
        
        return data, oot_datasets
    except Exception as e:
        msg=traceback.format_exc()
        print(msg)
        raise


  

#============================数据切分主程序==========================================  
def data_split_func(orl_data, config, oot_files=None):
    try:
        ## 在这里加载数据清洗流程
        data, oot_datasets = data_clear_func(orl_data, config, oot_files)

        save_dir = config['output_path']['dst_data']
        dst_rp = config['output_path']['dst_rp']
        dst_rp_path = config['output_path']['dst_rp_path']
        clean_json_path = config['output_path']['clean_json_path']
        # 解包配置
        id_col = config['settings']['id_col']
        date_col = config['settings']['date_col']
        channel_col = config['settings']['channel_col']
        label = config['settings']['label']
        method = config['settings']['method'].lower()
        random_state = config['settings']['random_state']
        test_size = config['settings']['test_size']
        train_range = config['settings'].get('train_split_range', [])
        oot_range = config['settings'].get('oot_split_range', [])
        psi_range = config['settings'].get('psi_split_range', [])
        
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(dst_rp, exist_ok=True)
        
        # 检查是否存在清洗后的oot_main数据（优先使用，支持parquet和csv格式）
        # 注意：这里只是预检查，实际使用时会在保存psi时再次检查
        from utils.data_utils import get_data_path
        oot_main_base = os.path.join(save_dir, 'oot_main')
        oot_main_path = get_data_path(oot_main_base)
        use_oot_main_for_psi = os.path.exists(oot_main_path)
    
        # 划分数据集
        if method == 'date_range':
            train_df, test_df, psi_df = split_by_date_range(data, date_col, train_range, oot_range, psi_range) 
        elif method == 'date_proportion':
            train_df, test_df, psi_df = split_by_date_proportion(data, date_col, test_size)
        else:  # method == 'random'
            train_df, test_df, psi_df = split_randomly(data, test_size, random_state)
        
        train_time = f"{train_df[date_col].min()} - {train_df[date_col].max()}"
        test_time = f"{test_df[date_col].min()} - {test_df[date_col].max()}"
        print(f"[数据切分方式] {method} | 训练集时间范围: {train_time} | 测试集时间范围: {test_time}")

        # 保存清洗后的数据为 Parquet 文件（高压缩、快速读取）
        dev_path = save_data(drop_baseline_optional_cols(train_df), config['output_path']['dev_data_path'])
        oot_path = save_data(drop_baseline_optional_cols(test_df), config['output_path']['oot_data_path'])
        
        # 处理psi数据：优先使用oot_main，否则根据psi_df是否为None决定
        psi_data_path = config['output_path']['psi_data_path']
        import shutil
        
        # 查找oot_main文件（支持parquet和csv格式）
        from utils.data_utils import get_data_path
        oot_main_base = os.path.join(save_dir, 'oot_main')
        oot_main_actual = get_data_path(oot_main_base)
        use_oot_main_for_psi = os.path.exists(oot_main_actual)
        
        if use_oot_main_for_psi:
            # 如果存在oot_main，复制为psi（保持相同格式）
            psi_base = os.path.splitext(psi_data_path)[0]
            ext = os.path.splitext(oot_main_actual)[1]
            shutil.copy(oot_main_actual, psi_base + ext)
            print(f"使用 {os.path.basename(oot_main_actual)} 作为 psi 数据集")
        elif psi_df is None:
            # 如果psi_df为None，说明psi_df等于test_df，复制test数据
            psi_base = os.path.splitext(psi_data_path)[0]
            ext = os.path.splitext(oot_path)[1]
            shutil.copy(oot_path, psi_base + ext)
            print(f"使用 TEST 数据集作为 psi 数据集")
        else:
            # 只有真正需要单独保存psi_df时才保存
            save_data(drop_baseline_optional_cols(psi_df), psi_data_path)
 
        # 构建开发数据集列表：DEV和TEST一定有
        dev_datasets = [(train_df, 'DEV'), (test_df, 'TEST')]
        
        # 生成数据集分布报告
        df_all = spliter_report(dev_datasets, oot_datasets, date_col, label, dst_rp_path)
        
        # 基线模型评估（在所有数据集确认之后）
        baseline_metrics_df = pd.DataFrame()
        baseline_status_info = None
        from utils.utils_baseline_model import evaluate_baseline_model, clear_baseline_output_dir
        # 每次评估前先清理历史基线产物，避免旧结果污染当前报告
        clear_baseline_output_dir()
        # 调用基线模型评估（直接使用已清洗的oot_datasets，避免重复读取）
        try:
            baseline_metrics_df, baseline_status_info = evaluate_baseline_model(
                dev_df=train_df,
                test_df=test_df,
                oot_datasets=oot_datasets
            )
        except Exception as e:
            print(f"警告: 基线模型评估失败: {str(e)}")
            import traceback
            traceback.print_exc()
            baseline_status_info = {"状态": "评估异常", "说明": f"基线模型评估过程中发生异常: {str(e)}"}
        
        # 生成JSON报告-用于页面展示
        from utils.generate_json_reports import generate_clear_split_json
        generate_clear_split_json(
            orl_data,
            data_cleared=data,  # 这里假设data已经是清洗后的数据
            method=method,
            dev_datasets=dev_datasets,
            oot_datasets=oot_datasets,
            df_all=df_all,
            output_path=clean_json_path,
            channel_col=channel_col,
            date_col=date_col,
            baseline_evaluation_info=baseline_status_info,
            baseline_metrics_df=baseline_metrics_df
        )
        
        return True
    except Exception as e:
        msg=traceback.format_exc()
        print(msg)
        return False
