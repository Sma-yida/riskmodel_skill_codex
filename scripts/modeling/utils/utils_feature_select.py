import gc
import pandas as pd
import numpy as np
from multiprocessing.pool import Pool
from .pyce1 import *
from utils.data_utils import identify_types
from utils.utils1 import ks_score
import os
import warnings
import json
from concurrent.futures import ProcessPoolExecutor
import logging
import math
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import time
import traceback

warnings.filterwarnings("ignore")

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 常量定义
STATISTIC_BOUND = -9990
BASELINE_OPTIONAL_COLS = {"baseline_prob", "baseline_score"}
MISSING_VALUES = [-99999, -9999, -9998, -9997, -9996]
DEFAULT_STEPWISE_LGB_PARAMS = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'random_state': 42,
    'max_depth': 8,
    'n_jobs': -1,
    'verbose': -1,
    'class_weight': None,
    'scale_pos_weight': 1.0,
    'subsample_for_bin': 200000,
}
MIN_STEPWISE_KS_GAIN = 0.0001
SINGLE_VAR_CACHE_VERSION = 1
NULL_IMPORTANCE_CONFIG_PATH = os.path.join('data', 'null_importance_config.json')


def _load_single_var_cache(cache_path, single_var_path, mono_var_path, input_hashes):
    """Return cached single-variable results only when both input files match."""
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache = json.load(f)
        cache_matches = (
            cache.get('single_var_cache_version') == SINGLE_VAR_CACHE_VERSION
            and cache.get('train_sha256') == input_hashes['train_sha256']
            and cache.get('psi_sha256') == input_hashes['psi_sha256']
        )
        if not cache_matches:
            return None
        single_var = pd.read_csv(single_var_path)
        mono_var = pd.read_csv(mono_var_path)
        required_columns = {'Feature', 'IV', 'PSI', 'missing'}
        if not required_columns.issubset(single_var.columns):
            logger.info("single_var.csv缺少必要列，将重新计算单变量统计")
            return None
        return single_var, mono_var
    except (FileNotFoundError, KeyError, OSError, ValueError, pd.errors.ParserError) as exc:
        logger.info(f"单变量缓存不可用，将重新计算: {exc}")
        return None


def _save_single_var_cache(cache_path, input_hashes):
    cache = {
        'single_var_cache_version': SINGLE_VAR_CACHE_VERSION,
        'train_sha256': input_hashes['train_sha256'],
        'psi_sha256': input_hashes['psi_sha256'],
    }
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=4)


def _load_null_importance_config(config_path):
    """Load the optional user-uploaded NI configuration."""
    if not os.path.exists(config_path):
        return {
            'lgb_params': {},
            'baseline_fea': [],
            'feature_pools': [],
        }
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            sidecar = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{config_path} is not valid JSON at line {exc.lineno}, column {exc.colno}. "
            "Use double quotes for property names and string values."
        ) from exc

    lgb_params = sidecar.get('lgb_params', {})
    baseline_fea = sidecar.get('baseline_fea', [])
    feature_pools = sidecar.get('feature_pools', [])
    if not isinstance(lgb_params, dict):
        raise ValueError("null_importance_config.json: lgb_params must be an object")
    if not isinstance(baseline_fea, list) or not all(isinstance(feature, str) for feature in baseline_fea):
        raise ValueError("null_importance_config.json: baseline_fea must be a list of feature names")
    if not isinstance(feature_pools, list) or any(not isinstance(pool, list) for pool in feature_pools):
        raise ValueError("null_importance_config.json: feature_pools must be a list of lists")
    if any(not isinstance(feature, str) for pool in feature_pools for feature in pool):
        raise ValueError("null_importance_config.json: feature_pools must contain feature names")
    feature_pools = [pool for pool in feature_pools if pool]
    return {
        'lgb_params': lgb_params,
        'baseline_fea': baseline_fea,
        'feature_pools': feature_pools,
    }


def _write_null_importance_config(output_dst, lgb_params=None, baseline_fea=None, feature_pools=None):
    """Save the effective NI configuration as a user-editable template."""
    applied_config = {
        'lgb_params': lgb_params or DEFAULT_STEPWISE_LGB_PARAMS,
        'baseline_fea': baseline_fea or [],
        'feature_pools': feature_pools or [],
    }
    with open(os.path.join(output_dst, 'null_importance_config.json'), 'w', encoding='utf-8') as f:
        json.dump(applied_config, f, ensure_ascii=False, indent=4)

# ══════════════════════════════════════════════════════════════════════
#  相关性计算辅助函数（Method2 / 向量化贪心剔除）
# ══════════════════════════════════════════════════════════════════════
 
def _greedy_drop_vectorized(corr_matrix: np.ndarray, n: int, threshold: float) -> np.ndarray:
    """
    numpy 向量化贪心剔除。
    预计算上三角布尔矩阵，外层保留 O(n) Python 循环（顺序依赖无法消除），
    内层用一行 numpy 广播替代整个 j 循环。
    k=1 保证对角线及下三角不参与，不会把 i 自身误标为 drop。
    注意：corr_matrix 中的 NaN（零方差或有效样本不足）会被 > threshold 判定为 False，
    等效于视为"不相关"，与 df.corr() 的行为一致。
    """
    above_thresh = np.triu(corr_matrix > threshold, k=1)  # (n, n) bool，NaN→False
    to_drop = np.zeros(n, dtype=bool)
    for i in range(n):
        if to_drop[i]:
            continue
        to_drop |= above_thresh[i]
    return to_drop
 
 
def _my_corrcoef_fast(df: pd.DataFrame, min_periods: int = 1) -> np.ndarray:
    """
    矩阵化 pairwise Pearson 相关性（精确版，等价于 df.corr(method='pearson')）。
    支持 NaN（pairwise deletion），底层用 BLAS 矩阵乘法加速。
 
    输入: 已完成缺失值替换（MISSING_VALUES→NaN）的 DataFrame
    输出: (p, p) float64 ndarray，绝对值相关性矩阵（含 NaN 表示无效对）
 
    内存布局（n=行数, p=列数, 以 500k×1000 float64 为例，每数组约 4 GB）：
    ┌─────────────────────────────────────────────────────────────┐
    │ 阶段               活着的大数组           峰值              │
    │ 原地清零后         X0(4G) + M0(4G)        8  G             │
    │ 计算 sum_x/count   X0(4G) + M0(4G)        8  G             │
    │ 计算 sum_x2        X0(4G) + X0²(4G)       8  G  ← 优化点   │
    │   （M0 在 sum_x2   算完后立即删除）                         │
    │ 计算 sum_xy        X0(4G)                 4  G             │
    │ 后续公式           仅 (p,p) 小矩阵        可忽略            │
    └─────────────────────────────────────────────────────────────│
    优化关键：
    1. 原地清零（X0[nan_mask]=0）代替 np.where，省掉一份 (n,p) 拷贝
    2. X0² 计算完 sum_x2 后立即 del，M0 在 sum_x2 完成后立即 del
    3. 始终只有≤2个大数组同时存活，峰值约 n*p*16 字节
    """
    # ── Step1: 构建 X0（原地清零）和 M0 ────────────────────────────────
    # 直接 copy 为 X0，NaN 位置原地填 0，避免 np.where 产生额外副本
    X0 = df.to_numpy(dtype=np.float64, copy=True)  # (n, p)  活: X0
    nan_mask = np.isnan(X0)                         # (n, p) bool  活: X0 + nan_mask
    M0 = (~nan_mask).astype(np.float64)             # (n, p) float 活: X0 + nan_mask + M0
    X0[nan_mask] = 0.0                              # 原地清零，不产生新数组
    del nan_mask                                    # 释放 bool mask  活: X0 + M0
 
    # ── Step2: pairwise 统计量（均为 (p,p) 小矩阵）────────────────────
    # count[i,j] = 列 i、j 同时非 NaN 的行数
    count  = M0.T @ M0                              # 活: X0 + M0
 
    # sum_x[i,j] = 在 i∧j 同时非 NaN 的行中，列 i 的求和
    sum_x  = X0.T @ M0                              # 活: X0 + M0
 
    # sum_x2[i,j] = 在 i∧j 同时非 NaN 的行中，列 i 的平方和
    X0_sq  = X0 * X0                               # (n,p) 临时  活: X0 + M0 + X0²
    sum_x2 = X0_sq.T @ M0                          # 活: X0 + M0 + X0²
    del X0_sq                                      # sum_x2 已得，释放 X0²  活: X0 + M0
    del M0                                         # ← 关键：M0 到此全部用完，提前释放
    #                                                活: X0
 
    # sum_xy[i,j] = 在 i∧j 同时非 NaN 的行中，x_i * x_j 的求和
    # （X0 中 NaN 位已填 0，乘积自然为 0，等价于只累加两列都有效的行）
    sum_xy = X0.T @ X0                              # 活: X0
    del X0                                          # 活: 仅 (p,p) 小矩阵
 
    # ── Step3: 由四个统计量推导 Pearson 相关系数 ──────────────────────
    # 协方差分子: Cov_num[i,j] = Σxy - (Σx)(Σy)/n
    # 方差分子  : Var_x[i,j]  = Σx² - (Σx)²/n  （沿 j 方向 pairwise）
    with np.errstate(divide='ignore', invalid='ignore'):
        cov_num = sum_xy - (sum_x * sum_x.T) / count
        var_x   = sum_x2 - (sum_x * sum_x)  / count
        var_y   = var_x.T
        corr    = cov_num / np.sqrt(var_x * var_y)
 
    # 有效性过滤：样本数不足或方差为 0 → NaN（与 df.corr() 行为一致）
    valid       = (count >= max(min_periods, 2)) & (var_x > 0) & (var_y > 0)
    corr[~valid] = np.nan
 
    corr = np.clip(corr, -1.0, 1.0)                # 修正浮点误差（理论范围 [-1,1]）
    return np.abs(corr)                             # 返回绝对值，与 calculate_correlation 对齐
 
 
def calculate_correlation_v2(data: pd.DataFrame, sorted_features: list, threshold: float) -> list:
    """
    Method2: 矩阵化 pairwise Pearson（精确） + numpy 向量化贪心剔除。
    接口与 calculate_correlation (Method1) 完全相同，可直接替换。
 
    优点: 相关性矩阵计算速度比 df.corr() 快约 5~20x（BLAS 加速）；
          贪心剔除速度比双层 Python 循环快约 30~88x。
    精度: 与 df.corr() 数值偏差 < 1e-10（浮点舍入），筛选结果完全一致。
 
    ⚠️ 注意: 大数据（500k 行×2000 列）内存峰值约 25~32 GB，见 _my_corrcoef_fast 说明。
    """
    data_corr = data[sorted_features].replace(MISSING_VALUES, np.nan)
    corr_mat  = _my_corrcoef_fast(data_corr)       # (p, p) abs corr，NaN 表示无效对
    n         = len(sorted_features)
    to_drop   = _greedy_drop_vectorized(corr_mat, n, threshold)
    return [sorted_features[i] for i in range(n) if to_drop[i]]
 
 
# ══════════════════════════════════════════════════════════════════════
#  临时对比函数（用真实数据验证 v1 vs v2，确认无误后可删除）
# ══════════════════════════════════════════════════════════════════════
 
def benchmark_correlation_methods(
    data: pd.DataFrame,
    sorted_features: list,
    threshold: float,
    sample_rows: int = None,
) -> dict:
    """
    对比 calculate_correlation (Method1) 与 calculate_correlation_v2 (Method2) 的：
      - 各阶段耗时（矩阵计算 / 贪心剔除 / 总耗时）
      - 剔除特征列表是否完全一致
      - 相关性矩阵上三角数值偏差（最大值 / 均值）
 
    参数:
        data            : 原始 DataFrame（未做 MISSING_VALUES 替换）
        sorted_features : 已按 IV 降序排列的特征列表
        threshold       : 相关性剔除阈值
        sample_rows     : 若指定，则随机抽样该行数再比较（加速大数据验证）
                          None = 使用全量数据
 
    返回:
        result dict，包含耗时、偏差统计和一致性标志
 
    使用示例:
        result = benchmark_correlation_methods(dev_data, sorted_features, corr_thresh)
        if result['drop_match']:
            print("两种方法结果完全一致，可安全替换")
    """
    import time as _time
 
    if sample_rows is not None and sample_rows < len(data):
        logger.info(f"[benchmark] 随机抽样 {sample_rows} 行（共 {len(data)} 行）进行对比")
        data_eval = data.sample(n=sample_rows, random_state=42)
    else:
        data_eval = data
 
    p = len(sorted_features)
    logger.info(f"[benchmark] 特征数={p}, 样本数={len(data_eval)}, 阈值={threshold}")
 
    # ── Method1: df.corr + 双层 Python 循环 ──────────────────────────
    data_corr1 = data_eval[sorted_features].replace(MISSING_VALUES, np.nan)
 
    t0 = _time.perf_counter()
    corr_mat1  = data_corr1.corr(method='pearson').abs().values
    t_corr1    = _time.perf_counter() - t0
 
    t0 = _time.perf_counter()
    n          = len(sorted_features)
    to_drop1   = np.zeros(n, dtype=bool)
    for i in range(n):
        if to_drop1[i]:
            continue
        for j in range(i + 1, n):
            if to_drop1[j]:
                continue
            if corr_mat1[i, j] > threshold:
                to_drop1[j] = True
    t_drop1    = _time.perf_counter() - t0
    drop1      = [sorted_features[i] for i, v in enumerate(to_drop1) if v]
 
    # ── Method2: 矩阵化 pairwise + 向量化贪心 ────────────────────────
    data_corr2 = data_eval[sorted_features].replace(MISSING_VALUES, np.nan)
 
    t0 = _time.perf_counter()
    corr_mat2  = _my_corrcoef_fast(data_corr2)
    t_corr2    = _time.perf_counter() - t0
 
    t0 = _time.perf_counter()
    to_drop2   = _greedy_drop_vectorized(corr_mat2, n, threshold)
    t_drop2    = _time.perf_counter() - t0
    drop2      = [sorted_features[i] for i, v in enumerate(to_drop2) if v]
 
    # ── 数值偏差（上三角，跳过 NaN 对）─────────────────────────────
    idx         = np.triu_indices(p, k=1)
    v1          = corr_mat1[idx]
    v2          = corr_mat2[idx]
    both_valid  = ~(np.isnan(v1) | np.isnan(v2))
    diff        = np.abs(v1[both_valid] - v2[both_valid])
    max_diff    = float(diff.max())  if diff.size > 0 else float('nan')
    mean_diff   = float(diff.mean()) if diff.size > 0 else float('nan')
 
    nan_only_v1 = int(np.isnan(v1).sum())
    nan_only_v2 = int(np.isnan(v2).sum())
 
    # ── 剔除列表一致性 ─────────────────────────────────────────────
    drop_match  = set(drop1) == set(drop2)
    only_in_1   = sorted(set(drop1) - set(drop2))
    only_in_2   = sorted(set(drop2) - set(drop1))
 
    t_total1    = t_corr1 + t_drop1
    t_total2    = t_corr2 + t_drop2
    speedup     = t_total1 / t_total2 if t_total2 > 0 else float('inf')
 
    # ── 打印报告 ──────────────────────────────────────────────────
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  相关性方法对比  特征={p}  样本={len(data_eval)}  阈值={threshold}")
    print(f"{sep}")
    print(f"  {'':30s} {'矩阵计算':>10s}  {'贪心剔除':>10s}  {'总耗时':>10s}")
    print(f"  {'-'*60}")
    print(f"  {'Method1  df.corr + 双层循环':30s} {t_corr1:>10.4f}s {t_drop1:>10.4f}s {t_total1:>10.4f}s")
    print(f"  {'Method2  矩阵化 + 向量化':30s} {t_corr2:>10.4f}s {t_drop2:>10.4f}s {t_total2:>10.4f}s")
    print(f"  加速比（Method2 vs Method1）: {speedup:.1f}x")
    print()
    print(f"  剔除特征数量   : Method1={len(drop1)}, Method2={len(drop2)}")
    print(f"  剔除列表一致性 : {'✅ 完全一致' if drop_match else '❌ 不一致，见 only_in_1 / only_in_2'}")
    if not drop_match:
        print(f"    仅 Method1 剔除: {only_in_1[:10]}")
        print(f"    仅 Method2 剔除: {only_in_2[:10]}")
    print()
    print(f"  相关性矩阵数值偏差（有效对，上三角）")
    print(f"    最大偏差: {max_diff:.2e}  均值偏差: {mean_diff:.2e}  （预期 < 1e-10）")
    print(f"    NaN 对数: Method1={nan_only_v1}  Method2={nan_only_v2}")
    print(f"{sep}\n")
 
    return {
        # 耗时
        't_corr1': t_corr1, 't_drop1': t_drop1, 't_total1': t_total1,
        't_corr2': t_corr2, 't_drop2': t_drop2, 't_total2': t_total2,
        'speedup': speedup,
        # 剔除结果
        'drop1': drop1, 'drop2': drop2,
        'drop_match': drop_match,
        'only_in_1': only_in_1, 'only_in_2': only_in_2,
        # 数值偏差
        'max_diff': max_diff, 'mean_diff': mean_diff,
        'nan_pairs_v1': nan_only_v1, 'nan_pairs_v2': nan_only_v2,
    }
 

def get_process_num():
    """获取CPU核心数"""
    import multiprocessing
    return max(1, multiprocessing.cpu_count() - 1)

def bining4iv(s: pd.Series, flag: int, binning_number: int):
    """IV分箱"""
    max_n = s.max()
    min_n = s[s > max(MISSING_VALUES)].min()
    init_bins = [-10000, -9999]
    binning_number = min(binning_number, len(set(s)))
    return _bining(s, max_n, min_n, init_bins, flag, binning_number)

def bining4psi(s1: pd.Series, s2: pd.Series, flag: int, binning_number: int):
    """PSI分箱"""
    max_n = max(s1.max(), s2.max()) + 0.1
    min_n = min(s1[s1 > -1].min(), s2[s2 > -1].min())
    init_bins = [-10000, min_n]
    return _bining(s1, max_n, min_n, init_bins, flag, binning_number)

def _bining(s: pd.Series, max_n: float, min_n: float, init_bins: list, flag: int, binning_number: int):
    """分箱实现"""
    bins_split = init_bins.copy()
    bin_step = 1.0 / binning_number
    
    # 等频分箱
    if flag == 1:
        perc = np.arange(0, 1, bin_step)
        node = list(s[s >= min_n].quantile(perc[1:]))
        bins_split.extend(node)
        bins_split.append(max_n)
    
    # 等距分箱
    if flag == 2:
        step_n = (max_n - min_n) / binning_number
        for i in range(binning_number):
            bins_split.append(min_n + i * step_n)
        tmp_max = max(min_n + binning_number * step_n, max_n)
        bins_split.append(tmp_max)
    
    return [x for x in np.unique(bins_split) if not math.isnan(x)]

def calculate_iv(data: pd.DataFrame, feature: str, target: str, flag: int, binning_number: int, is_categorical: bool = False) -> tuple:
    """计算IV值
    
    Args:
        data: 数据DataFrame
        feature: 特征名
        target: 目标变量名
        flag: 分箱类型（1=等频，2=等距）
        binning_number: 分箱数量
        is_categorical: 是否为类别特征，如果是，则有多少类分多少箱
    """
    df = data[[feature, target]].copy()
    df = df[df[target].notna()]  # 过滤掉目标变量为空的行
    
    # 判断是否为类别特征
    if is_categorical or df[feature].dtype == 'object':
        # 类别特征：有多少类分多少箱，每个类别一个箱
        # 统一处理所有形式的缺失值（包括数值和字符串形式）
        missing_values_all = MISSING_VALUES + [str(v) for v in MISSING_VALUES] + [' ', '']
        
        # 先将所有形式的缺失值统一替换为np.nan
        s = df[feature].replace(missing_values_all, np.nan)
        
        # 然后将np.nan填充为字符串'_MISSING_'（使用特殊标记，避免与正常值混淆）
        s = s.fillna('_MISSING_').astype(str)
        
        # 直接按类别分组
        result = df.groupby(s)[target].agg(["count", "sum"]).rename(
            columns={"count": 'total_count', "sum": "bad_count"})
    else:
        # 数值特征：使用原来的分箱逻辑
        # 分箱
        bin_split = bining4iv(df[feature], flag, binning_number)
        
        # 替换缺失值为-9999
        s = df[feature].replace(MISSING_VALUES + [np.nan], -9999)
        
        # 按照给定边界进行分箱
        binned_fc = pd.cut(s, bins=bin_split)
        result = df.groupby(binned_fc)[target].agg(["count", "sum"]).rename(
            columns={"count": 'total_count', "sum": "bad_count"})
    
    result['good_count'] = result['total_count'] - result['bad_count']
    
    all_total = result.total_count.sum()
    good_total = result.good_count.sum()
    bad_total = result.bad_count.sum()
    
    # 计算分布
    result['total_pcnt'] = result.apply(lambda x: "{:.2f}%".format(x.total_count / all_total * 100) \
        if all_total > 0 else "0.00%", axis=1)
    result['good_gtotal_pcnt'] = result.apply(lambda x: x.good_count / good_total if good_total > 0 else 0.0, axis=1)
    result['bad_btotal_pcnt'] = result.apply(lambda x: x.bad_count / bad_total if bad_total > 0 else 0.0, axis=1)
    
    # 计算单调性
    result['bad_rate'] = result.apply(lambda x: "{:.2f}%".format(x.bad_count / x.total_count * 100) \
        if x.total_count > 0 else "0.00%", axis=1)
    
    # 计算WOE和IV
    result['woe'] = result.apply(lambda x: np.log(x.good_gtotal_pcnt / x.bad_btotal_pcnt) \
        if x.good_gtotal_pcnt > 0 and x.bad_btotal_pcnt > 0 else 0.0, axis=1)
    result['iv'] = result.apply(lambda x: (x.good_gtotal_pcnt - x.bad_btotal_pcnt) * x.woe \
        if x.good_gtotal_pcnt > 0 and x.bad_btotal_pcnt > 0 else 0.0, axis=1)
    
    iv_value = result['iv'].sum()
    
    # 生成分箱点信息
    dist_mono_points = result[["total_count", "bad_count", "total_pcnt", "bad_rate"]].reset_index().rename(
        columns={feature: "binned"})
    dist_mono_points.insert(0, "Feature", feature)
    
    return iv_value, dist_mono_points

def calculate_psi(dev_data: pd.Series, psi_data: pd.Series, flag: int, binning_number: int, is_categorical: bool = False) -> tuple:
    """计算PSI值
    
    Args:
        dev_data: 训练集数据
        psi_data: 测试集数据
        flag: 分箱类型（1=等频，2=等距）
        binning_number: 分箱数量
        is_categorical: 是否为类别特征，如果是，则有多少类分多少箱
    """
    if is_categorical or dev_data.dtype == 'object':
        # 类别特征：统一处理所有形式的缺失值（包括数值和字符串形式）
        missing_values_all = MISSING_VALUES + [str(v) for v in MISSING_VALUES] + [' ', '']
        
        # 计算包含缺失值的PSI
        dev_data_str = dev_data.replace(missing_values_all, np.nan).fillna('_MISSING_').astype(str)
        psi_data_str = psi_data.replace(missing_values_all, np.nan).fillna('_MISSING_').astype(str)
        all_categories = sorted(set(dev_data_str.unique()) | set(psi_data_str.unique()))
        psi_value = _calculate_psi_value_categorical(dev_data_str, psi_data_str, all_categories)[0]
        
        # 计算剔除缺失值的PSI（将缺失值替换为np.nan后排除）
        dev_data_clean = dev_data.replace(missing_values_all, np.nan)
        psi_data_clean = psi_data.replace(missing_values_all, np.nan)
        # 只使用非缺失值计算PSI
        if len(dev_data_clean.dropna()) > 0 and len(psi_data_clean.dropna()) > 0:
            dev_data_clean_str = dev_data_clean.dropna().astype(str)
            psi_data_clean_str = psi_data_clean.dropna().astype(str)
            all_categories_clean = sorted(set(dev_data_clean_str.unique()) | set(psi_data_clean_str.unique()))
            psi_value_without_missing = _calculate_psi_value_categorical(dev_data_clean_str, psi_data_clean_str, all_categories_clean)[0]
        else:
            psi_value_without_missing = 0.0
        
        return psi_value, psi_value_without_missing
    else:
        # 数值特征：使用原来的分箱逻辑
        # 分箱
        bins_split = bining4psi(dev_data, psi_data, flag, binning_number)
        
        # 计算PSI
        psi_value = _calculate_psi_value(dev_data, psi_data, bins_split)
        psi_value_without_missing = _calculate_psi_value(
            dev_data.replace(MISSING_VALUES, np.nan),
            psi_data.replace(MISSING_VALUES, np.nan),
            bins_split
        )
        
        return psi_value, psi_value_without_missing

def _calculate_psi_value(s1: pd.Series, s2: pd.Series, bins_split: list) -> float:
    """计算PSI值的具体实现（数值特征）"""
    train_num = s1.size
    test_num = s2.size
    
    tr_count_ = s1.groupby(pd.cut(s1, bins_split, right=False)).count().rename("train")
    te_count_ = s2.groupby(pd.cut(s2, bins_split, right=False)).count().rename("test")
    
    _psi_train_prec = tr_count_.apply(lambda x: np.round((x if x > 0 else x + 1) / train_num, 8))
    _psi_test_prec = te_count_.apply(lambda x: np.round((x if x > 0 else x + 1) / test_num, 8))
    
    psi_result = (_psi_train_prec - _psi_test_prec) * (_psi_train_prec / _psi_test_prec).apply(
        lambda x: math.log(x, math.e)).round(4)
    
    psi_result.replace(to_replace=np.inf, value=0.0, inplace=True)
    psi_result.fillna(0.0, inplace=True)
    
    return psi_result.sum()

def _calculate_psi_value_categorical(s1: pd.Series, s2: pd.Series, all_categories: list) -> tuple:
    """计算PSI值的具体实现（类别特征）"""
    train_num = s1.size
    test_num = s2.size
    
    # 计算每个类别的分布
    tr_count_ = s1.value_counts().reindex(all_categories, fill_value=0).rename("train")
    te_count_ = s2.value_counts().reindex(all_categories, fill_value=0).rename("test")
    
    _psi_train_prec = tr_count_.apply(lambda x: np.round((x if x > 0 else x + 1) / train_num, 8))
    _psi_test_prec = te_count_.apply(lambda x: np.round((x if x > 0 else x + 1) / test_num, 8))
    
    psi_result = (_psi_train_prec - _psi_test_prec) * (_psi_train_prec / _psi_test_prec).apply(
        lambda x: math.log(x, math.e) if x > 0 else 0.0).round(4)
    
    psi_result.replace(to_replace=np.inf, value=0.0, inplace=True)
    psi_result.fillna(0.0, inplace=True)
    
    psi_value = psi_result.sum()
    
    # 对于类别特征，PSI(剔特殊值)和PSI相同（因为类别特征没有特殊值概念）
    return psi_value, psi_value

def process_feature(args):
    """处理单个特征的统计信息"""
    feature, data, psi_data, target, iv_binning_number, iv_binning_type, psi_binning_number, psi_binning_type,idx,total = args
    try:
        print(f"正在处理特征: {feature} ({idx+1}/{total})")
        
        # 判断是否为类别特征
        is_categorical = data[feature].dtype == 'object'
        if is_categorical:
            print("该特征为类别特征")
        
        # 计算IV值
        iv_value, mono_stats = calculate_iv(data, feature, target, iv_binning_type, iv_binning_number, is_categorical=is_categorical)
        
        # 计算PSI值
        psi_value, psi_value_without_missing = calculate_psi(
            data[feature], psi_data[feature], psi_binning_type, psi_binning_number, is_categorical=is_categorical
        )
        
        # 计算缺失率
        if is_categorical:
            # 类别特征：需要处理字符串形式的缺失值
            # 定义完整的缺失值列表（包含字符串形式）
            missing_values_all = MISSING_VALUES + [str(v) for v in MISSING_VALUES] + [' ', '']
            # 对于类别特征，使用 isin 来匹配所有形式的缺失值
            missing_rate = data[feature].isin(missing_values_all + [np.nan]).sum() / len(data)
        else:
            # 数值特征：使用原来的逻辑
            missing_rate = data[feature].replace(MISSING_VALUES, np.nan).isna().mean()
        
        # 计算其他统计量
        if is_categorical:
            # 类别特征：数值型统计量置空
            mean_value = np.nan
            std_value = np.nan
            min_value = np.nan
            max_value = np.nan
            neg_rate = np.nan
            # 计算单一值占比（类别特征）
            unique_rate = data[feature].value_counts(dropna=True).max() / len(data) if len(data[feature].value_counts(dropna=True)) > 0 else 0.0
        else:
            # 数值特征：正常计算
            valid_data = data[feature][data[feature] > STATISTIC_BOUND]
            mean_value = valid_data.mean()
            std_value = valid_data.std(ddof=1)
            min_value = valid_data.min()
            max_value = valid_data.max()
            unique_rate = valid_data.value_counts(dropna=True).max() / len(data)
            neg_rate = (data[feature] < 0).mean()
        
        # 格式化负值占比
        if is_categorical:
            neg_rate_str = np.nan # 类别特征没有负值概念
        else:
            neg_rate_str = f"{neg_rate:.2%}"
        
        return {
            'Feature': feature,
            'IV': iv_value,
            'PSI': psi_value,
            'PSI(剔特殊值)': psi_value_without_missing,
            'missing': f"{missing_rate:.2%}",
            'mean': mean_value,
            'std': std_value,
            'min': min_value,
            'max': max_value,
            '单一值占比': f"{unique_rate:.2%}",
            '<0': neg_rate_str,
            'mono_stats': mono_stats
        }
    except Exception as e:
        logger.error(f"Error processing feature {feature}: {str(e)}")
        return None

# def calculate_correlation(data: pd.DataFrame, sorted_features: list, threshold: float) -> list:
#     """
#     计算特征相关性并返回需要删除的特征，sorted_features需已按IV降序、特征名升序排序。
#     相关性计算前将-9999, -9998, -9997, -9996替换为np.nan，
#     只有当pearson和spearman相关性都大于阈值时才剔除后面的特征。
#     """
#     # 替换特殊缺失值为 NaN
#     data_corr = data[sorted_features].replace(MISSING_VALUES, np.nan)
#     # 计算两种相关性
#     pearson_corr = data_corr.corr(method='pearson').abs()
#     spearman_corr = data_corr.corr(method='spearman').abs()
#     to_drop = set()
#     for i in range(len(sorted_features)):
#         for j in range(i + 1, len(sorted_features)):
#             f1, f2 = sorted_features[i], sorted_features[j]
#             if f1 in to_drop or f2 in to_drop:
#                 continue
#             if (pearson_corr.loc[f1, f2] > threshold) or (spearman_corr.loc[f1, f2] > threshold):
#                 to_drop.add(f2)
#     return list(to_drop)

def calculate_correlation(data: pd.DataFrame, sorted_features: list, threshold: float) -> list:
    """
    计算特征相关性并返回需要删除的特征，sorted_features需已按IV降序、特征名升序排序。
    相关性计算前将-9999, -9998, -9997, -9996替换为np.nan，
    只有当pearson和spearman相关性都大于阈值时才剔除后面的特征。
    优化：用numpy下标访问加速。
    """
    # 替换特殊缺失值为 NaN
    data_corr = data[sorted_features].replace(MISSING_VALUES, np.nan)
    pearson_corr = data_corr.corr(method='pearson').abs().values
    # spearman_corr = data_corr.corr(method='spearman').abs().values
    n = len(sorted_features)
    to_drop = np.zeros(n, dtype=bool)
    for i in range(n):
        if to_drop[i]:
            continue
        for j in range(i + 1, n):
            if to_drop[j]:
                continue
            # if (pearson_corr[i, j] > threshold) or (spearman_corr[i, j] > threshold):
            if (pearson_corr[i, j] > threshold):
                to_drop[j] = True
    return [sorted_features[i] for i in range(n) if to_drop[i]]
    
def get_feature_importances(data: pd.DataFrame, target: str, features: list, shuffle: bool, seed: int = None) -> pd.DataFrame:
    """Train one Null Importance model and return feature importances."""
    data = data[data[target].isin([0, 1])].copy()
    X = data[features].copy()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    for feature in categorical_features:
        X[feature] = X[feature].astype('category')

    y = data[target].copy()
    if shuffle:
        y = y.sample(frac=1.0, random_state=seed).to_numpy()
    else:
        y = y.to_numpy()

    lgb_params = {
        'objective': 'binary',
        'boosting_type': 'rf',
        'subsample': 0.6,
        'colsample_bytree': 0.8,
        'num_leaves': 31,
        'max_depth': 4,
        'seed': seed,
        'bagging_freq': 1,
        'verbose': -1,
        'n_jobs': -1,
    }
    dtrain = lgb.Dataset(X, y, categorical_feature=categorical_features or 'auto', free_raw_data=False)
    clf = lgb.train(params=lgb_params, train_set=dtrain, num_boost_round=60)

    imp_df = pd.DataFrame()
    imp_df['feature'] = features
    imp_df['importance_gain'] = clf.feature_importance(importance_type='gain')
    imp_df['importance_split'] = clf.feature_importance(importance_type='split')
    imp_df['trn_score'] = roc_auc_score(y, clf.predict(X))
    return imp_df


def get_null_importance(data: pd.DataFrame, target: str, features: list, nb_runs: int = 80) -> pd.DataFrame:
    """Train Null Importance models with different shuffled labels."""
    null_imp_list = []
    start = time.time()
    for i in range(nb_runs):
        imp_df = get_feature_importances(data, target, features, shuffle=True, seed=817 + i)
        imp_df['run'] = i + 1
        null_imp_list.append(imp_df)
        spent = (time.time() - start) / 60
        print('Done with %4d of %4d (Spent %5.1f min)' % (i + 1, nb_runs, spent), end='\r', flush=True)
    return pd.concat(null_imp_list, axis=0, ignore_index=True)


def null_importance_select(data: pd.DataFrame, target: str, features: list, null_result_path: str, top_n: int = 500) -> list:
    """Run Null Importance and save one compact feature-level score table."""
    features = list(features)
    if features:
        actual_imp_df = get_feature_importances(data, target, features, shuffle=False, seed=817)
        null_imp_df = get_null_importance(data, target, features)
        feature_scores = []
        for feature in actual_imp_df['feature'].unique():
            null_gain = null_imp_df.loc[null_imp_df['feature'] == feature, 'importance_gain'].values
            actual_gain = actual_imp_df.loc[actual_imp_df['feature'] == feature, 'importance_gain'].mean()
            null_gain_p75 = np.percentile(null_gain, 75)
            gain_score = np.log(1e-10 + actual_gain / (1 + null_gain_p75))

            null_split = null_imp_df.loc[null_imp_df['feature'] == feature, 'importance_split'].values
            actual_split = actual_imp_df.loc[actual_imp_df['feature'] == feature, 'importance_split'].mean()
            null_split_p75 = np.percentile(null_split, 75)
            split_score = np.log(1e-10 + actual_split / (1 + null_split_p75))
            feature_scores.append((
                feature, actual_gain, null_gain_p75, gain_score,
                actual_split, null_split_p75, split_score,
            ))
        scores_df = pd.DataFrame(feature_scores, columns=[
            'feature', 'actual_gain', 'null_gain_p75', 'gain_score',
            'actual_split', 'null_split_p75', 'split_score',
        ])
        scores_df = scores_df.sort_values(by='gain_score', ascending=False).reset_index(drop=True)
    else:
        scores_df = pd.DataFrame(columns=[
            'feature', 'actual_gain', 'null_gain_p75', 'gain_score',
            'actual_split', 'null_split_p75', 'split_score',
        ])

    scores_df['rank_gain'] = np.arange(1, len(scores_df) + 1)
    scores_df['selected'] = scores_df['rank_gain'] <= top_n
    scores_df.to_csv(os.path.join(null_result_path, 'scores_df.csv'), index=False, encoding='utf-8-sig')

    for filename in (
        'actual_imp_df.csv',
        'null_imp_df.csv',
        'null_importance_scores.csv',
        'null_importance_selected_features.json',
    ):
        legacy_path = os.path.join(null_result_path, filename)
        if os.path.exists(legacy_path):
            os.remove(legacy_path)

    selected_features = scores_df.loc[scores_df['selected'], 'feature'].to_list()
    return selected_features


def _train_and_ks(feas, X_train_df, y_train, X_oot_df, y_oot, lgb_params):
    """Train one model with the current feature set and return train/OOT KS."""
    if not feas:
        return np.nan, 0.0

    train_mask = y_train.isin([0, 1])
    oot_mask = y_oot.isin([0, 1])
    X_train = X_train_df.loc[train_mask, feas].copy()
    X_oot = X_oot_df.loc[oot_mask, feas].copy()
    y_train = y_train.loc[train_mask]
    y_oot = y_oot.loc[oot_mask]
    if y_train.nunique() < 2 or y_oot.nunique() < 2:
        return np.nan, 0.0

    categorical_features = []
    for feature in feas:
        if X_train[feature].dtype == 'object' or X_oot[feature].dtype == 'object' or str(X_train[feature].dtype) == 'category' or str(X_oot[feature].dtype) == 'category':
            categories = pd.concat([X_train[feature], X_oot[feature]], ignore_index=True).dropna().unique()
            X_train[feature] = pd.Categorical(X_train[feature], categories=categories)
            X_oot[feature] = pd.Categorical(X_oot[feature], categories=categories)
            categorical_features.append(feature)

    clf = lgb.LGBMClassifier(**lgb_params)
    clf.fit(X_train, y_train, categorical_feature=categorical_features or 'auto')
    train_pred = clf.predict_proba(X_train)[:, 1]
    oot_pred = clf.predict_proba(X_oot)[:, 1]
    return ks_score(y_train, train_pred), ks_score(y_oot, oot_pred)


def stepwise_forward_ks(
    X_oot, y_oot, X_train, y_train,
    base_features, feature_pools,
    min_ks_gain=MIN_STEPWISE_KS_GAIN, lgb_params=None,
):
    """Use NI baseline features, then select each configured pool by KS gain."""
    lgb_params = lgb_params or DEFAULT_STEPWISE_LGB_PARAMS
    selected_features = list(base_features)

    if selected_features:
        ks_train, ks_baseline = _train_and_ks(
            selected_features, X_train, y_train, X_oot, y_oot, lgb_params
        )
        print(f"[baseline] features={len(selected_features)} | KS={ks_baseline:.6f}")
    else:
        ks_train, ks_baseline = np.nan, 0.0
        print("[baseline] baseline_pool为空，KS从0.000000开始，直接进入分层KS增益筛选")

    all_history = [{
        'record_type': 'baseline',
        'pool': 'baseline',
        'round': 0,
        'feature': None,
        'features': json.dumps(selected_features, ensure_ascii=False),
        'reason': None,
        'train_ks': round(ks_train, 6) if pd.notna(ks_train) else None,
        'oot_ks': round(ks_baseline, 6),
        'ks_gain': None,
    }]

    for i, pool in enumerate(feature_pools, start=1):
        pool_name = f'pool{i}'
        candidate_features = list(pool)
        round_idx = 0
        all_history.append({
            'record_type': 'pool_candidates',
            'pool': pool_name,
            'round': None,
            'feature': None,
            'features': json.dumps(candidate_features, ensure_ascii=False),
            'reason': None,
            'train_ks': None,
            'oot_ks': None,
            'ks_gain': None,
        })

        while candidate_features:
            round_idx += 1
            ks_candidates = []
            for feature in candidate_features:
                train_ks, oot_ks = _train_and_ks(
                    selected_features + [feature], X_train, y_train, X_oot, y_oot, lgb_params
                )
                if pd.notna(train_ks) and train_ks >= oot_ks and oot_ks >= ks_baseline:
                    ks_candidates.append((feature, train_ks, oot_ks))

            if not ks_candidates:
                break

            best_feature, best_train_ks, best_ks = max(ks_candidates, key=lambda item: item[2])
            ks_gain = best_ks - ks_baseline
            if ks_gain < min_ks_gain:
                break

            selected_features.append(best_feature)
            candidate_features.remove(best_feature)
            ks_baseline = best_ks
            all_history.append({
                'record_type': 'selected',
                'pool': pool_name,
                'round': round_idx,
                'feature': best_feature,
                'features': None,
                'reason': None,
                'train_ks': round(best_train_ks, 6),
                'oot_ks': round(ks_baseline, 6),
                'ks_gain': round(ks_gain, 6),
            })
            print(f"[{pool_name}] round {round_idx}: +{best_feature} | KS={ks_baseline:.6f} (+{ks_gain:.6f})")

    return selected_features, pd.DataFrame(all_history)


def layered_null_importance_select(
    dev_data: pd.DataFrame,
    oot_data: pd.DataFrame,
    target: str,
    corr_fea: list,
    hard_filter_fea: list,
    sidecar: dict,
    output_dst: str,
    top_n: int,
) -> tuple:
    """Build pools from the sidecar, run NI for the baseline, then run KS selection."""
    if oot_data is None:
        raise ValueError("Layered Null Importance requires OOT or test data for KS selection")

    baseline_fea = sidecar['baseline_fea']
    feature_pools = sidecar['feature_pools']
    pool_features = {feature for pool in feature_pools for feature in pool}
    overlap = sorted(set(baseline_fea) & pool_features)
    if overlap:
        raise ValueError(
            f"null_importance_config.json: baseline_fea and feature_pools overlap: {overlap}"
        )

    corr_set = set(corr_fea)
    hard_filter_set = set(hard_filter_fea)
    train_features = set(dev_data.columns)
    oot_features = set(oot_data.columns)
    configured_features = set()
    layered_pools = []
    skipped_features = []

    for pool_index, raw_pool in enumerate(feature_pools, start=1):
        pool_name = f'pool{pool_index}'
        pool = []
        for feature in raw_pool:
            if feature in configured_features:
                skipped_features.append({'pool': pool_name, 'feature': feature, 'reason': 'duplicate_in_sidecar'})
                continue
            configured_features.add(feature)
            if feature not in train_features:
                skipped_features.append({'pool': pool_name, 'feature': feature, 'reason': 'not_in_train_data'})
            elif feature not in oot_features:
                skipped_features.append({'pool': pool_name, 'feature': feature, 'reason': 'not_in_oot_data'})
            elif feature not in hard_filter_set:
                skipped_features.append({'pool': pool_name, 'feature': feature, 'reason': 'filtered_by_iv_psi_missing'})
            elif feature not in corr_set:
                skipped_features.append({'pool': pool_name, 'feature': feature, 'reason': 'filtered_by_correlation'})
            else:
                pool.append(feature)
        layered_pools.append(pool)

    baseline_source = baseline_fea or corr_fea
    baseline_pool = []
    seen_baseline = set()
    for feature in baseline_source:
        if feature in seen_baseline:
            skipped_features.append({'pool': 'baseline_pool', 'feature': feature, 'reason': 'duplicate_in_sidecar'})
            continue
        seen_baseline.add(feature)
        if not baseline_fea and feature in configured_features:
            continue
        if feature not in train_features:
            skipped_features.append({'pool': 'baseline_pool', 'feature': feature, 'reason': 'not_in_train_data'})
        elif feature not in oot_features:
            skipped_features.append({'pool': 'baseline_pool', 'feature': feature, 'reason': 'not_in_oot_data'})
        elif feature not in hard_filter_set:
            skipped_features.append({'pool': 'baseline_pool', 'feature': feature, 'reason': 'filtered_by_iv_psi_missing'})
        elif feature not in corr_set:
            skipped_features.append({'pool': 'baseline_pool', 'feature': feature, 'reason': 'filtered_by_correlation'})
        else:
            baseline_pool.append(feature)

    baseline_features = null_importance_select(dev_data, target, baseline_pool, output_dst, top_n=top_n)

    stepwise_params = DEFAULT_STEPWISE_LGB_PARAMS.copy()
    stepwise_params.update(sidecar['lgb_params'])
    selected_features, history_df = stepwise_forward_ks(
        X_oot=oot_data,
        y_oot=oot_data[target],
        X_train=dev_data,
        y_train=dev_data[target],
        base_features=baseline_features,
        feature_pools=layered_pools,
        min_ks_gain=MIN_STEPWISE_KS_GAIN,
        lgb_params=stepwise_params,
    )

    skip_records = [{
        'record_type': 'skip_summary',
        'pool': None,
        'round': None,
        'feature': None,
        'features': None,
        'reason': None,
        'skipped_count': len(skipped_features),
        'train_ks': None,
        'oot_ks': None,
        'ks_gain': None,
    }]
    for item in skipped_features:
        skip_records.append({
            'record_type': 'skipped',
            'pool': item['pool'],
            'round': None,
            'feature': item['feature'],
            'features': None,
            'reason': item['reason'],
            'skipped_count': None,
            'train_ks': None,
            'oot_ks': None,
            'ks_gain': None,
        })

    history_df['skipped_count'] = None
    history_df = pd.concat([pd.DataFrame(skip_records), history_df], ignore_index=True)
    history_df = history_df.reindex(columns=[
        'record_type', 'pool', 'round', 'feature', 'features', 'reason',
        'skipped_count', 'train_ks', 'oot_ks', 'ks_gain',
    ])
    history_df.to_csv(os.path.join(output_dst, 'ks_stepwise_history.csv'), index=False, encoding='utf-8-sig')

    skipped_path = os.path.join(output_dst, 'null_importance_skipped_features.csv')
    if os.path.exists(skipped_path):
        os.remove(skipped_path)
    return selected_features, baseline_pool, layered_pools, stepwise_params

def select_features_func(dev_data, psi_data, config, oot_df=None, input_hashes=None):
    """特征选择主函数"""
    try:
        # 从config中获取参数
        settings = config['settings']
        output_path = config['output_path']
        
        id_col = settings['id_col']
        date_col = settings['date_col']
        label_col = settings['label']
        base_fea = settings['base_fea']
        iv_thresh = settings['iv_thresh']
        psi_thresh = settings['psi_thresh']
        miss_thresh = settings['miss_thresh']
        corr_thresh = settings['corr_thresh']
        use_null_importance = settings['use_null_importance']
        top_n = settings['top_n']
        upload_fea_list = settings['upload_fea_list']
        remove_fea = list(dict.fromkeys(list(base_fea) + list(BASELINE_OPTIONAL_COLS)))
        
        output_dst = output_path['dst_outs']
        use_fea_path = output_path['use_feature_path']
        feature_json_path = output_path['feature_json_path']
        single_var_path = output_path['single_var_path']
        mono_var_path = output_path['mono_var_path']
        single_var_cache_path = os.path.join(output_dst, 'feature_select_cache.json')
        applied_lgb_params = DEFAULT_STEPWISE_LGB_PARAMS
        applied_baseline_fea = []
        applied_feature_pools = []
        
        # 创建输出目录
        os.makedirs(output_dst, exist_ok=True)
        
        # 可选：人工上传覆盖特征
        if upload_fea_list:
            logger.info("已选择自定义的入模特征")
            
            # 定义可用的特征列表（排除基础特征和基线对比列）
            available_features = [feat for feat in dev_data.columns if feat not in remove_fea]
            
            # 检查特征是否存在于数据中（排除基础特征）
            missing_features = [feat for feat in upload_fea_list if feat not in available_features]
            if missing_features:
                error_msg = f"以下特征在数据中不存在：{missing_features}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # 支持类别特征：仅校验特征存在性，不限制数据类型
            use_fea = upload_fea_list

            # 检查是否需要重新计算统计信息
            need_recalc = True
            cached_results = (
                _load_single_var_cache(single_var_cache_path, single_var_path, mono_var_path, input_hashes)
                if input_hashes else None
            )
            if cached_results:
                single_var, mono_var = cached_results
                if set(upload_fea_list).issubset(set(single_var['Feature'])):
                    need_recalc = False
                    logger.info("train、psi SHA-256一致，复用single_var.csv和mono_var.csv")

            if need_recalc:
                # 计算自定义特征的统计信息
                n_cpus = min(get_process_num(), 4)
                args_list = [(feat,  dev_data[[feat, label_col]], psi_data[[feat]], label_col, 10, 1, 10, 1, idx, len(upload_fea_list)) 
                            for idx, feat in enumerate(upload_fea_list)]
                
                if len(args_list) < 100:
                    results = [process_feature(args) for args in args_list]
                else:
                    with ProcessPoolExecutor(max_workers=n_cpus) as executor:
                        results = list(executor.map(process_feature, args_list))
                
                single_var = pd.DataFrame([{k: v for k, v in r.items() if k != 'mono_stats'} for r in results if r is not None])
                mono_var = pd.concat([r['mono_stats'] for r in results if r is not None])
                
                # 保存特征统计分析结果
                single_var.to_csv(single_var_path, index=False, encoding='utf-8-sig')
                mono_var.to_csv(mono_var_path, index=False, encoding='utf-8-sig')
                if os.path.exists(single_var_cache_path):
                    os.remove(single_var_cache_path)
        else:
            # 数据识别
            num_fea, cat_fea = identify_types(df=dev_data, remove_fea=remove_fea)

            # 合并数值特征和类别特征，统一处理
            all_fea = num_fea + cat_fea
            # all_fea = cat_fea
            logger.info(f"数值特征数量: {len(num_fea)}, 类别特征数量: {len(cat_fea)}, 总特征数量: {len(all_fea)}")

            cached_results = (
                _load_single_var_cache(single_var_cache_path, single_var_path, mono_var_path, input_hashes)
                if input_hashes else None
            )
            if cached_results:
                single_var, mono_var = cached_results
                logger.info("train、psi SHA-256一致，复用single_var.csv和mono_var.csv")
            else:
                logger.info("单变量缓存不存在或输入变化，重新计算IV、PSI和缺失率")
                n_cpus = min(get_process_num(), 4)
                args_list = [
                    (feat, dev_data[[feat, label_col]], psi_data[[feat]], label_col, 10, 1, 10, 1, idx, len(all_fea))
                    for idx, feat in enumerate(all_fea)
                ]
                if len(args_list) < 100:
                    results = [process_feature(args) for args in args_list]
                else:
                    with ProcessPoolExecutor(max_workers=n_cpus) as executor:
                        results = list(executor.map(process_feature, args_list))

                single_var = pd.DataFrame([
                    {key: value for key, value in result.items() if key != 'mono_stats'}
                    for result in results if result is not None
                ])
                mono_var = pd.concat([result['mono_stats'] for result in results if result is not None])
                single_var.to_csv(single_var_path, index=False, encoding='utf-8-sig')
                mono_var.to_csv(mono_var_path, index=False, encoding='utf-8-sig')
                if input_hashes:
                    _save_single_var_cache(single_var_cache_path, input_hashes)

            single_var[["IV", "PSI"]] = single_var[["IV", "PSI"]].astype(float)
            single_var["missing"] = single_var["missing"].str.rstrip('%').astype(float) / 100
            
            filter_fea = single_var.loc[
                (single_var.IV >= iv_thresh) & 
                (single_var.missing < miss_thresh) & 
                (single_var.PSI < psi_thresh),
                'Feature'
            ].values
            
            logger.info(f"After IV>{iv_thresh}, PSI<{psi_thresh}, Missingrate<{miss_thresh}, use_fea counts: {len(filter_fea)}")
        

            # 相关性分析（只处理数值特征，类别特征不参与相关性筛选）
            filter_fea_num = [f for f in filter_fea if dev_data[f].dtype != 'object']
            iv_dict = {f: single_var.loc[single_var['Feature'] == f, 'IV'].values[0] for f in filter_fea_num}
            sorted_features = sorted(iv_dict.keys(), key=lambda x: (-iv_dict[x], x))
            corr_drop = calculate_correlation_v2(dev_data, sorted_features, corr_thresh)  

            # calculate_correlation_v2() 空间换时间（Method2 矩阵化 pairwise Pearson + 向量化贪心剔除）
            # 速度：14.8x vs Method1（0.11s vs 1.65s），结果完全一致，偏差 < 1e-13
            # 内存：函数内峰值额外持有 ~3×(n_rows × p_cols × 8B)，函数返回后完全释放
            #       其中 Method1 基准线为 1×(n,p)，Method2 额外多 2×(n,p)（X0 + X0²）500k行×1000列时额外峰值约 +8 GB，函数结束后归零
            #误差： 矩阵数值最大偏差: 6.31e-14，均值偏差: 9.84e-16，剔除特征数量✅ 完全一致
            # result = benchmark_correlation_methods(
            #     dev_data,
            #     sorted_features,
            #     corr_thresh,
            #     sample_rows=50000, 
            # )

            corr_fea = [i for i in filter_fea if i not in corr_drop]
            logger.info(f"Corr<{corr_thresh}, use_fea counts: {len(corr_fea)}")

            # 如果启用Null Importance特征选择，先相关性过滤，再进入NI/分层KS逻辑
            if use_null_importance:
                null_output_dst = os.path.join(output_dst, 'null_importance')
                os.makedirs(null_output_dst, exist_ok=True)
                for filename in (
                    'actual_imp_df.csv',
                    'null_imp_df.csv',
                    'null_importance_scores.csv',
                    'null_importance_selected_features.json',
                    'null_importance_skipped_features.csv',
                    'scores_df.csv',
                    'ks_stepwise_history.csv',
                ):
                    legacy_path = os.path.join(output_dst, filename)
                    if os.path.exists(legacy_path):
                        os.remove(legacy_path)

                sidecar = _load_null_importance_config(NULL_IMPORTANCE_CONFIG_PATH)
                has_baseline_config = bool(sidecar['baseline_fea'])
                has_pool_config = bool(sidecar['feature_pools'])
                if not has_baseline_config and not has_pool_config:
                    logger.info("Using unified Null Importance for feature selection")
                    use_fea = null_importance_select(dev_data, label_col, corr_fea, null_output_dst, top_n)
                    applied_baseline_fea = list(corr_fea)
                    for filename in ('ks_stepwise_history.csv', 'null_importance_skipped_features.csv'):
                        stale_path = os.path.join(null_output_dst, filename)
                        if os.path.exists(stale_path):
                            os.remove(stale_path)
                    logger.info(f"After Null Importance selection, use_fea counts: {len(use_fea)}")
                elif not has_pool_config:
                    baseline_set = set(sidecar['baseline_fea'])
                    applied_baseline_fea = [feature for feature in corr_fea if feature in baseline_set]
                    logger.info("Using configured Null Importance baseline features")
                    use_fea = null_importance_select(
                        dev_data, label_col, applied_baseline_fea, null_output_dst, top_n
                    )
                    for filename in ('ks_stepwise_history.csv', 'null_importance_skipped_features.csv'):
                        stale_path = os.path.join(null_output_dst, filename)
                        if os.path.exists(stale_path):
                            os.remove(stale_path)
                    logger.info(f"After Null Importance selection, use_fea counts: {len(use_fea)}")
                else:
                    logger.info(f"Using layered Null Importance config: {NULL_IMPORTANCE_CONFIG_PATH}")
                    use_fea, applied_baseline_fea, applied_feature_pools, applied_lgb_params = layered_null_importance_select(
                        dev_data=dev_data,
                        oot_data=oot_df,
                        target=label_col,
                        corr_fea=corr_fea,
                        hard_filter_fea=list(filter_fea),
                        sidecar=sidecar,
                        output_dst=null_output_dst,
                        top_n=top_n,
                    )
                    logger.info(f"After layered Null Importance and KS selection, use_fea counts: {len(use_fea)}")
            else:
                use_fea = corr_fea
            logger.info(f"Final selected feature counts: {len(use_fea)}")

        _write_null_importance_config(
            output_dst,
            lgb_params=applied_lgb_params,
            baseline_fea=applied_baseline_fea,
            feature_pools=applied_feature_pools,
        )
        
        # 保存结果
        with open(use_fea_path, "w", encoding="utf-8") as f:
            json.dump(use_fea, f, ensure_ascii=False, indent=4)
        
        # 生成特征筛选报告
        from utils.generate_json_reports import generate_feature_select_json
        generate_feature_select_json(dev_data, use_fea, single_var, feature_json_path, base_fea)
        
        print(f"筛选后特征数量: {len(use_fea)}")
        print(f"入模特征为：", use_fea)
        return True
        
    except Exception as e:
        msg = traceback.format_exc()
        print(msg)
        return False
