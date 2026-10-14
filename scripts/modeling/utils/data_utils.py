import sys
import os
from .utils1 import *
from .pyce1 import *
import traceback
from copy import deepcopy
import pandas as pd
import json
from multiprocessing.pool import Pool
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import warnings
import gc
import glob
warnings.filterwarnings("ignore")

# ==================== 数据格式配置 ====================
# Parquet: 压缩率最高(小5-10倍)，读写较快，跨平台兼容性好，推荐用于持久化存储
# Feather: 读写最快，压缩率中等，适合临时文件和进程间数据交换
# CSV: 兼容性最好，但文件大、读写慢
# 支持的文件格式（按读取优先级排序）
SUPPORTED_FORMATS = ['.parquet', '.feather', '.csv']

# Parquet压缩算法: 'snappy'(默认，平衡), 'gzip'(最高压缩), 'brotli'(高压缩), 'zstd'(推荐), None(无压缩)
PARQUET_COMPRESSION = 'gzip'  # 使用gzip获得最大压缩比


# ==================== 通用格式化函数 ====================

def format_elapsed_hms(elapsed_seconds: float) -> str:
    """
    将耗时秒数格式化为“X小时Y分Z秒”。
    """
    total_seconds = int(round(elapsed_seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}小时{minutes}分{seconds}秒"


# ==================== 格式识别函数 ====================

def get_file_format(file_path: str) -> str:
    """
    获取文件格式
    
    返回: 'csv', 'parquet', 'feather' 或 'unknown'
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.csv':
        return 'csv'
    elif ext == '.parquet':
        return 'parquet'
    elif ext in ['.feather', '.ftr']:
        return 'feather'
    return 'unknown'


def get_data_path(base_path: str) -> str:
    """
    按优先级查找文件，返回完整路径（公开接口）
    
    参数:
        base_path: 基础路径（可带或不带扩展名）
        
    返回:
        找到的文件完整路径，如果找不到返回原路径
    """
    # 如果路径已存在（带扩展名），直接返回
    if os.path.exists(base_path):
        return base_path
    
    # 移除可能的扩展名
    base_no_ext = os.path.splitext(base_path)[0]
    
    # 按优先级查找
    for ext in SUPPORTED_FORMATS:
        candidate = base_no_ext + ext
        if os.path.exists(candidate):
            return candidate
    
    # 找不到则返回原路径
    return base_path


# ==================== 基础读取函数 ====================

def _read_csv_file(file_path: str, **kwargs) -> pd.DataFrame:
    """读取CSV文件，自动尝试UTF-8和GBK编码"""
    try:
        return pd.read_csv(file_path, encoding='utf-8', **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding='gbk', **kwargs)


def _read_parquet_file(file_path: str, columns: List[str] = None) -> pd.DataFrame:
    """
    读取Parquet文件
    
    参数:
        file_path: 文件路径
        columns: 只读取指定的列（Parquet列式存储的优势）
    """  
    if columns:
        return pd.read_parquet(file_path, columns=columns, engine='pyarrow')
    return pd.read_parquet(file_path, engine='pyarrow')


def _read_feather_file(file_path: str, columns: List[str] = None) -> pd.DataFrame:
    """
    读取Feather文件
    
    参数:
        file_path: 文件路径
        columns: 只读取指定的列
    """
    if columns:
        return pd.read_feather(file_path, columns=columns)
    return pd.read_feather(file_path)


def _read_csv_chunked(file_path: str, chunk_size: int) -> pd.DataFrame:
    """分批读取大CSV文件"""
    chunks = []
    total_rows = 0
    
    for chunk in _read_csv_file(file_path, chunksize=chunk_size):
        chunks.append(chunk)
        total_rows += len(chunk)
        print(f"已处理 {total_rows} 行")
        
        if len(chunks) % 10 == 0:
            gc.collect()
    
    print("开始合并数据...")
    return pd.concat(chunks, ignore_index=True)


# ==================== 核心读取函数 ====================

def _get_file_columns(file_path: str, file_format: str) -> List[str]:
    """快速获取文件的列名（不读取全部数据）"""
    if file_format == 'parquet':
        import pyarrow.parquet as pq
        return pq.read_schema(file_path).names
    elif file_format == 'feather':
        import pyarrow.feather as feather
        return feather.read_table(file_path).column_names
    else:  # CSV
        return pd.read_csv(file_path, nrows=0, encoding='utf-8').columns.tolist()


def get_data(file_path: str, required_columns: List[str] = None, chunk_size: int = 100000) -> pd.DataFrame:
    """
    智能读取数据文件，支持CSV、Parquet、Feather格式
    
    参数:
        file_path: 数据文件路径（可带或不带扩展名）
                   - 无扩展名：按 parquet → feather → csv 优先级查找
                   - 有扩展名：直接识别格式读取
        required_columns: 必需的列名列表（用于校验，前置校验避免无效读取）
        chunk_size: CSV分批读取的行数（仅对大CSV有效）
        
    返回:
        pd.DataFrame
    """
    # 1. 查找文件
    actual_path = get_data_path(file_path)
    if not os.path.exists(actual_path):
        raise FileNotFoundError(f"找不到数据文件: {file_path} (尝试了 {SUPPORTED_FORMATS})")
    file_format = get_file_format(actual_path)
    file_name = os.path.basename(actual_path)
    
    # 2. 前置校验必需列（避免读完才发现列缺失）
    if required_columns:
        try:
            columns = _get_file_columns(actual_path, file_format)
            missing_cols = [col for col in required_columns if col not in columns]
            if missing_cols:
                raise ValueError(f"文件 '{file_name}' 缺少必需列: {missing_cols}")
        except Exception as e:
            if "缺少必需列" in str(e):
                raise
            # 如果获取列名失败，继续读取后再校验
    
    # 3. 根据格式读取
    if file_format == 'parquet':
        df = _read_parquet_file(actual_path)
        
    elif file_format == 'feather':
        df = _read_feather_file(actual_path)
        
    else:  # CSV
        file_size_mb = os.path.getsize(actual_path) / (1024 * 1024)
        if file_size_mb > 500:
            print(f"文件较大 ({file_size_mb:.1f}MB)，开始分批读取...")
            df = _read_csv_chunked(actual_path, chunk_size)
        else:
            df = _read_csv_file(actual_path)
    
    print(f"读取 '{file_name}': {df.shape[0]} 行 × {df.shape[1]} 列")
    # 将 string dtype 列的 pd.NA 统一转为 np.nan，兼容后续所有模块的 == 比较
    string_cols = df.select_dtypes(include='string').columns
    for col in string_cols:
        df[col] = df[col].astype(object).where(df[col].notna(), other=np.nan)
    return df


# ==================== 数据保存函数 ====================

def _prepare_df_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """
    准备DataFrame用于Parquet保存，处理不支持的数据类型
    """
    df_copy = df.copy()
    
    for col in df_copy.columns:
        dtype = df_copy[col].dtype
        
        # Period类型转字符串
        if str(dtype).startswith('period'):
            df_copy[col] = df_copy[col].astype(str)
        # object类型统一转字符串（避免混合类型问题）
        elif dtype == 'object':
            df_copy[col] = df_copy[col].astype(str).replace('nan', pd.NA)
    
    return df_copy


def save_data(df: pd.DataFrame, file_path: str, format: str = 'parquet') -> str:
    """
    保存数据，支持Parquet、Feather、CSV格式
    
    参数:
        df: 要保存的DataFrame
        file_path: 保存路径（不带扩展名）
        format: 保存格式 ('parquet', 'feather', 'csv')
        
    返回:
        实际保存的文件路径（带扩展名）
    """
    # 移除可能的扩展名
    base_path = os.path.splitext(file_path)[0]
    
    # 确保目录存在
    dir_path = os.path.dirname(base_path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    
    file_name = os.path.basename(base_path)
    
    if format == 'parquet':
        save_path = base_path + '.parquet'
        df_safe = _prepare_df_for_parquet(df)
        df_safe.to_parquet(save_path, index=False, engine='pyarrow', compression=PARQUET_COMPRESSION)
        
    elif format == 'feather':
        save_path = base_path + '.feather'
        df_safe = _prepare_df_for_parquet(df)
        df_safe.to_feather(save_path)
        
    else:  # csv
        save_path = base_path + '.csv'
        df.to_csv(save_path, index=False, encoding='utf-8-sig')
    
    # print(f"已保存: {os.path.basename(save_path)}")
    return save_path


# ==================== OOT文件查找函数 ====================

def find_oot_files(data_dir: str = 'data', prefix: str = 'oot_') -> List[str]:
    """
    查找指定目录下的OOT文件
    
    参数:
        data_dir: 数据目录路径
        prefix: 文件名前缀，默认'oot_'
        
    返回:
        OOT文件路径列表（不带扩展名，去重后）
        
    示例:
        # 查找原始OOT文件
        find_oot_files('data')
        
        # 查找清洗后的OOT文件
        find_oot_files('outputs/data_clean_split/data')
    """
    oot_files = set()
    
    for ext in SUPPORTED_FORMATS:
        pattern = os.path.join(data_dir, f'{prefix}*{ext}')
        for f in glob.glob(pattern):
            # 返回不带扩展名的路径
            base = os.path.splitext(f)[0]
            oot_files.add(base)
    
    # 排序：oot_main优先，其他按字母顺序
    def sort_key(path):
        name = os.path.basename(path)
        if name == 'oot_main':
            return (0, name)  # oot_main排第一
        return (1, name)  # 其他按字母顺序
    
    return sorted(list(oot_files), key=sort_key)


# ==================== 数据类型识别函数 ====================

def identify_types(df, remove_fea=None):
    """
    识别数值型和类别型变量
    
    参数:
        df: DataFrame，输入数据
        remove_fea: list，需要排除的字段（如ID列、日期列）
        
    返回:
        num_fea: list，数值型变量名列表（保持原始列顺序）
        cat_fea: list，类别型变量名列表（保持原始列顺序）
    """
    # 将 remove_fea 转换为 set 以提高查找效率
    remove_fea_set = set(remove_fea) if remove_fea else set()
    
    cat_fea = []
    num_fea = []
    
    # 遍历所有列，保持原始顺序，只遍历一次以提高性能
    for col in df.columns:
        if col not in remove_fea_set:
            if df[col].dtype == 'object':
                cat_fea.append(col)
            else:
                num_fea.append(col)
    
    return num_fea, cat_fea


# ==================== 兼容旧代码的函数 ====================

def _read_csv_with_encoding(file_path, **kwargs):
    """兼容旧代码：读取CSV文件"""
    return _read_csv_file(file_path, **kwargs)

def need_batch_read(file_path, size_threshold_mb=500):
    """判断是否需要分批读取"""
    if not os.path.exists(file_path):
        return False
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    return file_size_mb > size_threshold_mb

def read_one_csv(file_path, required_columns):
    """兼容旧代码：读取单个CSV文件"""
    return get_data(file_path, required_columns)

def merge_alldata(df_list, feature_list, base_fea):
    """合并多个DataFrame"""
    all_fea_names = []
    processed_dfs = []
    for df, fea in zip(df_list, feature_list):
        all_fea_names += fea
        processed_dfs.append(df[base_fea + fea])
    merged_df = reduce(lambda left, right: pd.merge(left, right, on=base_fea, how='inner'), processed_dfs)
    return merged_df, all_fea_names
