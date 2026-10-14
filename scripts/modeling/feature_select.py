import argparse
import hashlib
import json
import os
import pandas as pd
import sys
import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.utils_feature_select import select_features_func
from utils.data_utils import get_data, get_data_path, format_elapsed_hms

NULL_IMPORTANCE_CONFIG_PATH = os.path.join('data', 'null_importance_config.json')
KS_OOT_MAIN_PATH = os.path.join('outputs', 'data_clean_split', 'data', 'oot_main')
KS_TEST_PATH = os.path.join('outputs', 'data_clean_split', 'data', 'test')


def _file_sha256(file_path):
    """Return a content fingerprint without loading the whole file into memory."""
    digest = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def run(config_path, session_id=None):
    """
    运行特征筛选任务
    
    参数:
        config_path (str): 配置文件JSON的路径
        session_id (str, optional): 用于跟踪运行的会话ID
    """
    try:
        # 1.加载配置文件
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        # 2. 写入启动指示器
        startup_file = config['output_path']['feature_startup_path']
        with open(startup_file, 'w') as f:
            f.write('1')
          
        # 2.5 打印启动信息和当前时间
        start_time = datetime.datetime.now()
        log_message = f"[START] 【特征筛选】启动于 {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        print(log_message)
        
        # 3.读取数据
        dev_data_path = config['data_path']['dev_data_path']
        psi_data_path = config['data_path'].get('psi_data_path', config['data_path']['oot_data_path'])
        
        # 读取训练集和PSI数据集
        dev_data_file = get_data_path(dev_data_path)
        psi_data_file = get_data_path(psi_data_path)
        dev_df = get_data(dev_data_file)
        psi_df = get_data(psi_data_file)
        input_hashes = {
            'train_sha256': _file_sha256(dev_data_file),
            'psi_sha256': _file_sha256(psi_data_file),
        }

        oot_df = None
        if config.get('settings', {}).get('use_null_importance') and os.path.exists(NULL_IMPORTANCE_CONFIG_PATH):
            oot_data_path = KS_OOT_MAIN_PATH if os.path.exists(get_data_path(KS_OOT_MAIN_PATH)) else KS_TEST_PATH
            oot_df = get_data(oot_data_path)
        
        # 4.运行特征筛选
        result = select_features_func(dev_df, psi_df, config, oot_df=oot_df, input_hashes=input_hashes)
        
        if result:
            end_time = datetime.datetime.now()
            elapsed_seconds = (end_time - start_time).total_seconds()
            elapsed_hms = format_elapsed_hms(elapsed_seconds)
            log_message = f"[END] 【特征筛选】完成于 {end_time.strftime('%Y-%m-%d %H:%M:%S')}，耗时: {elapsed_hms}\n"
            print(log_message)
            return True
        else:
            end_time = datetime.datetime.now()
            elapsed_seconds = (end_time - start_time).total_seconds()
            elapsed_hms = format_elapsed_hms(elapsed_seconds)
            log_message = f"[FAIL] 【特征筛选】执行失败于 {end_time.strftime('%Y-%m-%d %H:%M:%S')}，耗时: {elapsed_hms}\n"
            print(log_message)
            return False
            
    except Exception as e:
        print(f"运行特征筛选时出错: {str(e)}")
        if 'start_time' in locals():
            end_time = datetime.datetime.now()
            elapsed_seconds = (end_time - start_time).total_seconds()
            elapsed_hms = format_elapsed_hms(elapsed_seconds)
            print(f"[FAIL] 【特征筛选】异常于 {end_time.strftime('%Y-%m-%d %H:%M:%S')}，耗时: {elapsed_hms}")
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='运行特征筛选')
    parser.add_argument('-conf', '--config', required=True, help='配置文件JSON的路径')
    parser.add_argument('--session-id', help='用于跟踪运行的会话ID')
    
    args = parser.parse_args()
    
    run(args.config, args.session_id) 
