import argparse
import json
import os
import pandas as pd
from utils.utils_data_clear_split import data_clear_func, data_split_func
from utils.data_utils import get_data, find_oot_files, format_elapsed_hms
import datetime

def run(config_path, session_id=None):
    """
    运行数据清洗与切分任务
    
    参数:
        config_path (str): 配置文件JSON的路径
        session_id (str, optional): 用于跟踪运行的会话ID
    """
    try:
        # 1.加载配置文件
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        # 2. 写入启动指示器
        startup_file = config['output_path']['clean_startup_path']
        with open(startup_file, 'w') as f:
            f.write('1')

        # 2.5 打印启动信息和当前时间
        start_time = datetime.datetime.now()
        log_message = f"[START] 【数据清洗和切分】启动于 {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        print(log_message)
        
        # 3.读取数据
        data_path = config['data_path']
        required_columns = config['settings']['base_fea']
        df = get_data(data_path, required_columns)
        
        # 3.5 扫描OOT文件路径（支持parquet/feather/csv格式）
        oot_files = find_oot_files('data')
        
        # 4.运行数据清洗与切分 OOT数量不确定，延迟读取更灵活
        result = data_split_func(df, config, oot_files)
        
        if result:
            end_time = datetime.datetime.now()
            elapsed_seconds = (end_time - start_time).total_seconds()
            elapsed_hms = format_elapsed_hms(elapsed_seconds)
            log_message = f"[END] 【数据清洗与切分】完成于 {end_time.strftime('%Y-%m-%d %H:%M:%S')}，耗时: {elapsed_hms}\n"
            print(log_message)
            return True
        else:
            end_time = datetime.datetime.now()
            elapsed_seconds = (end_time - start_time).total_seconds()
            elapsed_hms = format_elapsed_hms(elapsed_seconds)
            log_message = f"[FAIL] 【数据清洗与切分】执行失败于 {end_time.strftime('%Y-%m-%d %H:%M:%S')}，耗时: {elapsed_hms}\n"
            print(log_message)
            return False
            
    except Exception as e:
        print(f"运行数据清洗与切分时出错: {str(e)}")
        if 'start_time' in locals():
            end_time = datetime.datetime.now()
            elapsed_seconds = (end_time - start_time).total_seconds()
            elapsed_hms = format_elapsed_hms(elapsed_seconds)
            print(f"[FAIL] 【数据清洗与切分】异常于 {end_time.strftime('%Y-%m-%d %H:%M:%S')}，耗时: {elapsed_hms}")
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='运行数据清洗与切分')
    parser.add_argument('-conf', '--config', required=True, help='配置文件JSON的路径')
    parser.add_argument('--session-id', help='用于跟踪运行的会话ID')
    
    args = parser.parse_args()
    
    run(args.config, args.session_id) 
