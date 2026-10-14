import argparse
import json
import os
import pandas as pd
import sys
import datetime
import glob
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.utils_model_evaluation import evaluation_func
from utils.data_utils import get_data, find_oot_files, format_elapsed_hms

def run(config_path, session_id=None):
    """
    运行模型评估任务
    
    参数:
        config_path (str): 配置文件JSON的路径
        session_id (str, optional): 用于跟踪运行的会话ID
    """
    try:
        # 1.加载配置文件
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        # 2. 写入启动指示器
        startup_file = config['output_path']['eval_startup_path']
        with open(startup_file, 'w') as f:
            f.write('1')
         
        # 2.5 打印启动信息和当前时间
        start_time = datetime.datetime.now()
        log_message = f"[START] 【模型评估】启动于 {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        print(log_message)
        
        # 3.读取数据
        dev_data_path = config['data_path']['dev_data_path']
        test_data_path = config['data_path']['oot_data_path']  # 注意：配置中仍使用oot_data_path，但变量名改为test
        psi_data_path = config['data_path']['psi_data_path']
        # 读取训练集和测试集数据
        dev_df = get_data(dev_data_path)
        test_df = get_data(test_data_path)
        psi_df = get_data(psi_data_path)
        
        # 3.5 扫描OOT文件路径（清洗后的OOT文件，支持parquet/feather/csv格式）
        oot_files = find_oot_files('outputs/data_clean_split/data')
        if oot_files:
            print(f"检测到 {len(oot_files)} 个OOT数据集: {[os.path.basename(f) for f in oot_files]}")
      
        # 4.运行模型评估
        result = evaluation_func(dev_df, test_df, psi_df, config, oot_files=oot_files)
        
        if result is not None:
            end_time = datetime.datetime.now()
            elapsed_seconds = (end_time - start_time).total_seconds()
            elapsed_hms = format_elapsed_hms(elapsed_seconds)
            log_message = f"[END] 【模型评估】完成于 {end_time.strftime('%Y-%m-%d %H:%M:%S')}，耗时: {elapsed_hms}\n"
            print(log_message)
            return True
        else:
            end_time = datetime.datetime.now()
            elapsed_seconds = (end_time - start_time).total_seconds()
            elapsed_hms = format_elapsed_hms(elapsed_seconds)
            log_message = f"[FAIL] 【模型评估】执行失败或未执行于 {end_time.strftime('%Y-%m-%d %H:%M:%S')}，耗时: {elapsed_hms}\n"
            print(log_message)
            return False
            
    except Exception as e:
        print(f"运行模型评估时出错: {str(e)}")
        if 'start_time' in locals():
            end_time = datetime.datetime.now()
            elapsed_seconds = (end_time - start_time).total_seconds()
            elapsed_hms = format_elapsed_hms(elapsed_seconds)
            print(f"[FAIL] 【模型评估】异常于 {end_time.strftime('%Y-%m-%d %H:%M:%S')}，耗时: {elapsed_hms}")
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='运行模型评估')
    parser.add_argument('-conf', '--config', required=True, help='配置文件JSON的路径')
    parser.add_argument('--session-id', help='用于跟踪运行的会话ID')
    
    args = parser.parse_args()
    
    run(args.config, args.session_id) 
