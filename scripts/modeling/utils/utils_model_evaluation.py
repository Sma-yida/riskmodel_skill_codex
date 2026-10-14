import os  
import json
import pandas as pd
import numpy as np
import traceback
import warnings
from scipy import stats
warnings.filterwarnings("ignore")

# import sys
# sys.path.append('/PycharmProjects/autoML-pipline/utils/')
from .utils1 import *
from .pyce1 import *
from utils.data_utils import identify_types
from copy import deepcopy
import pickle
from utils.data_utils import get_data

def get_auc_ks(dataset_type,df, target, col, pred):
    auc, ks, flag = [], [], []
    total_num, good_num, bad_num, bad_ratio = [], [], [], []
    item = list(sorted(df[col].unique()))
    for i in item:
        tmp = df[df[col] == i]
        total_num.append(tmp.shape[0])
        bad_num.append(tmp[target].sum())
        good_num.append(tmp.shape[0] - tmp[target].sum())
        bad_ratio.append(tmp[target].mean())
        # 新增：判断类别数
        if tmp[target].nunique() < 2:
            auc.append(None)
            ks.append(None)
        else:
            auc.append(roc_auc_score(tmp[target], tmp[pred]))
            ks.append(ks_score(tmp[target], tmp[pred]))
        flag.append(i)
    return pd.DataFrame({
        'dataset':dataset_type,
        'col': flag,
        'total': total_num,
        'good': good_num,
        'bad': bad_num,
        'bad_ratio': bad_ratio,
        'auc': auc,
        'ks': ks
    })

def get_final_result(dataset_type, df, target, pred, col, dst):
    """
    获取模型评估结果（合并整体和分渠道结果）
    
    参数:
        dataset_type : str,数据类型[dev|oot]
        df: DataFrame, 输入数据
        target: str, 目标变量列名
        pred: str, 预测概率列名
        col: str, 分组列名
        dst: str, 输出目录
    """
    # 获取整体表现
    df['overall'] = 'overall'
    overall_df = get_auc_ks(dataset_type, df, col='overall', target=target, pred=pred)
    # 获取分渠道表现
    channel_df = get_auc_ks(dataset_type, df, col=col, target=target, pred=pred)
    
    # 合并结果：overall放在上面，channel放在下面
    combined_df = pd.concat([overall_df, channel_df], ignore_index=True)
    
    # 保存合并后的结果（只保存一个CSV文件）
    combined_df.to_csv(os.path.join(dst, f'{dataset_type}_{col}_auc_ks.csv'), index=False)
    
    return combined_df

def evaluate_psi(train_df, test_df, pred_col):
    psi = Psi()
    psi.fit(train_df[pred_col].values)
    return psi.transform(test_df[pred_col].values)[0]


def model_pred(df, mod, mod_var, prob_col, is_autogluon=False, model_name=None):
    """
    对输入的df数据进行模型预测并打分

    参数：
        df: 原始数据 DataFrame
        mod: 已训练模型对象
            - 当is_autogluon=False时：LightGBM模型对象
            - 当is_autogluon=True时：TabularPredictor对象（即predictor）
        mod_var: 模型使用的特征名列表
        prob_col: 预测概率列名
        is_autogluon: bool, 是否为AutoGluon模型
        model_name: str, AutoGluon模型名称（如'XGBoost'），当is_autogluon=True时必需
                   用于指定使用哪个模型进行预测

    返回：
        包含预测概率和评分的新 df
    """
    df = df.copy()
    if is_autogluon:
        # AutoGluon模型预测：使用predictor.predict_proba，指定模型
        # mod此时是TabularPredictor对象，用于加载模型和执行数据预处理
        # 注意：AutoGluon只需要特征列，会自动处理数据预处理
        pred_data = df[mod_var].copy()
        
        # 使用指定模型进行预测
        predictions = mod.predict_proba(pred_data, model=model_name)
        # predictions是DataFrame，包含两列（0和1），取正类概率
        if isinstance(predictions, pd.DataFrame):
            df[prob_col] = predictions.iloc[:, 1].values  # 获取正类概率
        else:
            # 如果是numpy数组，取第二列
            df[prob_col] = predictions[:, 1]
    else:
        # 传统LightGBM模型预测
        df[prob_col] = mod.predict_proba(df[mod_var])[:, 1]
    return df

def lift_analysis(test_df, label_col, pred_col,bin_num):
    base = test_df[label_col].mean()
    # lift_low = test_df[test_df[score_col] <= 450][label_col].mean() / base
    # lift_high = test_df[test_df[score_col] > 700][label_col].mean() / base
    # lift_ratio = lift_low / lift_high
    # print("score>700:",lift_high,"score<=450:",lift_low)

    lift_20 = sta_groups(test_df[label_col], test_df[pred_col], bins_num=bin_num, labels=list(range(bin_num, 0, -1)))
    return {
        'base_rate': base,
        'lift_top_20': lift_20
    }

def get_model_description(model_name):
    """
    获取模型的描述信息
    
    参数:
        model_name: str, 模型名称（可能包含后缀，如CatBoost_r49）
    
    返回:
        str, 模型描述
    """
    # 提取基础模型名称（去掉后缀，如_r49, _r12等）
    import re
    base_model_name = re.sub(r'_r\d+$', '', model_name)  # 去掉_r数字后缀
    base_model_name = re.sub(r'Large$', '', base_model_name)  # 去掉Large后缀
    
    # 检查是否有变体编号
    variant_match = re.search(r'_r(\d+)$', model_name)
    variant_info = f"（变体编号：{variant_match.group(1)}）" if variant_match else ""
    
    model_descriptions = {
        'XGBoost': 'XGBoost是一种高效的梯度提升决策树算法，采用梯度提升框架，通过集成多个弱学习器提升模型性能。内置正则化机制，有效防止过拟合，支持并行计算，训练速度快，对缺失值有良好的处理能力，在结构化数据上表现优异。',
        'CatBoost': 'CatBoost是Yandex开发的梯度提升算法，特别擅长处理类别特征。自动处理类别特征，无需手动编码，采用有序提升方法减少过拟合，内置类别特征组合自动发现特征交互，对类别特征和数值特征都有良好支持，训练过程稳定不易过拟合。',
        'LightGBM': '该LightGBM模型，是使用多模型对比得到的，是微软开发的高效梯度提升框架，采用基于直方图的算法，训练速度快，内存占用低。支持类别特征直接输入，适合大规模数据，支持并行和GPU加速，在保持高精度的同时训练效率高。',
        'LightGBMXT': 'LightGBM的极端树变体，采用更激进的剪枝策略，模型更简洁，训练速度更快，适合对模型大小有要求的场景。',
        'RandomForestEntr': '使用熵作为分裂准则的随机森林模型，适合处理类别不平衡问题。集成多个决策树提高模型稳定性，对过拟合有较好的抵抗能力，可以评估特征重要性，训练过程可并行化。',
        'RandomForestGini': '使用Gini系数作为分裂准则的随机森林模型，计算效率高。集成多个决策树提高模型稳定性，对过拟合有较好的抵抗能力，可以评估特征重要性，训练过程可并行化。',
        'ExtraTreesEntr': '使用熵分裂的极端随机树，分裂点选择更随机，减少过拟合。训练速度比随机森林更快，对噪声数据更鲁棒，适合高维数据，模型方差较小。',
        'ExtraTreesGini': '使用Gini分裂的极端随机树，计算效率高且过拟合风险低。训练速度比随机森林更快，对噪声数据更鲁棒，适合高维数据，模型方差较小。',
        'KNeighborsDist': '基于距离加权的K近邻算法，距离越近的样本权重越大。无需训练过程，基于实例学习，对局部模式敏感，适合非线性问题，但计算复杂度随样本数增加。',
        'KNeighborsUnif': '基于均匀加权的K近邻算法，所有K个近邻样本权重相等。无需训练过程，基于实例学习，实现简单易于理解，适合非线性问题，但计算复杂度随样本数增加。',
        'NeuralNetTorch': '基于PyTorch实现的深度神经网络，能够学习复杂的非线性关系。支持多层网络结构，适合大规模数据，可以通过调整网络结构优化性能，但训练时间相对较长。',
        'NeuralNetFastAI': '基于FastAI框架的神经网络，训练效率高，适合快速原型开发。内置多种优化技巧，对超参数不敏感，在结构化数据上表现良好。',
        'WeightedEnsemble_L2': 'AutoGluon自动构建的加权集成模型，通过组合多个基础模型提升性能。通过加权平均提升预测性能，通常比单个模型表现更好，但模型复杂度较高，预测时间相对较长。',
        'RandomForest': '随机森林模型，集成多个决策树提高模型稳定性。对过拟合有较好的抵抗能力，可以评估特征重要性，训练过程可并行化。',
        'ExtraTrees': '极端随机树模型，分裂点选择更随机，减少过拟合。训练速度比随机森林更快，对噪声数据更鲁棒，适合高维数据，模型方差较小。'
    }
    
    base_description = model_descriptions.get(base_model_name, f'{base_model_name}模型，由AutoGluon自动训练和优化，模型参数已自动调优，适合当前数据集特点。')
    
    # 如果有变体编号，添加到描述中
    if variant_info:
        base_description = f'{base_description}{variant_info}'
    
    # 所有AutoGluon模型都加上部署限制说明
    full_description = f'{base_description} 该模型受限于模型开发环境，暂不支持部署于生产环境。'
    
    return full_description


def evaluation_func(dev_df, test_df, psi_df, config, oot_files=None):
    """
    模型评估函数
    
    参数:
        dev_df: DataFrame, 训练集数据
        test_df: DataFrame, 测试集数据  
        psi_df: DataFrame, PSI数据集
        config: dict, 配置参数
        oot_files: list, OOT文件路径列表（可选）
    """
    msg='evaluation: succ'
    try:
        print("=" * 60)
        print("开始模型评估流程")
         ## 基本参数导入
        settings = config['settings']
        id_col = settings['id_col']
        date_col = settings['date_col']
        label = settings['label']
        channel_col = settings['channel_col']
        base_fea = settings['base_fea']
        bin_num = settings['bin_num']     
        OUT_REPORT = settings['gen_report']
        OUTPUT = settings['save_output']
        IF_RUN = settings['run_eval']

        if IF_RUN:
            ## 导入模型 - 支持LightGBM和AutoGluon模型
            # 获取评估模型类型和名称（如果用户未指定，从模型训练JSON中读取）
            eval_model_type = settings.get('eval_model_type') ### 默认 null，可选'lightgbm'或'autogluon'
            eval_model_name = settings.get('eval_model_name') ### 默认 null，可选'None'或'XGBoost'、'CatBoost'、'LightGBM'等已经训练好的模型
            autogluon_predictor_path = config['model_file'].get('autogluon_predictor_path')
            model_train_config_path = config['model_file'].get('model_train_config_path')

            # 如果用户未指定eval_model_type（为None或null字符串），从模型训练配置文件中读取model_select
            if eval_model_type is None or (isinstance(eval_model_type, str) and eval_model_type.lower() in ['null', 'none']):
                # 直接从model_train_config.json读取model_select字段
                if os.path.exists(model_train_config_path):
                    with open(model_train_config_path, 'r', encoding='utf-8') as f:
                        train_config = json.load(f)
                    model_select = train_config.get('settings', {}).get('model_select')
                    eval_model_type = model_select.lower()  # 转换为小写，统一处理
                    print(f"从模型训练配置文件读取eval_model_type: {eval_model_type}")
                else:
                    eval_model_type = 'lightgbm'
                    print(f"模型训练配置文件不存在 ({model_train_config_path})，使用默认值lightgbm")
            else:
                eval_model_type = eval_model_type.lower()

            if eval_model_type == 'lrscorecard':
                from utils.generate_json_reports import generate_lr_scorecard_evaluation_json

                print('检测到LrScorecard，第5阶段直接复用训练阶段Excel报告，不重新预测或计算指标')
                generate_lr_scorecard_evaluation_json(config)
                return msg
            
            model_info = {'model_type': 'lightgbm'}
            is_autogluon = False
            
            if eval_model_type == 'autogluon':
                # 加载AutoGluon模型
                from autogluon.tabular import TabularPredictor
                
                if not os.path.exists(autogluon_predictor_path):
                    raise FileNotFoundError(f"错误: AutoGluon predictor路径不存在: {autogluon_predictor_path}")
                
                # 检查模型是否存在（通过加载predictor后检查）
                predictor = TabularPredictor.load(autogluon_predictor_path)
                # 获取可用模型列表（通过leaderboard获取，更可靠）
                leaderboard = predictor.leaderboard(silent=True)
                available_models = leaderboard['model'].tolist() if 'model' in leaderboard.columns else []
                
                # 如果用户未指定模型名称，使用leaderboard第一名的模型（最好的模型）
                if not eval_model_name or eval_model_name == 'None' or eval_model_name.lower() == 'none':
                    if len(available_models) > 0:
                        eval_model_name = available_models[0]
                        print(f"未指定模型名称，使用leaderboard第一名模型: {eval_model_name}")
                    else:
                        raise FileNotFoundError("错误: 未找到可用的AutoGluon模型")
                elif eval_model_name not in available_models:
                    # 只显示前20个模型，避免输出过长
                    models_display = available_models[:20] if len(available_models) > 20 else available_models
                    raise FileNotFoundError(
                        f"错误: AutoGluon模型 '{eval_model_name}' 不存在\n"
                        f"可用的模型列表（前20个）: {models_display}\n"
                        f"共{len(available_models)}个模型，请检查模型名称是否正确"
                    )
                
                # 对于AutoGluon，mod就是predictor（TabularPredictor对象）
                mod = predictor
                is_autogluon = True
                
                model_info = {
                    'model_type': 'autogluon',
                    'model_name': eval_model_name
                }
                print(f'步骤1/9: AutoGluon模型导入完成 ({eval_model_name})')
            else:
                # 加载传统LightGBM模型
                pkl_model_path = config['model_file']['pkl_model_path']
                if not os.path.exists(pkl_model_path):
                    raise FileNotFoundError(
                        f"错误: LightGBM模型文件不存在: {pkl_model_path}\n"
                        f"请先运行模型训练，或设置eval_model_type='autogluon'使用AutoGluon模型"
                    )
                with open(pkl_model_path, 'rb') as f:
                    mod = pickle.load(f)
                print('步骤1/9: 常规LightGBM模型导入完成')
            
            # 加载特征列表
            with open(config['model_file']['model_fea_path'], encoding='utf-8') as f:
                mod_var = json.load(f)

            ## 模型分数转换参数
            pdo = settings['pdo']
            theta = settings['theta']
            p0 = settings['p0']
            cut_points = settings.get('cut_points', [])
            # 固定列名：不从配置读取
            prob_col = "prob_"
            score_col = "score_"

            ## 输出路径 - 根据模型类型动态构建
            base_dst = config['output_path']['dst']  # outputs/model_eval
            if is_autogluon and eval_model_name:
                # AutoGluon: outputs/model_eval/AutoGluon/{model_name}/
                model_subdir = os.path.join('Autogluon', eval_model_name)
            else:
                # LightGBM: outputs/model_eval/LightGBM/
                model_subdir = 'LightGBM'
            
            dst_output = f"{base_dst}/{model_subdir}/outputs"
            dst_report = f"{base_dst}/{model_subdir}/report"
            dst_data = f"{base_dst}/{model_subdir}/retrain_data"
            
            os.makedirs(dst_output, exist_ok=True)
            os.makedirs(dst_report, exist_ok=True)
            os.makedirs(dst_data, exist_ok=True)

            eval_json_path = config['output_path']['eval_json_path']     
            final_report_path = os.path.join(dst_report, 'model_evaluation_report.xlsx')
            print(f'步骤2/9: 输出目录创建完成 (评估模型类型: {model_subdir})')

            # 加载OOT数据集（从传入的oot_files）
            oot_datasets = {}
            if oot_files:
                for oot_file in oot_files:
                    dataset_name = os.path.splitext(os.path.basename(oot_file))[0]  # 例如: oot_main, oot_1
                    try:
                        oot_data = get_data(oot_file)
                        oot_datasets[dataset_name] = oot_data
                    except Exception as e:
                        print(f"  - {dataset_name}: 加载失败 - {str(e)}")
                        continue

            ## 区分数值型特征和类别型特征
            num_fea, cat_fea = identify_types(dev_df[mod_var])
            for i in cat_fea:
                dev_df[i] = dev_df[i].astype('category')
                test_df[i] = test_df[i].astype('category')
                psi_df[i] = psi_df[i].astype('category')
                # 处理所有OOT数据集
                for oot_data in oot_datasets.values():
                    if i in oot_data.columns:
                        oot_data[i] = oot_data[i].astype('category')
            
            # 模型概率预测：先预测所有数据集（避免重复预测）
            print("\n开始模型预测...")
            df_dev = model_pred(df=dev_df, mod=mod, mod_var=mod_var, prob_col=prob_col, is_autogluon=is_autogluon, model_name=eval_model_name)
            df_test = model_pred(df=test_df, mod=mod, mod_var=mod_var, prob_col=prob_col, is_autogluon=is_autogluon, model_name=eval_model_name)
            df_psi = model_pred(df=psi_df, mod=mod, mod_var=mod_var, prob_col=prob_col, is_autogluon=is_autogluon, model_name=eval_model_name)
            
            # 预测所有OOT数据集
            all_predictions = {'test': df_test}  # 包含test和所有OOT数据集
            if oot_datasets:
                for dataset_name, oot_data in oot_datasets.items():
                    df_oot_pred = model_pred(df=oot_data, mod=mod, mod_var=mod_var, prob_col=prob_col, is_autogluon=is_autogluon, model_name=eval_model_name)
                    all_predictions[dataset_name] = df_oot_pred

            ##模型PSI
            psi_value = evaluate_psi(df_dev, df_psi, pred_col=prob_col)
            psi_details = {
                "PSI集(psi)": float(psi_value),
                "测试集(test)": float(evaluate_psi(df_dev, df_test, pred_col=prob_col))
            }
            if 'oot_main' in all_predictions:
                psi_details["验证集(oot_main)"] = float(evaluate_psi(df_dev, all_predictions['oot_main'], pred_col=prob_col))
            for psi_name, psi_metric in psi_details.items():
                print(f"         {psi_name:<18} PSI={psi_metric:.4f}")
            print('步骤3/9: 模型稳定性(PSI)评估完成')

            ### 7.04 KS 结果计算 + 7.05 AUC 结果计算
            # 训练集评估
            train_temp = df_dev[base_fea+[prob_col]].copy()  # 使用.copy()避免SettingWithCopyError
            train_temp['overall'] = 'overall'
            train_combined_df = get_final_result('dev', train_temp, target=label, pred=prob_col, col=channel_col, dst=dst_output) 
            
            # 评估所有数据集（test + 所有OOT数据集），同时确定eval_dataset和计算时间范围
            all_eval_results = {}  # {dataset_name: (combined_df, display_name, date_range)}
            eval_dataset_key = 'oot_main' if 'oot_main' in all_predictions else 'test'  # 确定评估数据集key
            
            # 仅在计算时间范围时临时解析日期，避免修改原始date列格式
            dev_dates = pd.to_datetime(df_dev[date_col], format='%Y%m%d', errors='coerce')
            train_date_range = f"{dev_dates.min().strftime('%Y%m%d')}-{dev_dates.max().strftime('%Y%m%d')}"
            
            for dataset_name, df_pred in all_predictions.items():
                # 评估数据集
                dataset_temp = df_pred[base_fea+[prob_col]].copy()
                dataset_temp['overall'] = 'overall'
                if dataset_name == 'oot_main':
                    dataset_display_name = "验证集(oot_main)"
                elif dataset_name == 'test':
                    dataset_display_name = "测试集(test)"
                else:
                    dataset_display_name = f"验证集({dataset_name})"
                combined_df = get_final_result(dataset_name, dataset_temp, target=label, pred=prob_col, col=channel_col, dst=dst_output)
                
                # 计算时间范围（不修改原始date列）
                pred_dates = pd.to_datetime(df_pred[date_col], format='%Y%m%d', errors='coerce')
                date_range = f"{pred_dates.min().strftime('%Y%m%d')}-{pred_dates.max().strftime('%Y%m%d')}"
                
                # 存储评估结果和时间范围
                all_eval_results[dataset_name] = (combined_df, dataset_display_name, date_range)
                # 从合并的DataFrame中筛选overall行
                overall_row = combined_df[combined_df['col'] == 'overall'].iloc[0]
                print(f"         {dataset_display_name:<18} AUC={overall_row['auc']:.6f}, KS={overall_row['ks']:.6f}")
            
            # 获取eval_dataset的结果和名称（用于后续Lift分析等）
            df_eval = all_predictions[eval_dataset_key]
            eval_combined_df, eval_dataset_name, _ = all_eval_results[eval_dataset_key]
            eval_temp = df_eval[base_fea+[prob_col]].copy()
            eval_temp['overall'] = 'overall'
            print(f"评估数据集: {eval_dataset_name}")
            print('步骤4/9: 模型性能(KS/AUC)评估完成')

            #### 7.06 Lift 结果计算
            lift_res = lift_analysis(eval_temp, label_col=label, pred_col=prob_col, bin_num=bin_num)
            lift_res['lift_top_20'].to_csv(os.path.join(dst_output, f'bin{bin_num}_lift_res.csv'), index=False)
            
            # 计算 Lift 指标（基于原始数据，按预测概率排序）
            base_bad_rate = lift_res['base_rate']
            df_sorted = eval_temp.sort_values(prob_col, ascending=False)
            n = len(df_sorted)
            
            if base_bad_rate:
                # 尾部=坏人=概率最高的（排序后的开头）
                lift_res['lift_tail10'] = df_sorted[label].iloc[:max(1, int(n * 0.10))].mean() / base_bad_rate
                # 头部=好人=概率最低的（排序后的末尾）
                lift_res['lift_head10'] = df_sorted[label].iloc[-max(1, int(n * 0.10)):].mean() / base_bad_rate
                lift_res['lift_head5'] = df_sorted[label].iloc[-max(1, int(n * 0.05)):].mean() / base_bad_rate
                lift_res['lift_head2'] = df_sorted[label].iloc[-max(1, int(n * 0.02)):].mean() / base_bad_rate
                lift_res['lift_head1'] = df_sorted[label].iloc[-max(1, int(n * 0.01)):].mean() / base_bad_rate
            else:
                lift_res['lift_tail10'] = lift_res['lift_head10'] = lift_res['lift_head5'] = lift_res['lift_head2'] = lift_res['lift_head1'] = 0
            
            print('步骤5/9: 预测概率的Lift分析完成')

            
            # 模型分数转换
            # 清理异常值，防止除以零错误
            def safe_transfer_score(x, p0, pdo, theta):
                if pd.isna(x) or x < 0 or x > 1:
                    x = 0.5  # 使用默认值
                # 统一处理边界值：确保 x 在安全范围内 [0.00000001, 0.99999999]
                else:
                    x = max(0.00000001, min(0.99999999, x))
                return Transfer2Score(x, p0=p0, PDO=pdo, theta=theta)
            
            # 模型分数转换：只对评估数据集
            df_dev[score_col] = df_dev[prob_col].apply(lambda x: safe_transfer_score(x, p0, pdo, theta))
            df_eval[score_col] = df_eval[prob_col].apply(lambda x: safe_transfer_score(x, p0, pdo, theta))
            df_psi[score_col] = df_psi[prob_col].apply(lambda x: safe_transfer_score(x, p0, pdo, theta))
            
            # 保存包含基础特征、入模特征、预测概率和分数的数据集，用于复现结果（仅LightGBM需要，用于串行验证）
            if not is_autogluon:
                # 添加入模特征，并将日期列重命名为month
                dev_reproduce = df_dev[base_fea + mod_var + [prob_col, score_col]].copy()
                eval_reproduce = df_eval[base_fea + mod_var + [prob_col, score_col]].copy()
                psi_reproduce = df_psi[base_fea + mod_var + [prob_col, score_col]].copy()
                
                # 将日期列重命名为month，并精确到月份
                dev_reproduce['month'] = pd.to_datetime(dev_reproduce[date_col], format='%Y%m%d', errors='coerce').dt.to_period('M')
                eval_reproduce['month'] = pd.to_datetime(eval_reproduce[date_col], format='%Y%m%d', errors='coerce').dt.to_period('M')
                psi_reproduce['month'] = pd.to_datetime(psi_reproduce[date_col], format='%Y%m%d', errors='coerce').dt.to_period('M')
                
                # 保存复现用数据集（仅用于串行验证，使用eval_dataset）
                dev_reproduce.to_csv(os.path.join(dst_data, 'df_dev_retrain.csv'), index=False)
                eval_reproduce.to_csv(os.path.join(dst_data, 'df_oot_retrain.csv'), index=False)
                psi_reproduce.to_csv(os.path.join(dst_data, 'df_psi_retrain.csv'), index=False)
            
            # 分数转换的序行结果（使用eval_dataset）
            # 如果 cut_points 为空，则自动生成等距10分箱
            if cut_points is None or len(cut_points) == 0:
                score_min = df_eval[score_col].min()
                score_max = df_eval[score_col].max()
                # 等距分成10箱
                step = (score_max - score_min) / 10
                cut_points = [score_min + i * step for i in range(11)]
                # 最小值替换为0，最大值替换为1000
                cut_points[0] = 0
                cut_points[-1] = 1000
            
            df_score = get_sta_groups(df_eval[label], df_eval[score_col], cut_points)
            
            # 分析：坏率是否单调递减（分数越高，坏率越低）
            bad_rates = df_score['bad_ration'].tolist()
            is_bad_rate_monotonic = all(bad_rates[i] >= bad_rates[i+1] for i in range(len(bad_rates)-1))

            # 调用并保存图片
            sorting_ability_plot(df_score, 'bad_rate', text='Score Distribution', is_asce=False,save_path=dst_output+'/score_distribution.png')
            
            # 分析：样本数是否符合正态分布（使用 Shapiro-Wilk 检验，p>0.05 认为正态）
            nums = df_score['num'].tolist()
            if len(nums) >= 3:
                _, p_value = stats.shapiro(nums)
                is_num_normal = p_value > 0.05
            else:
                is_num_normal = False
            
            # 构建分数转换结果
            score_transform_result = {
                'params': f"p0={p0}, pdo={pdo}, theta={theta}",
                'cut_points': cut_points,
                'df_score': df_score,
                'is_bad_rate_monotonic': is_bad_rate_monotonic,
                'is_num_normal': is_num_normal
            }
            print('步骤6/9: 分数转换完成')

            # 独立压测评估：
            # - df_eval 是本流程已选定的主评估集（优先 oot_main，否则 test），已含标签；
            # - df_eval[prob_col] 是步骤4已完成的基准预测概率；
            # - mod 是步骤1已加载的当前模型，仅Top2缺失场景会通过下方回调再次预测。
            # 压测失败或场景字段缺失均不影响主评估流程。
            try:
                from .utils_model_stress import STRESS_TEST_ENABLED, run_model_stress_tests

                if STRESS_TEST_ENABLED:
                    stress_output_dir = os.path.join(base_dst, model_subdir, 'stress_test')

                    def stress_predict_func(stress_df):
                        stress_result = model_pred(
                            df=stress_df,
                            mod=mod,
                            mod_var=mod_var,
                            prob_col='__stress_prob__',
                            is_autogluon=is_autogluon,
                            model_name=eval_model_name,
                        )
                        return stress_result['__stress_prob__'].to_numpy()

                    run_model_stress_tests(
                        eval_df=df_eval,
                        baseline_prob=df_eval[prob_col],
                        predict_func=stress_predict_func,
                        model_features=mod_var,
                        label_col=label,
                        feature_importance_path=config['report_all']['feature_importance_path'],
                        output_dir=stress_output_dir,
                        score_params={'p0': p0, 'pdo': pdo, 'theta': theta},
                        dataset_name=eval_dataset_key,
                    )
            except Exception as stress_error:
                print(f"压测评估执行失败，已跳过且不影响主评估: {stress_error}")
                print(traceback.format_exc())
            
            # 生成模型评估JSON报告（在分数转换后，包含完整信息）
            from utils.generate_json_reports import generate_evaluation_json
            generate_evaluation_json(
                train_combined_df=train_combined_df,
                all_eval_results=all_eval_results,
                psi_value=psi_value,
                lift_res=lift_res,
                output_path=eval_json_path,
                train_date_range=train_date_range,
                eval_dataset_key=eval_dataset_key,
                date_col=date_col,
                model_info=model_info,
                score_transform_result=score_transform_result,
                psi_details=psi_details
            )
            print('步骤7/9: 评估报告JSON生成完成')
            
            ### 报告汇总
            if OUT_REPORT:
                print('步骤8/9: 开始生成Excel模型报告...')
                dic_rp = config['report_all']
                # ========== 固定保持不变的文件 【数据质量|违约率|单变量分箱|lift分箱】==========
                if os.path.exists(dic_rp['data_quality_report_path']):
                    df_quality = pd.read_excel(dic_rp['data_quality_report_path']) #数据质量报告
                else:
                    df_quality = pd.DataFrame()
                if os.path.exists(dic_rp["target_stats_report_path"]):
                    df_default_dist = pd.read_csv(dic_rp["target_stats_report_path"])##违约率报告
                else:
                    df_default_dist = pd.DataFrame()
                
                mono_var_path = dic_rp['mono_var_path']
                if os.path.exists(mono_var_path):
                    df_mono_var = pd.read_csv(mono_var_path)   ##取单变量分箱结果
                else:
                    df_mono_var = pd.DataFrame()

                df_lift = lift_res['lift_top_20']
                # ========== 入模特征分析：合并特征分析结果 ==========
                single_var_path = dic_rp['single_var_path']
                feature_importance_path = dic_rp['feature_importance_path']
                
                # 读取文件（如果存在），保证有文件就有Feature列
                df_feat_analysis = pd.read_csv(single_var_path) if os.path.exists(single_var_path) else pd.DataFrame()
                df_importance = pd.read_csv(feature_importance_path) if os.path.exists(feature_importance_path) else pd.DataFrame()
                
                # 合并分析和重要性（用Feature字段）
                if not df_importance.empty and not df_feat_analysis.empty:
                    df_feat_merged = df_importance.merge(df_feat_analysis, on='Feature', how='left')
                elif not df_importance.empty:
                    df_feat_merged = df_importance
                elif not df_feat_analysis.empty:
                    df_feat_merged = df_feat_analysis
                else:
                    df_feat_merged = pd.DataFrame()
                
                # ========== 模型效果合并 ==========
                # 合并所有数据集的结果（已经是合并后的DataFrame，直接concat即可）
                all_combined_dfs = [train_combined_df]
                for dataset_name, (combined_df, _, _) in all_eval_results.items():
                    all_combined_dfs.append(combined_df)
                df_model_effect_all = pd.concat(all_combined_dfs, ignore_index=True)
                # ========== 汇总写入 Excel ==========
                with pd.ExcelWriter(final_report_path,engine='openpyxl') as writer:
                    df_quality.to_excel(writer, sheet_name="01_数据质量报告", index=False)
                    df_default_dist.to_excel(writer, sheet_name="02_违约率分布", index=False)
                    df_feat_merged.to_excel(writer, sheet_name="03_入模特征分析", index=False)
                    df_mono_var.to_excel(writer, sheet_name="04_单变量分箱结果", index=False)
                    df_model_effect_all.to_excel(writer, sheet_name="05_模型效果", index=False)
                    df_lift.to_excel(writer, sheet_name="06_Lift分箱", index=False)
                    
                    # 创建分数转换参数说明DataFrame
                    score_params_df = pd.DataFrame([
                        ['分数转换参数说明', ''],
                        ['基准分数点(p0)', p0],
                        ['分数翻倍所需的好样本比例(pdo)', pdo],
                        ['基准好坏比(theta)', theta],
                        ['分数切点', str(cut_points)],
                        ['', ''],
                        ['分数转换结果', '']
                    ])
                    score_params_df.to_excel(writer, sheet_name="07_分数转换", index=False, header=False)
                    # 获取当前sheet的writer对象
                    workbook = writer.book
                    worksheet = writer.sheets["07_分数转换"]
                    # 找到下一个空行
                    startrow = score_params_df.shape[0]
                    # 写df_score，保留表头
                    df_score.to_excel(writer, sheet_name="07_分数转换", index=False, header=True, startrow=startrow)      
                print('步骤8/9: 模型报告生成完成')
                
                # 步骤9/9: 自动生成并执行串行验证（仅适用于传统LightGBM模型）
                if not is_autogluon:
                    print('步骤9/9: 【常规模型专属】开始自动生成并执行串行验证...')
                    try:
                        # 生成100条预测csv、5条json和特征csv（常规模型专属）
                        TEST_PRED_100_CSV = os.path.join(dst_data, 'test_pred_100.csv')
                        TEST_PRED_5_JSON = os.path.join(dst_data, 'test_pred_5.json')
                        USE_FEA_CSV = os.path.join(dst_data, 'use_fea_list.csv')
                        # 1. 取df_eval前100行，按mod_var和prob_col输出
                        df_eval_100 = df_eval.head(100).copy()
                        cols = mod_var + [prob_col]
                        df_eval_100[cols].to_csv(TEST_PRED_100_CSV, index=False)
                        # 风控字段映射只读取刚生成的样本和最终入模特征，不会重新预测。
                        # 映射文件缺失或内容不完整时仅输出提示，不能影响后续串行验证。
                        try:
                            from .utils_model_feature_mapping import generate_risk_variable_mapping

                            generate_risk_variable_mapping(
                                model_features=mod_var,
                                retrain_data_dir=dst_data,
                                test_pred_path=TEST_PRED_100_CSV,
                                prediction_column=prob_col,
                            )
                        except Exception as mapping_error:
                            print(f'风控变量映射提示：生成映射产物失败，已跳过：{mapping_error}')
                        # 2. 随机抽5行保存为json
                        sample_df = df_eval_100[cols].sample(n=5, random_state=42)
                        sample_json = sample_df.to_dict(orient='records')
                        with open(TEST_PRED_5_JSON, 'w', encoding='utf-8') as f:
                            json.dump(sample_json, f, ensure_ascii=False, indent=2)
                        # 3. mod_var转csv
                        use_fea_df = pd.DataFrame({'Feature': mod_var})
                        use_fea_df.to_csv(USE_FEA_CSV, index=False)
                        print(f'已生成用于部署验证的：100条预测csv、5条json和特征csv，路径可在代码中自定义。')
                        
                        from .retrain_code_generator import auto_generate_and_execute_retrain
                        with open('outputs/model_train/outputs/model_params.json', 'r', encoding='utf-8') as f:
                            model_params = json.load(f)
                        # 使用字符串拼接（正斜杠），避免在生成的代码中被误解析为转义字符
                        dev_data_path = f"{dst_data}/df_dev_retrain.csv"
                        oot_data_path = f"{dst_data}/df_oot_retrain.csv"
                        psi_data_path = f"{dst_data}/df_psi_retrain.csv"
                        output_path = f"{dst_data}/retrain_validationRes.xlsx"
                        success = auto_generate_and_execute_retrain(
                            dev_data_path=dev_data_path,
                            oot_data_path=oot_data_path,
                            psi_data_path=psi_data_path,
                            use_features=mod_var,
                            model_params=model_params,
                            p0=p0,
                            pdo=pdo,
                            theta=theta,
                            label=label,
                            output_path=output_path,
                            output_dir=dst_data
                        )
                        if success:
                            print("✓ 串行验证自动生成并执行成功!")
                        else:
                            print("✗ 串行验证自动生成并执行失败!")
                    except Exception as e:
                        print(f"自动生成并执行串行验证时发生错误: {str(e)}")
                        print(traceback.format_exc())
                else:
                    print('步骤9/9: 串行验证仅适用于传统LightGBM模型，已跳过')
            return True
        return False
    except:
        msg=traceback.format_exc()
        print(msg)
        return False
