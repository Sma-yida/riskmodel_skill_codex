import json
import os
import traceback
import warnings

import lightgbm as lgb
import optuna
from sklearn.linear_model import LogisticRegression

from utils.data_utils import get_data, identify_types
from .utils1 import ks_score
from .utils_model_train_autogluon import autogluon_model_comparison
from .utils_model_train_lightgbm import (
    save_model_pkl,
    save_model_pmml,
    train_lgb_with_optuna,
)

warnings.filterwarnings("ignore")


def _load_model_train_inputs(config, model_select):
    """
    入模特征优先级：
    1) 当前算法专属model_train_spec的use_features，存在且非空则使用
    2) config['data_path']['use_fea_path']（默认特征筛选输出）
    同时读取可选的 model_params；非空时由训练主流程覆盖基础参数并跳过 HPO。
    返回: (use_fea, feature_source, uploaded_model_params)
    """
    def _clean_feature_list(value, field_name):
        if not isinstance(value, list):
            raise ValueError(f"{field_name} 必须是字符串列表")
        invalid_values = [item for item in value if not isinstance(item, str) or not item.strip()]
        if invalid_values:
            raise ValueError(f"{field_name} 只能包含非空字符串")
        return list(dict.fromkeys(value.strip() for value in value))

    def _load_feature_list(path):
        if not path or (not os.path.exists(path)):
            return None
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if obj is None:
            return []
        if isinstance(obj, list):
            return _clean_feature_list(obj, path)
        # 兼容极少数场景：可能是 {"features":[...]}
        if isinstance(obj, dict) and isinstance(obj.get("features"), list):
            return _clean_feature_list(obj["features"], f"{path}中的features")
        raise ValueError(f"入模特征文件格式错误: {path}")

    # 复现配置放在工作目录的 data/ 下，按运行目录查找（原为按代码所在目录查找）
    project_root = os.getcwd()
    spec_names = {
        "lightgbm": "model_train_spec_lightgbm.json",
        "lrscorecard": "model_train_spec_lrscorecard.json",
    }
    spec_name = spec_names.get(model_select)
    model_train_spec_path = (
        os.path.join(project_root, "data", spec_name) if spec_name else None
    )
    uploaded_features = []
    uploaded_model_params = {}

    if model_train_spec_path and os.path.exists(model_train_spec_path):
        try:
            with open(model_train_spec_path, "r", encoding="utf-8") as f:
                model_train_spec = json.load(f)
        except Exception as exc:
            raise ValueError(f"读取 data/{spec_name} 失败: {exc}") from exc
        if not isinstance(model_train_spec, dict):
            raise ValueError(f"data/{spec_name}顶层必须是JSON对象")

        uploaded_features_value = model_train_spec.get("use_features", [])
        if uploaded_features_value is None:
            uploaded_features_value = []
        uploaded_features = _clean_feature_list(
            uploaded_features_value, f"data/{spec_name}中的use_features"
        )

        uploaded_model_params = model_train_spec.get("model_params", {})
        if uploaded_model_params is None:
            uploaded_model_params = {}
        if not isinstance(uploaded_model_params, dict):
            raise ValueError(f"data/{spec_name}中的model_params必须是JSON对象")

    if uploaded_features:
        return (
            uploaded_features,
            f"用户上传{spec_name}",
            uploaded_model_params,
        )

    # 组合文件不存在或未指定特征时，使用特征筛选结果。
    default_path = config["data_path"]["use_fea_path"]
    default_list = _load_feature_list(default_path)
    if default_list is None:
        raise FileNotFoundError(f"未找到特征筛选输出文件: {default_path}")
    if not default_list:
        raise ValueError(f"特征筛选输出的入模特征为空: {default_path}")
    return default_list, "特征筛选输出", uploaded_model_params
# LR模型超参数寻优
def train_lr_with_optuna(X_train, y_train, X_test, y_test, dic_p):
    n_trials = dic_p["n_trials"]
    ks_gap_threshold = dic_p['ks_gap_threshold']
    config_model_params = dic_p['model_param']  # 默认参数

    def objective(trial):
        C = trial.suggest_loguniform("C", 1e-3, 100)
        penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
        solver = "liblinear" if penalty == "l1" else "lbfgs"
        model = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=1000)

        model.fit(X_train, y_train)
        pred_train = model.predict_proba(X_train)[:, 1]
        pred_test = model.predict_proba(X_test)[:, 1]
        ks_train = ks_score(y_train, pred_train)
        ks_test = ks_score(y_test, pred_test)

        return ks_test if (abs(ks_train - ks_test) <= ks_gap_threshold and ks_train > ks_test) else 0

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_value = study.best_trial.value
    best_params = study.best_trial.params
    
    print("模型寻优结束！")
    if best_value == 0:
        print("未能找到满足条件的参数，使用默认参数为：", config_model_params)
        best_params = config_model_params
    else:
        best_params = {**config_model_params, **best_params}
        print("寻找的最佳LR模型结果KS值", best_value, "，参数为：", best_params)

    return best_params

# LightGBM + Optuna 模型超参数寻优

### Step 4: 保存 PMML 模型



        
#### 特征重要性计算


##模块6：模型训练
# output_dst/
# ├── model.pkl                  # 训练好的模型文件（pkl格式）
# ├── model.pmml                 # PMML格式模型（若有需求）
# ├── use_features.json         # 入模特征 list
# ├── best_params.json          # 最优参数
# ├── model_fea/                # 特征重要性分析输出目录（已有）
####---------------------主程序---------------------------------    
def model_train_func(dev_data, oot_data, config, oot_files=None):
    try:
        ##读取参数
        feature_dic_path = config['data_path']['feature_dic_path']
        model_select = config['settings'].get('model_select', 'LightGBM').lower()
        use_fea, use_fea_source, uploaded_model_params = _load_model_train_inputs(
            config, model_select
        )
        print(f"入模特征来源: {use_fea_source}，共 {len(use_fea)} 个特征")
        label_col = config['settings']['label']
        dst_model = config['output_path']['dst_model']
        dst_outs = config['output_path']['dst_outs']
        os.makedirs(dst_model, exist_ok=True)
        os.makedirs(dst_outs, exist_ok=True)
        pkl_model_path = config['output_path']['pkl_model_path']
        pmml_model_path = config['output_path']['pmml_model_path']
        feature_importance_path = config['output_path']['feature_importance_path']
        model_params_path = config['output_path']['model_params_path']
        model_fea_path = config['output_path']['model_fea_path']
        model_json_path = config['output_path']['model_json_path']
        optuna_trials_csv_path = os.path.join(dst_outs, 'optuna_trials.csv')
        model_train_spec_output_path = os.path.join(
            dst_outs, 'model_train_spec_lightgbm.json'
        )

        # ====== 执行模式控制（只用 model_select 控制） ======

        # 评分卡分支在任何LightGBM专用类型转换和输出写入之前执行。
        # 惰性导入保证未选择LrScorecard时不加载scorecardpipeline及其依赖。
        if model_select == 'lrscorecard':
            from .utils_model_train_scorecard import train_scorecard_model

            train_scorecard_model(
                dev_data=dev_data,
                test_data=oot_data,
                config=config,
                use_features=use_fea,
                feature_source=use_fea_source,
                uploaded_model_params=uploaded_model_params,
                oot_files=oot_files,
            )
            return True

        ## 区分数值型特征和类别型特征
        num_fea, cat_fea = identify_types(dev_data[use_fea])
        for i in cat_fea:
            dev_data[i] = dev_data[i].astype('category')
            oot_data[i] = oot_data[i].astype('category')

        # 加载OOT数据集（从传入的oot_files）
        oot_datasets = {}
        if oot_files:
            for oot_file in oot_files:
                dataset_name = os.path.splitext(os.path.basename(oot_file))[0]  # 例如: oot_main, oot_1
                try:
                    oot_df = get_data(oot_file)
                    # 确保列名一致
                    for i in cat_fea:
                        if i in oot_df.columns:
                            oot_df[i] = oot_df[i].astype('category')
                    X_oot = oot_df[use_fea]
                    y_oot = oot_df[label_col]
                    oot_datasets[dataset_name] = (X_oot, y_oot)
                except Exception as e:
                    print(f"  - {dataset_name}: 加载失败 - {str(e)}")
                    continue

        if model_select == 'lightgbm':
            # ====== LightGBM模型训练 ======
            print("\n>>> LightGBM模型训练开始 ...")
            X_train = dev_data[use_fea]
            y_train = dev_data[label_col]
            
            # LightGBM早停固定使用test；优化目标由HPO配置决定。
            X_test = oot_data[use_fea]
            y_test = oot_data[label_col]
            early_stopping_dataset_name = "测试集(test)"
            if 'oot_main' in oot_datasets:
                X_final_eval, y_final_eval = oot_datasets['oot_main']
                eval_dataset_name = "验证集(oot_main)"
            else:
                X_final_eval, y_final_eval = X_test, y_test
                eval_dataset_name = early_stopping_dataset_name
            
            print(f"LightGBM早停数据集: {early_stopping_dataset_name}")
            print(f"最终模型评估数据集: {eval_dataset_name}")
            
            ## 模型寻优：【参数配置：optuna寻优|使用默认参数】
            dic_p = config['settings'].copy()
            dic_p['optuna_trials_csv_path'] = optuna_trials_csv_path
            dic_p['oot_datasets'] = oot_datasets  # 传递OOT数据集字典（用于寻优时的OOT评估）
            if uploaded_model_params:
                best_params = {
                    **config['settings']['model_param'],
                    **uploaded_model_params,
                }
                is_optuna = "使用用户上传model_train_spec_lightgbm.json中的模型参数"
                optimization_info = None
                model_params_source = "用户上传model_train_spec_lightgbm.json"
                print("检测到用户上传的模型参数，跳过Optuna超参数寻优")
                print(f"用户指定并合并后的模型参数为: {best_params}")
            else:
                best_params, is_optuna, optimization_info = train_lgb_with_optuna(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    cat_fea=cat_fea,
                    dic_p=dic_p
                )
                model_params_source = (
                    "Optuna自动寻优"
                    if optimization_info is not None
                    else "模型训练配置"
                )
            # 保存模型最优参数
            with open(model_params_path, 'w', encoding='utf8') as f:
                json.dump(best_params, f, indent=4, ensure_ascii=False)
                
            ## 使用调参构造模型
            model_lgb = lgb.LGBMClassifier(**best_params)
            print(f"入模特征数量: {len(use_fea)}")
            print(f"其中数值型特征数量: {len(num_fea)}")
            print(f"其中类别型特征数量: {len(cat_fea)}，具体为：{cat_fea}")

            # === Step 5.1. 评估模型 + 导出PKL + 计算特征重要性 ===
            extra_eval_datasets = {}
            if 'oot_main' in oot_datasets:
                extra_eval_datasets["测试集(test)"] = (X_test, y_test)
            model_metrics, feature_imp = save_model_pkl(
                model_lgb, X_train, y_train, X_final_eval, y_final_eval,
                use_fea, pkl_model_path, cat_fea, eval_dataset_name,
                feature_dic_path, feature_importance_path,
                extra_eval_datasets=extra_eval_datasets
            )
            
            # === Step 5.2. 评估模型 + 导出PMML ===
            save_model_pmml(model_lgb, X_train, y_train, X_final_eval, y_final_eval, use_fea, pmml_model_path, cat_fea, eval_dataset_name)
            
            # 保存最终的入模特征列表
            with open(model_fea_path, 'w', encoding='utf8') as f:
                json.dump(use_fea, f, indent=4, ensure_ascii=False)

            # 保存可直接用于下一次训练的组合复现配置。
            with open(model_train_spec_output_path, 'w', encoding='utf8') as f:
                json.dump(
                    {
                        "use_features": use_fea,
                        "model_params": best_params,
                    },
                    f,
                    indent=4,
                    ensure_ascii=False,
                )
                
            # 生成模型训练JSON报告
            from utils.generate_json_reports import generate_model_train_json
            generate_model_train_json(
                model_metrics=model_metrics,
                best_params=best_params,
                feature_imp=feature_imp,
                output_path=model_json_path,
                is_optuna=is_optuna,
                use_fea=use_fea,
                use_fea_source=use_fea_source,
                model_params_source=model_params_source,
                optimization_info=optimization_info,
            )
            print(">>> LightGBM模型训练完成！\n")
        elif model_select == 'autogluon':
            # ====== AutoGluon多模型对比 ======
            print("\n>>> AutoGluon多模型对比开始 ...")
            autogluon_supported = ['CAT', 'GBM','XGB','NN_TORCH','RF','XT','KNN','FASTAI']
            save_predictor = config['settings'].get('save_autogluon_predictor', True)# 是否保存模型由config控制，默认True
            time_limit = config['settings'].get('autogluon_time_limit', 60*5)
            output_path = os.path.join(dst_outs, 'autogluon_leaderboard.csv')
            predictor_path = os.path.join(dst_model, 'autogluon_predictor')
            os.makedirs(predictor_path, exist_ok=True)
            autogluon_log_path = os.path.join(dst_outs, 'autogluon_log.txt')
            leaderboard = autogluon_model_comparison(
                train_df=dev_data[use_fea+[label_col]],
                test_df=oot_data[use_fea+[label_col]],
                target_col=label_col,
                include_model_types=autogluon_supported,
                time_limit=time_limit,
                eval_metric='Ks',
                output_path=output_path,
                predictor_path=predictor_path,
                log_path=autogluon_log_path,
                save_predictor=save_predictor
            )
            
            # 保存最终的入模特征列表（与LightGBM保持一致）
            with open(model_fea_path, 'w', encoding='utf8') as f:
                json.dump(use_fea, f, indent=4, ensure_ascii=False)
            
            # 保存为json用于页面展示，包含对比模型列表
            from utils.generate_json_reports import generate_autogluon_train_json
            generate_autogluon_train_json(
                leaderboard=leaderboard,
                output_path=model_json_path,
                autogluon_supported=autogluon_supported,
                use_fea=use_fea,
                use_fea_source=use_fea_source,
            )
            print(">>> AutoGluon多模型对比完成！\n")
        else:
            print(f"错误: model_select 参数只能为 'LightGBM'、'AutoGluon' 或 'LrScorecard'，当前为: {model_select}")
            return False
        return True   
    except Exception:
        msg = traceback.format_exc()
        print(msg)
        return False
