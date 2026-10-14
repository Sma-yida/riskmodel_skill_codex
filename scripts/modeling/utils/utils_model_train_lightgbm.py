import json
import os
import pickle
from copy import deepcopy

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn2pmml import PMMLPipeline, sklearn2pmml
from sklearn2pmml.decoration import CategoricalDomain
from sklearn2pmml.preprocessing import PMMLLabelEncoder
from sklearn_pandas import DataFrameMapper

from .utils1 import (
    auc_score,
    compute_psi,
    decile_analysis,
    evaluate_on_dataset,
    ks_metric,
    ks_score,
    monotonicity_violation,
)


OPTIMIZATION_METRIC_DIRECTIONS = {
    "ks": "maximize",
    "lift_tail10": "maximize",
    "lift_head10": "minimize",
}

TRIAL_METRIC_NAMES = (
    "ks", "auc", "ks_gap", "psi",
    "lift_tail10", "lift_head10", "lift_head1", "lift_head2", "lift_head5",
    "bins_df_json", "bad_rates",
    "mono_violation", "mono_jump_count", "overall_bad_rate",
)


def resolve_optimization_targets(dic_p, available_datasets):
    """根据 single/multi 解析 Optuna 优化目标。"""
    optimization_mode = dic_p["optimization_mode"]
    resolved_targets = dic_p.get("optimization_targets") or []

    if not resolved_targets:
        if optimization_mode == "single":
            resolved_targets = [{"dataset": "test", "metric": "ks"}]
        else:
            resolved_targets = [
                {"dataset": "test", "metric": "ks"},
                {"dataset": "test", "metric": "lift_tail10"},
                {"dataset": "test", "metric": "lift_head10"},
            ]

    available_datasets = set(available_datasets)
    try:
        target_keys = [
            (target["dataset"], target["metric"])
            for target in resolved_targets
        ]
        valid_count = (
            len(target_keys) == 1
            if optimization_mode == "single"
            else 2 <= len(target_keys) <= 3
        )
        is_valid = (
            valid_count
            and len(target_keys) == len(set(target_keys))
            and all(dataset in available_datasets for dataset, _ in target_keys)
            and all(metric in OPTIMIZATION_METRIC_DIRECTIONS for _, metric in target_keys)
        )
    except (KeyError, TypeError):
        is_valid = False

    if not is_valid:
        raise ValueError(
            "optimization_targets 配置错误：single 需填写1个目标，multi需填写2～3个不重复目标；"
            f"dataset 可选 {sorted(available_datasets)}，metric 可选 "
            f"{list(OPTIMIZATION_METRIC_DIRECTIONS)}"
        )

    resolved_targets = [
        {"dataset": dataset, "metric": metric}
        for dataset, metric in target_keys
    ]
    directions = [
        OPTIMIZATION_METRIC_DIRECTIONS[target["metric"]]
        for target in resolved_targets
    ]
    mode_text = "单目标" if optimization_mode == "single" else "多目标"
    target_text = "、".join(
        f"{target['dataset']}.{target['metric']}({direction})"
        for target, direction in zip(resolved_targets, directions)
    )
    print(f"Optuna寻优方式：{mode_text}；优化目标：{target_text}")
    return optimization_mode, resolved_targets, directions


def _metric_value(trial, key, default):
    value = trial.user_attrs.get(key, default)
    if value is None:
        return default
    try:
        if not isinstance(value, (list, tuple, dict)) and pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    return value


def _all_conditions_ok(
    trial,
    dataset_name,
    ks_train,
    ks_gap_threshold,
    psi_threshold_max,
):
    ks_value = _metric_value(trial, f"ks_{dataset_name}", 0.0)
    ks_gap_value = _metric_value(
        trial, f"ks_gap_{dataset_name}", float("inf")
    )
    mono_value = _metric_value(
        trial, f"mono_violation_{dataset_name}", float("inf")
    )
    psi_value = _metric_value(trial, f"psi_{dataset_name}", float("inf"))
    return (
        ks_gap_value <= ks_gap_threshold
        and mono_value <= 0
        and ks_train > ks_value
        and psi_value <= psi_threshold_max
    )


def _export_trials_to_csv(
    trials,
    csv_path,
    pareto_numbers,
    selection_dataset,
    oot_dataset_names,
    ks_gap_threshold,
    psi_threshold_max,
):
    """从 trial.user_attrs 导出 CSV，并返回相同 records 供后续选解。"""
    if not trials:
        return []

    pareto_numbers = pareto_numbers or set()
    records = []
    for trial in trials:
        ks_train = _metric_value(trial, "ks_train", 0.0)
        all_conditions_ok = _all_conditions_ok(
            trial,
            selection_dataset,
            ks_train,
            ks_gap_threshold,
            psi_threshold_max,
        )
        record = {
            "trial_number": trial.number,
            "state": getattr(trial.state, "name", str(trial.state)),
            "objective_values": json.dumps(
                list(trial.values) if trial.values is not None else None,
                ensure_ascii=False,
            ),
            "ks_train": ks_train,
            "auc_train": trial.user_attrs.get("auc_train", 0.0),
            "params_json": json.dumps(
                trial.user_attrs.get("params", {}),
                ensure_ascii=False,
            ),
            "all_conditions_ok": int(all_conditions_ok),
            "is_pareto_front": int(trial.number in pareto_numbers),
            "selection_dataset": selection_dataset,
        }
        for metric in TRIAL_METRIC_NAMES:
            key = f"{metric}_test"
            record[key] = trial.user_attrs.get(key)

        for dataset_name in oot_dataset_names:
            for metric in TRIAL_METRIC_NAMES:
                key = f"{metric}_{dataset_name}"
                record[key] = trial.user_attrs.get(key)

        records.append(record)

    if csv_path and records:
        output_dir = os.path.dirname(csv_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        pd.DataFrame(records).to_csv(
            csv_path,
            index=False,
            encoding="utf-8-sig",
        )
        print(f"已保存 {len(records)} 条Optuna可行解到 {csv_path}")

    return records


def _select_best_trial(pareto_records, all_records, selection_dataset):
    """保持原有三级降级规则，按选解数据集 KS 选择 trial。"""
    ks_key = f"ks_{selection_dataset}"
    pareto_valid = [
        record for record in pareto_records if record["all_conditions_ok"]
    ]
    all_valid = [
        record for record in all_records if record["all_conditions_ok"]
    ]

    def record_selection_ks(record):
        value = record.get(ks_key, 0.0)
        if value is None:
            return float("-inf")
        try:
            if pd.isna(value):
                return float("-inf")
        except (TypeError, ValueError):
            pass
        return value

    if pareto_valid:
        best_record = max(pareto_valid, key=record_selection_ks)
        print(
            f"从 {len(pareto_valid)} 个满足条件的Pareto front解中选择最佳解"
            f"（{ks_key}最高）"
        )
        is_optuna = f"Optuna自动寻优：Pareto满足条件，按{ks_key}最高选择"
    elif all_valid:
        best_record = max(all_valid, key=record_selection_ks)
        print(
            f"Pareto front无满足条件解，从 {len(all_valid)} 个全量满足条件"
            f"trial中选择{ks_key}最高的解"
        )
        is_optuna = f"Optuna自动寻优：全量满足条件，按{ks_key}最高选择"
    else:
        best_record = max(all_records, key=record_selection_ks)
        print(f"未找到满足条件的trial，退化为从全量trials中选择{ks_key}最高的解")
        is_optuna = f"Optuna自动寻优：无满足条件trial，兜底按{ks_key}最高选择"

    return best_record, is_optuna


def train_lgb_with_optuna(X_train, y_train, X_test, y_test, cat_fea, dic_p):
    IF_RUN = dic_p['RUN_HPO']
    config_model_params = dic_p['model_param']
    optuna_trials_csv_path = dic_p.get('optuna_trials_csv_path')
    # 1. 如果RUN_HPO为True，则进行模型超参数寻优
    if  IF_RUN:
        print("=" * 60)
        print("开始模型超参数寻优 (Optuna)")
        n_trials=dic_p["n_trials"]
        ks_gap_threshold = dic_p['ks_gap_threshold']
        psi_threshold_max = 0.2 
        print(f"   - 寻优次数: {n_trials}")
        print(f"   - KS差值阈值: {ks_gap_threshold:.3f}")
        print(f"   - PSI最大阈值: {psi_threshold_max:.3f}")
        print(f"   - 训练集样本: {len(X_train)}")
        print(f"   - 测试集样本: {len(X_test)}")

        # 1.5. 解析优化目标；空列表时保持原有 single/multi 默认行为
        oot_datasets = dic_p.get('oot_datasets', {})
        available_datasets = {'test', *oot_datasets.keys()}
        optimization_mode, optimization_targets, optimization_directions = resolve_optimization_targets(
            dic_p, available_datasets
        )
        # 2. 定义约束函数
        def constraints(trial: optuna.Trial):
            ks_gap = trial.user_attrs["ks_gap_test"]
            mono_violation = trial.user_attrs["mono_violation_test"]          
            # 约束1：KS gap <= ks_gap_threshold
            g1 = ks_gap - ks_gap_threshold         
            # 约束2：单调性违背程度 <= 0（完全单调则 mono_violation=0，否则>0）
            g2 = mono_violation  # 已经是 >=0 的量，要求 <=0           
            return (g1, g2)

        def suggest_int_or_fixed(trial, param_name, bounds):
            """从整数范围取值；上下界相同表示固定参数，不创建 Optuna 维度。"""
            low, high, *extra = bounds
            if low == high:
                return int(low)
            if extra:
                return trial.suggest_int(param_name, low, high, step=extra[0])
            return trial.suggest_int(param_name, low, high)

        def suggest_float_or_fixed(trial, param_name, bounds):
            """从浮点范围取值；上下界相同表示固定参数，不创建 Optuna 维度。"""
            low, high, *extra = bounds
            if low == high:
                return float(low)
            if extra:
                return trial.suggest_float(param_name, low, high, step=extra[0])
            return trial.suggest_float(param_name, low, high)

        # 3. 定义目标函数
        def objective(trial):
            ## 获取当前试验的参数
            search_space = dic_p['model_selection_params']
            param_grid = {
                ####  补充参数：min_split_gain
                "min_split_gain": suggest_int_or_fixed(trial, "min_split_gain", search_space["min_split_gain"]), # 最小分割增益
                #### 大模型可调参数如下
                "learning_rate": suggest_float_or_fixed(trial, "learning_rate", search_space["learning_rate"]), # 学习率
                "num_leaves": suggest_int_or_fixed(trial, "num_leaves", search_space["num_leaves"]), # 叶子节点数
                "max_bin": suggest_int_or_fixed(trial, "max_bin", search_space["max_bin"]), # 最大bin数
                "min_child_samples": suggest_int_or_fixed(trial, "min_child_samples", search_space["min_child_samples"]), # 叶节点最少样本数
                "reg_alpha": suggest_int_or_fixed(trial, "reg_alpha", search_space["reg_alpha"]), # 正则化系数
                "reg_lambda": suggest_int_or_fixed(trial, "reg_lambda", search_space["reg_lambda"]), # 正则化系数
                "subsample": suggest_float_or_fixed(trial, "subsample", search_space["subsample"]), # 子采样比例
                "colsample_bytree": suggest_float_or_fixed(trial, "colsample_bytree", search_space["colsample_bytree"]), # 列采样比例
                "subsample_freq": suggest_int_or_fixed(trial, "subsample_freq", search_space["subsample_freq"]), # 子采样频率
                "min_child_weight": suggest_float_or_fixed(trial, "min_child_weight", search_space["min_child_weight"]), # 最小子权重
                "n_estimators": trial.suggest_categorical("n_estimators", search_space["n_estimators"]), # 迭代次数
                "max_depth": suggest_int_or_fixed(trial, "max_depth", search_space["max_depth"]), # 最大深度
                "scale_pos_weight": suggest_float_or_fixed(trial, "scale_pos_weight", search_space["scale_pos_weight"]), # 正样本权重
            }
            ## 训练模型
            print(f"============== Trial #{trial.number} ==============")
            model_params = config_model_params.copy()
            model_params.update(param_grid)
            model = lgb.LGBMClassifier(**model_params)
            # 使用ks早停和ks评估，来增加ks的优化权重
            eval_result = {}
            model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], eval_metric=ks_metric, categorical_feature=cat_fea,
                     callbacks=[lgb.early_stopping(50),lgb.log_evaluation(0),lgb.record_evaluation(eval_result)])          
            
            ## 通过日志寻找最佳的ks所在的真实迭代轮次
            best_score = model.best_score_['valid_1']['ks']   #最佳迭代的ks
        
            # 通过日志寻找最佳的ks所在的真实迭代轮次（全量）
            # 获取所有指标列表
            train_loss_list = eval_result['training']['binary_logloss']   # 训练集的 loss
            train_ks_list = eval_result['training']['ks']   # 训练集的 ks
            valid_loss_list = eval_result['valid_1']['binary_logloss']   # 验证集的 loss
            valid_ks_list = eval_result['valid_1']['ks']   # 验证集的 ks

            # 去掉连续重复，按照 training's binary_logloss 进行去重
            # 保留"真正变化的一步一步的历史
            kept = []
            removed = []
            prev_train_loss = None
            new_iter = 0         # 去重后的新迭代轮次，从 1 开始计

            for i in range(len(train_loss_list)):
                log_iter = i + 1   # 日志里轮次从 1 开始
                train_loss = train_loss_list[i]
                train_ks = train_ks_list[i]
                valid_loss = valid_loss_list[i]
                valid_ks = valid_ks_list[i]
                       
                # first round or different training loss
                tolerance = 1e-12   # 阈值
                if prev_train_loss is None or abs(train_loss - prev_train_loss) > tolerance:
                    new_iter += 1
                    kept.append((log_iter, new_iter, train_loss, train_ks, valid_loss, valid_ks))
                else:
                    # 重复（training loss 相同）
                    removed.append((log_iter, train_loss, train_ks, valid_loss, valid_ks))
                    # print(f"重复删除: {log_iter}, train_loss={train_loss}, train_ks={train_ks}, valid_loss={valid_loss}, valid_ks={valid_ks}")
                # 更新 prev
                prev_train_loss = train_loss
                # 到最佳 ks 就停止
                if valid_ks == best_score:
                    break

            best_iteration = kept[-1][1]
            # 使用 booster 的 predict 方法，指定最佳轮次
            pred_train = model.booster_.predict(X_train, num_iteration=best_iteration)
            pred_test = model.booster_.predict(X_test, num_iteration=best_iteration)
            ## 计算各种指标，包括KS指标、head/tail 10% lift + decile 坏人占比 以及单调性违背程度 
            ks_train = ks_score(y_train, pred_train)
            # 评估test数据集（标准化命名：{指标}_test）
            ks_test = ks_score(y_test, pred_test) 
            auc_train = auc_score(y_train, pred_train)
            auc_test = auc_score(y_test, pred_test)
            ks_gap_test = ks_train - ks_test
            psi_value = compute_psi(pred_train, pred_test)
            df_lift, bad_rates, lift_tail10_test, lift_head10_test, overall_bad_rate_test, lift_head1_test, lift_head2_test, lift_head5_test, bins_df_json_test = decile_analysis(y_test, pred_test, n_bins=10)
            mono_violation_test, mono_jump_count_test = monotonicity_violation(bad_rates)

            # 评估所有OOT数据集（如果存在）
            for dataset_name, (X_oot, y_oot) in oot_datasets.items():
                # 使用相同的最佳迭代轮次预测
                pred_oot = model.booster_.predict(X_oot, num_iteration=best_iteration)
                
                # 计算PSI（train vs oot）
                psi_oot = compute_psi(pred_train, pred_oot)
                trial.set_user_attr(f"psi_{dataset_name}", psi_oot)
                
                # 计算指标（传入ks_train而不是重新计算）
                metrics = evaluate_on_dataset(y_oot, pred_oot, ks_train)
                
                # 保存到trial属性（命名格式：{指标}_{数据集名称}）
                for metric_name, metric_value in metrics.items():
                    trial.set_user_attr(f"{metric_name}_{dataset_name}", metric_value)

            # ⑤ 打印最关键的调试信息
            if best_score == ks_test:
                print("KS是否一致？一致")
            else:
                # 红色高亮警告
                print(f"重大警告：去重后的 KS（{ks_test}）与 best_score（{best_score}）仍旧不一致！")

            ## 设置trial的属性-⑥ 把所有中间结果存到 user_attrs（标准化命名：{指标}_{数据集}）
            param_grid_final = param_grid.copy()
            param_grid_final["n_estimators"] = model.best_iteration_  # 将 param_grid 中的 n_estimators 替换为实际的 best_iteration
            trial.set_user_attr("params", param_grid_final)
            
            # train数据集指标
            trial.set_user_attr("ks_train", ks_train)
            trial.set_user_attr("auc_train", auc_train)
            
            # test数据集指标（标准化命名）
            trial.set_user_attr("ks_test", ks_test)
            trial.set_user_attr("auc_test", auc_test)
            trial.set_user_attr("ks_gap_test", ks_gap_test)
            trial.set_user_attr("psi_test", psi_value)
            trial.set_user_attr("lift_tail10_test", lift_tail10_test)
            trial.set_user_attr("lift_head10_test", lift_head10_test)
            trial.set_user_attr("lift_head1_test", lift_head1_test)
            trial.set_user_attr("lift_head2_test", lift_head2_test)
            trial.set_user_attr("lift_head5_test", lift_head5_test)
            trial.set_user_attr("overall_bad_rate_test", overall_bad_rate_test)
            trial.set_user_attr("bad_rates_test", bad_rates)
            trial.set_user_attr("bins_df_json_test", bins_df_json_test)
            trial.set_user_attr("mono_violation_test", mono_violation_test)
            trial.set_user_attr("mono_jump_count_test", mono_jump_count_test)
            
            # test数据集PSI指标（标准化命名）
            
            objective_values = [
                trial.user_attrs.get(f"{target['metric']}_{target['dataset']}")
                for target in optimization_targets
            ]
            invalid_objective = any(
                value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value))
                for value in objective_values
            )

            # 分箱失败或目标值无效时，按每个目标方向返回最差值
            if np.isnan(lift_tail10_test) or invalid_objective:
                print("警告: 模型退化（预测值无差异），跳过此trial")
                objective_values = [
                    0.0 if direction == "maximize" else float("inf")
                    for direction in optimization_directions
                ]

            if optimization_mode == 'single':
                return objective_values[0]
            return tuple(objective_values)

        # 4. 确定调参数据集、选解数据集，并导出CSV
        selection_dataset = 'oot_main' if 'oot_main' in oot_datasets else 'test'
        print(f"最终trial选解数据集: {selection_dataset}")

        # # 单目标
        # objective_with_params = partial(objective)
        # study = optuna.create_study(direction="maximize")
        # study.optimize(objective_with_params, n_trials=n_trials)

        # 6. 根据实际目标及方向创建study
        sampler = optuna.samplers.NSGAIISampler(constraints_func=constraints) 
        study = optuna.create_study(
            directions=optimization_directions,
            sampler=sampler
        )
        study.optimize(objective, n_trials=n_trials)  # n_jobs 可并行
        
        ## 优化结果分析
        print("\n" + "=" * 60)
        print(f"{optimization_mode}模式优化结果分析")
        print("=" * 60)
        
        # 获取Pareto front中的解（多目标优化返回的是best_trials列表）
        pareto_trials = study.best_trials
        pareto_trial_numbers = {trial.number for trial in pareto_trials}
        print(f"Pareto front 解数量: {len(pareto_trials)}")
        
        # 导出所有完成的trial到CSV，并同时获取records用于后续选择最佳解
        all_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        all_records = _export_trials_to_csv(
            all_trials,
            optuna_trials_csv_path,
            pareto_trial_numbers,
            selection_dataset,
            oot_datasets.keys(),
            ks_gap_threshold,
            psi_threshold_max,
        )
        if len(all_records) == 0:
            raise ValueError("没有完成的trials，无法选择最佳解")
        
        # 筛选出pareto解用于后续选择
        pareto_records = [record for record in all_records if record["is_pareto_front"]]
        
        # 7. 选择最佳解
        # 1）Pareto front中all_conditions_ok=True，选selection_dataset上KS最高
        # 2）全量trials中all_conditions_ok=True，选selection_dataset上KS最高
        # 3）全量trials中不管条件，选selection_dataset上KS最高
        ks_key = f"ks_{selection_dataset}"
        best_record, is_optuna = _select_best_trial(
            pareto_records,
            all_records,
            selection_dataset,
        )
        
        # 8. 提取最佳解的参数和指标（全部从best_record获取）
        best_params = json.loads(best_record["params_json"])  # 从params_json恢复params字典
        best_params = {**config_model_params, **best_params}
        
        # 获取所有指标（全部从best_record获取）
        ks_train = best_record["ks_train"]
        ks_test = best_record["ks_test"]
        ks_eval = best_record[ks_key]
        auc_eval = best_record[f"auc_{selection_dataset}"]
        ks_gap_eval = best_record[f"ks_gap_{selection_dataset}"]
        psi_eval = best_record[f"psi_{selection_dataset}"]
        lift_tail10_eval = best_record[f"lift_tail10_{selection_dataset}"]
        lift_head10_eval = best_record[f"lift_head10_{selection_dataset}"]
        lift_head1_eval = best_record[f"lift_head1_{selection_dataset}"]
        lift_head2_eval = best_record[f"lift_head2_{selection_dataset}"]
        lift_head5_eval = best_record[f"lift_head5_{selection_dataset}"]
        mono_violation_eval = best_record[f"mono_violation_{selection_dataset}"]
        gap_ok = ks_gap_eval <= ks_gap_threshold
        mono_ok = mono_violation_eval <= 0
        order_ok = ks_train > ks_eval
        psi_ok = psi_eval <= psi_threshold_max
        
        # 9. 输出详细结果
        mode_str = "多目标优化" if optimization_mode == 'multi' else "单目标优化"
        print(f"模型optuna{mode_str}完成! (状态: {is_optuna})")
        print("\n"+"-" * 60)       
        print(f"选解数据集: {selection_dataset}，推荐的模型指标和参数")
        print("  数据集                 KS        AUC       KS_Gap    PSI       Mono      条件状态")
        print(f"  训练集(train)        {ks_train:.6f}  {best_record['auc_train']:.6f}  {'-':>8}  {'-':>8}  {'-':>8}  -")
        print(f"  测试集(test)         {ks_test:.6f}  {best_record['auc_test']:.6f}  {best_record['ks_gap_test']:.6f}  {best_record['psi_test']:.6f}  {best_record['mono_violation_test']:.6f}  调参集")
        print(f"  {selection_dataset:<18} {ks_eval:.6f}  {auc_eval:.6f}  {ks_gap_eval:.6f}  {psi_eval:.6f}  {mono_violation_eval:.6f}  {'通过' if best_record['all_conditions_ok'] else '未通过'}")
        print(f"  Lift({selection_dataset}): tail10={lift_tail10_eval:.4f}, head10={lift_head10_eval:.4f}, head5={lift_head5_eval:.4f}, head2={lift_head2_eval:.4f}, head1={lift_head1_eval:.4f}")
        print(f"  选解条件: KS_Gap<={ks_gap_threshold}({'OK' if gap_ok else 'FAIL'}), Mono<=0({'OK' if mono_ok else 'FAIL'}), KS_train>KS_{selection_dataset}({'OK' if order_ok else 'FAIL'}), PSI<={psi_threshold_max}({'OK' if psi_ok else 'FAIL'})")
        print("最终选择的模型参数",best_params)
        print("-" * 60)
        optimization_info = {
            "optimization_mode": optimization_mode,
            "n_trials": n_trials,
            "optimization_targets": [
                {
                    "dataset": target["dataset"],
                    "metric": target["metric"],
                    "direction": direction,
                }
                for target, direction in zip(optimization_targets, optimization_directions)
            ],
            "selection_dataset": selection_dataset,
        }
    else:
        print("=" * 60)
        print("跳过超参数寻优，使用默认参数")  
        best_params = config_model_params  # 返回默认参数
        print(f"默认参数为: {best_params}")
        is_optuna = "使用默认参数"
        optimization_info = None
        print("=" * 60)

    return best_params, is_optuna, optimization_info

def save_model_pmml(model_lgb, X_train, y_train, X_test, y_test, use_fea, model_save_path, cat_fea=[], eval_dataset_name="测试集"):
    """
    训练 LightGBM 模型、评估指标、保存PMML模型。
    
    参数：
    - X_train, y_train: 训练集特征和标签
    - X_test, y_test: 测试集或验证集特征和标签
    - use_fea: 使用的特征列表
    - model_save_path: PMML模型保存路径
    - cat_fea: 类别型特征列表
    - eval_dataset_name: 评估数据集名称（用于输出显示）
    返回：
    - model_metrics: 包含训练集和评估集评估指标的字典
    """
    X_train = X_train[use_fea]
    X_test = X_test[use_fea] 
    print(f"类别型特征: {cat_fea}")

    tmp_mapper = []
    cat_indices = []
    for index,i in enumerate(X_train.columns.tolist()):
        if i  in cat_fea:
            tmp_mapper.append((i,[CategoricalDomain(invalid_value_treatment='as_missing'),PMMLLabelEncoder()]))
            cat_indices.append(index)
            # 新增 V260421同步替换测试集中训练集未见过的类别值为nan，避免PMMLLabelEncoder报错
            known_vals = set(X_train[i].dropna().unique())
            unknown_mask = ~X_test[i].isin(known_vals) & X_test[i].notna()
            if unknown_mask.sum() > 0:
                print(f"  {i}: 替换 {unknown_mask.sum()} 个未知类别为nan")
                X_test.loc[unknown_mask, i] = np.nan
            # END
        else:
            tmp_mapper.append((i,None))
    
    mapper = DataFrameMapper(tmp_mapper)
    pipeline = PMMLPipeline([("mapper", mapper), ("classifier", deepcopy(model_lgb))])
    # 传递类别特征索引，让LightGBM使用原生类别特征处理方式
    pipeline.fit(X_train, y_train, classifier__categorical_feature=cat_indices)

    # mapper = DataFrameMapper([(f, None) for f in use_fea])
    # pipeline = PMMLPipeline([("mapper", mapper), ("classifier", deepcopy(model_lgb))])
    # pipeline.fit(X_train, y_train)

    # 预测 + 计算指标
    pmml_pred_train = pipeline.predict_proba(X_train)[:, 1]
    pmml_auc_train = auc_score(y_train, pmml_pred_train)
    pmml_ks_train = ks_score(y_train, pmml_pred_train)

    pmml_pred_test = pipeline.predict_proba(X_test)[:, 1]
    pmml_auc_test = auc_score(y_test, pmml_pred_test)
    pmml_ks_test = ks_score(y_test, pmml_pred_test)
            
    # 计算PSI
    psi_value = compute_psi(pmml_pred_train, pmml_pred_test)      
    # 计算KS差值
    ks_gap = pmml_ks_train- pmml_ks_test
     # 计算lift
    df_lift, bad_rates, lift_tail10, lift_head10, overall_bad_rate, lift_head1, lift_head2, lift_head5, bins_df_json = decile_analysis(y_test, pmml_pred_test, n_bins=10)
    # 计算分箱单调性
    mono_violation, mono_jump_count_test = monotonicity_violation(bad_rates)
   
            
    # 保存模型评估指标
    model_metrics = {
        'eval_dataset_name': eval_dataset_name,  # 评估数据集名称
        'ks_train': pmml_ks_train,
        'ks_test': pmml_ks_test,
        'auc_train': pmml_auc_train,
        'auc_test': pmml_auc_test,
        'psi':psi_value,
        'ks_gap':ks_gap,
        'lift_tail10': lift_tail10,
        'lift_head10': lift_head10,
        'lift_head1': lift_head1,
        'lift_head2': lift_head2,
        'lift_head5': lift_head5,
        'mono_violation':mono_violation,
        'mono_jump_count_test': mono_jump_count_test,
        'overall_bad_rate':overall_bad_rate,
        'df_lift': df_lift.to_dict(orient="records"), # DataFrame转为字典列表（用于JSON报告）
    }
    
    # 尝试保存PMML模型
    try:
        sklearn2pmml(pipeline, model_save_path, with_repr=True)
        print("   PMML模型训练完成并保存!")
    except Exception as e:
        print(f"PMML模型无法保存,原因是{e}-模型预测结果：\n")
    
    print("   PMML模型性能指标:")
    print("   ┌───────────────────────────────────────────────┐")
    print(f"   │ {'训练集(train)':<16} AUC={pmml_auc_train:.6f}, KS={pmml_ks_train:.6f} │")
    print(f"   │ {eval_dataset_name:<16} AUC={pmml_auc_test:.6f}, KS={pmml_ks_test:.6f} │") 
    print("   └───────────────────────────────────────────────┘")
    print(f"   模型稳定性: PSI = {psi_value:.6f}")
    return model_metrics

def save_model_pkl(model_lgb, X_train, y_train, X_test, y_test, use_fea, model_save_path, cat_fea=[], eval_dataset_name="测试集", feature_dict_path=None, feature_importance_path=None, extra_eval_datasets=None):
    """
    训练 LightGBM 模型、评估指标、保存模型，并计算特征重要性。
    
    参数：
    - X_train, y_train: 训练集特征和标签
    - X_test, y_test: 测试集或验证集特征和标签
    - use_fea: 使用的特征列表
    - model_save_path: 模型保存路径
    - cat_fea: 类别型特征列表
    - eval_dataset_name: 评估数据集名称（用于输出显示）
    - feature_dict_path: 特征字典路径（可选，用于特征重要性分析）
    - feature_importance_path: 特征重要性保存路径（可选，用于特征重要性分析）
    返回：
    - model_metrics: 包含模型评估指标的字典
    - feature_imp: 特征重要性DataFrame（如果提供了feature_dict_path和feature_importance_path）
    """
    X_train = X_train[use_fea]
    X_test = X_test[use_fea]
    model_lgb = deepcopy(model_lgb) 
    model_lgb.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        eval_metric=ks_metric)
    
    # 预测 + 计算指标
    lgb_pred_train = model_lgb.predict_proba(X_train)[:, 1]
    lgb_auc_train = auc_score(y_train, lgb_pred_train)
    lgb_ks_train = ks_score(y_train, lgb_pred_train)

    pmml_test = model_lgb.predict_proba(X_test)[:, 1]
    pmml_auc_test = auc_score(y_test, pmml_test)
    pmml_ks_test = ks_score(y_test, pmml_test)

     # 计算PSI
    psi_value = compute_psi(lgb_pred_train, pmml_test)      
    # 计算KS差值
    ks_gap = lgb_ks_train- pmml_ks_test

    dataset_metrics_compare = [
        {
            "数据集": "训练集(train)",
            "KS值": f"{lgb_ks_train:.6f}",
            "AUC值": f"{lgb_auc_train:.6f}",
            "KS差值": "-",
            "PSI值": "-"
        }
    ]
    extra_eval_datasets = extra_eval_datasets or {}
    for dataset_name, (X_extra, y_extra) in extra_eval_datasets.items():
        X_extra = X_extra[use_fea]
        pred_extra = model_lgb.predict_proba(X_extra)[:, 1]
        ks_extra = ks_score(y_extra, pred_extra)
        auc_extra = auc_score(y_extra, pred_extra)
        dataset_metrics_compare.append({
            "数据集": dataset_name,
            "KS值": f"{ks_extra:.6f}",
            "AUC值": f"{auc_extra:.6f}",
            "KS差值": f"{(lgb_ks_train - ks_extra):.6f}",
            "PSI值": f"{compute_psi(lgb_pred_train, pred_extra):.6f}"
        })
    dataset_metrics_compare.append({
        "数据集": eval_dataset_name,
        "KS值": f"{pmml_ks_test:.6f}",
        "AUC值": f"{pmml_auc_test:.6f}",
        "KS差值": f"{ks_gap:.6f}",
        "PSI值": f"{psi_value:.6f}"
    })
    # 计算lift
    df_lift, bad_rates, lift_tail10, lift_head10, overall_bad_rate, lift_head1, lift_head2, lift_head5, bins_df_json = decile_analysis(y_test, pmml_test, n_bins=10)
    # 计算分箱单调性
    mono_violation, mono_jump_count_test = monotonicity_violation(bad_rates)

    # 保存模型评估指标
    model_metrics = {
        'eval_dataset_name': eval_dataset_name,  # 评估数据集名称
        'ks_train': lgb_ks_train,
        'ks_test': pmml_ks_test,
        'auc_train': lgb_auc_train,
        'auc_test': pmml_auc_test,
        'psi': psi_value,
        'ks_gap':ks_gap,
        'lift_tail10': lift_tail10,
        'lift_head10': lift_head10,
        'lift_head1': lift_head1,
        'lift_head2': lift_head2,
        'lift_head5': lift_head5,
        'mono_violation':mono_violation,
        'mono_jump_count_test': mono_jump_count_test,
        'overall_bad_rate':overall_bad_rate,
        'df_lift': df_lift.to_dict(orient="records"),  # DataFrame转为字典列表（用于JSON报告）
        'dataset_metrics_compare': dataset_metrics_compare,
    }
    
    # 判断并保存模型
    pickle.dump(model_lgb, open(model_save_path, 'wb'))
    print("   PKL模型训练完成并保存!，模型性能指标:")
    print("   ┌───────────────────────────────────────────────┐")
    print(f"   │ {'训练集(train)':<16} AUC={lgb_auc_train:.6f}, KS={lgb_ks_train:.6f} │")
    print(f"   │ {eval_dataset_name:<16} AUC={pmml_auc_test:.6f}, KS={pmml_ks_test:.6f} │")
    print("   └───────────────────────────────────────────────┘")
    print(f"   模型稳定性: PSI = {psi_value:.6f}")

    # 计算特征重要性（如果提供了路径）
    feature_imp = None
    if feature_dict_path and feature_importance_path:
        feature_imp = feature_importance_analysis(model_lgb, use_fea, feature_dict_path, feature_importance_path)
    
    return model_metrics, feature_imp

def feature_importance_analysis(model_lgb, use_fea, feature_dict_path, imp_path):
    """
    计算并保存特征重要性（split + gain），并输出筛选后的有效特征列表。

    参数：
    - model_lgb: 训练好的 LightGBM 模型
    - use_fea: 使用的特征列表
    - feature_dict_path: 特征英文-中文映射表路径（CSV）
    - output_dir: 输出路径

    返回：
    - final_features: 最终保留的特征列表（按 importance_score_split > 0）
    """


     # 加载特征字典
    if feature_dict_path and os.path.exists(feature_dict_path):
        feature_map = pd.read_csv(feature_dict_path)
        if 'Feature' not in feature_map.columns:
            print(f"特征字典文件 {feature_dict_path} 缺少主键列 'Feature'，请检查表头命名。将使用空表。")
            feature_map = pd.DataFrame(columns=['Feature', 'feature_cn', 'data_source'])
    else:
        print(f"特征字典文件 {feature_dict_path} 不存在，将使用空表。")
        feature_map = pd.DataFrame(columns=['Feature', 'feature_cn', 'data_source'])
        
    # 获取特征重要性
    feature_imp_split = pd.DataFrame({
        'Feature': use_fea,
        'importance_score_split': model_lgb.booster_.feature_importance(importance_type='split')
    })
    feature_imp_gain = pd.DataFrame({
        'Feature': use_fea,
        'importance_score_gain': model_lgb.booster_.feature_importance(importance_type='gain')
    })
    feature_imp = pd.merge(feature_imp_split, feature_imp_gain, on='Feature', how='inner')

    # 合并中文名称
    feature_imp = pd.merge(feature_map, feature_imp, on='Feature', how='right')
    feature_imp.sort_values(by='importance_score_split', ascending=False, inplace=True)
    feature_imp['imp_split_ratio'] = feature_imp['importance_score_split'] / feature_imp['importance_score_split'].sum()
    feature_imp['imp_gain_ratio'] = feature_imp['importance_score_gain'] / feature_imp['importance_score_gain'].sum()

    # 保存特征重要性文件
    feature_imp.to_csv(imp_path, index=False)

    # 输出 importance_score_split > 0 的有效特征[不做筛选]
    final_feature_df = feature_imp[feature_imp['importance_score_split'] > 0][['Feature']]
    print("=" * 60)
    print(f"特征重要性大于0的特征数量: {len(final_feature_df)}")
    print("特征重要性大于0的特征(按重要性排序)：",list(final_feature_df['Feature']) )
    return feature_imp
