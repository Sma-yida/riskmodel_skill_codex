import shutil

from .utils1 import ks_score


def autogluon_model_comparison(
    train_df,
    test_df,
    target_col,
    include_model_types=None,  # 只用 config['settings']['model_type_list'] 传递的模型类型
    time_limit=300,
    eval_metric=None,
    output_path=None,
    predictor_path=None,
    log_path=None,
    save_predictor=True  # 新增参数，是否保存模型文件
):
    """
    用AutoGluon对比多模型效果，输出leaderboard。
    关于 include_model_types：
      - 支持的 AutoGluon 模型类型及含义：
          "CAT"      : CatBoost（高效的梯度提升决策树）
          "GBM"      : LightGBM（微软的高效梯度提升树）
          "XGB"      : XGBoost（流行的梯度提升树）
          "NN_TORCH" : 神经网络 (PyTorch 实现)
          "RF"       : 随机森林 (Random Forest)
          "XT"       : Extra Trees（极端随机树）
          "KNN"      : K-Nearest Neighbors（K近邻）
          "FASTAI"   : FastAI 神经网络

    参数：
        train_df: 训练集DataFrame
        test_df: 验证集DataFrame
        target_col: 目标列名
        include_model_types: 要对比的模型类型list（如None/[]/空则用AutoGluon默认）
        time_limit: 训练时间限制（秒）
        eval_metric: 评估指标
            - None: 使用AutoGluon默认指标
            - 'ks'/'KS': 使用自定义KS指标
            - 其他字符串: 使用AutoGluon内置指标（如'roc_auc', 'f1', 'precision'等）
        output_path: leaderboard保存路径（如有）
        predictor_path: 模型保存路径（如有）
        log_path: 日志文件路径（如有）
        save_predictor: bool, 是否保存模型文件（默认True）。为False时训练后会删除predictor_path目录。
    返回：
        leaderboard DataFrame
    """
    import warnings
    warnings.filterwarnings('ignore')
    from autogluon.tabular import TabularPredictor
    from autogluon.core.metrics import make_scorer
    
    # eval_metric处理逻辑
    if eval_metric is None:
        # 使用AutoGluon默认指标
        metric = None
        print("使用AutoGluon默认评估指标")
    elif isinstance(eval_metric, str) and eval_metric.lower() in ['ks', 'k-s']:
        # 使用自定义KS指标
        ks_scorer = make_scorer(
            name='ks',
            score_func=ks_score,  # 使用模块级别的函数
            optimum=1,
            greater_is_better=True,
            needs_proba=True
        )
        metric = ks_scorer
        print("使用自定义KS评估指标")
    else:
        # 使用AutoGluon内置指标
        metric = eval_metric
        print(f"使用AutoGluon内置评估指标: {eval_metric}")

    train_data = train_df.copy()
    test_data = test_df.copy()
    
    # 构建TabularPredictor参数
    predictor_kwargs = {
        'label': target_col,
        'eval_metric': metric,
        'problem_type': 'binary',
        'verbosity': 2  # 标准日志等级
    }
    
    # 添加可选参数
    if predictor_path:
        predictor_kwargs['path'] = predictor_path
    if log_path:
        predictor_kwargs['log_to_file'] = True
        predictor_kwargs['log_file_path'] = log_path
    
    predictor = TabularPredictor(**predictor_kwargs)

    fit_kwargs = dict(
        train_data=train_data,
        tuning_data=test_data,
        presets='best_quality',
        time_limit=time_limit,
        auto_stack=False,
        included_model_types=include_model_types
    )

    predictor.fit(**fit_kwargs)

    leaderboard = predictor.leaderboard(silent=True)
    if output_path:
        leaderboard.to_csv(output_path, index=False)
    print(leaderboard)
     # 新增：如不保存模型，删除目录
    if predictor_path and not save_predictor:
        try:
            shutil.rmtree(predictor_path)
            print("不保存多模型对比文件")
        except Exception as e:
            print(f"Warning: 删除AutoGluon模型目录失败: {predictor_path}, {e}")
    return leaderboard
