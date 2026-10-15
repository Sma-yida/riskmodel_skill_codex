# 环节 4：模型训练（model_train）

<!-- 参数表开始 -->
## 可修改参数

只能修改下表中的参数；用户没提到的参数保持当前值。默认值来自原 params.json。

`make_config.py --set` 的写法：`{"model_train": {"model_select": 值}}`

| 参数 | 中文名 | 默认值 | 说明 |
|---|---|---|---|
| `model_select` | 模型算法 | `"LightGBM"` | str类型，训练用到的模型，可选LightGBM、AutoGluon或LrScorecard。其中LightGBM为梯度提升树模型，选择后可以调整lightgbm_params下的相关模型参数，而AutoGluon代表执行多模型对比，选择后可以调整autogluon_params下的相关模型参数，LrScorecard为基于LR算法的标准评分卡模型，选择后可以调整lr_scorecard_params下的相关参数，默认为LightGBM |
| `n_trials` | 寻优次数 | `10` | int类型，超参数优化迭代次数，默认为10 |
| `ks_gap_threshold` | 训练/测试KS差值上限 | `1.0` | float类型，超参寻优时，训练集与测试集KS差值最大阈值，默认为1.0 |
| `run_hpo` | 是否超参寻优 | `false` | bool类型，是否启用超参寻优；设为true时采用Optuna，可通过optimization_mode设置单目标或多目标模式、optimization_targets指定优化目标、hpo_params设置参数搜索范围，默认为false |
| `optimization_mode` | 寻优模式 | `"multi"` | str类型，超参寻优模式，可选single或multi；single表示单目标，multi表示多目标，默认为multi |
| `optimization_targets` | 寻优目标 | `[]` | list[object]类型，超参寻优目标，如果指定必须按[{"dataset":"test","metric":"ks"}]格式填写；dataset取test或实际OOT名称（如oot_main、oot_1），metric仅取ks、lift_tail10、lift_head10；single填写1项，multi填写2~3项；非必填，空列表使用默认目标，默认为[] |

### autogluon_params（AutoGluon参数）

model_select 为 AutoGluon 时生效。写法：`{"model_train": {"autogluon_params": {"autogluon_time_limit": 值}}}`

| 参数 | 中文名 | 默认值 | 说明 |
|---|---|---|---|
| `autogluon_params.autogluon_time_limit` | AutoGluon时间限制(秒) | `300` | int类型，AutoGluon多模型对比的时间限制（秒），默认为300秒 |
| `autogluon_params.save_autogluon_predictor` | 是否保存AutoGluon模型文件 | `true` | bool类型，是否保存AutoGluon多模型对比的模型文件，默认true |

### lr_scorecard_params（评分卡参数）

model_select 为 LrScorecard 时生效。写法：`{"model_train": {"lr_scorecard_params": {"binning_method": 值}}}`

| 参数 | 中文名 | 默认值 | 说明 |
|---|---|---|---|
| `lr_scorecard_params.binning_method` | 分箱方法 | `"chi"` | str类型，评分卡自动分箱方法，可选chi、dt、quantile或step，分别表示卡方、决策树、等频和等宽分箱，默认为chi |
| `lr_scorecard_params.adj_rules` | 人工分箱规则 | `{}` | dict类型，人工指定的变量分箱规则；未指定变量继续自动分箱，默认为{} |
| `lr_scorecard_params.fine_selection_corr` | 细筛相关性阈值 | `0.6` | float类型，WOE后细筛的相关性阈值，范围为(0,1]，默认为0.6 |
| `lr_scorecard_params.stepwise_criterion` | 逐步回归准则 | `"aic"` | str类型，双向逐步回归准则，可选aic或bic，默认为aic |
| `lr_scorecard_params.score_scale` | 分数刻度 | `{"base_score": 500, "pdo": 30, "base_odds": 20}` | dict类型，评分卡转分数参数，包含base_score、pdo和base_odds，默认分别为500、30和20 |

### lightgbm_params（LightGBM参数）

model_select 为 LightGBM 时生效；run_hpo 为 false 时直接使用这些参数。写法：`{"model_train": {"lightgbm_params": {"random_state": 值}}}`

| 参数 | 中文名 | 默认值 | 说明 |
|---|---|---|---|
| `lightgbm_params.random_state` | 随机种子 | `42` | int类型，随机种子，默认为42 |
| `lightgbm_params.n_jobs` | CPU线程数 | `16` | int类型，LightGBM训练使用的CPU线程数；-1表示使用全部可见逻辑核，正整数表示指定线程数，默认为16 |
| `lightgbm_params.min_split_gain` | 最小分割增益 | `0` | int类型，最小分割增益，非负整数，用于防止过拟合，默认为0 |
| `lightgbm_params.learning_rate` | 学习率 | `0.033` | float类型，学习率，范围为(0,1]的浮点数，控制每次迭代的步长，默认为0.033 |
| `lightgbm_params.num_leaves` | 每棵树最大叶子数 | `11` | int类型，每棵树的最大叶子数，正整数，需要小于2^max_depth，默认为11 |
| `lightgbm_params.max_bin` | 最大分箱数 | `108` | int类型，特征离散化的最大箱数，正整数，影响训练速度和精度，默认为108 |
| `lightgbm_params.min_child_samples` | 叶节点最少样本数 | `20` | int类型，叶节点最少样本数，正整数，默认为20 |
| `lightgbm_params.reg_alpha` | L1正则 | `5.0` | float类型，L1正则化参数，非负浮点数，值越大模型权重越稀疏，默认为5.0 |
| `lightgbm_params.reg_lambda` | L2正则 | `125.0` | float类型，L2正则化参数，非负浮点数，值越大模型越平滑，默认为125.0 |
| `lightgbm_params.subsample` | 样本采样比例 | `0.539` | float类型，训练样本采样比例，范围为(0,1]的浮点数，用于防止过拟合，默认为0.539 |
| `lightgbm_params.subsample_freq` | 子采样频率 | `2` | int类型，子采样频率，正整数，表示每隔多少次迭代进行一次子采样，默认为2 |
| `lightgbm_params.colsample_bytree` | 特征采样比例 | `0.229` | float类型，每棵树特征采样比例，范围为(0,1]的浮点数，用于防止过拟合，默认为0.229 |
| `lightgbm_params.min_child_weight` | 叶节点最小权重和 | `17.0` | float类型，叶节点最小权重和，非负浮点数，用于防止过拟合，默认为17.0 |
| `lightgbm_params.scale_pos_weight` | 正负样本权重比 | `1.0` | float类型，正负样本权重比，>0，默认为1.0 |
| `lightgbm_params.n_estimators` | 树的数量 | `5` | int类型，提升迭代次数，即树的数量，默认为5 |
| `lightgbm_params.max_depth` | 最大深度 | `-1` | int类型，树的最大深度，-1表示不限制深度，正整数表示具体深度限制，默认为-1 |

### hpo_params（寻优搜索范围）

run_hpo 为 true 时，Optuna 在这些范围内搜索 LightGBM 参数。写法：`{"model_train": {"hpo_params": {"min_split_gain": 值}}}`

| 参数 | 中文名 | 默认值 | 说明 |
|---|---|---|---|
| `hpo_params.min_split_gain` | 最小分割增益范围 | `[0, 10, 1]` | list[int]类型，最小分割增益网格范围，每个值都是非负整数，用于防止过拟合，默认为[0,10,1] |
| `hpo_params.learning_rate` | 学习率范围 | `[0.03, 0.2]` | list[float]类型，学习率网格范围，每个值都是范围为(0,1]的浮点数，控制每次迭代的步长，默认为[0.03,0.2] |
| `hpo_params.num_leaves` | 叶子数范围 | `[8, 60]` | list[int]类型，每棵树的最大叶子数网格范围，每个值都是正整数，需要小于2^max_depth，默认为[8,60] |
| `hpo_params.max_bin` | 最大分箱数范围 | `[20, 350]` | list[int]类型，特征离散化的最大箱数网格范围，每个值都是正整数，影响训练速度和精度，默认为[20,350] |
| `hpo_params.min_child_samples` | 叶节点最少样本数范围 | `[20, 100, 10]` | list[int]类型，叶节点最少样本数的网格范围；格式为[下界,上界,步长]，上下界相同表示固定值，默认为[20,100,10] |
| `hpo_params.reg_alpha` | L1正则范围 | `[0, 20.0]` | list[float]类型，L1正则化参数网格范围，每个值都是非负浮点数，值越大模型权重越稀疏，默认为[0,20.0] |
| `hpo_params.reg_lambda` | L2正则范围 | `[0, 200.0, 5]` | list[float]类型，L2正则化参数网格范围，每个值都是非负浮点数，值越大模型越平滑，默认为[0,200.0,5] |
| `hpo_params.subsample` | 样本采样比例范围 | `[0.2, 1.0]` | list[float]类型，训练样本采样比例网格范围，每个值都是范围为(0,1]的浮点数，用于防止过拟合，默认为[0.2,1.0] |
| `hpo_params.colsample_bytree` | 特征采样比例范围 | `[0.2, 1.0]` | list[float]类型，每棵树特征采样比例网格范围，每个值都是范围为(0,1]的浮点数，用于防止过拟合，默认为[0.2,1.0] |
| `hpo_params.subsample_freq` | 子采样频率范围 | `[1, 10]` | list[int]类型，子采样频率网格范围，每个值都是非负整数，表示每隔多少次迭代进行一次子采样，0表示不进行子采样，默认为[1,10] |
| `hpo_params.min_child_weight` | 叶节点最小权重和范围 | `[1.0, 20.0, 0.5]` | list[float]类型，叶节点最小权重和网格范围；格式为[下界,上界,步长]，每个值都是非负浮点数，默认为[1.0,20.0,0.5] |
| `hpo_params.n_estimators` | 树的数量范围 | `[500]` | list[int]类型，提升迭代次数网格范围，每个值都是正整数，即树的数量，默认为[500] |
| `hpo_params.max_depth` | 最大深度范围 | `[-1, -1]` | list[int]类型，树的最大深度网格范围，每个值都是正整数或-1(不限制深度)，默认为[-1,-1] |
| `hpo_params.scale_pos_weight` | 正负样本权重比范围 | `[1.0, 1.0, 0.1]` | list[float]类型，正负样本权重比的网格范围；格式为[下界,上界,步长]，上下界相同表示固定值，默认为[1.0,1.0,0.1] |

### 参数之间的规则（make_config.py 会检查）

- optimization_targets：single 模式写 1 项，multi 模式写 2～3 项；dataset 只能是 test 或 data/ 下真实存在的 OOT 名称（如 oot_main）。
- lightgbm_params.num_leaves 不能超过 2^max_depth（max_depth 为 -1 时不限制）。
- 分组参数（lightgbm_params 等）只写要改的子参数，没写的保持当前值；lr_scorecard_params.adj_rules 会整体替换。
<!-- 参数表结束 -->

## 作用与前置条件

要求前三个环节成功且未过期，且入模特征非空。同一工作目录保存当前一次训练结果；切换模型会重建训练产物，并使旧评估过期。需要保留多模型历史时使用不同工作目录，不能把旧文件混入当前结果。

## 三类模型的实际行为

| 模型 | 使用的参数与主要产物 | 解读重点 |
|---|---|---|
| LightGBM | lightgbm_params；可选 Optuna 单/多目标及 hpo_params；model.pkl、model.pmml、final_model_fea.json | train/test/主验证集表现、寻优目标和入模特征来源 |
| AutoGluon | autogluon_params；比较后台支持的八类算法；leaderboard 和可选 predictor | 使用本次实际模型名称和排名，未成功训练的算法不算完成 |
| LrScorecard | lr_scorecard_params；自动或人工分箱、WOE、细筛、逐步回归、LR；LR 子目录中的模型、分箱表、评分表和 Excel | 每一步特征数量、实际入模变量、分箱、系数及验证口径 |

LightGBM 的 run_hpo、n_trials、hpo_params 不控制评分卡或 AutoGluon。AutoGluon 未保存 predictor 时，允许查看训练对比，但不能继续模型评估。
训练早停 test、寻优目标集、最终报告主验证集可能不同，按输出明确名称；不能把使用 OOT 寻优后的 OOT 当成从未参与选择的独立验证集。

## 用户说法与参数

| 用户说法 | 提交给 make_config 的变化 |
|---|---|
| “用 LightGBM，100 棵树，2 线程” | `{"model_train":{"model_select":"LightGBM","lightgbm_params":{"n_estimators":100,"n_jobs":2}}}` |
| “单目标，以 test 的 KS 寻优 5 次” | `{"model_train":{"model_select":"LightGBM","run_hpo":true,"n_trials":5,"optimization_mode":"single","optimization_targets":[{"dataset":"test","metric":"ks"}]}}` |
| “多模型比较 300 秒，保存后评估” | `{"model_train":{"model_select":"AutoGluon","autogluon_params":{"autogluon_time_limit":300,"save_autogluon_predictor":true}}}` |
| “LR 评分卡，用等频分箱” | `{"model_train":{"model_select":"LrScorecard","lr_scorecard_params":{"binning_method":"quantile"}}}` |

人工分箱 adj_rules 整体替换；若用户要求只调整一个变量，先读现有规则并保留其余变量，再展示完整变化。score_scale 逐项合并，不能与模型评估的 pdo/p0 混为一组。

## 复现文件与参数优先级

训练前检查 `data/model_train_spec_lightgbm.json` 或 `data/model_train_spec_lrscorecard.json`（只读对应算法）。非空 use_features 优先于筛选名单；非空 model_params 会覆盖基础参数，LightGBM 还会跳过 HPO。
将这种影响放入需求确认。若与本次请求冲突，先让用户选择复现现有文件或更新/移走该文件，不宣称普通参数覆盖了复现文件。输出 JSON 中的特征与参数来源是最终解读依据。

## 结果与明细

LightGBM 读取模型性能指标、寻优信息和特征重要性；重要性 JSON 可能只列前 15 个，完整表为 `outputs/model_train/outputs/feature_imp.csv`。
AutoGluon JSON 排名可能只列前 15 个，完整表为 `outputs/model_train/outputs/autogluon_leaderboard.csv`。具体模型名从这次排名读取。
评分卡的 IV 来自训练评分卡分箱，PSI 按训练切点比较主验证集（优先 oot_main，否则 test）与 train；不能与特征筛选阶段数字直接对照判错。查看 `outputs/model_train/LR/outputs/` 的 binning_review.csv、binning_detail.csv、scorecard_points.csv；完整报告在 `LR/report/评分卡模型报告.xlsx`。

## 推荐下一步与失败处理

- “评估本次训练模型”：先核对类型和模型是否保存，默认继承训练类型。
- “查看完整模型报告/分箱明细/模型排名”：只读当前产物。
- “调整指定参数重新训练”：列出具体参数、预算和评估将过期的影响后确认；未明确值则补充信息。
- 第 4 项为自定义输入。

PMML 缺失是 LightGBM 训练交付失败，即使已有 JSON 也不能放行。评分卡 list.remove 异常属于已知依赖边界，不自行删样本、改算法或升级包。训练数据单一标签、空特征、内存不足、Java 异常均按日志证据处理。AutoGluon 限时导致部分模型失败时如实说明实际排名，不承诺所有算法都已成功。
