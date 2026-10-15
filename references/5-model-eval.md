# 环节 5：模型评估（model_eval）

<!-- 参数表开始 -->
## 可修改参数

只能修改下表中的参数；用户没提到的参数保持当前值。默认值来自原 params.json。

`make_config.py --set` 的写法：`{"model_eval": {"eval_model_type": 值}}`

| 参数 | 中文名 | 默认值 | 说明 |
|---|---|---|---|
| `eval_model_type` | 评估模型类型 | `null` | str类型，选择评估的模型类型，可选LightGBM、AutoGluon或LrScorecard。当选择AutoGluon时，可设置eval_model_name选择具体算法；LrScorecard复用训练阶段的完整Excel报告生成评估摘要。默认为null（自动继承上一步训练模型）。 |
| `eval_model_name` | AutoGluon评估模型名 | `null` | str类型，只有当eval_model_type为AutoGluon时，用户可以指定想要评估具体模型名称，默认为null（默认最佳模型）。 |
| `bin_num_lift` | Lift分箱数 | `10` | int类型，分箱数量（如用于Lift图等），默认为10 |
| `pdo` | PDO(赔率翻倍分数) | `91` | int类型，PDO参数（使赔率翻倍所需的分数点数），默认为91 |
| `theta` | 每倍赔率分数差(theta) | `8.6` | float类型，每一倍赔率对应的分数差，默认为8.6 |
| `p0` | 基准分(p0) | `410` | int类型，基准分数，默认为410 |
| `cut_points` | 分数切分点 | `[]` | list[int]类型，分数转换的切分点列表，空列表表示为取分数的最小最大值等距十分箱划分，默认为空列表[] |
| `gen_eval_report` | 是否生成评估报告 | `true` | bool类型，是否生成模型评估报告，默认为true |
| `save_eval_output` | 是否保存评估结果 | `true` | bool类型，是否将评估的输出结果保存为CSV，默认为true |

### 参数之间的规则（make_config.py 会检查）

- eval_model_name 只有在 eval_model_type 为 AutoGluon 时才能指定。
<!-- 参数表结束 -->

## 作用与前置条件

要求四个前置环节成功、未过期且所需模型产物存在。`eval_model_type=null` 继承当前训练模型；显式类型必须与当前有效训练一致。旧目录中其他算法文件不构成有效模型。
AutoGluon 的 eval_model_name 从本次实际 leaderboard 读取，不能凭算法简称猜模型名；指定时同时将 eval_model_type 设为 AutoGluon。

## LightGBM 评估包含的复现验证

LightGBM 默认 `gen_eval_report=true`：后台生成评估 Excel 后，会自动生成并执行 `retrain_data/retrain_code.py`。该脚本会对训练数据再次 `fit`，用于复现验证；不替换第四环节保存的模型。执行清单应说明这个额外拟合，不能把“只运行第五环节”解释为“完全不会再拟合”。AutoGluon 和评分卡分支没有这个步骤。

用户明确要求不发生任何重训/拟合时，先说明与当前设置的冲突，提供读取已有报告，或确认关闭 `gen_eval_report` 后重新评估的选项。关闭后仍生成 JSON、Markdown、HTML，但不生成后台评估 Excel，也不执行这次复现拟合。不能未经确认关闭报告开关，也不能承诺本版同时保留新 Excel 和禁止拟合。

复现验证失败可能被后台作为提示保留，主评估仍成功。应查看日志，把主评估、复现验证、映射表是否齐备分开说明；主评估成功不等于具备部署条件。

## 用户说法与参数

| 用户说法 | 提交给 make_config 的变化 |
|---|---|
| “评估刚训练好的模型” | 沿用参数；如历史显式类型不同，展示 `{"model_eval":{"eval_model_type":null,"eval_model_name":null}}` 的变化后确认 |
| “评估 AutoGluon 排名里的某个模型” | `{"model_eval":{"eval_model_type":"AutoGluon","eval_model_name":"本次排名中的准确名称"}}`，补充实际名称后检查 |
| “不生成后台评估 Excel” | `{"model_eval":{"gen_eval_report":false}}`，仅适用于实际读取此开关的分支 |

LightGBM/AutoGluon 的 pdo、theta、p0、cut_points 控制后台相应评分转换；空 cut_points 表示按后台规则划分。不要把这些参数当成评分卡训练的 score_scale。

## 评分卡评估的边界

LrScorecard 第五环节读取本次训练 JSON 和训练 Excel，形成评估摘要，不重新加载新数据预测或重算指标。此分支不会按评估阶段 pdo、p0、theta、cut_points 或报告开关重新生成评分卡结果；不能推荐修改这些参数来改变评分卡效果。
用户要调整评分刻度、分箱或模型参数，应回到评分卡训练；数据变化则先处理受影响上游。它不会额外导出 LightGBM PMML。

## 结果与解读

优先看模型信息、模型效果评估、模型稳定性评估、分渠道效果、Lift 分析和评分分箱。每个结论注明数据集和参照集；报告未提供的数值不可补算。
KS/AUC 描述区分能力，PSI 描述相对分布变化，Lift 随头部/尾部与排序口径变化。引用本次报告的具体定义和数值，不把单一阈值作为部署通过结论。
基线列缺失表示未提供对比，不是基线指标为零。映射表缺失可能只影响部署辅助文件，应说明实际遗漏的文件，不把缺映射误称为全部建模失败。
后台产物位于 `outputs/model_eval/` 对应模型子目录；评分卡完整 Excel 仍位于训练的 LR/report。通过当前报告产物链接获取文件。

## 推荐下一步与失败处理

- “查看完整评估报告及产物”：只读本次文件。
- “解释某个 OOT/渠道/评分段的表现”：基于已有明细解释。
- “调整明确参数重新训练并评估”或“结束本次建模并汇总”：按用户目标选择；调参必须给出具体范围并检查。
- 第 4 项为自定义输入。

模型未保存、指定模型名不存在、模型类型不匹配：返回训练或选择当前有效模型。评分卡 Excel 与 JSON 不一致/缺少工作表：不能拼接历史文件冒充成功，指出日志并回到训练。评估失败保留成功训练结果，可修正评估参数后单独重试。

显示精度：训练 JSON 的 KS 为 `0.48`，Excel 保存为 `0.475`，再次格式化为两位小数可能成为 `0.47`。评估的一致性检查按 JSON 两位、Excel 四位小数的精度容差比较 AUC/KS；样本数、坏样本率仍严格比较，数据集和特征名单仍检查一致。各报告保留原展示值，不重算指标。超过容差仍报错，不能修改结果文件或换分箱方式掩盖不一致。
