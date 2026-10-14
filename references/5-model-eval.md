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
