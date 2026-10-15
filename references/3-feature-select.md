# 环节 3：特征筛选（feature_select）

<!-- 参数表开始 -->
## 可修改参数

只能修改下表中的参数；用户没提到的参数保持当前值。默认值来自原 params.json。

`make_config.py --set` 的写法：`{"feature_select": {"iv_thresh": 值}}`

| 参数 | 中文名 | 默认值 | 说明 |
|---|---|---|---|
| `iv_thresh` | IV阈值 | `0.02` | float类型，IV阈值，范围0~1的小数，低于该值特征将被剔除，默认为0.02 |
| `psi_thresh` | PSI阈值 | `0.2` | float类型，PSI阈值，范围0~1的小数，高于该值说明特征稳定性差，将被剔除，默认为0.2 |
| `miss_thresh` | 缺失率阈值 | `0.9` | float类型，缺失率阈值，范围0~1的小数，超过该比例的特征将被剔除，默认为0.9 |
| `corr_thresh` | 相关性阈值 | `0.8` | float类型，相关性阈值，范围0~1的小数，高度相关的特征将被剔除，默认为0.8 |
| `use_null_importance` | 是否启用Null Importance | `false` | bool类型，是否启用 Null Importance 筛选特征，默认为false |
| `top_n` | Null Importance保留Top N | `500` | int类型，Null Importance 筛选特征的Top N，默认为500 |
| `user_feature_list` | 人工指定特征 | `[]` | list[str]类型，人工指定的特征列表，若用户指定则以列表的形式填充，严禁省略、缩写、添加省略号或修改任何用户指定的特征名称字符。若未指定使用默认值，默认为空列表 |

### 参数之间的规则（make_config.py 会检查）

- user_feature_list 中的特征名必须与数据列名完全一致，不能省略、缩写或加省略号；不能包含 id/date/label/channel。
<!-- 参数表结束 -->

## 作用与前置条件

要求数据分析、清洗切分成功且未过期。使用清洗后的 train 与 psi 计算单变量信息并筛选。
普通筛选依次考虑 IV、缺失、PSI、数值变量相关性，以及可选的 Null Importance。`user_feature_list` 非空时采用人工指定名单，不应描述为这些变量都通过普通阈值筛选。
单变量统计缓存由后台按 train/psi 内容摘要核对，参数重筛时可复用；输入变化会重算。

## 用户说法与参数

| 用户说法 | 提交给 make_config 的变化 |
|---|---|
| “IV 至少 0.03，其他不变” | `{"feature_select":{"iv_thresh":0.03}}` |
| “PSI 上限 0.1，相关性阈值 0.7” | `{"feature_select":{"psi_thresh":0.1,"corr_thresh":0.7}}` |
| “只用 f1、f3” | `{"feature_select":{"user_feature_list":["f1","f3"]}}`，须查真实列名 |
| “取消人工名单，恢复规则筛选” | `{"feature_select":{"user_feature_list":[]}}` |

“更严格”不对应唯一参数；需明确 IV/PSI/缺失/相关性中的哪项和新值。
`use_null_importance=true` 可使用 `data/null_importance_config.json`。该辅助文件可能定义特征池和后续 KS 逐步筛选，不属于 make_config 的公开参数表；只识别并说明用户已有文件，不临时创造高级配置。

## 结果与解读

读取“特征筛选概览”的完整入选名单、数量及“统计分析”，详细 IV/PSI 等在 `outputs/feature_select/outputs/single_var.csv`，分箱明细在 `mono_var.csv`，入选名单在 `use_fea.json`。
JSON 的特征详情只列 IV 较高的前 10 个，不能把它当成全部入选变量。没有逐变量淘汰记录时，只说明规则和现有证据，不编造某特征被删的唯一原因。
这里的分箱、IV、PSI 不等同于评分卡训练期 WOE 分箱的统计口径。相关性阈值只适用于后台实际处理的数值变量。

## 推荐下一步

- “沿用当前模型与参数训练”：有效前置结果齐全后直接执行，并在选项中写明模型名。
- “查看全部入选特征和单变量统计”：只读文件。
- 需要不同模型时给“改用 LR 评分卡训练”或“改用 AutoGluon 对比，保存模型，限时 300 秒”等明确选项，先用训练指南检查。
- 第 4 项为自定义输入。阈值调整建议必须说明本次结果依据，不承诺提升效果。

## 常见失败

筛选为空：展示现有数量与阈值，供用户选择调整或指定有效特征。特征不存在/包含基础列：要求准确名称。单变量计算异常：引用失败变量和日志；不要悄悄跳过后宣称全部成功。
