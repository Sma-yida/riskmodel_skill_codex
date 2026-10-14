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
