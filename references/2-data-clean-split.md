# 环节 2：数据清洗与切分（data_clean_split）

<!-- 参数表开始 -->
## 可修改参数

只能修改下表中的参数；用户没提到的参数保持当前值。默认值来自原 params.json。

`make_config.py --set` 的写法：`{"data_clean_split": {"train_split_range": 值}}`

| 参数 | 中文名 | 默认值 | 说明 |
|---|---|---|---|
| `train_split_range` | 训练集时间范围 | `[20230401, 20240228]` | list[int]类型，训练数据集的时间范围，只有在split_method为date_range，才需要配置该参数，且必须配置。格式为包含两个时间值的列表，分别代表起始时间和结束时间，时间格式为yyyyMMdd，时间类型为int，取值需要根据用户的描述和数据质量分析中对时间列的分析结论进行提取 |
| `oot_split_range` | 测试集(OOT)时间范围 | `[20240301, 20240331]` | list[int]类型，OOT测试数据集的时间范围，只有在split_method为date_range，才需要配置该参数，且必须配置。格式为包含两个时间值的列表，分别代表起始时间和结束时间，时间格式为yyyyMMdd，时间类型为int，取值需要根据用户的描述和数据质量分析中对时间列的分析结论进行提取 |
| `psi_split_range` | PSI数据集时间范围 | `[20240301, 20240331]` | list[int]类型，PSI数据集的时间范围，一般情况下，psi测试集和oot测试集时间范围可以相同，除非用户明确指出psi测试集的时间范围。格式为包含两个时间值的列表，分别代表起始时间和结束时间，时间格式为yyyyMMdd，时间类型为int，取值需要根据用户的描述和数据质量分析中对时间列的分析结论进行提取 |
| `channel_keep_list` | 保留渠道 | `[]` | list[str]类型，需保留的渠道名单列表，如提供列表，则只保留对应渠道。渠道包括但不限于APP流量、饿了么、百度闪付、中信零售、银联等，根据用户输入的渠道进行填充，如果用户未提及，则默认为空列表，表示保留所有渠道 |
| `fill_val` | 缺失值填充值 | `-9999` | int类型，缺失值默认填充值，默认为-9999 |
| `split_method` | 切分方式 | `"date_proportion"` | str类型，数据切分方式，目前提供date_range、date_proportion和random三种，即按指定时间范围切分、按时间排序比例切分、以及随机切分，如果用户提供了train_split_range或oot_split_range，则split_method必为date_range，默认为date_proportion |
| `test_size` | 测试集占比 | `0.2` | float类型，测试集的占比，值为0~1的小数。如果split_method为date_proportion或者random，则需要配置该参数，默认为0.2 |
| `random_state` | 随机种子 | `42` | int类型，随机状态参数的配置，默认为42 |

### 参数之间的规则（make_config.py 会检查）

- 提供了任一时间范围时，切分方式自动为 date_range；如果同时明确写了别的 split_method，会报错。
- 只给了 oot_split_range、没给 psi_split_range 时，PSI 时间范围自动与 OOT 相同。
- 按时间范围切分时，train_split_range 和 oot_split_range 都必须明确给出；有数据质量分析结果时，还会核对是否落在数据日期内。
- 日期写成 yyyyMMdd 整数，例如 20240301。
<!-- 参数表结束 -->
