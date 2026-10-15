# 环境与工作目录

控制脚本仅依赖 Python 标准库，可用 Python 3.8+ 启动；模型环境统一为原生 CPython 3.9，支持 macOS / Linux。下面命令中的 `SKILL` 指本 `risk-modeling` 文件夹的绝对路径。脚本最终向 stdout 输出 JSON；进度向 stderr 输出，失败返回非零退出码。

## 检查、选择与复用

```bash
python3 "$SKILL/scripts/setup_env.py" --list
python3 "$SKILL/scripts/setup_env.py" --check --python /absolute/env/bin/python
# 已记住环境后
python3 "$SKILL/scripts/setup_env.py" --check
```

`--python` 也接受 PATH 中的命令或已发现的 conda 环境名称。列表只表示解释器可以启动，不代表建模验收通过。检测到多个环境时不擅自选择。macOS ARM 上的 Intel/Rosetta Python 会报告架构不匹配。

检查分为独立子进程：解释器与包版本、Java、OpenMP/LightGBM、`pip check`、后台导入、LightGBM/PMML、评分卡、AutoGluon。首次接入必须全部通过；后续解释器、依赖、Java、指定动态库或验证代码发生变化时重新完整验收。未变化时仍检查依赖、Java 和原生运行库，复用上次小模型验证结果。`--full` 强制重验，`--no-save` 只验证、不记录环境。

完整验收包括：600 条合成数据训练；PMML 导出、读取及 150 条概率对比；评分卡 chi/dt/quantile/step 四种分箱与 pickle 重载；AutoGluon GBM/CAT/XGB/RF/XT/KNN/NN_TORCH/FASTAI 八类模型训练及逐模型重载预测。某个模型被跳过会判为失败。这不等于五环节端到端或行内一致性验收。

可选运行库设置：

```bash
python3 "$SKILL/scripts/setup_env.py" --check --python /absolute/env/bin/python \
  --library-dir /absolute/openmp/lib --java-home /absolute/jdk \
  --mapping-dir /absolute/mapping
```

只有完全通过才更新 `~/.risk-modeling/settings.json`，保存 `python`、`checked_at`、`mapping_dir`、所需运行库位置和验证指纹。已有设置中的其他字段保留。测试可用环境变量 `RISKMODEL_HOME` 指定临时设置目录；不要将临时目录中的验证环境记录为正式生产环境。

`run_stage.py` 复用 `env_support.runtime_env()`：指定 Python、原生库、Java 和映射表路径，禁用向 Skill 写入 Python 缓存，使用非交互绘图后端。每次开始建模会话先执行上述环境检查；执行器要求本机记录已完成完整验收，不自动安装环境。

## 联网新建

```bash
# 默认用 conda，创建 ~/.risk-modeling/envs/riskmodel-py39
python3 "$SKILL/scripts/setup_env.py" --create

# 已有原生 Python 3.9、Java、OpenMP 时，可用 venv
python3 "$SKILL/scripts/setup_env.py" --create \
  --base-python /absolute/python3.9 --prefix /absolute/riskmodel-py39 \
  --library-dir /absolute/openmp/lib
```

conda 路径使用 conda-forge 并选择本机原生子平台，环境内安装 Python 3.9、Java 17、OpenMP，再用 pip 安装固定依赖。`--conda` 可指定 conda 命令。venv 路径复用已有 Java/OpenMP，不下载系统运行库。目标目录存在时拒绝创建，绝不将安装或升级操作施加于用户已有环境；安装失败保留目录和日志，重新创建需指定新的目录。

支持 `--index-url https://mirror.example/simple`，不修改全局 pip/conda 配置。该参数只控制 pip；conda 创建仍访问 conda-forge。主要版本在 `scripts/requirements.txt`，本次 macOS ARM64 的完整实测版本在 `scripts/requirements/verified-macos-arm64-py39.txt`，安装时作为传递依赖约束；pip/setuptools/wheel 的引导版本在 `requirements/bootstrap.txt`。

核心组合：NumPy 1.26.4、pandas 2.2.3、scikit-learn 1.6.1、LightGBM 4.5.0、AutoGluon 1.3.0、scorecardpipeline 0.1.39、toad 0.1.7、sklearn2pmml 0.110.0。已有 AutoGluon 旧模型继续使用其原环境；官方不支持跨版本直接加载，不能为接入本工具而升级旧模型环境。[AutoGluon 1.3 说明](https://auto.gluon.ai/dev/whats_new/v1.3.0.html)

## 离线包与离线安装

在与目标机系统、处理器架构和 CPython 3.9 一致的联网机器执行：

```bash
python3 "$SKILL/scripts/setup_env.py" --download-wheels /absolute/wheels \
  --python /absolute/native/python3.9
```

输出目录须为空或不存在。工具使用 `pip wheel` 下载或构建所有依赖及引导工具，写入含目标信息和 SHA-256 的 `manifest.json`。构建阶段可能需要网络和构建工具。把整个目录原样复制到目标机，不要混入其他 wheel。

```bash
python3 "$SKILL/scripts/setup_env.py" --create \
  --base-python /absolute/python3.9 --prefix /absolute/riskmodel-py39 \
  --wheel-dir /absolute/wheels --library-dir /absolute/openmp/lib \
  --java-home /absolute/jdk
```

离线范围是已有 Python 3.9、Java，以及可用的原生 OpenMP 运行库。包内不包含 Python/conda、Java 或系统库。先校验目标信息、清单和 wheel 内容，再创建全新 venv；安装强制 `--no-index --only-binary=:all: --no-deps --no-cache-dir`，全部依赖从已校验的本地目录读取。禁用外部 pip 配置；不接收依赖 URL，不进行源代码构建。安装后仍执行完整验收。

Mac 包不能用于 Linux；Linux 包需要在对应 Linux 架构上重新生成，并在目标机验收。相同系统和架构仍可能受系统版本、glibc、驱动等影响，wheel 标签检查和实际模型验收都必须通过。当前没有 Linux 实机通过记录。离线安装参数参见 [pip 官方文档](https://pip.pypa.io/en/stable/cli/pip_install/#install-no-index)。

## 日志与失败处理

每次运行单独保存 `~/.risk-modeling/logs/setup-*/report.json` 和各阶段日志；`--log-dir` 可改根目录。模型探针产物也保存在该次日志目录。运行期间每 15 秒报告耗时和日志大小。每项检查默认 180 秒，安装/下载默认 1800 秒，可分别用 `--timeout`、`--install-timeout` 调整。超时或中断会清理进程组（包括 Java）；受限系统不允许清理时明确报错，不能算验证成功。

失败 JSON 的 `error_kind` 区分缺包、版本冲突、处理器架构、Java/OpenMP、网络、磁盘、wheel 不匹配、超时和安装失败，并提供处理选项。不自动切换镜像、不自动升级现有环境、不删除失败环境。

两项已知边界：

- macOS ARM 的 grpcio 1.67.1 wheel 内部架构标记会导致 `pip check` 报错。仅当完整输出恰好只有这一条已知错误，且实际本机 gRPC 请求/响应通过时，作为显式警告放行；原日志和非零退出码保留。出现任何额外依赖错误均失败。
- scorecardpipeline 0.1.39 / toad 0.1.7 在一组接近均衡的合成数据上，逐步回归出现 `list.remove(x): x not in list`。本次模型探针使用坏样本占比约 20% 的固定样本；均衡数据复现保留于开发目录，Step 2 未修改后台算法，不承诺任意数据训练成功。

## 工作目录

```bash
python3 "$SKILL/scripts/init_workspace.py"
python3 "$SKILL/scripts/init_workspace.py" --workdir /absolute/my_project
```

默认创建启动目录下的 `riskmodel_work/`；当前目录已经是工作目录时直接复用。工作目录不能位于 Skill 内，也不能使用包含 Skill 的父目录。

| 子目录 | 用途 |
|---|---|
| `data/` | 主数据 `data.csv` / `data.parquet` / `data.feather`；时间外数据 `oot_*`；可选映射表 `mapping/` |
| `config/` | 五份环节配置，只补充缺失文件；以后通过 `make_config.py` 修改 |
| `outputs/` | 后台生成的模型、中间数据和原始报告 |
| `results/` | 五个环节同名 JSON、Markdown、HTML |
| `logs/` | 环节日志、task.json、stages/ 下的环节执行记录，以及运行锁 |

默认基础列为 `id`、`date`、`label`、`channel`；建模标签为二分类 0/1。初始化不加载全量数据，不覆盖已存在的数据、配置或结果；无效 JSON 只报告，不修复。有旧结果但没有执行记录时标为 `untracked`，不能当作已验证前置结果。`startup_marker_present` 仅表示存在旧启动标记。初始化不会启动建模。

## 执行、等待与停止

以下操作须在用户确认环节和参数后执行；`--check`、`--status` 是只读检查。

```bash
python3 "$SKILL/scripts/run_stage.py" -w "$WORK" --start data_analysis data_clean_split --check
python3 "$SKILL/scripts/run_stage.py" -w "$WORK" --start data_analysis data_clean_split
python3 "$SKILL/scripts/run_stage.py" -w "$WORK" --wait --seconds 30
python3 "$SKILL/scripts/run_stage.py" -w "$WORK" --status
python3 "$SKILL/scripts/run_stage.py" -w "$WORK" --stop
python3 "$SKILL/scripts/make_report.py" -w "$WORK" --stage data_analysis
```

`WORK` 是独立工作目录的绝对路径。五个环节必须按标准顺序列出，不能重复；未纳入本次执行范围的前置环节必须有效。同目录同时只能有一个任务，运行中配置工具拒绝写入。可用不同工作目录保留不同试验。

`ok=true` 表示命令/查询成功；实际任务读取 `task.status`，环节读取 `stages.<环节>.state`：

| 状态 | 含义与处理 |
|---|---|
| starting / running | 已启动或执行中，继续查询；不要再次启动 |
| succeeded | 计算完成；还需留意 report_error 和具体输出警告 |
| failed / stopped | 执行失败或已停止，当前环节不能作为前置结果 |
| stale | 配置、输入、上游或产物变化，重跑受影响环节 |
| untracked / invalid_result | 历史结果未验证或结果损坏；可查原文件，不跳过验证 |
| interrupted | 执行记录未正常结束且锁已释放，查日志后重跑 |
| unresponsive | 超过 15 秒无执行器心跳但仍持锁；不启动重复任务，不宣称已停止 |
| report_failed | 计算成功，报告生成失败；补生成报告后接续剩余环节 |
| ready | 报告已修复，remaining_stages 列出原任务尚未执行的环节；按原确认范围接续 |

任务在独立进程运行，关闭聊天后可以查询同目录接续。`--wait` 默认 30 秒、最多 60 秒；它不会设置模型的运行时限。停止通过当前任务的停止请求交给执行器清理子进程；系统拒绝进程组操作时明确报错，不绕过权限或误停其他进程。

## 重跑与结果有效性

结果仍使用固定文件名，没有 run_id 或自动历史版本。重跑会清理该环节的旧 JSON/Markdown/HTML 及标准生成目录内旧产物；特征统计缓存保留，由后台按输入内容摘要核对。用户输入目录不清理。欲保留旧模型请先选择另一个工作目录。

记录各环节当前配置摘要、相关输入及上游完成时间/结果摘要。大数据与普通产物用路径、大小、纳秒修改时间检查；辅助 JSON 另加 SHA-256。它能检测正常替换、增删和修改，不承诺识别人为保持大小及时间戳不变的篡改。输入在执行过程中变化会判失败。

训练参数变化只使训练和评估失效；主数据变化从分析开始失效；外部 OOT 变化从清洗开始失效。只更改一个环节不会自动替用户重跑全部下游。

报告完整展示 JSON，后台 Top N 摘要保持原口径并提供当前原始产物链接。只生成摘要不触发模型计算，报告缺失可以单独补生成。历史/过期结果不能被重新包装成当前报告。

报告标注计算完成时间，并提醒它是当次运行的快照。静态 HTML 本身不会实时刷新结果状态；数据、参数或产物变化后，先让助手查询当前状态，再引用或分享旧报告。
