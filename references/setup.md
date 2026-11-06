# 环境与工作目录

控制脚本仅依赖 Python 标准库，可用 Python 3.8+ 启动；模型环境统一为原生 CPython 3.9，已适配 macOS / Linux / Windows x64。下面 `SKILL` 为本 Skill 绝对路径，`WORK` 为已明确的工作目录绝对路径。脚本最终向 stdout 输出 JSON；进度向 stderr 输出，失败返回非零退出码。

当前源码已加入 Windows 原生环境发现、路径、文件锁、进程树停止和执行器启动交接；旧的 rc.1 ZIP 没有这些改动。Windows/Linux 的完整实机验收尚未完成，不能仅凭适配代码或 Python 可启动就宣称五环节可用。目标机器必须通过完整 `--check`，再按已确认的范围建模；失败时报告实际失败项，不擅自降级为“验证通过”。

Windows 环境自身是否可用，与本 Skill 的平台适配是两回事；执行器缺陷不应归因于用户环境损坏。不要在业务建模中临时改程序绕过检查。WSL 是另一个 Linux 环境，不能复用 Windows 的 python.exe，WSL 的结果也不能代表 Windows 原生通过。Windows ARM/32 位尚未适配；依赖版本保持原有 CPython 3.9 固定版本，不为兼容而升级用户的 bx 环境。

## Windows 的调用方式

优先使用用户指定的 bx/conda 环境；`--python bx` 在可发现 conda 时按环境名解析，也可直接给解释器路径。Windows conda 为 `环境目录/python.exe`，venv 为 `环境目录/Scripts/python.exe`，JDK 需含 `bin/java.exe`。conda 的 `Library/bin` 自动加入运行 PATH；可用 `--library-dir`、`--java-home` 明确指定。

PowerShell 示例（换成实际路径）：

```powershell
$SkillRoot = 'D:\tools\risk-modeling'
$ModelPython = 'D:\Miniconda3\envs\bx\python.exe'
$WorkRoot = 'D:\建模项目\项目一'
& $ModelPython -B "$SkillRoot/scripts/init_workspace.py" --workdir $WorkRoot --check
& $ModelPython -B "$SkillRoot/scripts/make_config.py" -w $WorkRoot --show data_analysis
# 只有用户确认了目录、环境和实际业务方案后才执行：
& $ModelPython -B "$SkillRoot/scripts/init_workspace.py" --workdir $WorkRoot
& $ModelPython -B "$SkillRoot/scripts/setup_env.py" -w $WorkRoot --check --python $ModelPython --log-dir "$WorkRoot/logs"
& $ModelPython -B "$SkillRoot/scripts/run_stage.py" -w $WorkRoot --start data_analysis --check
& $ModelPython -B "$SkillRoot/scripts/run_stage.py" -w $WorkRoot --start data_analysis
& $ModelPython -B "$SkillRoot/scripts/run_stage.py" -w $WorkRoot --wait --seconds 30
```

带空格/中文路径保持为独立参数。PowerShell 5.1 向本机程序传内联 JSON 可能丢失双引号，参数变更优先保存为 UTF-8 JSON 请求文件，再用 `make_config.py --set-file 文件路径 --check`；确认后去掉 `--check` 写入。新目录确认前将请求文件放系统临时目录，不为保存它提前创建工作目录。该文件是参数请求，不是手写五份工作配置。不要照抄 Bash 的 `\` 续行或 `PYTHONPATH=...` 前缀。若需指定 conda，使用 `Scripts/conda.exe`，不经 `cmd /c` 拼接用户路径。

可在 Skill 目录运行 `python -B -m unittest discover -s tests -v` 检查执行器。它在临时目录使用替身后台，覆盖启动、断开接续、并发、停止、超时及 Windows 执行器崩溃清理，不读用户业务数据、不训练模型、不改正式环境设置。通过仍需完整 `setup_env --check` 和真实五环节验收；测试结果应注明运行系统。

## 检查、选择与复用

用户明确指定的目录和环境优先。未指定时先检查项目记录、公共记录和实际候选，给出具体可行的推荐；不要先让用户选择是否准备方案。首次准备按 [交互指南](first-use.md) 合并目录、环境、数据位置及能确定的业务范围；同一已准备项目后续省略重复说明，检查仍需进行。新安装说明名称、位置、方式和联网需求；单独同意安装/选环境不授权业务运行。

```bash
python3 -B "$SKILL/scripts/setup_env.py" --list
# 可选：确认前验证已有候选，不创建工作目录、不改变环境绑定
python3 -B "$SKILL/scripts/setup_env.py" -w "$WORK" --check --python /absolute/env/bin/python --no-save
# 确认后，已初始化工作目录：验证并记录本项目环境
python3 -B "$SKILL/scripts/setup_env.py" -w "$WORK" --check --python /absolute/env/bin/python --log-dir "$WORK/logs"
# 以后复用本项目环境
python3 -B "$SKILL/scripts/setup_env.py" -w "$WORK" --check --log-dir "$WORK/logs"
```

`--python` 也接受 PATH 中的命令或已发现的 conda 环境名称，优先于项目记录；记录保存实际解释器路径。列表只表示解释器可以启动，不代表建模验收通过。没有唯一合理候选时让用户选择，不擅自换掉指定环境；指定或绑定路径失效时明确报错。macOS ARM 上的 Intel/Rosetta Python 会报告架构不匹配。

检查分为独立子进程：解释器与包版本、Java、OpenMP/LightGBM、`pip check`、后台导入、LightGBM/PMML、评分卡、AutoGluon。首次接入必须全部通过；后续解释器、依赖、Java、指定动态库或验证代码发生变化时重新完整验收。未变化时仍检查依赖、Java 和原生运行库，复用上次小模型验证结果。`--full` 强制重验，`--no-save` 只验证、不记录环境。

完整验收包括：600 条合成数据训练；PMML 导出、读取及 150 条概率对比；评分卡 chi/dt/quantile/step 四种分箱与 pickle 重载；AutoGluon GBM/CAT/XGB/RF/XT/KNN/NN_TORCH/FASTAI 八类模型训练及逐模型重载预测。某个模型被跳过会判为失败。这不等于五环节端到端或行内一致性验收。

可选运行库设置：

```bash
python3 -B "$SKILL/scripts/setup_env.py" -w "$WORK" --check --python /absolute/env/bin/python \
  --library-dir /absolute/openmp/lib --java-home /absolute/jdk \
  --mapping-dir /absolute/mapping --log-dir "$WORK/logs"
```

带 `--workdir` 时，只有完全通过才写本项目 `config/project.json` 中的环境路径、运行库和验证记录，保留其他已有字段，不改本机公共选择。这是跨对话复用的项目运行配置，不是聊天会话配置，也不记录执行授权。未绑定的旧项目兼容读取 `~/.risk-modeling/settings.json`；不带 `--workdir` 的环境命令仍读写该公共设置。项目一旦绑定便优先使用自身环境，不受另一个项目切换公共环境影响。`--no-save` 不更新任何选择记录，但会产生诊断日志。测试用 `RISKMODEL_HOME` 隔离公共设置；正式环境不要推荐临时目录。

`run_stage.py`、参数校验读取 Parquet/Feather 表头及评估输入核对都优先使用项目环境。运行时禁用向 Skill 写入 Python 缓存，使用非交互绘图后端。执行前按上述方式核查环境，检查失败停止，不自动安装。曾经验证通过不代表解释器现在仍存在或依赖没有变化。

## 联网新建

先给用户具体方案：建议名 `riskmodel-py39`、展开后的绝对安装位置、联网下载说明及实际可行的创建方式。默认位置已存在则优先检查复用，或建议新的明确目录，不能覆盖安装。目录准备可与安装一起确认；现有离线包满足要求时可选离线创建，不把联网当成所有方案的必需条件。确认后先初始化工作目录，再带 `--workdir "$WORK" --log-dir "$WORK/logs"` 创建，通过后只绑定本项目。准备完成说明实际结果及准确数据放置路径；没有业务数据时不追加训练。

```bash
# 工作目录已按清单初始化；默认用 conda，创建 ~/.risk-modeling/envs/riskmodel-py39
python3 -B "$SKILL/scripts/setup_env.py" -w "$WORK" --create --log-dir "$WORK/logs"

# 已有原生 Python 3.9、Java、OpenMP 时，可用 venv
python3 -B "$SKILL/scripts/setup_env.py" -w "$WORK" --create --log-dir "$WORK/logs" \
  --base-python /absolute/python3.9 --prefix /absolute/riskmodel-py39 \
  --library-dir /absolute/openmp/lib
```

conda 路径使用 conda-forge 并选择本机原生子平台，环境内安装 Python 3.9、Java 17、OpenMP，再用 pip 安装固定依赖。`--conda` 可指定 conda 命令。venv 路径复用已有 Java/OpenMP，不下载系统运行库。目标目录存在时拒绝创建，绝不将安装或升级操作施加于用户已有环境；安装失败保留目录和日志，重新创建需指定新的目录。

支持 `--index-url https://mirror.example/simple`，不修改全局 pip/conda 配置。该参数只控制 pip；conda 创建仍访问 conda-forge。主要版本在 `scripts/requirements.txt`；macOS ARM64 安装用 `requirements/verified-macos-arm64-py39.txt` 的实测约束。其他目标使用 `requirements/portable-py39.txt`，不套用包含 Unix 专属包的 Mac 全量快照；它是关键依赖约束，尚非 Windows/Linux 完整实测锁文件。pip/setuptools/wheel 引导版本在 `requirements/bootstrap.txt`。

核心组合：NumPy 1.26.4、pandas 2.2.3、scikit-learn 1.6.1、LightGBM 4.5.0、AutoGluon 1.3.0、scorecardpipeline 0.1.39、toad 0.1.7、sklearn2pmml 0.110.0。已有 AutoGluon 旧模型继续使用其原环境；官方不支持跨版本直接加载，不能为接入本工具而升级旧模型环境。[AutoGluon 1.3 说明](https://auto.gluon.ai/dev/whats_new/v1.3.0.html)

## 离线包与离线安装

在与目标机系统、处理器架构和 CPython 3.9 一致的联网机器执行：

```bash
python3 -B "$SKILL/scripts/setup_env.py" --download-wheels /absolute/wheels \
  --python /absolute/native/python3.9
```

输出目录须为空或不存在。工具使用 `pip wheel` 下载或构建所有依赖及引导工具，写入含目标信息和 SHA-256 的 `manifest.json`。构建阶段可能需要网络和构建工具。把整个目录原样复制到目标机，不要混入其他 wheel。

```bash
python3 -B "$SKILL/scripts/setup_env.py" -w "$WORK" --create --log-dir "$WORK/logs" \
  --base-python /absolute/python3.9 --prefix /absolute/riskmodel-py39 \
  --wheel-dir /absolute/wheels --library-dir /absolute/openmp/lib \
  --java-home /absolute/jdk
```

离线范围是已有 Python 3.9、Java，以及可用的原生 OpenMP 运行库。包内不包含 Python/conda、Java 或系统库。先校验目标信息、清单和 wheel 内容，再创建全新 venv；安装强制 `--no-index --only-binary=:all: --no-deps --no-cache-dir`，全部依赖从已校验的本地目录读取。禁用外部 pip 配置；不接收依赖 URL，不进行源代码构建。安装后仍执行完整验收。

Mac、Linux、Windows 的离线包不能互换，需要在对应系统和架构上生成，并在目标机验收。相同系统和架构仍可能受系统版本、运行库、驱动等影响，wheel 标签检查和实际模型验收都必须通过。当前没有 Windows/Linux 完整实机通过记录。离线安装参数参见 [pip 官方文档](https://pip.pypa.io/en/stable/cli/pip_install/#install-no-index)。

## 日志与失败处理

工作目录已创建后，项目环境命令统一加 `--log-dir "$WORK/logs"`，将各次 `setup-*/report.json`、日志和模型探针产物保存在本项目。确认前发现/临时检查未创建目录时，不为放日志提前创建工作区，可用临时日志目录；不带 `--log-dir` 的 CLI 仍兼容默认 `~/.risk-modeling/logs/`。项目进度与环境绑定以本目录记录为准，不跨项目寻找其他任务日志。

运行期间每 15 秒报告耗时和日志大小。每项检查默认 180 秒，安装/下载默认 1800 秒，可分别用 `--timeout`、`--install-timeout` 调整。超时或中断会清理本次进程树（包括 Java）；Windows 使用 Job Object，POSIX 使用进程组。受限系统不允许创建 Job 或清理进程时明确失败，不改用无保护的进程启动，也不算验证成功。

失败 JSON 的 `error_kind` 区分缺包、版本冲突、处理器架构、Java/OpenMP、网络、磁盘、wheel 不匹配、超时和安装失败，并提供处理选项。不自动切换镜像、不自动升级现有环境、不删除失败环境。

两项已知边界：

- macOS ARM 的 grpcio 1.67.1 wheel 内部架构标记会导致 `pip check` 报错。仅当完整输出恰好只有这一条已知错误，且实际本机 gRPC 请求/响应通过时，作为显式警告放行；原日志和非零退出码保留。出现任何额外依赖错误均失败。
- scorecardpipeline 0.1.39 / toad 0.1.7 在一组接近均衡的合成数据上，逐步回归出现 `list.remove(x): x not in list`。本次模型探针使用坏样本占比约 20% 的固定样本；均衡数据复现保留于开发目录，Step 2 未修改后台算法，不承诺任意数据训练成功。

## 工作目录

```bash
# 确认前预览；目录尚不存在时也可读取默认参数
python3 -B "$SKILL/scripts/init_workspace.py" --workdir "$WORK" --check
python3 -B "$SKILL/scripts/make_config.py" -w "$WORK" --show data_analysis
# 确认后仅补缺失目录/模板
python3 -B "$SKILL/scripts/init_workspace.py" --workdir /absolute/my_project
```

初始化和环境命令维护项目记录，不手写它。旧版 `--label-zero/--label-one` 和 `label_meaning` 仅保留兼容，正常建模不调用、不追问，不将它们视为可改模型参数。

用户路径优先，助手按确认过的绝对路径调用。CLI 不传路径时仍保留兼容规则：当前目录同时有 `data/` 和 `config/` 则复用，否则使用其下 `riskmodel_work/`。只有 `data/` 的项目应明确指定其根目录，不靠默认解析猜。首次创建前用 `--check`，在需求清单中显示将创建/补齐的位置；用户确认后才创建。工作目录不能位于 Skill 内，也不能使用包含 Skill 的父目录。

Skill/脚本是共用程序，正常建模不修改；用 `python -B` 启动控制脚本，避免向安装目录写 Python 缓存。业务文件都写当前工作目录。用户从不同目录发起的项目互相隔离；同一父目录要做两个独立任务时，按主入口推荐不同未占用的子目录并显式传入。重新对话读取原工作目录恢复状态，不每次重建，也不把聊天窗口编号当成工作目录隔离机制。

工作目录自身是项目记录的根，不在 project.json 再复制一份绝对目录。每次从用户指定路径或当前打开项目定位，显式传 `--workdir` 后加载该根下的环境、参数和状态；定位不唯一才询问。目录迁移后不能盲用任务日志里的旧绝对产物路径，需重新核查环境和结果有效性。

| 子目录 | 用途 |
|---|---|
| `data/` | 主数据 `data.csv` / `data.parquet` / `data.feather`；时间外数据 `oot_*`；可选映射表 `mapping/` |
| `config/` | 五份建模环节配置；`project.json` 保存已验证的项目环境，旧标签字段兼容保留，不重复保存运行进度 |
| `outputs/` | 后台生成的模型、中间数据和原始报告 |
| `results/` | 五个环节同名 JSON、Markdown、HTML |
| `logs/` | 环节日志、task.json、stages/ 下的环节执行记录，以及运行锁 |

默认基础列为 `id`、`date`、`label`、`channel`；建模标签为二分类 0/1。初始化不加载全量数据，不覆盖已存在的数据、配置或结果；无效 JSON 只报告，不修复。有旧结果但没有执行记录时标为 `untracked`，不能当作已验证前置结果。`startup_marker_present` 仅表示存在旧启动标记。初始化不会启动建模。

开始分析/清洗前，执行器会拦截主数据缺失、主数据多格式并存以及清洗时同名 OOT 多格式并存；不会先清空旧结果再报这些错误。数据文件选择与标签方向要求见[交互指南](first-use.md)。记录与配置损坏只报告错误，不自动恢复成默认值。

接续时先用 `init_workspace.py --check` 读项目环境及缺失目录，已初始化项目再查 `run_stage.py --status`；只需准备的部分由实际缺项推导。`project.json` 不保存密码、完整聊天、待执行方案或确认凭证；执行进度仍在原有任务记录。项目记录损坏会阻止初始化/新运行，但状态查询返回 `project_error`，仍可停止正在运行的任务。没有旧项目记录时正常读取现有状态，不强制重建项目或模型。

## 执行、等待与停止

以下操作须在用户确认环节和参数后执行；`--check`、`--status` 是只读检查。

```bash
python3 -B "$SKILL/scripts/run_stage.py" -w "$WORK" --start data_analysis data_clean_split --check
python3 -B "$SKILL/scripts/run_stage.py" -w "$WORK" --start data_analysis data_clean_split
python3 -B "$SKILL/scripts/run_stage.py" -w "$WORK" --wait --seconds 30
python3 -B "$SKILL/scripts/run_stage.py" -w "$WORK" --status
python3 -B "$SKILL/scripts/run_stage.py" -w "$WORK" --stop
python3 -B "$SKILL/scripts/make_report.py" -w "$WORK" --stage data_analysis
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

任务在独立进程运行，会话断开后可以查询同目录接续；宿主退出或系统终止进程时可能中断，不承诺关闭整个应用后仍运行。`--wait` 默认 30 秒、最多 60 秒；它不会设置模型的运行时限。停止请求交给执行器清理本次进程树；系统拒绝进程管理时明确报错，不绕过权限或误停其他进程。

## 重跑与结果有效性

结果仍使用固定文件名，没有 run_id 或自动历史版本。重跑会清理该环节的旧 JSON/Markdown/HTML 及标准生成目录内旧产物；特征统计缓存保留，由后台按输入内容摘要核对。用户输入目录不清理。欲保留旧模型请先选择另一个工作目录。

记录各环节当前配置摘要、相关输入及上游完成时间/结果摘要。大数据与普通产物用路径、大小、纳秒修改时间检查；辅助 JSON 另加 SHA-256。它能检测正常替换、增删和修改，不承诺识别人为保持大小及时间戳不变的篡改。输入在执行过程中变化会判失败。

训练参数变化只使训练和评估失效；主数据变化从分析开始失效；外部 OOT 变化从清洗开始失效。只更改一个环节不会自动替用户重跑全部下游。

报告完整展示 JSON，后台 Top N 摘要保持原口径并提供当前原始产物链接。只生成摘要不触发模型计算，报告缺失可以单独补生成。历史/过期结果不能被重新包装成当前报告。

报告标注计算完成时间，并提醒它是当次运行的快照。静态 HTML 本身不会实时刷新结果状态；数据、参数或产物变化后，先让助手查询当前状态，再引用或分享旧报告。
