"""各脚本共用的小工具：环节列表、目录位置、读写 JSON、输出结果、读取数据列名。

只用 Python 标准库，Python 3.8+ 可运行。
"""
import csv
import json
import os
import subprocess
import sys

SKILL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELING_DIR = os.path.join(SKILL_DIR, "scripts", "modeling")
TEMPLATE_DIR = os.path.join(SKILL_DIR, "assets", "config_templates")

# 本机设置：记住用哪个 Python 环境跑建模、映射表目录（RISKMODEL_HOME 仅用于测试时改位置）
SETTINGS_PATH = os.path.join(
    os.environ.get("RISKMODEL_HOME") or os.path.join(os.path.expanduser("~"), ".risk-modeling"),
    "settings.json",
)

# 五个环节，按执行顺序排列：(环节 id, 中文名)
STAGES = [
    ("data_analysis", "数据质量分析"),
    ("data_clean_split", "数据清洗与切分"),
    ("feature_select", "特征筛选"),
    ("model_train", "模型训练"),
    ("model_eval", "模型评估"),
]
STAGE_IDS = [s for s, _ in STAGES]
STAGE_NAMES = dict(STAGES)

# 配置文件里结果 JSON 和启动标记的键名（后台代码按这些键读写）
RESULT_KEYS = {
    "data_analysis": "dq_json_path",
    "data_clean_split": "clean_json_path",
    "feature_select": "feature_json_path",
    "model_train": "model_json_path",
    "model_eval": "eval_json_path",
}
STARTUP_KEYS = {
    "data_analysis": "dq_startup_path",
    "data_clean_split": "clean_startup_path",
    "feature_select": "feature_startup_path",
    "model_train": "model_startup_path",
    "model_eval": "eval_startup_path",
}

DEFAULT_WORKDIR = "riskmodel_work"
WORKDIR_SUBDIRS = ["data", "config", "outputs", "results", "logs"]
BASE_COLUMNS = ["id", "date", "label", "channel"]
DATA_EXTS = [".parquet", ".feather", ".csv"]  # 与后台代码 data_utils.SUPPORTED_FORMATS 的查找顺序一致


def emit(obj, code=None):
    """以 JSON 输出结果并退出；ok 为 False 时退出码为 1。"""
    print(json.dumps(obj, ensure_ascii=False, indent=2))
    if code is None:
        code = 0 if obj.get("ok", True) else 1
    sys.exit(code)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)
    os.replace(tmp, path)


def is_workdir(path):
    return all(os.path.isdir(os.path.join(path, d)) for d in ("data", "config"))


def resolve_workdir(arg):
    """确定工作目录：显式指定 > 当前目录本身是工作目录 > 当前目录下的 riskmodel_work。找不到返回 None。"""
    if arg:
        return os.path.abspath(os.path.expanduser(arg))
    cwd = os.getcwd()
    if is_workdir(cwd):
        return cwd
    default = os.path.join(cwd, DEFAULT_WORKDIR)
    if is_workdir(default):
        return default
    return None


def config_path(workdir, stage):
    return os.path.join(workdir, "config", "%s_config.json" % stage)


def template_path(stage):
    return os.path.join(TEMPLATE_DIR, "%s_config.json" % stage)


def load_settings():
    if os.path.exists(SETTINGS_PATH):
        try:
            return load_json(SETTINGS_PATH)
        except (ValueError, OSError):
            return {}
    return {}


def find_data_file(directory, base):
    """按 parquet > feather > csv 的顺序查找 <directory>/<base>.<ext>，找不到返回 None。"""
    for ext in DATA_EXTS:
        p = os.path.join(directory, base + ext)
        if os.path.exists(p):
            return p
    return None


def list_oot_names(workdir):
    """data/ 下原始 OOT 文件名（不含扩展名），如 oot_main、oot_1。"""
    names = set()
    data_dir = os.path.join(workdir, "data")
    if os.path.isdir(data_dir):
        for fn in os.listdir(data_dir):
            base, ext = os.path.splitext(fn)
            if base.startswith("oot_") and ext in DATA_EXTS:
                names.add(base)
    return sorted(names)


def read_columns(path):
    """读取数据文件的列名。CSV 直接读表头；parquet/feather 借用建模环境的 pyarrow。读不到返回 None。"""
    if path.endswith(".csv"):
        for enc in ("utf-8-sig", "gbk"):
            try:
                with open(path, "r", encoding=enc, newline="") as f:
                    return next(csv.reader(f))
            except (UnicodeDecodeError, StopIteration):
                continue
        return None
    python = load_settings().get("python")
    if not python or not os.path.exists(python):
        return None
    code = (
        "import json,sys\n"
        "p=sys.argv[1]\n"
        "if p.endswith('.parquet'):\n"
        "    import pyarrow.parquet as pq; names=pq.read_schema(p).names\n"
        "else:\n"
        "    import pyarrow as pa\n"
        "    try:\n"
        "        names=pa.ipc.open_file(p).schema.names\n"
        "    except Exception:\n"
        "        import pyarrow.feather as pf; names=pf.read_table(p).schema.names\n"
        "print(json.dumps(names))\n"
    )
    try:
        out = subprocess.run([python, "-c", code, path], capture_output=True, text=True, timeout=120)
        if out.returncode == 0:
            return json.loads(out.stdout.strip().splitlines()[-1])
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    return None
