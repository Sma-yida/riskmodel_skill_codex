"""Render complete backend JSON into offline Markdown and HTML; never recompute metrics."""
import argparse
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import tempfile
from urllib.parse import quote

from common import STAGE_IDS, emit, resolve_workdir, save_json
from execution_state import (TaskError, file_stamp, mutation_lock, read_record, receipt_path,
                             result_path, stage_statuses, validate_result, within, work_root)


def scalar(value):
    if value is None:
        return "未提供（null）"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, allow_nan=True)
    return str(value)


def md_text(value):
    text = html.escape(scalar(value), quote=False).replace("\\", "\\\\")
    for char in ("|", "*", "`", "_", "[", "]", "#"):
        text = text.replace(char, "\\" + char)
    return text.replace("\n", "<br>")


def table_rows(value):
    if isinstance(value, list) and value and all(isinstance(x, dict) for x in value):
        keys = list(dict.fromkeys(k for row in value for k in row))
        if keys:
            return keys, [[row[k] if k in row else "未提供" for k in keys] for row in value]
    return None


def markdown(value, depth=2):
    table = table_rows(value)
    if table:
        keys, rows = table
        return "\n".join(["| " + " | ".join(map(md_text, keys)) + " |",
                          "| " + " | ".join("---" for _ in keys) + " |"] +
                         ["| " + " | ".join(map(md_text, row)) + " |" for row in rows]) + "\n"
    if isinstance(value, dict) and value:
        return "\n\n".join(("#" * min(depth, 6) + " " + md_text(k) + "\n\n" + markdown(v, depth + 1))
                            if isinstance(v, (dict, list)) else ("**" + md_text(k) + "：** " + md_text(v))
                            for k, v in value.items())
    if isinstance(value, list) and value:
        return "\n\n".join("- " + md_text(v) if not isinstance(v, (dict, list)) else markdown(v, depth + 1) for v in value)
    return md_text(value) + "\n"


def html_body(value, depth=2):
    table = table_rows(value)
    if table:
        keys, rows = table
        return ('<div class="table"><table><thead><tr>' +
                ''.join('<th scope="col">' + html.escape(str(k)) + '</th>' for k in keys) +
                '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + html.escape(scalar(v)) + '</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table></div>')
    if isinstance(value, dict) and value:
        level = min(depth, 6)
        return ''.join(('<section><h{n}>{k}</h{n}>{v}</section>'.format(n=level, k=html.escape(str(k)), v=html_body(v, depth + 1)))
                       if isinstance(v, (dict, list)) else ('<p><strong>' + html.escape(str(k)) + '：</strong> ' + html.escape(scalar(v)) + '</p>')
                       for k, v in value.items())
    if isinstance(value, list) and value:
        return '<ul>' + ''.join('<li>' + html_body(v, depth + 1) + '</li>' for v in value) + '</ul>'
    return '<p>' + html.escape(scalar(value)) + '</p>'


def atomic_text(path, content):
    # Keep existing report intact if rendering or writing fails.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=str(path.parent), delete=False) as stream:
        tmp = stream.name
        try:
            stream.write(content)
        except BaseException:
            os.unlink(tmp)
            raise
    try:
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def generate(root, stage):
    status = stage_statuses(root)[stage]
    if status["state"] != "succeeded":
        raise TaskError("result_not_current", "只能为已成功且未过期的结果生成报告，当前状态：" + status["state"])
    data = validate_result(result_path(root, stage), stage)
    record = read_record(receipt_path(root, stage))
    title = data["报告标题"]
    content = {k: v for k, v in data.items() if k != "报告标题"}
    links = [(stage + ".json", "原始结果 JSON")]
    artifacts = dict(record.get("artifacts", {}))
    if stage == "model_eval":
        artifacts.update(read_record(receipt_path(root, "model_train")).get("artifacts", {}))
    for name, stamp in artifacts.items():
        p = within(root, name)
        if p.is_file() and file_stamp(p) == stamp:
            links.append(("../" + name, name))
    note = "本报告完整呈现后台 JSON；其中标明的 Top N 或部分展示保留原口径。完整明细见下方现有产物，未重算指标。"
    completed = record.get("completed_at")
    snapshot = "本报告为当次运行快照；修改数据或参数、重跑项目后，请通过助手确认结果仍适用。"
    if completed is not None:
        completed_text = datetime.fromtimestamp(completed, timezone.utc).astimezone().isoformat(timespec="seconds")
        snapshot = "计算完成时间：" + completed_text + "。" + snapshot
    md = "# " + md_text(title) + "\n\n" + note + "\n\n" + md_text(snapshot) + "\n\n" + markdown(content)
    md += "\n\n## 结果与明细文件\n\n" + "\n".join("- [" + md_text(label) + "](" + quote(url, safe="/.") + ")" for url, label in links) + "\n"
    body = ('<h1>' + html.escape(title) + '</h1><p class="note">' + note + '</p><p>'
            + html.escape(snapshot) + '</p>' + html_body(content))
    body += '<section><h2>结果与明细文件</h2><ul>' + ''.join('<li><a href="' + quote(url, safe="/.") + '">' + html.escape(label) + '</a></li>' for url, label in links) + '</ul></section>'
    page = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>''' + html.escape(title) + '''</title><style>
body{max-width:1120px;margin:32px auto;padding:0 24px;font:16px/1.7 system-ui,-apple-system,"PingFang SC",sans-serif;color:#172b3a;background:#fff}h1{font-size:30px}h2{margin-top:32px;border-bottom:1px solid #dbe5ec;padding-bottom:8px}h3{font-size:19px}p,li,td{overflow-wrap:anywhere;white-space:pre-wrap}li p{margin:2px 0}.note{background:#f0f5f9;padding:16px;border-radius:8px}.table{overflow-x:auto;margin:16px 0}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #dbe5ec;padding:9px 12px;text-align:left;vertical-align:top}th{background:#eef4f7}tr:nth-child(even){background:#f9fbfc}a{color:#145f91}@media print{body{max-width:none;margin:0;padding:0;font-size:10pt}.table{overflow:visible}thead{display:table-header-group}tr{break-inside:avoid}h2,h3{break-after:avoid}}
</style></head><body>''' + body + '</body></html>'
    paths = {ext: str(within(root, "results/" + stage + "." + ext)) for ext in ("md", "html")}
    atomic_text(Path(paths["md"]), md)
    atomic_text(Path(paths["html"]), page)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-w", "--workdir")
    parser.add_argument("--stage", required=True, choices=STAGE_IDS)
    args = parser.parse_args()
    try:
        root = work_root(resolve_workdir(args.workdir) or "riskmodel_work")
        with mutation_lock(root):
            reports = generate(root, args.stage)
            record = read_record(receipt_path(root, args.stage))
            record.update(reports=reports, report_error=None)
            save_json(str(receipt_path(root, args.stage)), record)
            task = read_record(within(root, "logs/task.json"))
            if task.get("status") == "report_failed" and task.get("stage") == args.stage:
                remaining = task["stages"][task["stages"].index(args.stage) + 1:]
                task.update(status="ready" if remaining else "succeeded", remaining_stages=remaining, error=None)
                save_json(str(within(root, "logs/task.json")), task)
        emit({"ok": True, "stage": args.stage, "reports": reports})
    except (TaskError, OSError, ValueError, KeyError) as exc:
        emit({"ok": False, "error_kind": getattr(exc, "kind", "report_failed"), "error": str(exc)})


if __name__ == "__main__":
    main()
